# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wan-VAE: Video Variational Autoencoder for Wan2.1-T2V-1.3B.

This module provides a JAX/Flax implementation of the Wan-VAE decoder,
which converts latent representations to RGB video frames.

Architecture (based on reference implementation):
- Denormalization with learned mean/std
- Frame-by-frame decoding with temporal upsampling
- CausalConv3d for temporal coherence
- Spatial upsampling from 60x60 to 832x480
- Temporal upsampling from 21 to 81 frames
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array


class CausalConv3d(nnx.Module):
    """Causal 3D convolution that doesn't look into the future."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int, int] = (3, 3, 3), *, rngs: nnx.Rngs
    ):
        self.kernel_size = kernel_size
        # Causal padding: pad past frames, no future frames
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            padding="VALID",  # We'll handle padding manually
            rngs=rngs,
        )

    def __call__(self, x: Array) -> Array:
        # x: [B, T, H, W, C] (JAX channel-last format)
        # Causal padding: (kernel_t - 1, 0) for temporal, symmetric for spatial
        kt, kh, kw = self.kernel_size
        # Pad: (batch, temporal_past, temporal_future, height_before, height_after, width_before, width_after, channel)
        padding = ((0, 0), (kt - 1, 0), (kh // 2, kh // 2), (kw // 2, kw // 2), (0, 0))
        x_padded = jnp.pad(x, padding, mode="constant")
        out = self.conv(x_padded)
        return out


class RMSNorm(nnx.Module):
    """RMS Normalization."""

    def __init__(self, channels: int, eps: float = 1e-6, *, rngs: nnx.Rngs):
        self.eps = eps
        self.scale = nnx.Param(jnp.ones((1, 1, 1, 1, channels)))

    def __call__(self, x: Array) -> Array:
        # x: [B, T, H, W, C]
        rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return x / rms * self.scale.value


class ResidualBlock(nnx.Module):
    """Residual block with GroupNorm and SiLU activation."""

    def __init__(self, in_channels: int, out_channels: int, *, rngs: nnx.Rngs):
        self.norm1 = nnx.GroupNorm(num_groups=32, num_features=in_channels, rngs=rngs)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=(3, 3, 3), rngs=rngs)
        self.norm2 = nnx.GroupNorm(num_groups=32, num_features=out_channels, rngs=rngs)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=(3, 3, 3), rngs=rngs)

        if in_channels != out_channels:
            self.skip_conv = CausalConv3d(in_channels, out_channels, kernel_size=(1, 1, 1), rngs=rngs)
        else:
            self.skip_conv = None

    def __call__(self, x: Array) -> Array:
        # x: [B, T, H, W, C] - already in JAX format
        residual = x

        x = self.norm1(x)
        x = nnx.silu(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = nnx.silu(x)
        x = self.conv2(x)

        if self.skip_conv is not None:
            residual = self.skip_conv(residual)

        return x + residual


class AttentionBlock(nnx.Module):
    """Spatial attention block."""

    def __init__(self, channels: int, *, rngs: nnx.Rngs):
        self.norm = nnx.GroupNorm(num_groups=32, num_features=channels, rngs=rngs)
        self.qkv = nnx.Conv(
            in_features=channels, out_features=channels * 3, kernel_size=(1, 1), use_bias=True, rngs=rngs
        )
        self.proj = nnx.Conv(in_features=channels, out_features=channels, kernel_size=(1, 1), use_bias=True, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        # x: [B, T, H, W, C]
        b, t, h, w, c = x.shape
        residual = x

        # Process each frame independently
        x_frames = []
        for i in range(t):
            frame = x[:, i, :, :, :]  # [B, H, W, C]
            frame = self.norm(frame)

            # QKV projection
            qkv = self.qkv(frame)  # [B, H, W, 3*C]
            q, k, v = jnp.split(qkv, 3, axis=-1)

            # Reshape for attention
            q = q.reshape(b, h * w, c)
            k = k.reshape(b, h * w, c)
            v = v.reshape(b, h * w, c)

            # Scaled dot-product attention
            scale = c**-0.5
            attn = jax.nn.softmax(jnp.einsum("bic,bjc->bij", q, k) * scale, axis=-1)
            out = jnp.einsum("bij,bjc->bic", attn, v)

            # Reshape back
            out = out.reshape(b, h, w, c)
            out = self.proj(out)
            x_frames.append(out)

        x = jnp.stack(x_frames, axis=1)  # [B, T, H, W, C]
        return x + residual


class Upsample2D(nnx.Module):
    """2D spatial upsampling (no temporal upsampling)."""

    def __init__(self, channels: int, *, rngs: nnx.Rngs):
        self.conv = nnx.Conv(in_features=channels, out_features=channels, kernel_size=(3, 3), padding=1, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        # x: [B, T, H, W, C]
        b, t, h, w, c = x.shape
        # Upsample spatially only (nearest neighbor)
        frames = []
        for i in range(t):
            frame = x[:, i, :, :, :]  # [B, H, W, C]
            # Upsample 2x
            frame_up = jax.image.resize(frame, (b, h * 2, w * 2, c), method="nearest")
            # Refine with convolution
            frame_up = self.conv(frame_up)
            frames.append(frame_up)
        return jnp.stack(frames, axis=1)  # [B, T, H*2, W*2, C]


class Upsample3D(nnx.Module):
    """3D spatio-temporal upsampling."""

    def __init__(self, channels: int, *, rngs: nnx.Rngs):
        self.conv = CausalConv3d(channels, channels, kernel_size=(3, 3, 3), rngs=rngs)

    def __call__(self, x: Array) -> Array:
        # x: [B, T, H, W, C]
        b, t, h, w, c = x.shape
        # Upsample temporally (2x) and spatially (2x) - nearest neighbor
        x = jax.image.resize(x, (b, t * 2, h * 2, w * 2, c), method="nearest")
        # Refine with convolution
        x = self.conv(x)
        return x


class Decoder3D(nnx.Module):
    """
    3D Decoder matching reference implementation.
    Upsamples from [B, 1, 104, 60, 16] -> [B, 4, 832, 480, 3] (JAX format)
    """

    def __init__(self, *, rngs: nnx.Rngs):
        # Initial convolution: 16 -> 384
        self.conv_in = CausalConv3d(16, 384, kernel_size=(3, 3, 3), rngs=rngs)

        # Middle blocks (at lowest resolution)
        self.mid_block1 = ResidualBlock(384, 384, rngs=rngs)
        self.mid_attn = AttentionBlock(384, rngs=rngs)
        self.mid_block2 = ResidualBlock(384, 384, rngs=rngs)

        # Upsample stages
        # Stage 0: 384 -> 384, spatio-temporal (2x), [1, 104, 60] -> [2, 208, 120]
        self.up_blocks_0 = nnx.List(
            [
                ResidualBlock(384, 384, rngs=rngs),
                ResidualBlock(384, 384, rngs=rngs),
                ResidualBlock(384, 384, rngs=rngs),
            ]
        )
        self.up_sample_0 = Upsample3D(384, rngs=rngs)

        # Stage 1: 384 -> 192, spatio-temporal (2x), [2, 208, 120] -> [4, 416, 240]
        self.up_blocks_1 = nnx.List(
            [
                ResidualBlock(384, 192, rngs=rngs),
                ResidualBlock(192, 192, rngs=rngs),
                ResidualBlock(192, 192, rngs=rngs),
            ]
        )
        self.up_sample_1 = Upsample3D(192, rngs=rngs)

        # Stage 2: 192 -> 96, spatial only (2x), [4, 416, 240] -> [4, 832, 480]
        self.up_blocks_2 = nnx.List(
            [
                ResidualBlock(192, 96, rngs=rngs),
                ResidualBlock(96, 96, rngs=rngs),
                ResidualBlock(96, 96, rngs=rngs),
            ]
        )
        self.up_sample_2 = Upsample2D(96, rngs=rngs)

        # Stage 3: 96 -> 96, no upsample
        self.up_blocks_3 = nnx.List(
            [
                ResidualBlock(96, 96, rngs=rngs),
                ResidualBlock(96, 96, rngs=rngs),
                ResidualBlock(96, 96, rngs=rngs),
            ]
        )

        # Output head: 96 -> 3
        self.norm_out = RMSNorm(96, rngs=rngs)
        self.conv_out = CausalConv3d(96, 3, kernel_size=(3, 3, 3), rngs=rngs)

    def __call__(self, z: Array) -> Array:
        """
        Args:
            z: [B, T, H, W, C] latent (e.g., [1, 1, 104, 60, 16])
        Returns:
            x: [B, T_out, H_out, W_out, 3] RGB video (e.g., [1, 4, 832, 480, 3])
        """
        # Initial convolution
        x = self.conv_in(z)

        # Middle blocks
        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        # Upsample stage 0
        for block in self.up_blocks_0:
            x = block(x)
        x = self.up_sample_0(x)

        # Upsample stage 1
        for block in self.up_blocks_1:
            x = block(x)
        x = self.up_sample_1(x)

        # Upsample stage 2
        for block in self.up_blocks_2:
            x = block(x)
        x = self.up_sample_2(x)

        # Upsample stage 3 (no spatial upsample)
        for block in self.up_blocks_3:
            x = block(x)

        # Output
        x = self.norm_out(x)
        x = nnx.silu(x)
        x = self.conv_out(x)

        return x


class WanVAEDecoder(nnx.Module):
    """
    Wan-VAE Decoder: Converts video latents to RGB frames.

    Architecture matches reference (wan/modules/vae.py:544-568):
    1. Denormalize latents with learned mean/std
    2. Conv 1x1 projection (16 -> 16 channels)
    3. Frame-by-frame decode with Decoder3D
    4. Concatenate and clamp output

    Input: [B, T, H, W, C] = [1, 21, 104, 60, 16]
    Output: [B, T_out, H_out, W_out, 3] = [1, 81, 832, 480, 3]
    """

    def __init__(self, *, rngs: nnx.Rngs):
        # Learned denormalization parameters (channel-last)
        self.latent_mean = nnx.Param(jnp.zeros((1, 1, 1, 1, 16)))
        self.latent_std = nnx.Param(jnp.ones((1, 1, 1, 1, 16)))

        # 1x1 conv projection
        self.conv2 = CausalConv3d(16, 16, kernel_size=(1, 1, 1), rngs=rngs)

        # 3D decoder
        self.decoder = Decoder3D(rngs=rngs)

    @jax.jit
    def decode(self, latents: Array) -> Array:
        """
        Decode latents to RGB video.

        Args:
            latents: [B, T, H, W, C] latent representation (JAX format)
                    e.g., [1, 21, 104, 60, 16]

        Returns:
            video: [B, T_out, H_out, W_out, 3] RGB video (values in [-1, 1])
                  e.g., [1, 81, 832, 480, 3]
        """
        # Step 1: Denormalize
        z = latents * self.latent_std.value + self.latent_mean.value

        # Step 2: Conv 1x1
        z = self.conv2(z)

        # Step 3: Frame-by-frame decode
        _b, t, _h, _w, _c = z.shape
        frames = []
        for i in range(t):
            # Extract single frame: [B, 1, H, W, C]
            frame_latent = z[:, i : i + 1, :, :, :]
            # Decode: [B, 1, H, W, C] -> [B, 4, H_out, W_out, 3]
            frame_out = self.decoder(frame_latent)
            frames.append(frame_out)

        # Concatenate along time dimension
        x = jnp.concatenate(frames, axis=1)  # [B, T_total, H_out, W_out, 3]

        # Clamp to [-1, 1]
        x = jnp.clip(x, -1.0, 1.0)

        return x


def load_vae_from_checkpoint(checkpoint_path: str, rngs: nnx.Rngs) -> WanVAEDecoder:
    """
    Load Wan-VAE decoder from checkpoint.

    Args:
        checkpoint_path: Path to the VAE checkpoint directory
        rngs: Random number generators for initialization

    Returns:
        WanVAEDecoder with loaded weights
    """
    # Create VAE decoder structure
    vae_decoder = WanVAEDecoder(rngs=rngs)

    # TODO: Implement checkpoint loading
    # This will require mapping PyTorch VAE weights to JAX
    # Similar to what's done in params.py for the DiT model

    print("Warning: VAE checkpoint loading not yet implemented")
    print("Returning VAE with random weights")

    return vae_decoder


def decode_latents_to_video(vae_decoder: WanVAEDecoder, latents: Array, normalize: bool = True) -> Array:
    """
    Helper function to decode latents and post-process to video.

    Args:
        vae_decoder: WanVAEDecoder instance
        latents: [B, T, H, W, C] latent representation
        normalize: If True, normalize output from [-1, 1] to [0, 255] uint8

    Returns:
        video: [B, T, H_out, W_out, 3] video frames
    """
    # Decode
    video = vae_decoder.decode(latents)

    if normalize:
        # Convert from [-1, 1] to [0, 255] uint8
        video = (video + 1.0) * 127.5
        video = jnp.clip(video, 0, 255).astype(jnp.uint8)

    return video


__all__ = ["WanVAEDecoder", "decode_latents_to_video", "load_vae_from_checkpoint"]
