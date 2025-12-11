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

from dataclasses import dataclass
from typing import Tuple

import imageio
import jax
import jax.numpy as jnp
from flax import nnx
from jax.lax import Precision
from jaxtyping import Array, Union

CACHE_T = 2


@dataclass
class VAEConfig:
    """Configuration for Wan-VAE decoder.

    Latent denormalization constants from reference implementation.
    These are fixed constants computed during VAE training.
    """

    latent_mean: Tuple[float, ...] = (
        -0.7571,
        -0.7089,
        -0.9113,
        0.1075,
        -0.1745,
        0.9653,
        -0.1517,
        1.5508,
        0.4134,
        -0.0715,
        0.5517,
        -0.3632,
        -0.1922,
        -0.9497,
        0.2503,
        -0.2921,
    )

    latent_std: Tuple[float, ...] = (
        2.8184,
        1.4541,
        2.3275,
        2.6558,
        1.2196,
        1.7708,
        2.6052,
        2.0743,
        3.2687,
        2.1526,
        2.8652,
        1.5579,
        1.6382,
        1.1253,
        2.8251,
        1.9160,
    )


class CausalConv3d(nnx.Module):
    """Causal 3D convolution that doesn't look into the future."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        *,
        rngs: nnx.Rngs,
        padding: Tuple[int, int, int] = (0, 0, 0),
    ):
        self.kernel_size = kernel_size
        self.temporal_padding = padding[0]  # Save for cache size calculation
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            padding="VALID",  # We'll handle padding manually
            rngs=rngs,
            precision=Precision.HIGHEST,
        )
        self.padding = (
            (0, 0),
            (2 * padding[0], 0),
            (padding[1], padding[1]),
            (padding[2], padding[2]),
            (0, 0),
        )

    def __call__(self, x: Array, cache: Array | None = None) -> tuple[Array, Array | None]:
        """Forward pass with optional caching.

        Args:
            x: [B, T, H, W, C] input (JAX channel-last format)
            cache: [B, CACHE_T, H, W, C] cached frames from previous call, or None

        Returns:
            out: [B, T_out, H_out, W_out, C_out] output
            new_cache: [B, CACHE_T, H, W, C] cache for next call, or None
        """
        # Cache size is 2*padding because we pad left by (2*padding, 0) for causality
        cache_t = 2 * self.temporal_padding
        if cache is not None and cache_t > 0:
            x = jnp.concatenate([cache, x], axis=1)  # [B, T+CACHE_T, H, W, C]
            # jax.debug.print("feat cache in causalconv3d:{},{}", cache.shape, x.shape)
            padding = list(self.padding)
            padding[1] = (max(0, self.padding[1][0] - cache.shape[1]), 0)  # Reduce left padding
            padding = tuple(padding)
        else:
            padding = self.padding

        x_padded = jnp.pad(x, padding, mode="constant")
        out = self.conv(x_padded)

        # Extract cache for next iteration: last cache_t frames of INPUT (before conv)
        # Always create cache if we have temporal padding (even on first frame)
        if cache_t > 0:
            new_cache = x[:, -cache_t:, :, :, :]  # [B, <=CACHE_T, H, W, C]
            # Pad on the left if we do not yet have cache_t frames (e.g., first call with T=1).
            if new_cache.shape[1] < cache_t:
                pad_t = cache_t - new_cache.shape[1]
                new_cache = jnp.pad(new_cache, ((0, 0), (pad_t, 0), (0, 0), (0, 0), (0, 0)), mode="constant")
        else:
            new_cache = None

        return out, new_cache


class RMSNorm(nnx.Module):
    """RMS Normalization with L2 normalize and learned scale.

    Based on F.normalize approach: normalize to unit norm, then scale.
    For videos (images=False), uses 3D spatial+temporal normalization.
    """

    def __init__(self, dim: int, *, rngs: nnx.Rngs):
        self.scale_factor = dim**0.5
        # gamma shape: (dim,) will broadcast to [B, T, H, W, C] or [B, H, W, C]
        self.scale = nnx.Param(jnp.ones(dim))
        self.eps = 1e-12

    def __call__(self, x: Array) -> Array:
        # x: [B, T, H, W, C] for 3D or [B, H, W, C] for 2D
        # Normalize to unit RMS along the channel dimension manually since jax.nn.normalize is unavailable.
        rms = jnp.sqrt(jnp.sum(jnp.square(x), axis=-1, keepdims=True))
        rms = jnp.maximum(rms, self.eps)
        if jnp.isnan(x).any():
            nan_mask = jnp.isnan(x)
            nan_indices = jnp.argwhere(nan_mask)
            jax.debug.print("NaN at indices: {}", nan_indices[:10])
        x_normalized = x / rms
        if jnp.isnan(x_normalized).any():
            nan_mask = jnp.isnan(x_normalized)
            nan_indices = jnp.argwhere(nan_mask)
            jax.debug.print("NaN at indices: {}", nan_indices[:10])
            jax.debug.print("rms values: {}", rms.mean())
        # jax.debug.print("x_normalized has nan: {}", jnp.isnan(x_normalized).any())
        # jax.debug.print("scale values: {} {}", self.scale_factor, self.scale.value.mean())
        return x_normalized * self.scale_factor * self.scale.value


class ResidualBlock(nnx.Module):
    """Residual block with RMSNorm and SiLU activation."""

    def __init__(self, in_channels: int, out_channels: int, *, rngs: nnx.Rngs):
        self.norm1 = RMSNorm(in_channels, rngs=rngs)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1), rngs=rngs)
        self.norm2 = RMSNorm(out_channels, rngs=rngs)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1), rngs=rngs)

        if in_channels != out_channels:
            self.skip_conv = CausalConv3d(in_channels, out_channels, kernel_size=(1, 1, 1), rngs=rngs)
        else:
            self.skip_conv = None

    def __call__(
        self, x: Array, cache_list: tuple[Array | None, ...] | None = None, cache_idx: list[int] | None = None
    ) -> tuple[Array, tuple[Array | None, ...] | None]:
        residual = x
        x = self.norm1(x)
        # if self.skip_conv is not None:
        # jax.debug.print("Residual block norm1 output has nan:{}", jnp.isnan(x).any())
        x = nnx.silu(x)
        # if self.skip_conv is not None:
        # jax.debug.print("Residual block activation output has nan:{}", jnp.isnan(x).any())

        if cache_list is not None:
            idx = cache_idx[0]
            x, new_cache = self.conv1(x, cache_list[idx])
            cache_list = (*cache_list[:idx], new_cache, *cache_list[idx + 1 :])
            cache_idx[0] += 1
            # if self.skip_conv is not None:
            # jax.debug.print("Residual block conv1 output has nan:{}", jnp.isnan(x).any())
        else:
            x, _ = self.conv1(x, None)
            # if self.skip_conv is not None:
            # jax.debug.print("no cache: Residual block conv1 output has nan:{}", jnp.isnan(x).any())

        x = self.norm2(x)
        # if self.skip_conv is not None:
        # jax.debug.print("Residual block norm2 output has nan:{}", jnp.isnan(x).any())
        x = nnx.silu(x)
        # if self.skip_conv is not None:
        # jax.debug.print("Residual block activation2 output has nan:{}", jnp.isnan(x).any())

        if cache_list is not None:
            idx = cache_idx[0]
            x, new_cache = self.conv2(x, cache_list[idx])
            cache_list = (*cache_list[:idx], new_cache, *cache_list[idx + 1 :])
            cache_idx[0] += 1
            # if self.skip_conv is not None:
            # jax.debug.print("Residual block conv2 output has nan:{}", jnp.isnan(x).any())
        else:
            x, _ = self.conv2(x, None)
            # if self.skip_conv is not None:
            # jax.debug.print("no cache: Residual block conv2 output has nan:{}", jnp.isnan(x).any())

        if self.skip_conv is not None:
            residual, _ = self.skip_conv(residual, None)
            # jax.debug.print("Residual conv output has nan:{}", jnp.isnan(residual).any())

        # jax.debug.print("Residual block output has nan:{}", jnp.isnan(x).any())

        return x + residual, cache_list


class AttentionBlock(nnx.Module):
    """Spatial attention block with batched frame processing."""

    def __init__(self, channels: int, *, rngs: nnx.Rngs):
        self.norm = RMSNorm(channels, rngs=rngs)
        self.qkv = nnx.Conv(
            in_features=channels,
            out_features=channels * 3,
            kernel_size=(1, 1),
            use_bias=True,
            rngs=rngs,
            precision=Precision.HIGHEST,
        )
        self.proj = nnx.Conv(
            in_features=channels,
            out_features=channels,
            kernel_size=(1, 1),
            use_bias=True,
            rngs=rngs,
            precision=Precision.HIGHEST,
        )

    def __call__(self, x: Array) -> Array:
        # x: [B, T, H, W, C]
        b, t, h, w, c = x.shape
        residual = x

        x = x.reshape(b * t, h, w, c)
        x = self.norm(x)
        # QKV projection: [B*T, H, W, C] -> [B*T, H, W, 3*C]
        qkv = self.qkv(x)

        # Reshape for attention: [B*T, H, W, 3*C] -> [B*T, H*W, 3*C] -> split to Q, K, V
        qkv = qkv.reshape(b * t, h * w, 3 * c)
        q, k, v = jnp.split(qkv, 3, axis=-1)  # Each: [B*T, H*W, C]

        # Scaled dot-product attention
        scale = c**-0.5
        attn = jax.nn.softmax(jnp.einsum("bic,bjc->bij", q, k) * scale, axis=-1)  # [B*T, H*W, H*W]
        out = jnp.einsum("bij,bjc->bic", attn, v)  # [B*T, H*W, C]

        # Reshape back to spatial: [B*T, H*W, C] -> [B*T, H, W, C]
        out = out.reshape(b * t, h, w, c)

        # Output projection
        out = self.proj(out)

        # Reshape back to video: [B*T, H, W, C] -> [B, T, H, W, C]
        out = out.reshape(b, t, h, w, c)

        return out + residual


class Upsample2D(nnx.Module):
    """Spatial 2x upsample that also halves channels, mirroring torch Resample."""

    def __init__(self, in_channels: int, out_channels: int, *, rngs: nnx.Rngs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            padding=1,
            rngs=rngs,
            precision=Precision.HIGHEST,
        )

    def __call__(self, x: Array) -> Array:
        # x: [B, T, H, W, Cin]
        b, t, h, w, _ = x.shape
        orig_dtype = x.dtype
        x = x.reshape(b * t, h, w, self.in_channels)
        x = jax.image.resize(x.astype(jnp.float32), (b * t, h * 2, w * 2, self.in_channels), method="nearest").astype(
            orig_dtype
        )
        x = self.conv(x)
        return x.reshape(b, t, h * 2, w * 2, self.out_channels)


class Upsample3D(nnx.Module):
    """Temporal+spatial 2x upsample with channel reduction (like torch Resample)."""

    def __init__(self, in_channels: int, out_channels: int, *, rngs: nnx.Rngs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_conv = CausalConv3d(in_channels, in_channels * 2, kernel_size=(3, 1, 1), padding=(1, 0, 0), rngs=rngs)
        self.spatial_conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            padding=1,
            rngs=rngs,
            precision=Precision.HIGHEST,
        )

    def __call__(
        self, x: Array, cache_list: tuple[Array | None, ...] | None = None, cache_idx: list[int] | None = None
    ) -> tuple[Array, tuple[Array | None, ...] | None]:
        b, t, h, w, _ = x.shape
        orig_dtype = x.dtype

        if cache_list is not None:
            idx = cache_idx[0]

            # First frame: skip time_conv, only do spatial upsampling
            if cache_list[idx] is None:
                # Use zero array as sentinel with SAME shape as real cache
                # This ensures consistent pytree structure for JIT
                # We use zeros with shape [B, 2, H, W, C] where 2 = cache size for 3x1x1 conv
                sentinel = jnp.zeros((b, 2, h, w, self.in_channels), dtype=orig_dtype)
                cache_list = (*cache_list[:idx], sentinel, *cache_list[idx + 1 :])
                cache_idx[0] += 1
                t_out = t
            else:
                # Always pass the cached features (including the zero sentinel) so the
                # time_conv sees a length-2 cache and returns a length-2 cache, matching
                # the torch behavior where the sentinel seeds the cache.
                x, new_cache = self.time_conv(x, cache_list[idx])

                cache_list = (*cache_list[:idx], new_cache, *cache_list[idx + 1 :])
                cache_idx[0] += 1

                x = x.reshape(b, t, h, w, 2, self.in_channels)
                x = jnp.moveaxis(x, 4, 2)  # [B, T, 2, H, W, Cin] -> [B, 2, T, H, W, Cin]
                t_out = t * 2
                x = x.reshape(b, t_out, h, w, self.in_channels)

        # Spatial upsampling (always applied)
        bt = b * t_out
        x = x.reshape(bt, h, w, self.in_channels)
        x = jax.image.resize(x.astype(jnp.float32), (bt, h * 2, w * 2, self.in_channels), method="nearest").astype(
            orig_dtype
        )
        x = self.spatial_conv(x)
        return x.reshape(b, t_out, h * 2, w * 2, self.out_channels), cache_list


class Decoder3D(nnx.Module):
    """
    3D Decoder matching reference implementation.
    Upsamples from [B, 1, 104, 60, 16] -> [B, 4, 832, 480, 3] (JAX format)
    """

    def __init__(self, *, rngs: nnx.Rngs):
        # Initial convolution: 16 -> 384
        self.conv_in = CausalConv3d(16, 384, kernel_size=(3, 3, 3), rngs=rngs, padding=(1, 1, 1))

        # Middle blocks (at lowest resolution)
        self.mid_block1 = ResidualBlock(384, 384, rngs=rngs)
        self.mid_attn = AttentionBlock(384, rngs=rngs)
        self.mid_block2 = ResidualBlock(384, 384, rngs=rngs)

        # Upsample stages (match torch checkpoint shapes)
        # Stage 0: stay at 384, then upsample to 192 channels
        self.up_blocks_0 = nnx.List(
            [
                ResidualBlock(384, 384, rngs=rngs),
                ResidualBlock(384, 384, rngs=rngs),
                ResidualBlock(384, 384, rngs=rngs),
            ]
        )
        self.up_sample_0 = Upsample3D(384, 192, rngs=rngs)

        # Stage 1: 192 -> 384 (first block), remain at 384, then upsample to 192
        self.up_blocks_1 = nnx.List(
            [
                ResidualBlock(192, 384, rngs=rngs),
                ResidualBlock(384, 384, rngs=rngs),
                ResidualBlock(384, 384, rngs=rngs),
            ]
        )
        self.up_sample_1 = Upsample3D(384, 192, rngs=rngs)

        # Stage 2: stay at 192, then spatial-only upsample to 96
        self.up_blocks_2 = nnx.List(
            [
                ResidualBlock(192, 192, rngs=rngs),
                ResidualBlock(192, 192, rngs=rngs),
                ResidualBlock(192, 192, rngs=rngs),
            ]
        )
        self.up_sample_2 = Upsample2D(192, 96, rngs=rngs)

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
        self.conv_out = CausalConv3d(96, 3, kernel_size=(3, 3, 3), padding=(1, 1, 1), rngs=rngs)

    def __call__(
        self, z: Array, cache_list: tuple[Array | None, ...] | None = None, cache_idx: list[int] | None = None
    ) -> tuple[Array, tuple[Array | None, ...] | None]:
        """Forward pass with optional caching.

        Args:
            z: [B, T, H, W, C] latent (e.g., [1, 1, 104, 60, 16])
            cache_list: Tuple of cached features for all conv layers, or None
            cache_idx: List containing current index in cache_list (mutable), or None

        Returns:
            x: [B, T_out, H_out, W_out, 3] RGB video (e.g., [1, 4, 832, 480, 3])
            cache_list: Updated cache tuple
        """
        # Initial convolution
        if cache_list is not None:
            idx = cache_idx[0]
            x, new_cache = self.conv_in(z, cache_list[idx])
            cache_list = (*cache_list[:idx], new_cache, *cache_list[idx + 1 :])
            cache_idx[0] += 1
        else:
            x, _ = self.conv_in(z, None)

        # jax.debug.print("Decoder3D conv_in output has nan:{}", jnp.isnan(x).any())

        # Middle blocks
        x, cache_list = self.mid_block1(x, cache_list, cache_idx)
        x = self.mid_attn(x)  # Attention doesn't use cache
        x, cache_list = self.mid_block2(x, cache_list, cache_idx)

        # jax.debug.print("Decoder3D mid output has nan:{}", jnp.isnan(x).any())

        # Upsample stage 0
        for block in self.up_blocks_0:
            x, cache_list = block(x, cache_list, cache_idx)
        x, cache_list = self.up_sample_0(x, cache_list, cache_idx)

        # jax.debug.print("Decoder3D upsample0 output has nan:{}", jnp.isnan(x).any())

        # Upsample stage 1
        for block in self.up_blocks_1:
            x, cache_list = block(x, cache_list, cache_idx)
        x, cache_list = self.up_sample_1(x, cache_list, cache_idx)

        # jax.debug.print("Decoder3D upsample1 output has nan:{}", jnp.isnan(x).any())

        # Upsample stage 2
        for block in self.up_blocks_2:
            x, cache_list = block(x, cache_list, cache_idx)
        x = self.up_sample_2(x)  # Spatial-only upsample, no cache

        # jax.debug.print("Decoder3D upsample2 output has nan:{}", jnp.isnan(x).any())

        # Upsample stage 3 (no spatial upsample)
        for block in self.up_blocks_3:
            x, cache_list = block(x, cache_list, cache_idx)

        # jax.debug.print("Decoder3D upsample3 output has nan:{}", jnp.isnan(x).any())

        # Output
        x = self.norm_out(x)
        x = nnx.silu(x)

        # jax.debug.print("Decoder3D norm_out output has nan:{}", jnp.isnan(x).any())

        if cache_list is not None:
            idx = cache_idx[0]
            x, new_cache = self.conv_out(x, cache_list[idx])
            cache_list = (*cache_list[:idx], new_cache, *cache_list[idx + 1 :])
            cache_idx[0] += 1
        else:
            x, _ = self.conv_out(x, None)

        # jax.debug.print("Decoder3D conv_out output has nan:{}", jnp.isnan(x).any())

        return x, cache_list


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

    def __init__(self, cfg: VAEConfig = VAEConfig(), *, rngs: nnx.Rngs):
        # Store config tuples as Python values (not JAX arrays!)
        # They'll be converted to JAX arrays at runtime in decode()
        # This avoids ShapeDtypeStruct issues during nnx.eval_shape()
        self.latent_mean_tuple = cfg.latent_mean
        self.latent_std_tuple = cfg.latent_std

        # 1x1 conv projection
        self.conv2 = CausalConv3d(16, 16, kernel_size=(1, 1, 1), rngs=rngs)

        # 3D decoder
        self.decoder = Decoder3D(rngs=rngs)

    def decode(self, latents: Array) -> Array:
        """
        Decode latents to RGB video with feature caching.

        Args:
            latents: [B, T, H, W, C] latent representation (JAX format)
                    e.g., [1, 21, 104, 60, 16]

        Returns:
            video: [B, T_out, H_out, W_out, 3] RGB video (values in [-1, 1])
                  e.g., [1, 81, 832, 480, 3]
        """
        # Step 1: Denormalize
        # Convert Python tuples to JAX arrays at runtime (JIT treats them as static constants)
        latent_mean = jnp.array(self.latent_mean_tuple).reshape(1, 1, 1, 1, 16)
        latent_std = jnp.array(self.latent_std_tuple).reshape(1, 1, 1, 1, 16)
        z = latents * latent_std + latent_mean

        z, _ = self.conv2(z, None)

        # Scan over time dimension: z is [B, T, H, W, C], transpose to [T, B, H, W, C]
        z_frames = jnp.moveaxis(z, 1, 0)  # [T, B, H, W, C]
        # Add singleton time dimension for each frame: [T, B, 1, H, W, C]
        z_frames = z_frames[:, :, None, :, :, :]

        # jax.debug.print("z_frames has nan:{}", jnp.isnan(z_frames).any())

        # Warm-up pass: process first frame to initialize cache with correct shapes
        # This ensures consistent pytree structure for jax.lax.scan
        cache_idx = [0]
        cache_tuple = (None,) * 50
        first_frame_out, cache_tuple = self.decoder(z_frames[0], cache_tuple, cache_idx)
        num_arrays = sum(isinstance(x, jnp.ndarray) for x in cache_tuple)
        num_nones = sum(x is None for x in cache_tuple)
        print(f"cache Arrays: {num_arrays},cache Nones: {num_nones}")

        # JIT-compiled scan function for remaining frames (now cache has concrete shapes)
        @jax.jit
        def scan_frames(cache_tuple, frame_latent):
            """Process single frame with caching (JIT-compiled)."""
            cache_idx = [0]
            frame_out, new_cache_tuple = self.decoder(frame_latent, cache_tuple, cache_idx)
            # num_arrays = sum(isinstance(x, jnp.ndarray) for x in new_cache_tuple)
            # num_nones = sum(x is None for x in new_cache_tuple)
            # jax.debug.print("new cache Arrays: {},cache Nones: {}", num_arrays, num_nones)
            # jax.debug.print("frame_out shape:{}", frame_out.shape)
            # right_part_frame = frame_out[:, :, :, 235:, :]
            # jax.debug.print("frame_out Has NaN: {}", jnp.isnan(right_part_frame).any())
            return new_cache_tuple, frame_out

        # Process remaining frames with JIT
        if z_frames.shape[0] > 1:
            _final_cache, remaining_outputs = jax.lax.scan(scan_frames, cache_tuple, z_frames[1:])

            print(f"remaining output shape: {remaining_outputs.shape}")
            right_part_remaining = remaining_outputs[:, :, :, :, 235:, :]
            print(f"Has NaN: {jnp.isnan(right_part_remaining).any()}")
            print(f"Has Inf: {jnp.isinf(right_part_remaining).any()}")
            print(f"remaining output mean:{right_part_remaining.mean()} ")
            # Frame 0 outputs 1 frame: [B, 1, H, W, 3]
            # Frames 1+ each output 4 frames: [T-1, B, 4, H, W, 3]
            # Flatten temporal dimensions before concatenating
            b, h_out, w_out, c = (
                first_frame_out.shape[0],
                first_frame_out.shape[2],
                first_frame_out.shape[3],
                first_frame_out.shape[4],
            )

            # Flatten first frame: [B, 1, H, W, 3] -> [1, B, H, W, 3]
            first_flat = first_frame_out.transpose(1, 0, 2, 3, 4)  # [1, B, H, W, 3]

            # Flatten remaining frames: [T-1, B, 4, H, W, 3] -> [T-1*4, B, H, W, 3]
            t_minus_1 = remaining_outputs.shape[0]
            t_out_per_frame = remaining_outputs.shape[2]
            remaining_flat = remaining_outputs.transpose(0, 2, 1, 3, 4, 5).reshape(
                t_minus_1 * t_out_per_frame, b, h_out, w_out, c
            )
            print(f"remaining flat shape:{remaining_flat.shape}")
            print(f"remaining flat mean:{remaining_flat[:, 0, :, 235:, :].mean()} ")

            # Concatenate along time dimension: [1+T-1*4, B, H, W, 3]
            # Concatenate first frame with remaining frames
            x = jnp.concatenate([first_flat, remaining_flat], axis=0).transpose(1, 0, 2, 3, 4)
        else:
            x = first_frame_out

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
    print(f"video shape:{video.shape}")
    print(f"video mean:{video[0, 1:, :, 235:, :].mean()}")

    if normalize:
        video = (video + 1.0) / 2.0
        video = jnp.clip(video, 0.0, 1.0)

        video = jnp.round(video * 255.0)
        video = jnp.clip(video, 0, 255).astype(jnp.uint8)

    return video


def save_video(
    video: Array,
    save_path: str,
    fps: int = 30,
    codec: str = "libx264",
    quality: int = 8,
) -> str | None:
    try:
        # Handle batch dimension: take first video if batched
        assert video.ndim == 5
        video = video[0]  # [T, H, W, C]

        video_np = jax.device_get(video)

        # Write video
        writer = imageio.get_writer(save_path, fps=fps, codec=codec, quality=quality)
        for frame in video_np:
            writer.append_data(frame)
        writer.close()

        return save_path

    except Exception as e:
        print(f"Failed to save video: {e}")
        return None


__all__ = ["VAEConfig", "WanVAEDecoder", "decode_latents_to_video", "load_vae_from_checkpoint", "save_video"]
