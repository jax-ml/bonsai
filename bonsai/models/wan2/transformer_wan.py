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

"""Wan2.1-T2V-1.3B: Text-to-Video Diffusion Transformer Model.

This implements the Wan2.1-T2V-1.3B model, a 1.3B parameter diffusion transformer
for text-to-video generation using Flow Matching framework.

Architecture:
- 30-layer Diffusion Transformer with 1536 hidden dim
- 12 attention heads (128 dim each)
- Vision self-attention + text cross-attention
- AdaLN modulation conditioned on timestep
- UMT5 text encoder for multilingual prompts
- Wan-VAE for video encoding/decoding
"""

import dataclasses
import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.lax import Precision
from jaxtyping import Array

from .unipc_multistep_scheduler import FlaxUniPCMultistepScheduler, UniPCMultistepSchedulerState


@dataclasses.dataclass(frozen=True)
class TransformerWanModelConfig:
    """Configuration for Wan2.1-T2V-1.3B Diffusion Transformer."""

    weights_dtype: jnp.dtype = jnp.bfloat16
    num_layers: int = 30
    hidden_dim: int = 1536
    latent_input_dim: int = 16
    latent_output_dim: int = 16
    ffn_dim: int = 8960
    freq_dim: int = 256
    num_heads: int = 12
    head_dim: int = 128
    text_embed_dim: int = 4096
    max_text_len: int = 512
    num_frames: int = 21
    latent_size: Tuple[int, int] = (30, 30)
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    cross_attn_norm: bool = True
    qk_norm: Optional[str] = "rms_norm_across_heads"
    eps: float = 1e-6
    added_kv_proj_dim: Optional[int] = None  # None for T2V, set for I2V
    rope_max_seq_len: int = 1024

    def __post_init__(self):
        assert self.hidden_dim == self.num_heads * self.head_dim, "hidden_dim must equal num_heads * head_dim"


class TimestepEmbedding(nnx.Module):
    """Timestep embedding: sinusoidal -> MLP -> projection for AdaLN."""

    def __init__(self, cfg: TransformerWanModelConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.time_embedding = nnx.Sequential(
            nnx.Linear(cfg.freq_dim, cfg.hidden_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(cfg.hidden_dim, cfg.hidden_dim, rngs=rngs),
        )
        self.time_projection = nnx.Sequential(
            nnx.silu,
            nnx.Linear(cfg.hidden_dim, 6 * cfg.hidden_dim, rngs=rngs),
        )

    def __call__(self, t: Array) -> tuple[Array, Array]:
        t_freq = sinusoidal_embedding_1d(t, self.cfg.freq_dim)
        time_emb = self.time_embedding(t_freq)
        time_proj = self.time_projection(time_emb)
        return time_emb, time_proj


def sinusoidal_embedding_1d(timesteps: Array, embedding_dim: int, max_period: int = 10000) -> Array:
    half_dim = embedding_dim // 2
    freqs = jnp.exp(-math.log(max_period) * jnp.arange(0, half_dim) / half_dim)
    args = timesteps[:, None] * freqs[None, :]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if embedding_dim % 2:
        embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
    return embedding


def precompute_freqs_cis_3d(dim: int, theta: float = 10000.0, max_seq_len: int = 1024) -> tuple[Array, Array, Array]:
    """Precompute 3D RoPE frequencies split as T: dim-4*(dim//6), H: 2*(dim//6), W: 2*(dim//6)."""
    dim_base = dim // 6
    dim_t, dim_h, dim_w = dim - 4 * dim_base, 2 * dim_base, 2 * dim_base
    assert dim_t + dim_h + dim_w == dim
    return (
        rope_params(max_seq_len, dim_t, theta),
        rope_params(max_seq_len, dim_h, theta),
        rope_params(max_seq_len, dim_w, theta),
    )


def rope_params(max_seq_len: int, dim: int, theta: float = 10000.0) -> Array:
    freqs = 1.0 / jnp.power(theta, jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
    positions = jnp.arange(max_seq_len, dtype=jnp.float32)
    freqs = jnp.outer(positions, freqs)
    return jnp.stack([jnp.cos(freqs), jnp.sin(freqs)], axis=-1)


def rope_apply(x: Array, grid_sizes: tuple[int, int, int], freqs: tuple[Array, Array, Array]) -> Array:
    """Apply 3D RoPE to input tensor."""
    b, seq_len, num_heads, head_dim = x.shape
    f, h, w = grid_sizes
    freqs_t, freqs_h, freqs_w = freqs

    dim_base = head_dim // 6
    dim_t, dim_h, dim_w = head_dim - 4 * dim_base, 2 * dim_base, 2 * dim_base

    freqs_grid = jnp.concatenate(
        [
            jnp.broadcast_to(freqs_t[:f, None, None, :, :], (f, h, w, dim_t // 2, 2)),
            jnp.broadcast_to(freqs_h[None, :h, None, :, :], (f, h, w, dim_h // 2, 2)),
            jnp.broadcast_to(freqs_w[None, None, :w, :, :], (f, h, w, dim_w // 2, 2)),
        ],
        axis=3,
    ).reshape(seq_len, head_dim // 2, 2)[None, :, None, :, :]

    x_complex = x.reshape(b, seq_len, num_heads, head_dim // 2, 2)
    x_out = jnp.stack(
        [
            x_complex[..., 0] * freqs_grid[..., 0] - x_complex[..., 1] * freqs_grid[..., 1],
            x_complex[..., 0] * freqs_grid[..., 1] + x_complex[..., 1] * freqs_grid[..., 0],
        ],
        axis=-1,
    )

    return x_out.reshape(b, seq_len, num_heads, head_dim)


class WanLayerNorm(nnx.LayerNorm):
    """LayerNorm with float32 conversion for numerical stability."""

    def __init__(self, dim: int, eps: float = 1e-6, use_scale: bool = False, use_bias: bool = False, *, rngs: nnx.Rngs):
        super().__init__(dim, epsilon=eps, use_scale=use_scale, use_bias=use_bias, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        dtype = x.dtype
        return super().__call__(x.astype(jnp.float32)).astype(dtype)


class MultiHeadAttention(nnx.Module):
    def __init__(self, cfg: TransformerWanModelConfig, *, rngs: nnx.Rngs):
        self.num_heads, self.head_dim = cfg.num_heads, cfg.head_dim
        self.q_proj = nnx.Linear(cfg.hidden_dim, cfg.hidden_dim, rngs=rngs, precision=Precision.HIGHEST)
        self.k_proj = nnx.Linear(cfg.hidden_dim, cfg.hidden_dim, rngs=rngs, precision=Precision.HIGHEST)
        self.v_proj = nnx.Linear(cfg.hidden_dim, cfg.hidden_dim, rngs=rngs, precision=Precision.HIGHEST)
        self.out_proj = nnx.Linear(cfg.hidden_dim, cfg.hidden_dim, rngs=rngs, precision=Precision.HIGHEST)
        self.q_norm = nnx.RMSNorm(cfg.hidden_dim, rngs=rngs)
        self.k_norm = nnx.RMSNorm(cfg.hidden_dim, rngs=rngs)

    def __call__(self, x: Array, rope_state: tuple | None = None, deterministic: bool = True) -> Array:
        b, n = x.shape[:2]
        q = self.q_norm(self.q_proj(x)).reshape(b, n, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x)).reshape(b, n, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, n, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        if rope_state is not None:
            freqs, grid_sizes = rope_state
            q, k = jnp.transpose(q, (0, 2, 1, 3)), jnp.transpose(k, (0, 2, 1, 3))
            q, k = rope_apply(q, grid_sizes, freqs), rope_apply(k, grid_sizes, freqs)
            q, k = jnp.transpose(q, (0, 2, 1, 3)), jnp.transpose(k, (0, 2, 1, 3))

        attn = jax.nn.softmax(
            jnp.einsum("bhid,bhjd->bhij", q, k, precision=Precision.HIGHEST) / math.sqrt(self.head_dim), axis=-1
        )
        out = (
            jnp.einsum("bhij,bhjd->bhid", attn, v, precision=Precision.HIGHEST).transpose(0, 2, 1, 3).reshape(b, n, -1)
        )
        return self.out_proj(out)


class CrossAttention(nnx.Module):
    def __init__(self, cfg: TransformerWanModelConfig, *, rngs: nnx.Rngs):
        self.num_heads, self.head_dim = cfg.num_heads, cfg.head_dim
        self.q_proj = nnx.Linear(cfg.hidden_dim, cfg.hidden_dim, rngs=rngs, precision=Precision.HIGHEST)
        self.kv_proj = nnx.Linear(cfg.hidden_dim, 2 * cfg.hidden_dim, rngs=rngs, precision=Precision.HIGHEST)
        self.out_proj = nnx.Linear(cfg.hidden_dim, cfg.hidden_dim, rngs=rngs, precision=Precision.HIGHEST)
        self.q_norm = nnx.RMSNorm(cfg.hidden_dim, rngs=rngs)
        self.k_norm = nnx.RMSNorm(cfg.hidden_dim, rngs=rngs)

    def __call__(self, x: Array, context: Array, deterministic: bool = True) -> Array:
        b, n, m = x.shape[0], x.shape[1], context.shape[1]
        q = self.q_norm(self.q_proj(x)).reshape(b, n, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        kv = self.kv_proj(context)
        k, v = jnp.split(kv, 2, axis=-1)
        k = self.k_norm(k).reshape(b, m, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(b, m, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        attn = jax.nn.softmax(
            jnp.einsum("bhid,bhjd->bhij", q, k, precision=Precision.HIGHEST) / math.sqrt(self.head_dim), axis=-1
        )
        out = (
            jnp.einsum("bhij,bhjd->bhid", attn, v, precision=Precision.HIGHEST).transpose(0, 2, 1, 3).reshape(b, n, -1)
        )
        return self.out_proj(out)


def modulate(x: Array, shift: Array, scale: Array) -> Array:
    """Apply adaptive layer norm modulation."""
    return x * (1 + scale) + shift


class WanAttentionBlock(nnx.Module):
    """
    Wan Diffusion Transformer Block.

    Includes:
    - Vision self-attention with AdaLN modulation
    - Text-to-vision cross-attention
    - Feed-forward network with AdaLN modulation
    """

    def __init__(self, cfg: TransformerWanModelConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg

        self.norm1 = WanLayerNorm(cfg.hidden_dim, rngs=rngs)
        self.norm2 = WanLayerNorm(cfg.hidden_dim, rngs=rngs, use_scale=True, use_bias=True)
        self.norm3 = WanLayerNorm(cfg.hidden_dim, rngs=rngs)

        self.self_attn = MultiHeadAttention(cfg, rngs=rngs)
        self.cross_attn = CrossAttention(cfg, rngs=rngs)

        self.mlp = nnx.Sequential(
            nnx.Linear(cfg.hidden_dim, cfg.ffn_dim, rngs=rngs, precision=Precision.HIGHEST),
            nnx.gelu,
            nnx.Linear(cfg.ffn_dim, cfg.hidden_dim, rngs=rngs, precision=Precision.HIGHEST),
        )

        self.scale_shift_table = nnx.Param(
            jax.random.normal(rngs.params(), (1, 6, cfg.hidden_dim)) / (cfg.hidden_dim**0.5)
        )

    @jax.named_scope("wan_attention_block")
    def __call__(
        self,
        x: Array,
        text_embeds: Array,
        time_proj: Array,
        rope_state: tuple | None = None,
        deterministic: bool = True,
    ) -> Array:
        """
        Args:
            x: [B, N, D] video tokens
            text_embeds: [B, M, text_dim] text embeddings
            time_proj: [B, 6*D] time embedding
            rope_state: Optional tuple of (freqs, grid_sizes) for 3D RoPE
            deterministic: Whether to apply dropout
        Returns:
            [B, N, D] transformed tokens
        """
        # Get modulation from time embedding
        b = time_proj.shape[0]
        d = self.cfg.hidden_dim
        reshaped_time_proj = time_proj.reshape(b, 6, d)
        modulation = reshaped_time_proj + self.scale_shift_table.value
        modulation = modulation.reshape(b, -1)  # [B, 6*D]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(modulation, 6, axis=-1)

        # Self-attention with AdaLN modulation and RoPE
        print("video tokens:", x.shape)
        norm_x = self.norm1(x)
        norm_x = modulate(norm_x, shift_msa[:, None, :], scale_msa[:, None, :])
        attn_out = self.self_attn(norm_x, rope_state=rope_state, deterministic=deterministic)
        x = x + gate_msa[:, None, :] * attn_out

        # Cross-attention
        norm_x = self.norm2(x)
        cross_out = self.cross_attn(norm_x, text_embeds, deterministic=deterministic)
        x = x + cross_out

        # MLP with AdaLN modulation
        norm_x = self.norm3(x)
        norm_x = modulate(norm_x, shift_mlp[:, None, :], scale_mlp[:, None, :])
        mlp_out = self.mlp(norm_x)
        x = x + gate_mlp[:, None, :] * mlp_out

        return x


class FinalLayer(nnx.Module):
    """Final layer that predicts noise from DiT output."""

    def __init__(self, cfg: TransformerWanModelConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.norm = WanLayerNorm(cfg.hidden_dim, rngs=rngs)
        out_dim = math.prod(cfg.patch_size) * cfg.latent_output_dim  # expand out_dim here for unpatchify
        self.linear = nnx.Linear(cfg.hidden_dim, out_dim, rngs=rngs, precision=Precision.HIGHEST)

        self.scale_shift_table = nnx.Param(
            jax.random.normal(rngs.params(), (1, 2, cfg.hidden_dim)) / (cfg.hidden_dim**0.5)
        )

    @jax.named_scope("final_layer")
    def __call__(self, x: Array, time_emb: Array) -> Array:
        """
        Args:
            x: [B, N, D] DiT output
            time_emb: [B, D] time embedding from TimestepEmbedding
        Returns:
            [B, N, latent_output_dim] predicted noise
        """
        # [B, D] → [B, 1, D] + [1, 2, D] → [B, 2, D]
        e = self.scale_shift_table.value + time_emb[:, None, :]
        shift, scale = e[:, 0, :], e[:, 1, :]

        x = self.norm(x) * (1 + scale[:, None, :]) + shift[:, None, :]
        x = self.linear(x)
        return x


class Wan2DiT(nnx.Module):
    """
    Wan2.1-T2V-1.3B Diffusion Transformer.
    """

    def __init__(self, cfg: TransformerWanModelConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg

        # 3D Conv to patchify video latents
        # (T, H, W) → (T, H/2, W/2)
        self.patch_embed = nnx.Conv(
            in_features=cfg.latent_input_dim,
            out_features=cfg.hidden_dim,
            kernel_size=(1, 2, 2),
            strides=(1, 2, 2),
            padding="VALID",
            use_bias=True,
            rngs=rngs,
            precision=Precision.HIGHEST,
        )

        # Text embedding projection: UMT5 (4096) → DiT (1536)
        self.text_proj = nnx.Sequential(
            nnx.Linear(cfg.text_embed_dim, cfg.hidden_dim, rngs=rngs, precision=Precision.HIGHEST),
            nnx.gelu,
            nnx.Linear(cfg.hidden_dim, cfg.hidden_dim, rngs=rngs, precision=Precision.HIGHEST),
        )

        self.time_embed = TimestepEmbedding(cfg, rngs=rngs)

        self.blocks = nnx.List([WanAttentionBlock(cfg, rngs=rngs) for _ in range(cfg.num_layers)])

        self.final_layer = FinalLayer(cfg, rngs=rngs)

    @jax.named_scope("wan2_dit")
    @jax.jit
    def forward(self, latents: Array, text_embeds: Array, timestep: Array, deterministic: bool = True) -> Array:
        """
        Forward pass of the Diffusion Transformer.

        Args:
            latents: [B, T, H, W, C] noisy video latents from VAE
            text_embeds: [B, seq_len, 4096] from UMT5-XXL encoder (before projection)
            timestep: [B] diffusion timestep (0 to num_steps)
            deterministic: Whether to apply dropout

        Returns:
            predicted_noise: [B, T, H, W, C] predicted noise
        """
        text_embeds = self.text_proj(text_embeds)

        # Get time embeddings
        # time_emb: [B, D] for FinalLayer
        # time_proj: [B, 6*D] for AdaLN in blocks
        time_emb, time_proj = self.time_embed(timestep)

        x = self.patch_embed(latents)
        b, t_out, h_out, w_out, d = x.shape
        x = x.reshape(b, t_out * h_out * w_out, d)

        grid_sizes = (t_out, h_out, w_out)

        max_seq = max(grid_sizes)
        rope_freqs = tuple(
            jax.lax.stop_gradient(arr) for arr in precompute_freqs_cis_3d(dim=self.cfg.head_dim, max_seq_len=max_seq)
        )

        for block in self.blocks:
            x = block(x, text_embeds, time_proj, rope_state=(rope_freqs, grid_sizes), deterministic=deterministic)

        # Final projection to noise space
        x = self.final_layer(x, time_emb)  # [B, T*H*W, latent_output_dim]

        # Reshape back to video format
        predicted_noise = self.unpatchify(x, grid_sizes)

        return predicted_noise

    def unpatchify(self, x: Array, grid_sizes: tuple[int, int, int]) -> Array:
        """
        Reconstruct video tensors from patch embeddings.

        Args:
            x: [B, T*H*W, patch_t*patch_h*patch_w*C] flattened patch embeddings
            grid_sizes: (T_patches, H_patches, W_patches) grid dimensions

        Returns:
            [B, T, H, W, C] reconstructed video tensor (channel-last)
        """
        b, seq_len, feature_dim = x.shape
        t_patches, h_patches, w_patches = grid_sizes
        c = self.cfg.latent_output_dim
        patch_size = self.cfg.patch_size

        assert seq_len == t_patches * h_patches * w_patches, (
            f"expected: seq_len={seq_len} should be {t_patches * h_patches * w_patches}"
        )
        assert feature_dim == patch_size[0] * patch_size[1] * patch_size[2] * c, (
            f"expected: feature_dim={feature_dim} should be {patch_size[0] * patch_size[1] * patch_size[2] * c}"
        )

        x = x.reshape(
            b,
            t_patches,
            h_patches,
            w_patches,
            patch_size[0],
            patch_size[1],
            patch_size[2],
            c,
        )
        x = jnp.einsum("bthwpqrc->btphqwrc", x)
        x = x.reshape(
            b,
            t_patches * patch_size[0],
            h_patches * patch_size[1],
            w_patches * patch_size[2],
            c,
        )

        return x


__all__ = [
    "TransformerWanModelConfig",
    "Wan2DiT",
]
