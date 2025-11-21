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
- T5 text encoder for multilingual prompts
- Wan-VAE for video encoding/decoding
"""

import dataclasses
import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array


class WanLayerNorm(nnx.LayerNorm):
    # disable use_bias and use_scale because for AdaLN modulation

    def __init__(self, dim: int, eps: float = 1e-6, use_scale: bool = False, use_bias: bool = False, *, rngs: nnx.Rngs):
        super().__init__(dim, epsilon=eps, use_scale=use_scale, use_bias=use_bias, rngs=rngs)

    @jax.named_scope("wan_layer_norm")
    def __call__(self, x: Array) -> Array:
        original_dtype = x.dtype
        x_float = x.astype(jnp.float32)
        x_normed = super().__call__(x_float)
        return x_normed.astype(original_dtype)


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Configuration for Wan2.1-T2V-1.3B Diffusion Transformer."""

    # Model architecture
    num_layers: int = 5
    hidden_dim: int = 1536
    input_dim: int = 16  # VAE latent channels
    output_dim: int = 16  # Predicted noise channels
    ffn_dim: int = 8960
    freq_dim: int = 256  # Frequency embedding dimension
    num_heads: int = 12
    head_dim: int = 128  # hidden_dim / num_heads

    # Text encoder
    text_embed_dim: int = 4096  # T5-XXL embedding dimension (before projection)
    max_text_len: int = 512  # Maximum text sequence length

    # Video generation specs
    num_frames: int = 21  # Default frames (5 seconds)
    latent_size: Tuple[int, int] = (32, 32)  # Latent spatial size for 480p

    # Diffusion parameters
    num_inference_steps: int = 50
    guidance_scale: float = 5.0

    @classmethod
    def wan2_1_1_3b(cls, use_sharding: bool = False):
        """Default Wan2.1-T2V-1.3B configuration."""
        return cls()


def sinusoidal_embedding_1d(timesteps: Array, embedding_dim: int, max_period: int = 10000) -> Array:
    half_dim = embedding_dim // 2
    freqs = jnp.exp(-math.log(max_period) * jnp.arange(0, half_dim) / half_dim)
    args = timesteps[:, None] * freqs[None, :]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if embedding_dim % 2:
        embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
    return embedding


class TimestepEmbedding(nnx.Module):
    """
    Timestep embedding matching HuggingFace Wan implementation.

    Consists of two parts:
    1. time_embedding: Linear → SiLU → Linear (freq_dim → dim → dim)
    2. time_projection: SiLU → Linear (dim → 6*dim) for AdaLN modulation
    """

    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg

        # time_embedding: processes sinusoidal timestep
        self.time_embedding = nnx.Sequential(
            nnx.Linear(cfg.freq_dim, cfg.hidden_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(cfg.hidden_dim, cfg.hidden_dim, rngs=rngs),
        )

        # time_projection: projects to 6D for AdaLN
        self.time_projection = nnx.Sequential(nnx.silu, nnx.Linear(cfg.hidden_dim, 6 * cfg.hidden_dim, rngs=rngs))

    @jax.named_scope("timestep_embedding")
    def __call__(self, t: Array) -> tuple[Array, Array]:
        """
        Args:
            t: [B] timestep indices
        Returns:
            time_emb: [B, hidden_dim] base time embedding
            time_proj: [B, 6 * hidden_dim] projected for AdaLN modulation
        """
        # Generate sinusoidal embeddings
        t_freq = sinusoidal_embedding_1d(t, self.cfg.freq_dim)

        # Process through time_embedding MLP
        time_emb = self.time_embedding(t_freq)  # [B, D]

        # Project for AdaLN modulation
        time_proj = self.time_projection(time_emb)  # [B, 6*D]

        return time_emb, time_proj


def rope_params(max_seq_len: int, dim: int, theta: float = 10000.0) -> Array:
    # Create frequency schedule: 1 / theta^(2i/dim)
    freqs = 1.0 / jnp.power(theta, jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)

    # Outer product with positions
    positions = jnp.arange(max_seq_len, dtype=jnp.float32)
    freqs = jnp.outer(positions, freqs)  # [max_seq_len, dim // 2]

    # Convert to complex representation [cos, sin]
    freqs_cos = jnp.cos(freqs)
    freqs_sin = jnp.sin(freqs)
    freqs_cis = jnp.stack([freqs_cos, freqs_sin], axis=-1)

    return freqs_cis


def precompute_freqs_cis_3d(
    dim: int,
    theta: float = 10000.0,
    max_seq_len: int = 1024,
) -> tuple[Array, Array, Array]:
    """
    Precompute RoPE frequencies for 3D video (temporal + spatial).

    dimensions are split as:
    - Temporal: dim - 4*(dim//6)
    - Height: 2*(dim//6)
    - Width: 2*(dim//6)
    """
    # Split dimension according to Wan2 formula
    dim_base = dim // 6
    dim_t = dim - 4 * dim_base  # Temporal gets more dims
    dim_h = 2 * dim_base  # Height
    dim_w = 2 * dim_base  # Width

    # Verify the split is correct
    assert dim_t + dim_h + dim_w == dim, f"Dimension split error: {dim_t} + {dim_h} + {dim_w} != {dim}"

    # Create frequency parameters for each dimension
    freqs_t = rope_params(max_seq_len, dim_t, theta)  # [1024, dim_t // 2, 2]
    freqs_h = rope_params(max_seq_len, dim_h, theta)  # [1024, dim_h // 2, 2]
    freqs_w = rope_params(max_seq_len, dim_w, theta)  # [1024, dim_w // 2, 2]

    return freqs_t, freqs_h, freqs_w


def rope_apply(
    x: Array,
    grid_sizes: tuple[int, int, int],
    freqs: tuple[Array, Array, Array],
) -> Array:
    b, seq_len, num_heads, head_dim = x.shape
    f, h, w = grid_sizes

    # Verify sequence length matches grid
    assert f * h * w == seq_len, f"Grid size {f}x{h}x{w}={f * h * w} != seq_len {seq_len}"

    # Split freqs into temporal, height, width components
    freqs_t, freqs_h, freqs_w = freqs

    # Get dimension splits
    dim_base = head_dim // 6
    dim_t = head_dim - 4 * dim_base
    dim_h = 2 * dim_base
    dim_w = 2 * dim_base

    # Build 3D positional grid by concatenating T, H, W frequencies
    # Shape: [f, h, w, head_dim // 2, 2]
    freqs_grid = jnp.concatenate(
        [
            # Temporal: [f, 1, 1, dim_t // 2, 2] → [f, h, w, dim_t // 2, 2]
            jnp.broadcast_to(freqs_t[:f, None, None, :, :], (f, h, w, dim_t // 2, 2)),
            # Height: [1, h, 1, dim_h // 2, 2] → [f, h, w, dim_h // 2, 2]
            jnp.broadcast_to(freqs_h[None, :h, None, :, :], (f, h, w, dim_h // 2, 2)),
            # Width: [1, 1, w, dim_w // 2, 2] → [f, h, w, dim_w // 2, 2]
            jnp.broadcast_to(freqs_w[None, None, :w, :, :], (f, h, w, dim_w // 2, 2)),
        ],
        axis=3,
    )  # [f, h, w, head_dim // 2, 2]

    # Reshape to sequence: [seq_len, head_dim // 2, 2]
    freqs_grid = freqs_grid.reshape(seq_len, head_dim // 2, 2)

    # Reshape x to complex pairs: [B, seq_len, num_heads, head_dim // 2, 2]
    x_complex = x.reshape(b, seq_len, num_heads, head_dim // 2, 2)

    # Expand freqs for broadcasting: [1, seq_len, 1, head_dim // 2, 2]
    freqs_grid = freqs_grid[None, :, None, :, :]

    # Complex multiplication: (x_r + i*x_i) * (cos + i*sin)
    # Real part: x_r * cos - x_i * sin
    # Imag part: x_r * sin + x_i * cos
    x_out = jnp.stack(
        [
            x_complex[..., 0] * freqs_grid[..., 0] - x_complex[..., 1] * freqs_grid[..., 1],  # Real
            x_complex[..., 0] * freqs_grid[..., 1] + x_complex[..., 1] * freqs_grid[..., 0],  # Imag
        ],
        axis=-1,
    )

    # Reshape back to original shape
    x_out = x_out.reshape(b, seq_len, num_heads, head_dim)
    return x_out


class MultiHeadAttention(nnx.Module):
    """Multi-head self-attention with RoPE position embeddings."""

    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim

        # Q, K, V projections
        self.q_proj = nnx.Linear(cfg.hidden_dim, cfg.hidden_dim, rngs=rngs)
        self.k_proj = nnx.Linear(cfg.hidden_dim, cfg.hidden_dim, rngs=rngs)
        self.v_proj = nnx.Linear(cfg.hidden_dim, cfg.hidden_dim, rngs=rngs)
        self.out_proj = nnx.Linear(cfg.hidden_dim, cfg.hidden_dim, rngs=rngs)

        # RMSNorm for Q and K
        self.q_norm = nnx.RMSNorm(cfg.head_dim, rngs=rngs)
        self.k_norm = nnx.RMSNorm(cfg.head_dim, rngs=rngs)

    @jax.named_scope("multi_head_attention")
    def __call__(self, x: Array, rope_state: tuple | None = None, deterministic: bool = True) -> Array:
        """
        Args:
            x: [B, N, D] input tokens
            rope_state: Optional tuple of (freqs, grid_sizes) for RoPE
            deterministic: Whether to apply dropout
        Returns:
            [B, N, D] attended features
        """
        b, n, _d = x.shape

        # Compute Q, K, V
        q = self.q_proj(x).reshape(b, n, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(b, n, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(b, n, self.num_heads, self.head_dim)

        q = jnp.transpose(q, (0, 2, 1, 3))  # [B, H, N, D_head]
        k = jnp.transpose(k, (0, 2, 1, 3))  # [B, H, N, D_head]
        v = jnp.transpose(v, (0, 2, 1, 3))  # [B, H, N, D_head]

        # Apply RMSNorm to Q and K
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE to Q and K
        if rope_state is not None:
            freqs, grid_sizes = rope_state
            # Reshape for RoPE: [B, H, N, D] → [B, N, H, D]
            q = jnp.transpose(q, (0, 2, 1, 3))
            k = jnp.transpose(k, (0, 2, 1, 3))

            # Apply rotary embeddings with 3D grid
            q = rope_apply(q, grid_sizes, freqs)
            k = rope_apply(k, grid_sizes, freqs)

            # Reshape back: [B, N, H, D] → [B, H, N, D]
            q = jnp.transpose(q, (0, 2, 1, 3))
            k = jnp.transpose(k, (0, 2, 1, 3))

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = jnp.einsum("bhid,bhjd->bhij", q, k) * scale
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)

        # Apply attention to values
        attn_output = jnp.einsum("bhij,bhjd->bhid", attn_weights, v)
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))  # [B, N, H, D]
        attn_output = attn_output.reshape(b, n, -1)  # [B, N, H*D]

        return self.out_proj(attn_output)


class CrossAttention(nnx.Module):
    """Cross-attention from video tokens to text embeddings."""

    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim

        # Q from video, K,V from text (text already projected to hidden_dim)
        self.q_proj = nnx.Linear(cfg.hidden_dim, cfg.hidden_dim, rngs=rngs)
        self.kv_proj = nnx.Linear(cfg.hidden_dim, 2 * cfg.hidden_dim, rngs=rngs)
        self.out_proj = nnx.Linear(cfg.hidden_dim, cfg.hidden_dim, rngs=rngs)

        # RMSNorm for Q
        self.q_norm = nnx.RMSNorm(cfg.head_dim, rngs=rngs)

    @jax.named_scope("cross_attention")
    def __call__(self, x: Array, context: Array, deterministic: bool = True) -> Array:
        """
        Args:
            x: [B, N, D] video tokens (query)
            context: [B, M, text_dim] text embeddings (key, value)
            deterministic: Whether to apply dropout
        Returns:
            [B, N, D] cross-attended features
        """
        b, n, _d = x.shape
        _, m, _ = context.shape

        # Query from video
        q = self.q_proj(x)  # [B, N, D]
        q = q.reshape(b, n, self.num_heads, self.head_dim)
        q = jnp.transpose(q, (0, 2, 1, 3))  # [B, H, N, D]

        # Apply RMSNorm to Q
        q = self.q_norm(q)

        # Key, Value from text
        kv = self.kv_proj(context)  # [B, M, 2*D]
        kv = kv.reshape(b, m, 2, self.num_heads, self.head_dim)
        kv = jnp.transpose(kv, (2, 0, 3, 1, 4))  # [2, B, H, M, D]
        k, v = kv[0], kv[1]

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = jnp.einsum("bhid,bhjd->bhij", q, k) * scale
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)

        # Apply attention to values
        attn_output = jnp.einsum("bhij,bhjd->bhid", attn_weights, v)
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))  # [B, N, H, D]
        attn_output = attn_output.reshape(b, n, -1)  # [B, N, H*D]

        return self.out_proj(attn_output)


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

    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg

        #  Layer norms
        self.norm1 = WanLayerNorm(cfg.hidden_dim, rngs=rngs)
        self.norm2 = WanLayerNorm(cfg.hidden_dim, rngs=rngs, use_scale=True, use_bias=True)
        self.norm3 = WanLayerNorm(cfg.hidden_dim, rngs=rngs)

        # Attention layers
        self.self_attn = MultiHeadAttention(cfg, rngs=rngs)
        self.cross_attn = CrossAttention(cfg, rngs=rngs)

        # Feed-forward MLP: Linear -> GELU -> Linear
        self.mlp = nnx.Sequential(
            nnx.Linear(cfg.hidden_dim, cfg.ffn_dim, rngs=rngs),
            nnx.gelu,
            nnx.Linear(cfg.ffn_dim, cfg.hidden_dim, rngs=rngs),
        )

        # Learnable modulation parameter
        self.modulation = nnx.Param(jax.random.normal(rngs.params(), (1, 6, cfg.hidden_dim)) / (cfg.hidden_dim**0.5))

    @jax.named_scope("wan_attention_block")
    def __call__(
        self, x: Array, text_embeds: Array, time_emb: Array, rope_state: tuple | None = None, deterministic: bool = True
    ) -> Array:
        """
        Args:
            x: [B, N, D] video tokens
            text_embeds: [B, M, text_dim] text embeddings
            time_emb: [B, 6*D] time embedding
            rope_state: Optional tuple of (freqs, grid_sizes) for 3D RoPE
            deterministic: Whether to apply dropout
        Returns:
            [B, N, D] transformed tokens
        """
        # Get modulation parameters from time embedding
        b = time_emb.shape[0]
        d = self.cfg.hidden_dim

        # Reshape time embedding and add learnable modulation
        reshaped_time_emb = time_emb.reshape(b, 6, d)
        modulation = nnx.silu(reshaped_time_emb + self.modulation.value)
        modulation = modulation.reshape(b, -1)  # [B, 6*D]

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(modulation, 6, axis=-1)

        # Self-attention with AdaLN modulation and RoPE
        print("video tokens:", x.shape)
        norm_x = self.norm1(x)
        norm_x = modulate(norm_x, shift_msa[:, None, :], scale_msa[:, None, :])
        attn_out = self.self_attn(norm_x, rope_state=rope_state, deterministic=deterministic)
        x = x + gate_msa[:, None, :] * attn_out

        # Cross-attention (no modulation, following DiT design)
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

    def __init__(self, cfg: ModelConfig, patch_size, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.norm = WanLayerNorm(cfg.hidden_dim, rngs=rngs)
        out_dim = math.prod(patch_size) * cfg.output_dim  # expand out_dim here for unpatchify
        self.linear = nnx.Linear(cfg.hidden_dim, out_dim, rngs=rngs)

        # Learnable modulation parameter (matches HF: torch.randn / dim**0.5)
        self.modulation = nnx.Param(jax.random.normal(rngs.params(), (1, 2, cfg.hidden_dim)) / (cfg.hidden_dim**0.5))

    @jax.named_scope("final_layer")
    def __call__(self, x: Array, time_emb: Array) -> Array:
        """
        Args:
            x: [B, N, D] DiT output
            time_emb: [B, D] time embedding from TimestepEmbedding
        Returns:
            [B, N, output_dim] predicted noise
        """
        # [B, D] → [B, 1, D] + [1, 2, D] → [B, 2, D]
        e = self.modulation.value + time_emb[:, None, :]
        shift, scale = e[:, 0, :], e[:, 1, :]

        x = self.norm(x) * (1 + scale[:, None, :]) + shift[:, None, :]
        x = self.linear(x)
        return x


class Wan2DiT(nnx.Module):
    """
    Wan2.1-T2V-1.3B Diffusion Transformer.
    """

    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg

        # 3D Conv to patchify video latents
        # (T, H, W) → (T, H/2, W/2)
        self.patch_embed = nnx.Conv(
            in_features=cfg.input_dim,
            out_features=cfg.hidden_dim,
            kernel_size=(1, 2, 2),
            strides=(1, 2, 2),
            padding="VALID",
            use_bias=True,
            rngs=rngs,
        )

        # Text embedding projection: T5 (4096) → DiT (1536)
        # Linear(4096 → 1536) → GELU → Linear(1536 → 1536)
        self.text_proj = nnx.Sequential(
            nnx.Linear(cfg.text_embed_dim, cfg.hidden_dim, rngs=rngs),
            nnx.gelu,
            nnx.Linear(cfg.hidden_dim, cfg.hidden_dim, rngs=rngs),
        )

        # Time embedding
        self.time_embed = TimestepEmbedding(cfg, rngs=rngs)

        # Precompute RoPE frequencies for 3D position encoding
        self.freqs_t, self.freqs_h, self.freqs_w = precompute_freqs_cis_3d(
            dim=cfg.head_dim,
            theta=10000.0,
        )

        # Transformer blocks
        self.blocks = nnx.List([WanAttentionBlock(cfg, rngs=rngs) for _ in range(cfg.num_layers)])

        # Final layer
        self.final_layer = FinalLayer(cfg, patch_size=(1, 2, 2), rngs=rngs)

    @jax.named_scope("wan2_dit")
    def __call__(self, latents: Array, text_embeds: Array, timestep: Array, deterministic: bool = True) -> Array:
        """
        Forward pass of the Diffusion Transformer.

        Args:
            latents: [B, T, H, W, C] noisy video latents from VAE
            text_embeds: [B, seq_len, 4096] from T5-XXL encoder (before projection)
            timestep: [B] diffusion timestep (0 to num_steps)
            deterministic: Whether to apply dropout

        Returns:
            predicted_noise: [B, T, H, W, C] predicted noise
        """
        b, _t, _h, _w, _c = latents.shape

        # Project text embeddings: [B, 512, 4096] → [B, 512, 1536]
        text_embeds = self.text_proj(text_embeds)

        # Patchify video latents with 3D Conv
        # Input: [B, T, H, W, C] -> [B, T, H/2, W/2, 1536]
        x = self.patch_embed(latents)

        # Flatten spatial-temporal dimensions to sequence
        # [B, T, H/2, W/2, 1536] → [B, T*H/2*W/2, 1536]
        b, t_out, h_out, w_out, d = x.shape
        x = x.reshape(b, t_out * h_out * w_out, d)

        # Grid sizes for unpatchify later
        grid_sizes = (t_out, h_out, w_out)

        # RoPE frequencies for 3D position encoding
        rope_freqs = (self.freqs_t, self.freqs_h, self.freqs_w)

        # Get time embeddings
        # time_emb: [B, D] for FinalLayer
        # time_proj: [B, 6*D] for AdaLN in blocks
        time_emb, time_proj = self.time_embed(timestep)

        # Process through transformer blocks with RoPE
        for block in self.blocks:
            x = block(x, text_embeds, time_proj, rope_state=(rope_freqs, grid_sizes), deterministic=deterministic)

        # Final projection to noise space
        x = self.final_layer(x, time_emb)  # [B, T*H*W, output_dim]

        # Reshape back to video format
        predicted_noise = self.unpatchify(x, grid_sizes)

        return predicted_noise

    def unpatchify(self, x: Array, grid_sizes: tuple[int, int, int]) -> Array:
        """
        Reconstruct video tensors from patch embeddings.

        Args:
            x: [B, T*H*W, C] flattened patch embeddings
            grid_sizes: (T_patches, H_patches, W_patches) grid dimensions

        Returns:
            [B, T, H, W, C] reconstructed video tensor (channel-last)
        """
        b, _seq_len, _c = x.shape
        t_patches, h_patches, w_patches = grid_sizes
        patch_size = (1, 2, 2)  # Patch size from the 3D Conv kernel
        c = self.cfg.output_dim

        # Reshape from sequence to grid: [B, T*H*W, C] -> [B, T, H, W, C]
        x = x.reshape(b, t_patches, h_patches, w_patches, c * math.prod(patch_size))

        # Rearrange dimensions for unpatchify
        # [B, T, H, W, C] -> [B, T, H, W, patch_t, patch_h, patch_w, C]
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

        # Merge patches: einsum 'bthwpqrc->btphqwrc'
        # This interleaves the patch dimensions with the grid dimensions
        x = jnp.einsum("bthwpqrc->btphqwrc", x)

        # Reshape to final video format: [B, T*patch_t, H*patch_h, W*patch_w, C]
        x = x.reshape(
            b,
            t_patches * patch_size[0],
            h_patches * patch_size[1],
            w_patches * patch_size[2],
            c,
        )

        return x


# Flow Matching Scheduler (simplified version)
class FlowMatchingScheduler:
    """Flow matching scheduler for diffusion sampling."""

    def __init__(self, num_steps: int = 50):
        self.num_steps = num_steps
        # Linear schedule from 0 to 1
        self.timesteps = jnp.linspace(0, 1, num_steps)

    def add_noise(self, latents: Array, noise: Array, t: Array) -> Array:
        """Add noise to latents according to flow matching."""
        # Flow matching: x_t = (1-t) * x_0 + t * noise
        t = t.reshape(-1, 1, 1, 1, 1)
        return (1 - t) * latents + t * noise

    def step(self, model_output: Array, sample: Array, t_idx: int) -> Array:
        """Single denoising step."""
        t = self.timesteps[t_idx]
        t_next = self.timesteps[t_idx - 1] if t_idx > 0 else 0.0

        # Flow matching step
        dt = t - t_next
        sample = sample - dt * model_output

        return sample


def generate_video(
    model: Wan2DiT,
    text_embeds: Array,
    num_frames: int = 81,
    latent_size: Tuple[int, int] = (60, 60),
    num_steps: int = 50,
    guidance_scale: float = 5.5,
    key: Optional[jax.Array] = None,
) -> Array:
    """
    Generate video from text embeddings using the diffusion model.

    Args:
        model: Wan2DiT model
        text_embeds: [B, seq_len, text_dim] text embeddings from T5
        num_frames: Number of frames to generate
        latent_size: Spatial size of latents
        num_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale (5-6 recommended)
        key: JAX random key

    Returns:
        latents: [B, T, H, W, C] generated video latents
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    b = text_embeds.shape[0]
    h, w = latent_size
    c = model.cfg.input_dim

    # Initialize random noise
    latents = jax.random.normal(key, (b, num_frames, h, w, c))

    # Create scheduler
    scheduler = FlowMatchingScheduler(num_steps)

    # Denoising loop
    for t_idx in reversed(range(num_steps)):
        # Current timestep
        t = jnp.full((b,), t_idx, dtype=jnp.int32)

        # Classifier-free guidance
        if guidance_scale != 1.0:
            # Predict with text conditioning
            noise_pred_cond = model(latents, text_embeds, t, deterministic=True)

            # Predict without text (null text)
            null_embeds = jnp.zeros_like(text_embeds)
            noise_pred_uncond = model(latents, null_embeds, t, deterministic=True)

            # Apply guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = model(latents, text_embeds, t, deterministic=True)

        # Update latents
        latents = scheduler.step(noise_pred, latents, t_idx)

    return latents


__all__ = [
    "FlowMatchingScheduler",
    "ModelConfig",
    "Wan2DiT",
    "generate_video",
]
