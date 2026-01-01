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

"""Qwen3-VL Flax NNX Implementation.

A precise port of the Qwen3-VL multimodal model from PyTorch to Flax NNX,
supporting distributed inference with proper sharding configurations.
"""

import dataclasses
import math
from functools import partial
from typing import Any, Callable, List, Optional, Tuple, TypeAlias, Union

import jax
import jax.numpy as jnp
from flax import nnx
from jax import P
from jax.sharding import PartitionSpec, get_abstract_mesh, reshard
from jaxtyping import Array, ArrayLike

_K_MASK = jnp.finfo(jnp.bfloat16).min
ShardingSpec = PartitionSpec


# =============================================================================
# Sharding Configuration
# =============================================================================


@dataclasses.dataclass(slots=True, frozen=True)
class ShardingCfg:
    """Sharding configuration for distributed model parallel inference."""

    # Text Model Embeddings
    emb_vd: ShardingSpec
    emb_dv: ShardingSpec
    # Attention
    q_weight_ndh: ShardingSpec
    kv_weight_ndh: ShardingSpec
    o_weight_nhd: ShardingSpec
    # MLP
    ffw_weight_df: ShardingSpec
    ffw_weight_fd: ShardingSpec
    # Norms/Activation
    rms_norm: ShardingSpec
    act_btd: ShardingSpec
    act_btf: ShardingSpec
    act_btnh: ShardingSpec
    # Vision Model
    vision_conv: ShardingSpec
    vision_weight: ShardingSpec

    @staticmethod
    def no_sharding():
        """Configuration with no sharding (all None)."""
        return ShardingCfg(
            emb_vd=P(None, None),
            emb_dv=P(None, None),
            q_weight_ndh=P(None, None, None),
            kv_weight_ndh=P(None, None, None),
            o_weight_nhd=P(None, None, None),
            ffw_weight_df=P(None, None),
            ffw_weight_fd=P(None, None),
            rms_norm=P(None),
            act_btd=P(None, None, None),
            act_btf=P(None, None, None),
            act_btnh=P(None, None, None, None),
            vision_conv=P(None, None, None, None, None),
            vision_weight=P(None, None),
        )

    @staticmethod
    def default():
        return ShardingCfg(
            emb_vd=P("tp", "fsdp"),
            emb_dv=P("fsdp", "tp"),
            q_weight_ndh=P("tp", "fsdp", None),
            kv_weight_ndh=P("tp", "fsdp", None),
            o_weight_nhd=P("tp", None, "fsdp"),
            ffw_weight_df=P("fsdp", "tp"),
            ffw_weight_fd=P("tp", "fsdp"),
            rms_norm=P("tp"),
            act_btd=P("fsdp", None, "tp"),
            act_btf=P("fsdp", None, "tp"),
            act_btnh=P("fsdp", None, "tp", None),
            vision_conv=P(None, "fsdp", None, None, None),
            vision_weight=P("tp", "fsdp"),
        )


# =============================================================================
# Model Configurations
# =============================================================================


@dataclasses.dataclass(frozen=True)
class VisionConfig:
    """Configuration for Qwen3-VL Vision Encoder."""

    depth: int = 27
    hidden_size: int = 1152
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 3584
    num_position_embeddings: int = 2304  # 48*48
    deepstack_visual_indexes: Tuple[int, ...] = (8, 16, 24)
    initializer_range: float = 0.02


@dataclasses.dataclass(frozen=True)
class TextConfig:
    """Configuration for Qwen3-VL Text Decoder."""

    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 22016
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 128000
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    rope_theta: float = 5000000.0
    rope_scaling: Optional[dict] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    # M-RoPE: (time, height, width) sections, sum = head_dim / 2
    mrope_section: Tuple[int, int, int] = (24, 20, 20)


@dataclasses.dataclass(frozen=True)
class Qwen3VLConfig:
    """Top-level configuration for Qwen3-VL model."""

    text_config: TextConfig
    vision_config: VisionConfig
    shd_cfg: ShardingCfg = ShardingCfg.no_sharding()

    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    tie_word_embeddings: bool = False

    @classmethod
    def standard_2b(cls):
        """Qwen3-VL-2B Configuration"""
        text_config = TextConfig(
            hidden_size=2048,
            intermediate_size=6144,
            num_hidden_layers=28,
            num_attention_heads=16,
            num_key_value_heads=8,
            hidden_act="silu",
            max_position_embeddings=262144,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            rope_theta=5000000.0,
            mrope_section=(24, 20, 20),
        )
        vision_config = VisionConfig(
            depth=24,
            hidden_size=1024,
            intermediate_size=4096,
            num_heads=16,
            out_hidden_size=2048,
            deepstack_visual_indexes=(5, 11, 17),
        )
        return cls(text_config=text_config, vision_config=vision_config, tie_word_embeddings=True)

    @classmethod
    def standard_4b(cls):
        """Qwen3-VL-4B Configuration"""
        text_config = TextConfig(
            hidden_size=2560,
            intermediate_size=9728,
            num_hidden_layers=36,
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_act="silu",
            max_position_embeddings=262144,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            rope_theta=5000000.0,
            mrope_section=(24, 20, 20),
        )
        vision_config = VisionConfig(
            depth=24,
            hidden_size=1024,
            intermediate_size=4096,
            num_heads=16,
            out_hidden_size=2560,
            deepstack_visual_indexes=(5, 11, 17),
        )
        return cls(text_config=text_config, vision_config=vision_config, tie_word_embeddings=True)

    @classmethod
    def standard_8b(cls):
        """Qwen3-VL-8B Configuration"""
        text_config = TextConfig(
            hidden_size=4096,
            intermediate_size=12288,
            num_hidden_layers=36,
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_act="silu",
            max_position_embeddings=262144,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            rope_theta=5000000.0,
            mrope_section=(24, 20, 20),
        )
        vision_config = VisionConfig(
            depth=27,
            hidden_size=1152,
            intermediate_size=4304,
            num_heads=16,
            out_hidden_size=4096,
            deepstack_visual_indexes=(8, 16, 24),
        )
        return cls(text_config=text_config, vision_config=vision_config, tie_word_embeddings=False)

    @classmethod
    def standard_test(cls):
        """Small configuration for testing - uses minimal dimensions."""
        text_config = TextConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            hidden_act="silu",
            max_position_embeddings=512,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            rope_theta=10000.0,
            mrope_section=(8, 4, 4),  # Sum = 16 = head_dim
        )
        vision_config = VisionConfig(
            depth=2,
            hidden_size=64,
            intermediate_size=256,
            num_heads=4,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=64,
            num_position_embeddings=16,  # 4x4 grid
            deepstack_visual_indexes=(0, 1),
        )
        return cls(text_config=text_config, vision_config=vision_config, tie_word_embeddings=True)


# =============================================================================
# Utility Functions
# =============================================================================


def shard(x: jnp.ndarray, s: ShardingSpec):
    """Reshard tensor to specified partition spec if mesh is available."""
    mesh = get_abstract_mesh()
    if not mesh.empty and len(mesh.axis_names) > 0:
        return reshard(x, s)
    return x


class Einsum(nnx.Module):
    """Einsum layer with sharding support."""

    def __init__(self, einsum_str: str, shape: tuple[int, ...], *, shd: ShardingSpec, rngs: nnx.Rngs):
        self.einsum_str = einsum_str
        self.shape = shape
        self.w = shard(nnx.Param(nnx.initializers.normal()(rngs.params(), shape)), shd)

    @jax.named_scope("einsum")
    def __call__(self, x: ArrayLike) -> Array:
        return jnp.einsum(self.einsum_str, x, self.w.value)


# =============================================================================
# Vision Encoder Modules
# =============================================================================


class VisionPatchEmbed(nnx.Module):
    """3D Patch Embedding via Convolution.

    Converts pixel values into patch embeddings using 3D convolution
    with kernel size (temporal_patch_size, patch_size, patch_size).

    Input: (batch_size, in_channels, temporal, height, width)
    Output: (total_patches, hidden_size)
    """

    def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs):
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = (self.temporal_patch_size, self.patch_size, self.patch_size)
        # Flax Conv expects kernel (T, H, W, In, Out) and input (B, T, H, W, C)
        self.proj = nnx.Conv(
            in_features=self.in_channels,
            out_features=self.embed_dim,
            kernel_size=kernel_size,
            strides=kernel_size,
            use_bias=True,
            rngs=rngs,
        )

    def __call__(self, hidden_states: Array) -> Array:
        """Process pixel values into patch embeddings.

        PyTorch reference: view(-1, in_ch, tp, ps, ps) -> conv -> view(-1, embed_dim)

        Args:
            hidden_states: (batch_size, in_channels, temporal, height, width)
                           or pre-grouped patches

        Returns:
            Flattened patch embeddings: (total_patches, embed_dim)
        """
        # hidden_states: [B, C, T, H, W] or [N, C, tp, ps, ps]
        # Reshape to per-patch groups: [N, C, tp, ps, ps]
        # where N = B * (T/tp) * (H/ps) * (W/ps)
        if hidden_states.ndim == 5:
            b, c, t, h, w = hidden_states.shape
            # Verify dimensions are divisible by patch sizes
            n_t = t // self.temporal_patch_size
            n_h = h // self.patch_size
            n_w = w // self.patch_size

            # Reshape to group patches: [B, C, n_t, tp, n_h, ps, n_w, ps]
            x = hidden_states.reshape(b, c, n_t, self.temporal_patch_size, n_h, self.patch_size, n_w, self.patch_size)
            # Reorder to: [B, n_t, n_h, n_w, tp, ps, ps, C]
            x = x.transpose(0, 2, 4, 6, 3, 5, 7, 1)
            # Flatten batch dims: [B*n_t*n_h*n_w, tp, ps, ps, C]
            x = x.reshape(-1, self.temporal_patch_size, self.patch_size, self.patch_size, c)
        else:
            # Already in [N, C, tp, ps, ps] format - transpose to [N, tp, ps, ps, C]
            x = hidden_states.transpose(0, 2, 3, 4, 1)

        # Apply 3D conv: [N, tp, ps, ps, C] -> [N, 1, 1, 1, embed_dim]
        x = self.proj(x)

        # Flatten to [N, embed_dim]
        x = x.reshape(-1, self.embed_dim)
        return x


class VisionRotaryEmbedding(nnx.Module):
    """Vision Rotary Position Embedding.

    Computes rotary embedding frequencies for 2D spatial positions.
    """

    def __init__(self, dim: int, theta: float = 10000.0):
        self.dim = dim
        self.theta = theta

    def __call__(self, max_seqlen: int) -> Array:
        """Generate frequency table for vision RoPE.

        Args:
            max_seqlen: Maximum grid dimension (height or width)

        Returns:
            Frequency table: (max_seqlen, dim/2)
        """
        inv_freq = 1.0 / (self.theta ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        seq = jnp.arange(max_seqlen, dtype=jnp.float32)
        freqs = jnp.outer(seq, inv_freq)  # [max_seqlen, dim/2]
        return freqs


class VisionPatchMerger(nnx.Module):
    """Merges spatial patches based on spatial_merge_size.

    Takes spatially-grouped patch features and projects them to output dimension.
    """

    def __init__(self, config: VisionConfig, use_postshuffle_norm: bool = False, *, rngs: nnx.Rngs):
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm

        norm_dim = self.hidden_size if use_postshuffle_norm else config.hidden_size
        self.norm = nnx.LayerNorm(norm_dim, epsilon=1e-6, rngs=rngs)
        self.linear_fc1 = nnx.Linear(self.hidden_size, self.hidden_size, rngs=rngs)
        self.linear_fc2 = nnx.Linear(self.hidden_size, config.out_hidden_size, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        """Merge and project patch features.

        Args:
            x: (seq_len, hidden_size)

        Returns:
            (seq_len / merge_size^2, out_hidden_size)
        """
        if self.use_postshuffle_norm:
            # Reshape first, then normalize
            x = x.reshape(-1, self.hidden_size)
            x = self.norm(x)
        else:
            # Normalize first, then reshape
            x = self.norm(x)
            x = x.reshape(-1, self.hidden_size)

        x = nnx.gelu(self.linear_fc1(x))
        x = self.linear_fc2(x)
        return x


def rotate_half(x: Array) -> Array:
    """Rotate half the hidden dims of the input for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb_vision(q: Array, k: Array, cos: Array, sin: Array) -> Tuple[Array, Array]:
    """Apply rotary position embedding to vision query and key tensors.

    Args:
        q, k: Query and key tensors, shape (seq_len, num_heads, head_dim)
        cos, sin: Position embeddings, shape (seq_len, head_dim)

    Returns:
        Rotated q, k with same shapes as input
    """
    orig_dtype = q.dtype
    q = q.astype(jnp.float32)
    k = k.astype(jnp.float32)

    # cos, sin: (seq, head_dim) -> (seq, 1, head_dim) for broadcasting
    cos = cos[:, None, :].astype(jnp.float32)
    sin = sin[:, None, :].astype(jnp.float32)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed.astype(orig_dtype), k_embed.astype(orig_dtype)


class VisionAttention(nnx.Module):
    """Multi-head attention for vision encoder.

    Operates on 2D sequence input (no batch dimension) as in PyTorch reference.
    """

    def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs):
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nnx.Linear(self.dim, self.dim * 3, use_bias=True, rngs=rngs)
        self.proj = nnx.Linear(self.dim, self.dim, use_bias=True, rngs=rngs)

    def __call__(
        self,
        hidden_states: Array,
        cu_seqlens: Optional[Array],
        position_embeddings: Tuple[Array, Array],
    ) -> Array:
        """Forward pass for vision attention.

        Args:
            hidden_states: (seq_len, hidden_size)
            cu_seqlens: Cumulative sequence lengths for variable-length attention
            position_embeddings: (cos, sin) each of shape (seq_len, head_dim)

        Returns:
            (seq_len, hidden_size)
        """
        seq_length = hidden_states.shape[0]

        # QKV projection: (seq, dim) -> (seq, 3*dim)
        qkv = self.qkv(hidden_states)
        # Reshape: (seq, 3*dim) -> (seq, 3, heads, head_dim)
        qkv = qkv.reshape(seq_length, 3, self.num_heads, self.head_dim)
        # Transpose and unbind: (3, seq, heads, head_dim)
        qkv = qkv.transpose(1, 0, 2, 3)
        query, key, value = qkv[0], qkv[1], qkv[2]  # Each: (seq, heads, head_dim)

        # Apply rotary position embeddings
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb_vision(query, key, cos, sin)

        # Transpose for attention: (seq, heads, dim) -> (heads, seq, dim)
        query = query.transpose(1, 0, 2)
        key = key.transpose(1, 0, 2)
        value = value.transpose(1, 0, 2)

        # Attention: (heads, seq, dim) @ (heads, dim, seq) -> (heads, seq, seq)
        attn_weights = jnp.einsum("hqd,hkd->hqk", query, key) * self.scale
        attn_weights = nnx.softmax(attn_weights, axis=-1)

        # Output: (heads, seq, seq) @ (heads, seq, dim) -> (heads, seq, dim)
        attn_output = jnp.einsum("hqk,hkd->hqd", attn_weights, value)

        # Transpose back: (heads, seq, dim) -> (seq, heads, dim) -> (seq, hidden)
        attn_output = attn_output.transpose(1, 0, 2).reshape(seq_length, -1)
        return self.proj(attn_output)


class VisionMLP(nnx.Module):
    """MLP for vision encoder blocks."""

    def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(config.hidden_size, config.intermediate_size, use_bias=True, rngs=rngs)
        self.fc2 = nnx.Linear(config.intermediate_size, config.hidden_size, use_bias=True, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        return self.fc2(nnx.gelu(self.fc1(x)))


class VisionBlock(nnx.Module):
    """Transformer block for vision encoder."""

    def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs):
        self.norm1 = nnx.LayerNorm(config.hidden_size, epsilon=1e-6, rngs=rngs)
        self.norm2 = nnx.LayerNorm(config.hidden_size, epsilon=1e-6, rngs=rngs)
        self.attn = VisionAttention(config, rngs=rngs)
        self.mlp = VisionMLP(config, rngs=rngs)

    def __call__(
        self,
        hidden_states: Array,
        cu_seqlens: Optional[Array],
        position_embeddings: Tuple[Array, Array],
    ) -> Array:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens, position_embeddings)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class VisionModel(nnx.Module):
    """Qwen3-VL Vision Encoder.

    Processes image/video pixels through patch embedding, position embedding,
    transformer blocks, and patch merging for output to language model.
    """

    def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size**2

        self.patch_embed = VisionPatchEmbed(config, rngs=rngs)
        self.pos_embed = nnx.Embed(config.num_position_embeddings, config.hidden_size, rngs=rngs)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nnx.List([VisionBlock(config, rngs=rngs) for _ in range(config.depth)])
        self.merger = VisionPatchMerger(config, use_postshuffle_norm=False, rngs=rngs)

        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = nnx.List(
            [VisionPatchMerger(config, use_postshuffle_norm=True, rngs=rngs) for _ in config.deepstack_visual_indexes]
        )

    def rot_pos_emb(self, grid_thw: Array) -> Array:
        """Generate 2D rotary position embeddings for vision tokens.

        Args:
            grid_thw: (num_images, 3) tensor of (temporal, height, width) per image

        Returns:
            (total_tokens, head_dim) position embeddings
        """
        merge_size = self.spatial_merge_size

        # Get maximum grid dimension for frequency table
        max_hw = int(jnp.max(grid_thw[:, 1:]))
        head_dim = self.config.hidden_size // self.config.num_heads
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, head_dim/2)

        # Calculate total number of tokens
        # After patch embed and before merge: t * h * w tokens per image
        total_tokens = int(jnp.sum(grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]))
        pos_ids = jnp.zeros((total_tokens, 2), dtype=jnp.int32)

        # Generate position IDs for each image/video
        offset = 0
        for i in range(grid_thw.shape[0]):
            t, h, w = int(grid_thw[i, 0]), int(grid_thw[i, 1]), int(grid_thw[i, 2])
            merged_h, merged_w = h // merge_size, w // merge_size

            # Create block and intra-block indices
            # Block row/col indices
            block_rows = jnp.arange(merged_h)
            block_cols = jnp.arange(merged_w)
            # Intra-block offsets
            intra_row = jnp.arange(merge_size)
            intra_col = jnp.arange(merge_size)

            # Compute full-resolution positions
            # row_idx: (merged_h, 1, merge_size, 1) + (1, 1, merge_size, 1)
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            # Expand and flatten
            row_idx = jnp.broadcast_to(row_idx, (merged_h, merged_w, merge_size, merge_size)).reshape(-1)
            col_idx = jnp.broadcast_to(col_idx, (merged_h, merged_w, merge_size, merge_size)).reshape(-1)

            coords = jnp.stack([row_idx, col_idx], axis=-1)  # (h*w, 2)

            # Repeat for temporal dimension
            if t > 1:
                coords = jnp.tile(coords, (t, 1))

            num_tokens = coords.shape[0]
            pos_ids = pos_ids.at[offset : offset + num_tokens].set(coords)
            offset += num_tokens

        # Look up rotary embeddings: (total_tokens, 2) -> (total_tokens, 2, head_dim/2)
        embeddings = freq_table[pos_ids]  # (total_tokens, 2, head_dim/2)
        embeddings = embeddings.reshape(total_tokens, -1)  # (total_tokens, head_dim)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw: Array) -> Array:
        """Bilinear interpolation of learned position embeddings.

        Args:
            grid_thw: (num_images, 3) tensor of (temporal, height, width) per image

        Returns:
            (total_tokens, hidden_size) interpolated position embeddings
        """
        merge_size = self.spatial_merge_size
        num_grid = self.num_grid_per_side
        total_patches = int(jnp.sum(grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]))

        # For simplicity, compute position embeddings assuming uniform grid
        # In production, this should use bilinear interpolation as in PyTorch
        all_embeds = []
        for i in range(grid_thw.shape[0]):
            t, h, w = int(grid_thw[i, 0]), int(grid_thw[i, 1]), int(grid_thw[i, 2])

            # Generate normalized coordinates for this grid
            h_coords = jnp.linspace(0, num_grid - 1, h).astype(jnp.int32)
            w_coords = jnp.linspace(0, num_grid - 1, w).astype(jnp.int32)

            # Create 2D grid of position indices
            h_grid, w_grid = jnp.meshgrid(h_coords, w_coords, indexing="ij")
            pos_indices = h_grid * num_grid + w_grid  # (h, w)
            pos_indices = pos_indices.flatten()  # (h*w,)

            # Get position embeddings
            pos_embeds = self.pos_embed(pos_indices)  # (h*w, hidden)

            # Repeat for temporal dimension
            pos_embeds = jnp.tile(pos_embeds, (t, 1))  # (t*h*w, hidden)

            # Permute for spatial merge grouping: [t, h/m, w/m, m, m, hidden]
            pos_embeds = pos_embeds.reshape(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
            pos_embeds = pos_embeds.transpose(0, 1, 3, 2, 4, 5)  # [t, h/m, w/m, m, m, hidden]
            pos_embeds = pos_embeds.reshape(-1, self.config.hidden_size)

            all_embeds.append(pos_embeds)

        return jnp.concatenate(all_embeds, axis=0)

    def __call__(self, hidden_states: Array, grid_thw: Array) -> Tuple[Array, List[Array]]:
        """Forward pass through vision encoder.

        Args:
            hidden_states: Pixel values (batch, channels, temporal, height, width)
            grid_thw: (num_images, 3) of (temporal, height, width) per image

        Returns:
            Tuple of:
                - Final merged visual embeddings: (num_merged_tokens, out_hidden_size)
                - List of deepstack feature tensors for intermediate layer outputs
        """
        # Patch embedding: (B, C, T, H, W) -> (total_patches, hidden_size)
        hidden_states = self.patch_embed(hidden_states)

        # Add interpolated position embeddings
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        # Generate rotary position embeddings
        rot_emb = self.rot_pos_emb(grid_thw)  # (total_patches, head_dim)
        emb = jnp.concatenate([rot_emb, rot_emb], axis=-1)  # (total_patches, 2*head_dim)
        position_embeddings = (jnp.cos(emb), jnp.sin(emb))

        # Compute cu_seqlens for variable-length attention (optional)
        cu_seqlens = None  # Not used in eager attention

        # Transformer blocks
        deepstack_features = []
        for layer_num, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, cu_seqlens, position_embeddings)

            # Extract deepstack features at specified layers
            if layer_num in self.deepstack_visual_indexes:
                idx = list(self.deepstack_visual_indexes).index(layer_num)
                ds_feat = self.deepstack_merger_list[idx](hidden_states)
                deepstack_features.append(ds_feat)

        # Final merge
        hidden_states = self.merger(hidden_states)

        return hidden_states, deepstack_features


# =============================================================================
# Text Decoder Modules
# =============================================================================


class LayerCache(nnx.Module):
    """KV Cache for a single decoder layer."""

    def __init__(self, cfg: TextConfig, batch_size: int, cache_size: int, dtype: jnp.dtype):
        cache_shape = (batch_size, cache_size, cfg.num_key_value_heads, cfg.head_dim)
        self.k_cache = nnx.Cache(jnp.zeros(cache_shape, dtype=dtype))
        self.v_cache = nnx.Cache(jnp.zeros(cache_shape, dtype=dtype))
        self.cur_ind = nnx.Variable(jnp.zeros((), dtype=jnp.int32))


class RMSNorm(nnx.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6, *, rngs: nnx.Rngs):
        self.weight = nnx.Param(jax.nn.initializers.ones(rngs.params(), (dim,)))
        self.eps = eps

    def __call__(self, x: Array) -> Array:
        dtype = x.dtype
        x = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.eps)
        return (self.weight.value * x).astype(dtype)


def apply_interleaved_mrope(freqs: Array, mrope_section: Tuple[int, int, int]) -> Array:
    """Apply interleaved M-RoPE to 3D rotary embeddings.

    Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
    interleaved [THW THW THW...TT], preserving frequency continuity.

    Args:
        freqs: (3, batch, seq_len, head_dim // 2) frequencies for T, H, W
        mrope_section: (t_size, h_size, w_size) sections

    Returns:
        Interleaved frequencies: (batch, seq_len, head_dim // 2)
    """
    # Start with temporal frequencies as base
    freqs_t = freqs[0].copy()  # (batch, seq, head_dim/2)

    # Interleave height frequencies at positions 1, 4, 7, ...
    h_length = mrope_section[1] * 3
    for i in range(1, h_length, 3):
        if i < freqs_t.shape[-1]:
            freqs_t = freqs_t.at[..., i].set(freqs[1, ..., i])

    # Interleave width frequencies at positions 2, 5, 8, ...
    w_length = mrope_section[2] * 3
    for i in range(2, w_length, 3):
        if i < freqs_t.shape[-1]:
            freqs_t = freqs_t.at[..., i].set(freqs[2, ..., i])

    return freqs_t


class TextAttention(nnx.Module):
    """Multi-head attention with M-RoPE for text decoder."""

    def __init__(self, config: TextConfig, layer_idx: int, *, rngs: nnx.Rngs):
        self.config = config
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nnx.Linear(
            config.hidden_size, self.num_heads * self.head_dim, use_bias=config.attention_bias, rngs=rngs
        )
        self.k_proj = nnx.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, use_bias=config.attention_bias, rngs=rngs
        )
        self.v_proj = nnx.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, use_bias=config.attention_bias, rngs=rngs
        )
        self.o_proj = nnx.Linear(
            self.num_heads * self.head_dim, config.hidden_size, use_bias=config.attention_bias, rngs=rngs
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, rngs=rngs)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, rngs=rngs)

    def __call__(
        self,
        x: Array,
        cache: Optional[LayerCache],
        sin: Array,
        cos: Array,
        attention_mask: Optional[Array] = None,
    ) -> Array:
        """Forward pass for text attention.

        Args:
            x: (batch, seq, hidden_size)
            cache: Optional KV cache
            sin, cos: (batch, seq, head_dim) position embeddings
            attention_mask: Optional attention mask

        Returns:
            (batch, seq, hidden_size)
        """
        B, S, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).reshape(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, S, self.num_kv_heads, self.head_dim)

        # Apply QK norms
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply rotary position embeddings
        # cos, sin: (B, S, head_dim) -> (B, S, 1, head_dim)
        cos_expanded = cos[:, :, None, :]
        sin_expanded = sin[:, :, None, :]

        def apply_rope(x, c, s):
            # x: (B, S, H, head_dim)
            # c, s: (B, S, 1, head_dim)
            # Use rotate_half pattern: (x * cos) + (rotate_half(x) * sin)
            return (x * c) + (rotate_half(x) * s)

        q = apply_rope(q, cos_expanded, sin_expanded)
        k = apply_rope(k, cos_expanded, sin_expanded)

        # Update cache if provided
        if cache is not None:
            idx = cache.cur_ind.value
            cache.k_cache.value = jax.lax.dynamic_update_slice(cache.k_cache.value, k, (0, idx, 0, 0))
            cache.v_cache.value = jax.lax.dynamic_update_slice(cache.v_cache.value, v, (0, idx, 0, 0))
            cache.cur_ind.value = idx + S
            # Use cached K, V
            k = cache.k_cache.value[:, : idx + S]
            v = cache.v_cache.value[:, : idx + S]

        # Repeat K, V for grouped query attention
        if self.num_kv_groups > 1:
            k = jnp.repeat(k, self.num_kv_groups, axis=2)
            v = jnp.repeat(v, self.num_kv_groups, axis=2)

        # Transpose for attention: (B, S, H, D) -> (B, H, S, D)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Compute attention
        attn = jnp.einsum("bhid,bhjd->bhij", q, k) * self.scale

        # Apply attention mask if provided
        if attention_mask is not None:
            attn = attn + attention_mask

        attn = nnx.softmax(attn, axis=-1)
        out = jnp.einsum("bhij,bhjd->bhid", attn, v)

        # Transpose back and project
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.o_proj(out)


class TextMLP(nnx.Module):
    """Gated MLP for text decoder."""

    def __init__(self, config: TextConfig, *, rngs: nnx.Rngs):
        self.gate_proj = nnx.Linear(config.hidden_size, config.intermediate_size, use_bias=False, rngs=rngs)
        self.up_proj = nnx.Linear(config.hidden_size, config.intermediate_size, use_bias=False, rngs=rngs)
        self.down_proj = nnx.Linear(config.intermediate_size, config.hidden_size, use_bias=False, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        return self.down_proj(nnx.silu(self.gate_proj(x)) * self.up_proj(x))


class TextDecoderLayer(nnx.Module):
    """Transformer decoder layer for text."""

    def __init__(self, config: TextConfig, layer_idx: int, *, rngs: nnx.Rngs):
        self.self_attn = TextAttention(config, layer_idx, rngs=rngs)
        self.mlp = TextMLP(config, rngs=rngs)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, rngs=rngs)

    def __call__(
        self,
        x: Array,
        cache: Optional[LayerCache],
        sin: Array,
        cos: Array,
        attention_mask: Optional[Array] = None,
    ) -> Array:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, cache, sin, cos, attention_mask)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


# =============================================================================
# Complete Qwen3-VL Model
# =============================================================================


class Qwen3VLModel(nnx.Module):
    """Qwen3-VL Multimodal Model.

    Combines vision encoder and text decoder for multimodal understanding.
    """

    def __init__(self, config: Qwen3VLConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.visual = VisionModel(config.vision_config, rngs=rngs)

        text_cfg = config.text_config
        self.embed_tokens = nnx.Embed(text_cfg.vocab_size, text_cfg.hidden_size, rngs=rngs)
        self.layers = nnx.List([TextDecoderLayer(text_cfg, i, rngs=rngs) for i in range(text_cfg.num_hidden_layers)])
        self.norm = RMSNorm(text_cfg.hidden_size, eps=text_cfg.rms_norm_eps, rngs=rngs)

    def _compute_mrope_position_embeddings(self, x: Array, position_ids: Optional[Array] = None) -> Tuple[Array, Array]:
        """Compute M-RoPE position embeddings for text.

        Args:
            x: (batch, seq, hidden) input tensor
            position_ids: Optional (3, batch, seq) or None for simple linear positions

        Returns:
            (sin, cos) each of shape (batch, seq, head_dim)
        """
        B, S, _ = x.shape
        head_dim = self.config.text_config.head_dim
        theta = self.config.text_config.rope_theta
        mrope_section = self.config.text_config.mrope_section

        # Compute inverse frequencies
        inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))

        if position_ids is None:
            # Simple linear positions for all 3 dimensions (T, H, W)
            pos = jnp.arange(S, dtype=jnp.float32)
            pos = pos[None, :].repeat(B, axis=0)  # (B, S)
            # Expand to 3D: (3, B, S)
            position_ids = jnp.stack([pos, pos, pos], axis=0)

        # position_ids: (3, B, S)
        # inv_freq: (head_dim/2,)
        # Result freqs: (3, B, S, head_dim/2)
        freqs = jnp.einsum("dbs,h->dbsh", position_ids, inv_freq)

        # Apply interleaved M-RoPE
        final_freqs = apply_interleaved_mrope(freqs, mrope_section)  # (B, S, head_dim/2)

        # Double for full head_dim
        emb = jnp.concatenate([final_freqs, final_freqs], axis=-1)  # (B, S, head_dim)

        cos = jnp.cos(emb)
        sin = jnp.sin(emb)

        return sin, cos

    def _deepstack_process(
        self,
        hidden_states: Array,
        visual_pos_masks: Array,
        visual_embeds: Array,
    ) -> Array:
        """Add deepstack visual embeddings to hidden states at visual positions.

        Args:
            hidden_states: (batch, seq, hidden_size)
            visual_pos_masks: (batch, seq) boolean mask
            visual_embeds: (num_visual_tokens, hidden_size)

        Returns:
            Updated hidden_states with visual embeddings added
        """
        # In JAX, we need to scatter the visual embeddings to the correct positions
        # For simplicity, broadcast and multiply by mask
        B, S, D = hidden_states.shape

        # Find positions where visual tokens should be inserted
        # visual_embeds: (num_vis, D) -> need to project to (B, S, D)
        # Use mask to place embeddings

        # Flatten for scatter operation
        flat_mask = visual_pos_masks.flatten()  # (B*S,)
        num_visual = int(jnp.sum(flat_mask))

        if num_visual > 0 and visual_embeds is not None:
            flat_states = hidden_states.reshape(-1, D)  # (B*S, D)
            # Get indices where mask is True
            indices = jnp.where(flat_mask, size=num_visual)[0]
            # Add visual embeddings at those positions
            flat_states = flat_states.at[indices].add(visual_embeds)
            hidden_states = flat_states.reshape(B, S, D)

        return hidden_states

    def __call__(
        self,
        input_ids: Array,
        pixel_values: Optional[Array] = None,
        grid_thw: Optional[Array] = None,
        visual_pos_masks: Optional[Array] = None,
        position_ids: Optional[Array] = None,
        attention_mask: Optional[Array] = None,
        cache: Optional[List[LayerCache]] = None,
    ) -> Array:
        """Forward pass for Qwen3-VL.

        Args:
            input_ids: (batch, seq) token IDs
            pixel_values: Optional (batch, channels, temporal, height, width) pixels
            grid_thw: Optional (num_images, 3) grid dimensions
            visual_pos_masks: Optional (batch, seq) boolean mask for visual positions
            position_ids: Optional (3, batch, seq) for M-RoPE
            attention_mask: Optional attention mask
            cache: Optional list of LayerCache for KV caching

        Returns:
            (batch, seq, hidden_size) final hidden states
        """
        # 1. Process vision inputs if provided
        visual_embeds = None
        deepstack_embeds = None
        if pixel_values is not None and grid_thw is not None:
            visual_embeds, deepstack_embeds = self.visual(pixel_values, grid_thw)

        # 2. Text embedding
        x = self.embed_tokens(input_ids)

        # 3. Compute M-RoPE position embeddings
        sin, cos = self._compute_mrope_position_embeddings(x, position_ids)

        # 4. Apply text decoder layers
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x = layer(x, layer_cache, sin, cos, attention_mask)

            # 5. Apply deepstack visual embeddings at specified layers
            if deepstack_embeds is not None and visual_pos_masks is not None:
                if i < len(deepstack_embeds):
                    x = self._deepstack_process(x, visual_pos_masks, deepstack_embeds[i])

        # 6. Final normalization
        x = self.norm(x)

        return x
