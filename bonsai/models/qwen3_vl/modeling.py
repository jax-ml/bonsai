import math
from dataclasses import dataclass
from typing import Optional, Tuple, TypeAlias

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array, P
from jax.sharding import PartitionSpec, get_abstract_mesh, reshard

_K_MASK = jnp.finfo(jnp.bfloat16).min


# --- Sharding Configuration --- #
@dataclass(slots=True, frozen=True)
class VisionShardingCfg:
    """Sharding configuration for vision encoder components."""

    attn_qkv_kernel: PartitionSpec
    attn_proj_kernel: PartitionSpec
    mlp_fc1_kernel: PartitionSpec
    mlp_fc2_kernel: PartitionSpec
    layer_norm: PartitionSpec
    activation: PartitionSpec
    patch_embed_kernel: PartitionSpec
    pos_embed: PartitionSpec

    @staticmethod
    def no_sharding():
        return VisionShardingCfg(
            attn_qkv_kernel=P(None, None),
            attn_proj_kernel=P(None, None),
            mlp_fc1_kernel=P(None, None),
            mlp_fc2_kernel=P(None, None),
            layer_norm=P(None),
            activation=P(None, None),
            patch_embed_kernel=P(None, None, None, None, None),
            pos_embed=P(None, None),
        )

    @staticmethod
    def default(use_fsdp: bool, use_tp: bool):
        fsdp = "fsdp" if use_fsdp else None
        tp = "tp" if use_tp else None
        return VisionShardingCfg(
            attn_qkv_kernel=P(fsdp, tp),
            attn_proj_kernel=P(tp, fsdp),
            mlp_fc1_kernel=P(fsdp, tp),
            mlp_fc2_kernel=P(tp, fsdp),
            layer_norm=P(tp),
            activation=P(fsdp, tp),
            patch_embed_kernel=P(None, None, None, None, tp),
            pos_embed=P(None, tp),
        )


@dataclass(slots=True, frozen=True)
class TextShardingCfg:
    """Sharding configuration for text decoder components."""

    q_weight: PartitionSpec
    kv_weight: PartitionSpec
    o_weight: PartitionSpec
    mlp_gate_up_kernel: PartitionSpec
    mlp_down_kernel: PartitionSpec
    rms_norm: PartitionSpec
    embed_kernel: PartitionSpec
    cache: PartitionSpec
    act_btd: PartitionSpec
    act_btf: PartitionSpec
    act_btnh: PartitionSpec

    @staticmethod
    def no_sharding():
        return TextShardingCfg(
            q_weight=P(None, None),
            kv_weight=P(None, None),
            o_weight=P(None, None),
            mlp_gate_up_kernel=P(None, None),
            mlp_down_kernel=P(None, None),
            rms_norm=P(None),
            embed_kernel=P(None, None),
            cache=P(None, None, None, None),
            act_btd=P(None, None, None),
            act_btf=P(None, None, None),
            act_btnh=P(None, None, None, None),
        )

    @staticmethod
    def default(use_fsdp: bool, use_tp: bool):
        fsdp = "fsdp" if use_fsdp else None
        tp = "tp" if use_tp else None
        return TextShardingCfg(
            q_weight=P(fsdp, tp),
            kv_weight=P(fsdp, tp),
            o_weight=P(tp, fsdp),
            mlp_gate_up_kernel=P(fsdp, tp),
            mlp_down_kernel=P(tp, fsdp),
            rms_norm=P(tp),
            embed_kernel=P(tp, fsdp),
            cache=P(fsdp, None, tp, None),
            act_btd=P(fsdp, None, tp),
            act_btf=P(fsdp, None, tp),
            act_btnh=P(fsdp, None, tp, None),
        )


def shard(x, spec: PartitionSpec):
    """Reshard tensor according to partition spec if mesh is available."""
    mesh = get_abstract_mesh()
    if not mesh.empty and len(mesh.axis_names) > 0:
        return reshard(x, spec)
    return x


@dataclass(frozen=True)
class Qwen3VLVisionConfig:
    """Vision encoder configuration for Qwen3-VL."""

    depth: int = 24
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    out_hidden_size: int = 2048
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: tuple = (5, 11, 17)
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    shd_cfg: VisionShardingCfg = VisionShardingCfg.no_sharding()

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @classmethod
    def qwen3vl_2b(cls, use_fsdp: bool = False, use_tp: bool = False):
        shd = VisionShardingCfg.default(use_fsdp, use_tp) if (use_fsdp or use_tp) else VisionShardingCfg.no_sharding()
        return cls(
            depth=24,
            hidden_size=1024,
            intermediate_size=4096,
            num_heads=16,
            out_hidden_size=2048,
            deepstack_visual_indexes=(5, 11, 17),
            shd_cfg=shd,
        )

    @classmethod
    def qwen3vl_4b(cls, use_fsdp: bool = False, use_tp: bool = False):
        shd = VisionShardingCfg.default(use_fsdp, use_tp) if (use_fsdp or use_tp) else VisionShardingCfg.no_sharding()
        return cls(
            depth=24,
            hidden_size=1024,
            intermediate_size=4096,
            num_heads=16,
            out_hidden_size=2560,
            deepstack_visual_indexes=(5, 11, 17),
            shd_cfg=shd,
        )

    @classmethod
    def qwen3vl_8b(cls, use_fsdp: bool = False, use_tp: bool = False):
        shd = VisionShardingCfg.default(use_fsdp, use_tp) if (use_fsdp or use_tp) else VisionShardingCfg.no_sharding()
        return cls(
            depth=27,
            hidden_size=1152,
            intermediate_size=4304,
            num_heads=16,
            out_hidden_size=4096,
            deepstack_visual_indexes=(8, 16, 24),
            shd_cfg=shd,
        )

    @classmethod
    def qwen3vl_32b(cls, use_fsdp: bool = False, use_tp: bool = False):
        shd = VisionShardingCfg.default(use_fsdp, use_tp) if (use_fsdp or use_tp) else VisionShardingCfg.no_sharding()
        return cls(
            depth=27,
            hidden_size=1152,
            intermediate_size=4304,
            num_heads=16,
            out_hidden_size=5120,
            deepstack_visual_indexes=(8, 16, 24),
            shd_cfg=shd,
        )


@dataclass(frozen=True)
class Qwen3VLTextConfig:
    """Text decoder configuration for Qwen3-VL."""

    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5_000_000
    mrope_section: tuple = (24, 20, 20)  # T, H, W partitions of head_dim
    attention_bias: bool = False
    tie_word_embeddings: bool = True
    shd_cfg: TextShardingCfg = TextShardingCfg.no_sharding()

    @classmethod
    def qwen3vl_2b(cls, use_fsdp: bool = False, use_tp: bool = False):
        shd = TextShardingCfg.default(use_fsdp, use_tp) if (use_fsdp or use_tp) else TextShardingCfg.no_sharding()
        return cls(
            hidden_size=2048,
            intermediate_size=6144,
            num_hidden_layers=28,
            num_attention_heads=16,
            num_key_value_heads=8,
            tie_word_embeddings=True,
            shd_cfg=shd,
        )

    @classmethod
    def qwen3vl_4b(cls, use_fsdp: bool = False, use_tp: bool = False):
        shd = TextShardingCfg.default(use_fsdp, use_tp) if (use_fsdp or use_tp) else TextShardingCfg.no_sharding()
        return cls(
            hidden_size=2560,
            intermediate_size=9728,
            num_hidden_layers=36,
            num_attention_heads=32,
            num_key_value_heads=8,
            tie_word_embeddings=True,
            shd_cfg=shd,
        )

    @classmethod
    def qwen3vl_8b(cls, use_fsdp: bool = False, use_tp: bool = False):
        shd = TextShardingCfg.default(use_fsdp, use_tp) if (use_fsdp or use_tp) else TextShardingCfg.no_sharding()
        return cls(
            hidden_size=4096,
            intermediate_size=12288,
            num_hidden_layers=36,
            num_attention_heads=32,
            num_key_value_heads=8,
            tie_word_embeddings=False,
            shd_cfg=shd,
        )

    @classmethod
    def qwen3vl_32b(cls, use_fsdp: bool = False, use_tp: bool = False):
        shd = TextShardingCfg.default(use_fsdp, use_tp) if (use_fsdp or use_tp) else TextShardingCfg.no_sharding()
        return cls(
            hidden_size=5120,
            intermediate_size=25600,
            num_hidden_layers=64,
            num_attention_heads=64,
            num_key_value_heads=8,
            tie_word_embeddings=False,
            shd_cfg=shd,
        )


@dataclass(frozen=True)
class Qwen3VLConfig:
    """Combined configuration for Qwen3-VL model."""

    vision_config: Qwen3VLVisionConfig
    text_config: Qwen3VLTextConfig
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653

    @classmethod
    def qwen3vl_2b(cls, use_fsdp: bool = False, use_tp: bool = False):
        """Qwen3-VL 2B configuration."""
        return cls(
            vision_config=Qwen3VLVisionConfig.qwen3vl_2b(use_fsdp, use_tp),
            text_config=Qwen3VLTextConfig.qwen3vl_2b(use_fsdp, use_tp),
        )

    @classmethod
    def qwen3vl_4b(cls, use_fsdp: bool = False, use_tp: bool = False):
        """Qwen3-VL 4B configuration."""
        return cls(
            vision_config=Qwen3VLVisionConfig.qwen3vl_4b(use_fsdp, use_tp),
            text_config=Qwen3VLTextConfig.qwen3vl_4b(use_fsdp, use_tp),
        )

    @classmethod
    def qwen3vl_8b(cls, use_fsdp: bool = False, use_tp: bool = False):
        """Qwen3-VL 8B configuration."""
        return cls(
            vision_config=Qwen3VLVisionConfig.qwen3vl_8b(use_fsdp, use_tp),
            text_config=Qwen3VLTextConfig.qwen3vl_8b(use_fsdp, use_tp),
        )

    @classmethod
    def qwen3vl_32b(cls, use_fsdp: bool = False, use_tp: bool = False):
        return cls(
            vision_config=Qwen3VLVisionConfig.qwen3vl_32b(use_fsdp, use_tp),
            text_config=Qwen3VLTextConfig.qwen3vl_32b(use_fsdp, use_tp),
        )


class RMSNorm(nnx.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6, *, rngs: nnx.Rngs):
        self.eps = eps
        self.weight = nnx.Param(jnp.ones((dim,), dtype=jnp.float32))

    def __call__(self, x: Array) -> Array:
        x_f32 = x.astype(jnp.float32)
        rms = jax.lax.rsqrt(jnp.mean(x_f32**2, axis=-1, keepdims=True) + self.eps)
        out = (x_f32 * rms) * self.weight[...]
        return out.astype(x.dtype)


class Qwen3VLPatchEmbed(nnx.Module):
    """3D Convolutional patch embedding for vision input using nnx.Conv."""

    def __init__(self, config: Qwen3VLVisionConfig, *, rngs: nnx.Rngs):
        self.config = config
        kernel = (config.temporal_patch_size, config.patch_size, config.patch_size)
        self.proj = nnx.Conv(
            in_features=config.in_channels,
            out_features=config.hidden_size,
            kernel_size=kernel,
            strides=kernel,
            use_bias=True,
            rngs=rngs,
        )

    def __call__(self, hidden_states: Array) -> Array:
        # Input: (num_patches, in_channels * temporal_patch_size * patch_size * patch_size)
        cfg = self.config
        seq_len = hidden_states.shape[0]

        hidden_states = hidden_states.reshape(
            seq_len, cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size
        )
        # (seq, C, D, H, W) -> (seq, D, H, W, C)
        hidden_states = hidden_states.transpose(0, 2, 3, 4, 1)

        # Apply conv: input (seq, D, H, W, C) -> output (seq , hidden_size)
        return self.proj(hidden_states).reshape(seq_len, cfg.hidden_size)


class Qwen3VLVisionMLP(nnx.Module):
    """Vision encoder MLP with GELU activation."""

    def __init__(self, config: Qwen3VLVisionConfig, *, rngs: nnx.Rngs):
        self.shd_cfg = config.shd_cfg
        self.linear_fc1 = nnx.Linear(config.hidden_size, config.intermediate_size, use_bias=True, rngs=rngs)
        self.linear_fc2 = nnx.Linear(config.intermediate_size, config.hidden_size, use_bias=True, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        x = self.linear_fc1(x)
        x = shard(x, self.shd_cfg.activation)
        x = nnx.gelu(x, approximate=True)
        x = self.linear_fc2(x)
        return x


class Qwen3VLVisionAttention(nnx.Module):
    """Vision encoder multi-head attention with RoPE."""

    def __init__(self, config: Qwen3VLVisionConfig, *, rngs: nnx.Rngs):
        self.shd_cfg = config.shd_cfg
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        hidden_size = config.hidden_size
        self.qkv = nnx.Linear(hidden_size, 3 * hidden_size, use_bias=True, rngs=rngs)
        self.proj = nnx.Linear(hidden_size, hidden_size, use_bias=True, rngs=rngs)
        self.scale = self.head_dim**-0.5

    def apply_rope(self, cos: Array, sin: Array, q: Array, k: Array):
        # Apply RoPE - split cos/sin into halves for the half-rotation
        half_dim = self.head_dim // 2
        cos1, cos2 = cos[:, None, :half_dim], cos[:, None, half_dim:]  # (seq, head_dim//2)
        sin1, sin2 = sin[:, None, :half_dim], sin[:, None, half_dim:]

        q1, q2 = q[..., :half_dim], q[..., half_dim:]  # (seq, heads, half_dim)
        q = jnp.concatenate([q1 * cos1 - q2 * sin1, q2 * cos2 + q1 * sin2], axis=-1)
        k1, k2 = k[..., :half_dim], k[..., half_dim:]
        k = jnp.concatenate([k1 * cos1 - k2 * sin1, k2 * cos2 + k1 * sin2], axis=-1)

        return (q, k)

    def __call__(self, hidden_states: Array, position_embeddings: Tuple[Array, Array]) -> Array:
        seq_len = hidden_states.shape[0]
        cos, sin = position_embeddings  # (seq_len, head_dim)
        qkv = self.qkv(hidden_states).reshape(seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # (seq, heads, head_dim)

        q, k = self.apply_rope(cos, sin, q, k)

        q, k, v = q.transpose(1, 0, 2), k.transpose(1, 0, 2), v.transpose(1, 0, 2)
        attn_weights = jnp.matmul(q, k.transpose(0, 2, 1)) * self.scale
        attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(q.dtype)
        out = jnp.matmul(attn_weights, v).transpose(1, 0, 2).reshape(seq_len, -1)
        return shard(self.proj(out), self.shd_cfg.activation)


class Qwen3VLVisionBlock(nnx.Module):
    """Single transformer block for vision encoder."""

    def __init__(self, config: Qwen3VLVisionConfig, *, rngs: nnx.Rngs):
        self.norm1 = nnx.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps, rngs=rngs)
        self.norm2 = nnx.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps, rngs=rngs)
        self.attn = Qwen3VLVisionAttention(config, rngs=rngs)
        self.mlp = Qwen3VLVisionMLP(config, rngs=rngs)

    def __call__(self, hidden_states: Array, position_embeddings: Tuple[Array, Array]) -> Array:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(hidden_states, position_embeddings)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen3VLPatchMerger(nnx.Module):
    """Merge spatial patches after vision encoding."""

    def __init__(self, config: Qwen3VLVisionConfig, use_postshuffle_norm: bool = False, *, rngs: nnx.Rngs):
        self.config = config
        self.shd_cfg = config.shd_cfg
        merge_factor = config.spatial_merge_size**2
        hidden_merged = config.hidden_size * merge_factor
        norm_dim = hidden_merged if use_postshuffle_norm else config.hidden_size
        self.norm = nnx.LayerNorm(norm_dim, epsilon=config.layer_norm_eps, rngs=rngs)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.linear_fc1 = nnx.Linear(hidden_merged, hidden_merged, use_bias=True, rngs=rngs)
        self.linear_fc2 = nnx.Linear(hidden_merged, config.out_hidden_size, use_bias=True, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        if not self.use_postshuffle_norm:
            x = self.norm(x)
        merge_factor = self.config.spatial_merge_size**2
        n_patches = x.shape[0] // merge_factor
        x = x.reshape(n_patches, -1)
        if self.use_postshuffle_norm:
            x = self.norm(x)
        x = self.linear_fc1(x)
        x = shard(x, self.shd_cfg.activation)
        x = nnx.gelu(x)
        x = self.linear_fc2(x)
        return x


class Qwen3VLVisionModel(nnx.Module):
    """Complete vision encoder with deepstack feature extraction."""

    def __init__(self, config: Qwen3VLVisionConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.shd_cfg = config.shd_cfg
        self.patch_embed = Qwen3VLPatchEmbed(config, rngs=rngs)
        self.pos_embed = nnx.Embed(
            num_embeddings=config.num_position_embeddings, features=config.hidden_size, rngs=rngs
        )
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        self.blocks = nnx.List([Qwen3VLVisionBlock(config, rngs=rngs) for _ in range(config.depth)])
        self.merger = Qwen3VLPatchMerger(config, use_postshuffle_norm=False, rngs=rngs)
        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = nnx.List(
            [
                Qwen3VLPatchMerger(config, use_postshuffle_norm=True, rngs=rngs)
                for _ in range(len(config.deepstack_visual_indexes))
            ]
        )

    def _fast_pos_embed_interpolate(self, grid_thw: Array) -> Array:
        """Bilinear interpolation for position embeddings, matching PyTorch."""
        grid_h, grid_w = int(grid_thw[0, 1]), int(grid_thw[0, 2])

        # Create interpolation indices
        h_idxs = jnp.linspace(0, self.num_grid_per_side - 1, grid_h)
        w_idxs = jnp.linspace(0, self.num_grid_per_side - 1, grid_w)

        h_floor = jnp.floor(h_idxs).astype(jnp.int32)
        w_floor = jnp.floor(w_idxs).astype(jnp.int32)
        h_ceil = jnp.clip(h_floor + 1, 0, self.num_grid_per_side - 1)
        w_ceil = jnp.clip(w_floor + 1, 0, self.num_grid_per_side - 1)

        dh = h_idxs - h_floor
        dw = w_idxs - w_floor

        # 2D grid indices for 4 corners
        base_h = h_floor * self.num_grid_per_side
        base_h_ceil = h_ceil * self.num_grid_per_side

        idx00 = (base_h[:, None] + w_floor[None, :]).flatten()
        idx01 = (base_h[:, None] + w_ceil[None, :]).flatten()
        idx10 = (base_h_ceil[:, None] + w_floor[None, :]).flatten()
        idx11 = (base_h_ceil[:, None] + w_ceil[None, :]).flatten()

        # Weights for bilinear interpolation
        w00 = ((1 - dh)[:, None] * (1 - dw)[None, :]).flatten()
        w01 = ((1 - dh)[:, None] * dw[None, :]).flatten()
        w10 = (dh[:, None] * (1 - dw)[None, :]).flatten()
        w11 = (dh[:, None] * dw[None, :]).flatten()

        # Lookup and interpolate
        pos_embeds = (
            self.pos_embed(idx00) * w00[:, None]
            + self.pos_embed(idx01) * w01[:, None]
            + self.pos_embed(idx10) * w10[:, None]
            + self.pos_embed(idx11) * w11[:, None]
        )

        # Apply spatial merge permutation
        merge_size = self.config.spatial_merge_size
        grid_t = int(grid_thw[0, 0])

        # Reshape: (H*W, D) -> (H, W, D)
        pos_embeds = pos_embeds.reshape(grid_h, grid_w, -1)

        # Repeat for temporal dimension
        if grid_t > 1:
            pos_embeds = jnp.tile(pos_embeds[None], (grid_t, 1, 1, 1))  # (T, H, W, D)
        else:
            pos_embeds = pos_embeds[None]  # (1, H, W, D)

        # Permute for spatial merge: (T, H, W, D) -> (T, H//m, m, W//m, m, D) -> (T, H//m, W//m, m, m, D)
        merged_h, merged_w = grid_h // merge_size, grid_w // merge_size
        pos_embeds = pos_embeds.reshape(grid_t, merged_h, merge_size, merged_w, merge_size, -1)
        pos_embeds = pos_embeds.transpose(0, 1, 3, 2, 4, 5)  # (T, merged_h, merged_w, m, m, D)
        pos_embeds = pos_embeds.reshape(-1, pos_embeds.shape[-1])  # Flatten to (seq, D)

        return pos_embeds

    def _rot_pos_emb(self, grid_thw: Array) -> Tuple[Array, Array]:
        """Compute rotary position embeddings matching PyTorch rot_pos_emb."""
        merge_size = self.config.spatial_merge_size
        grid_h, grid_w = int(grid_thw[0, 1]), int(grid_thw[0, 2])
        grid_t = int(grid_thw[0, 0])

        # Compute merged dimensions
        merged_h, merged_w = grid_h // merge_size, grid_w // merge_size

        # Compute position indices for each patch
        block_rows = jnp.arange(merged_h)
        block_cols = jnp.arange(merged_w)
        intra_row = jnp.arange(merge_size)
        intra_col = jnp.arange(merge_size)

        # Full resolution positions
        row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
        col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

        # Expand and reshape to match spatial merge order: (merged_h, merged_w, m, m) -> (merged_h, merged_w, m, m)
        row_idx = jnp.broadcast_to(row_idx, (merged_h, merged_w, merge_size, merge_size)).reshape(-1)
        col_idx = jnp.broadcast_to(col_idx, (merged_h, merged_w, merge_size, merge_size)).reshape(-1)

        # Repeat for temporal dimension
        if grid_t > 1:
            row_idx = jnp.tile(row_idx, grid_t)
            col_idx = jnp.tile(col_idx, grid_t)

        # Create frequency table - PyTorch uses rotary_dim = head_dim // 2
        # And inv_freq has length rotary_dim // 2 = head_dim // 4
        max_hw = max(grid_h, grid_w)
        head_dim = self.config.head_dim
        rotary_dim = head_dim // 2  # = 32 for head_dim=64
        inv_freq_dim = rotary_dim // 2  # = 16
        inv_freq = 1.0 / (self.config.rope_theta ** (jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / rotary_dim))
        seq_positions = jnp.arange(max_hw, dtype=jnp.float32)
        freq_table = jnp.outer(seq_positions, inv_freq)  # (max_hw, rotary_dim//2) = (max_hw, 16)

        # Lookup embeddings for row and col: (seq, rotary_dim//2) each
        row_emb = freq_table[row_idx]  # (seq, 16)
        col_emb = freq_table[col_idx]  # (seq, 16)

        # Concatenate row and col: (seq, 32)
        emb = jnp.concatenate([row_emb, col_emb], axis=-1)

        # Double the embedding (matching PyTorch cat): (seq, 64)
        emb = jnp.concatenate([emb, emb], axis=-1)

        # Apply cos/sin
        cos = jnp.cos(emb)
        sin = jnp.sin(emb)
        return cos, sin

    def __call__(self, hidden_states: Array, grid_thw: Array) -> Tuple[Array, list[Array]]:
        hidden_states = self.patch_embed(hidden_states)
        seq_len = hidden_states.shape[0]

        # Position embeddings with bilinear interpolation
        pos_embeds = self._fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds[:seq_len]

        # RoPE embeddings
        cos, sin = self._rot_pos_emb(grid_thw)
        position_embeddings = (cos[:seq_len], sin[:seq_len])

        deepstack_features = []
        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, position_embeddings)
            if layer_idx in self.deepstack_visual_indexes:
                ds_idx = list(self.deepstack_visual_indexes).index(layer_idx)
                deepstack_features.append(self.deepstack_merger_list[ds_idx](hidden_states))

        hidden_states = self.merger(hidden_states)
        return hidden_states, deepstack_features


class LayerCache(nnx.Module):
    """KV-cache for a single decoder layer."""

    def __init__(self, config: Qwen3VLTextConfig, batch_size: int, cache_size: int, dtype: jnp.dtype = jnp.bfloat16):
        cache_shape = (batch_size, cache_size, config.num_key_value_heads, config.head_dim)
        shd = config.shd_cfg
        self.k_cache = nnx.Cache(shard(jnp.zeros(cache_shape, dtype=dtype), shd.cache))
        self.v_cache = nnx.Cache(shard(jnp.zeros(cache_shape, dtype=dtype), shd.cache))
        self.size = cache_size
        self.cur_ind = nnx.Variable(jnp.zeros((), dtype=jnp.int32))


Cache: TypeAlias = list[LayerCache]


def init_cache(
    config: Qwen3VLConfig, batch_size: int, token_len: int, generate_steps: int, dtype: jnp.dtype = jnp.bfloat16
) -> Cache:
    """Initialize KV-cache for all layers."""
    cache_size = 2 ** math.ceil(math.log2(max(token_len + generate_steps, 1)))
    return [
        LayerCache(config.text_config, batch_size, cache_size, dtype)
        for _ in range(config.text_config.num_hidden_layers)
    ]


class Qwen3VLMLP(nnx.Module):
    """SiLU-gated MLP for text decoder."""

    def __init__(self, config: Qwen3VLTextConfig, *, rngs: nnx.Rngs):
        self.shd_cfg = config.shd_cfg
        self.gate_proj = nnx.Linear(config.hidden_size, config.intermediate_size, use_bias=False, rngs=rngs)
        self.up_proj = nnx.Linear(config.hidden_size, config.intermediate_size, use_bias=False, rngs=rngs)
        self.down_proj = nnx.Linear(config.intermediate_size, config.hidden_size, use_bias=False, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        activations = nnx.silu(self.gate_proj(x)) * self.up_proj(x)
        activations = shard(activations, self.shd_cfg.act_btf)
        return self.down_proj(activations)


def _generate_rope(positions: Array, head_dim: int, rope_theta: float) -> Tuple[Array, Array]:
    """Generate RoPE cos/sin embeddings."""
    fraction = jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim
    timescale = rope_theta**fraction
    sinusoid_inp = jnp.einsum(
        "bt,k->btk", positions.astype(jnp.float32), 1.0 / timescale, precision=jax.lax.Precision.HIGHEST
    )
    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


def _apply_rope(x: Array, sin: Array, cos: Array) -> Array:
    """Apply rotary position embeddings."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    sin, cos = sin[:, :, None, :], cos[:, :, None, :]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(x.dtype)


def repeat_kv(hidden_states: Array, n_rep: int) -> Array:
    """Repeat KV heads for GQA."""
    if n_rep == 1:
        return hidden_states
    b, t, kv_heads, head_dim = hidden_states.shape
    return jnp.repeat(hidden_states, n_rep, axis=2)


class Qwen3VLAttention(nnx.Module):
    """Text decoder attention with GQA and Q/K normalization."""

    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int, *, rngs: nnx.Rngs):
        self.config = config
        self.shd_cfg = config.shd_cfg
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.n_rep = self.num_heads // self.num_kv_heads
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
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps, rngs=rngs)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps, rngs=rngs)

    def __call__(self, x: Array, cache: LayerCache, sin: Array, cos: Array, mask: Array | None) -> Array:
        batch, seq_len, _ = x.shape

        q = shard(
            self.q_norm(self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)), self.shd_cfg.act_btnh
        )
        k = shard(
            self.k_norm(self.k_proj(x).reshape(batch, seq_len, self.num_kv_heads, self.head_dim)), self.shd_cfg.act_btnh
        )
        v = shard(self.v_proj(x).reshape(batch, seq_len, self.num_kv_heads, self.head_dim), self.shd_cfg.act_btnh)

        q = _apply_rope(q, sin, cos)
        k = _apply_rope(k, sin, cos)

        # Update cache - cast to cache dtype
        cache_dtype = cache.k_cache[...].dtype
        k_cached = k.astype(cache_dtype)
        v_cached = v.astype(cache_dtype)
        slice_indices = (0, cache.cur_ind[...], 0, 0)
        cache.k_cache[...] = jax.lax.dynamic_update_slice(cache.k_cache[...], k_cached, slice_indices)
        cache.v_cache[...] = jax.lax.dynamic_update_slice(cache.v_cache[...], v_cached, slice_indices)

        k = repeat_kv(cache.k_cache[...], self.n_rep)
        v = repeat_kv(cache.v_cache[...], self.n_rep)

        q = q.transpose(0, 2, 1, 3)  # (B, heads, T, dim)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        attn_weights = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, _K_MASK)
        attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(q.dtype)
        attn_out = jnp.matmul(attn_weights, v).transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)

        cache.cur_ind[...] = cache.cur_ind[...] + seq_len
        return shard(self.o_proj(attn_out), self.shd_cfg.act_btd)


class Qwen3VLDecoderLayer(nnx.Module):
    """Single decoder layer for text model."""

    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int, *, rngs: nnx.Rngs):
        self.self_attn = Qwen3VLAttention(config, layer_idx, rngs=rngs)
        self.mlp = Qwen3VLMLP(config, rngs=rngs)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, rngs=rngs)

    def __call__(self, x: Array, cache: LayerCache, sin: Array, cos: Array, mask: Array | None) -> Array:
        x = x + self.self_attn(self.input_layernorm(x), cache, sin, cos, mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3VLTextModel(nnx.Module):
    """Text decoder model."""

    def __init__(self, config: Qwen3VLTextConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.shd_cfg = config.shd_cfg
        self.embed_tokens = nnx.Embed(num_embeddings=config.vocab_size, features=config.hidden_size, rngs=rngs)
        self.layers = nnx.List([Qwen3VLDecoderLayer(config, i, rngs=rngs) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, rngs=rngs)

    def __call__(self, inputs_embeds: Array, cache: Cache, sin: Array, cos: Array, mask: Array | None) -> Array:
        hidden_states = shard(inputs_embeds, self.shd_cfg.act_btd)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, cache[i], sin, cos, mask)
        return self.norm(hidden_states)


def merge_modalities(img_emb: Array, text_emb: Array, token_mask: Array) -> Array:
    """Merge image embeddings into text sequence at masked positions."""
    img_indices = jnp.cumsum(token_mask) - 1
    safe_indices = jnp.clip(img_indices, 0, img_emb.shape[0] - 1)
    aligned_images = img_emb[safe_indices]
    return jnp.where(token_mask[:, None], aligned_images, text_emb)


def batched_merge_modalities(img_emb: Array, text_emb: Array, token_mask: Array) -> Array:
    """Batched version of merge_modalities."""
    return jax.vmap(merge_modalities)(img_emb, text_emb, token_mask)


def make_causal_mask(cache: LayerCache, seq_len: int) -> Array:
    """Create causal attention mask."""
    cache_size = cache.size
    cur_pos = cache.cur_ind[...]
    seq_arange = jnp.arange(seq_len)
    cache_arange = jnp.arange(cache_size)
    mask = (seq_arange[:, None] + cur_pos) >= cache_arange[None, :]
    return mask[None, None, :, :]


class Qwen3VLForConditionalGeneration(nnx.Module):
    """Qwen3-VL model with language modeling head."""

    def __init__(self, config: Qwen3VLConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.model = Qwen3VLModel(config, rngs=rngs)
        if config.text_config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nnx.Linear(
                config.text_config.hidden_size, config.text_config.vocab_size, use_bias=False, rngs=rngs
            )

    def __call__(
        self,
        input_ids: Array,
        cache: Cache,
        pixel_values: Optional[Array] = None,
        image_grid_thw: Optional[Array] = None,
        token_type_ids: Optional[Array] = None,
    ) -> Array:
        """Forward pass with KV-cache."""
        batch, seq_len = input_ids.shape

        # Generate position embeddings
        positions = jnp.arange(seq_len)[None, :] + cache[0].cur_ind[...]
        positions = jnp.broadcast_to(positions, (batch, seq_len))
        sin, cos = _generate_rope(positions, self.config.text_config.head_dim, self.config.text_config.rope_theta)

        # Create causal mask
        mask = make_causal_mask(cache[0], seq_len)

        # Get text embeddings
        inputs_embeds = self.model.language_model.embed_tokens(input_ids)

        # Merge vision if provided
        if pixel_values is not None and image_grid_thw is not None and token_type_ids is not None:
            vision_embeds, _ = self.model.visual(pixel_values, image_grid_thw)
            # Broadcast vision_embeds for batch
            vision_embeds_batched = jnp.broadcast_to(
                vision_embeds[None], (batch, vision_embeds.shape[0], vision_embeds.shape[1])
            )
            inputs_embeds = batched_merge_modalities(vision_embeds_batched, inputs_embeds, token_type_ids)

        hidden_states = self.model.language_model(inputs_embeds, cache, sin, cos, mask)

        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            embedding = self.model.language_model.embed_tokens.embedding[...]
            logits = jnp.matmul(hidden_states, embedding.T)

        return logits


class Qwen3VLModel(nnx.Module):
    """Qwen3-VL backbone model."""

    def __init__(self, config: Qwen3VLConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.visual = Qwen3VLVisionModel(config.vision_config, rngs=rngs)
        self.language_model = Qwen3VLTextModel(config.text_config, rngs=rngs)


@jax.jit
def forward(model: Qwen3VLForConditionalGeneration, cache: Cache, input_ids: Array) -> Tuple[Array, Cache]:
    """JIT-compiled forward pass for text-only generation."""
    logits = model(input_ids, cache)
    return logits[:, -1, :], cache


def forward_vision(
    model: Qwen3VLForConditionalGeneration,
    cache: Cache,
    input_ids: Array,
    pixel_values: Array,
    image_grid_thw: Array,
    token_type_ids: Array,
) -> Tuple[Array, Cache]:
    """Forward pass with vision inputs (not JIT - vision has data-dependent shapes)."""
    logits = model(input_ids, cache, pixel_values, image_grid_thw, token_type_ids)
    return logits[:, -1, :], cache
