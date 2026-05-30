# Copyright 2026 The JAX Authors.
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

import dataclasses
import math
from functools import partial
from typing import cast, TypeAlias
from enum import Enum

import jax
from jax import P
from jax.sharding import PartitionSpec, get_abstract_mesh, reshard
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from flax import nnx
from flax.nnx.nn.linear import default_embed_init


class ShardMode(Enum):
    """Sharding Modes for Model Parameters"""

    FSDP = "fsdp"
    TP = "tp"


@dataclasses.dataclass(slots=True, frozen=True)
class LlamaShardCfg:
    # Embedding
    emb_weight: PartitionSpec | None = None
    activation: PartitionSpec | None = None
    logits: PartitionSpec | None = None

    # Attention
    q_proj: PartitionSpec | None = None
    k_proj: PartitionSpec | None = None
    v_proj: PartitionSpec | None = None
    o_proj: PartitionSpec | None = None

    attn_logits: PartitionSpec | None = None
    attn_out: PartitionSpec | None = None

    cache: PartitionSpec | None = None

    # MLP
    gate_proj: PartitionSpec | None = None
    up_proj: PartitionSpec | None = None
    down_proj: PartitionSpec | None = None

    # Head
    lm_head: PartitionSpec | None = None

    @classmethod
    def no_sharding(cls) -> "LlamaShardCfg":
        return LlamaShardCfg()

    @classmethod
    def default(cls, use_fsdp: bool, use_tp: bool) -> "LlamaShardCfg":
        if not (use_fsdp and use_tp):
            return cls.no_sharding()

        fsdp = ShardMode.FSDP.value if use_fsdp else None
        tp = ShardMode.TP.value if use_tp else None

        return cls(
            emb_weight=P(None, tp),
            activation=P(fsdp, None, tp),
            logits=P(fsdp, None, tp),
            q_proj=P(fsdp, tp),
            k_proj=P(fsdp, tp),
            v_proj=P(fsdp, tp),
            o_proj=P(tp, fsdp),
            attn_logits=P(fsdp, None, None, tp, None),
            attn_out=P(fsdp, None, tp, None, None),
            cache=P(fsdp, None, tp, None),
            gate_proj=P(fsdp, tp),
            up_proj=P(fsdp, tp),
            down_proj=P(tp, fsdp),
            lm_head=P(None, tp),
        )


def shard(x: jnp.ndarray, s: PartitionSpec | None):
    if s is None:
        return x
    mesh = get_abstract_mesh()
    if not mesh.empty and len(mesh.axis_names) > 0:
        return reshard(x, s)
    return x


@dataclasses.dataclass(frozen=True)
class RopeScalingConfig:
    factor: float
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_position_embeddings: int = 8192


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    head_dim: int
    num_key_value_heads: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: RopeScalingConfig | None
    tie_word_embeddings: bool
    shd_cfg: LlamaShardCfg
    dtype: jnp.dtype = jnp.bfloat16

    @classmethod
    def llama3_2_1b(cls, use_fsdp: bool, use_tp: bool) -> "ModelConfig":
        return cls(
            vocab_size=128256,
            hidden_size=2048,
            intermediate_size=8192,
            num_hidden_layers=16,
            num_attention_heads=32,
            head_dim=64,
            num_key_value_heads=8,
            max_position_embeddings=131072,
            rms_norm_eps=1e-5,
            rope_theta=500000.0,
            rope_scaling=RopeScalingConfig(factor=32.0),
            tie_word_embeddings=True,
            shd_cfg=LlamaShardCfg.default(use_fsdp, use_tp),
        )

    @classmethod
    def llama3_2_3b(cls, use_fsdp: bool, use_tp: bool) -> "ModelConfig":
        return cls(
            vocab_size=128256,
            hidden_size=3072,
            intermediate_size=8192,
            num_hidden_layers=28,
            num_attention_heads=24,
            head_dim=128,
            num_key_value_heads=8,
            max_position_embeddings=131072,
            rms_norm_eps=1e-5,
            rope_theta=500000.0,
            rope_scaling=RopeScalingConfig(factor=32.0),
            tie_word_embeddings=True,
            shd_cfg=LlamaShardCfg.default(use_fsdp, use_tp),
        )


class LayerCache(nnx.Module):
    """Key-Value Cache for Attention Layers"""

    def __init__(self, cfg: ModelConfig, batch_size: int, cache_size: int, dtype: jnp.dtype):
        cache_shape = (batch_size, cache_size, cfg.num_key_value_heads, cfg.head_dim)
        kv_shd = cfg.shd_cfg.cache
        self.k_cache = nnx.Cache(jnp.zeros(cache_shape, dtype=dtype, out_sharding=kv_shd))
        self.v_cache = nnx.Cache(jnp.zeros(cache_shape, dtype=dtype, out_sharding=kv_shd))
        self.size = self.k_cache.shape[1]
        start_ind_shd = None if kv_shd is None else P(kv_shd[0])
        self.start_ind = nnx.Variable(-1 * jnp.ones((batch_size,), dtype=jnp.int32, out_sharding=start_ind_shd))
        self.cur_ind = nnx.Variable(jnp.zeros((), dtype=jnp.int32))


Cache: TypeAlias = list[LayerCache]


class LlamaRMSNorm(nnx.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, hidden_size: int, eps: float, rngs: nnx.Rngs):
        self.scale = nnx.Param(nnx.initializers.ones_init()(rngs.params(), (hidden_size,)))
        self.norm_eps = eps

    @jax.named_scope("rms_norm")
    def __call__(self, x: Array) -> Array:
        dtype = x.dtype
        x_fp32 = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x_fp32), axis=-1, keepdims=True)
        inv_rms = jax.lax.rsqrt(variance + self.norm_eps)

        return (self.scale[...] * x_fp32 * inv_rms).astype(dtype)


# TODO: Replace with nnx.Linear once explicit sharding is supported.
class ShardedLinear(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        sharding: PartitionSpec | None,
        *,
        use_bias: bool = True,
        dtype=jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        kernel_initializer = jax.nn.initializers.lecun_normal()
        self.kernel = nnx.Param(
            kernel_initializer(rngs.params(), (in_dim, out_dim), dtype=dtype, out_sharding=sharding)
        )
        if use_bias:
            self.bias = nnx.Param(jnp.zeros((out_dim,), dtype=dtype))
        else:
            self.bias = None

    def __call__(self, x: ArrayLike, *, out_sharding: PartitionSpec | None = None) -> Array:
        out = jnp.matmul(x, self.kernel[...], out_sharding=out_sharding)
        if self.bias is None:
            return out
        return out + self.bias[...]


class ShardedEmbedding(nnx.Embed):
    """Sharded Embedding Layer"""

    def __call__(self, inputs: Array, *, out_sharding: PartitionSpec | None = None) -> Array:
        if not jnp.issubdtype(inputs.dtype, jnp.integer):
            raise ValueError("Input type must be an integer or unsigned integer.")
        (embedding,) = self.promote_dtype((self.embedding[...],), dtype=self.dtype, inexact=False)
        if self.num_embeddings == 1:
            return jnp.broadcast_to(embedding, (*inputs.shape, self.features))
        return embedding.at[inputs].get(out_sharding=out_sharding)

    def decode(self, query: Array, *, out_sharding: PartitionSpec | None = None) -> Array:
        query, embedding = self.promote_dtype((query, self.embedding[...]), dtype=self.dtype)
        return jnp.dot(query, embedding.T, out_sharding=out_sharding)


class LlamaMLP(nnx.Module):
    """Feed Forward Network"""

    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        super().__init__()
        self.config = cfg
        self.hidden_size = cfg.hidden_size
        self.intermediate_size = cfg.intermediate_size
        self.gate_proj = ShardedLinear(
            self.hidden_size,
            self.intermediate_size,
            sharding=cfg.shd_cfg.gate_proj,
            use_bias=False,
            dtype=cfg.dtype,
            rngs=rngs,
        )
        self.up_proj = ShardedLinear(
            self.hidden_size,
            self.intermediate_size,
            sharding=cfg.shd_cfg.up_proj,
            use_bias=False,
            dtype=cfg.dtype,
            rngs=rngs,
        )
        self.down_proj = ShardedLinear(
            self.intermediate_size,
            self.hidden_size,
            sharding=cfg.shd_cfg.down_proj,
            use_bias=False,
            dtype=cfg.dtype,
            rngs=rngs,
        )

    @jax.named_scope("feed_forward")
    def __call__(self, x: ArrayLike) -> Array:
        act_shd = self.config.shd_cfg.activation
        gated = nnx.silu(self.gate_proj(x, out_sharding=act_shd)) * self.up_proj(x, out_sharding=act_shd)
        return self.down_proj(gated, out_sharding=act_shd)


def _generate_pos_embeddings(
    positions: Array,
    head_dim: int,
    rope_theta: float = 500000.0,
    rope_scaling: RopeScalingConfig | None = None,
) -> tuple[Array, Array]:
    """Generate sin and cos for rotary position embeddings."""
    if rope_scaling is None:
        rope_scaling = RopeScalingConfig(factor=1.0)

    inv_freq = 1.0 / (rope_theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))

    factor = rope_scaling.factor
    low_freq_factor = rope_scaling.low_freq_factor
    high_freq_factor = rope_scaling.high_freq_factor
    old_context_len = rope_scaling.original_max_position_embeddings

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = (2.0 * math.pi) / inv_freq
    inv_freq_llama = jnp.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1.0 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = jnp.logical_and(
        wavelen >= high_freq_wavelen,
        wavelen <= low_freq_wavelen,
    )
    inv_freq_llama = jnp.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    positions_f32 = positions.astype(jnp.float32)
    sinusoid_inp = jnp.einsum("BT,k->BTk", positions_f32, inv_freq_llama, precision=jax.lax.Precision.HIGHEST)
    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


def apply_rope(x: Array, sin: Array, cos: Array) -> Array:
    """Apply rotary position embeddings to input tensor."""
    if x.ndim != 4 or sin.ndim != 3 or cos.ndim != 3:
        raise ValueError(
            "apply_rope expects x.ndim == 4 and sin.ndim == cos.ndim == 3; "
            f"got x.ndim={x.ndim}, sin.ndim={sin.ndim}, cos.ndim={cos.ndim}"
        )
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    # [B, T, head_dim] -> [B, h, T, head_dim]
    sin, cos = sin[:, :, None, :], cos[:, :, None, :]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(x.dtype)


def count_left_pads(x: Array) -> Array:
    """Count left padding tokens per batch element."""
    return jnp.sum(jnp.cumsum(x != 0, axis=-1) == 0, -1)


def count_right_pads(x: Array, pad_id: int) -> Array:
    """Count right padding tokens per batch element."""
    all_pad = jnp.all(x == pad_id, axis=1)
    right_pad = jnp.argmin(jnp.flip(x == pad_id, axis=1).astype(jnp.int32), axis=1)
    max_len = jnp.full_like(right_pad, x.shape[1])
    return jnp.where(all_pad, max_len, right_pad)


def count_right_pads_from_mask(attn_mask: Array) -> Array:
    """Count right padding tokens from a 0/1 attention mask."""
    mask = attn_mask.astype(jnp.int32)
    all_pad = jnp.all(mask == 0, axis=1)
    right_pad = jnp.argmax(jnp.flip(mask, axis=1), axis=1)
    max_len = jnp.full_like(right_pad, mask.shape[1])
    return jnp.where(all_pad, max_len, right_pad)


def compute_positions_from_segment_ids(seg_ids: Array) -> Array:
    """Compute position ids from segment ids with support for packed sequences."""
    seg_ids = seg_ids.astype(jnp.int32)
    pad_sentinel = 2**30

    def step(carry: tuple[Array, Array], seg_id: Array) -> tuple[tuple[Array, Array], Array]:
        prev_seg, prev_pos = carry
        is_pad = seg_id == 0
        is_new = seg_id != prev_seg
        zero = jnp.zeros_like(seg_id)
        pos = jnp.where(is_pad, zero, jnp.where(is_new, zero, prev_pos + 1))
        pad_val = jnp.full_like(seg_id, pad_sentinel)
        out = jnp.where(is_pad, pad_val, pos)
        new_prev_seg = jnp.where(is_pad, zero, seg_id)
        new_prev_pos = jnp.where(is_pad, zero, pos)
        return (new_prev_seg, new_prev_pos), out

    base = jnp.zeros_like(seg_ids[:, 0])
    init = (base, base)
    _, out = jax.lax.scan(step, init, seg_ids.T)
    return cast(Array, out.T)


def sharded_attention(
    q: Array,
    k: Array,
    v: Array,
    attn_mask: Array | None,
    scale: float,
    *,
    attn_logit_sharding: PartitionSpec | None,
    out_sharding: PartitionSpec | None,
) -> Array:
    """Compute scaled dot-product attention with optional masking."""
    attn_logits = jnp.einsum("BTKGH,BSKH->BTSKG", q, k, out_sharding=attn_logit_sharding) * scale

    if attn_mask is not None:
        if attn_mask.ndim == 2:
            attn_mask = attn_mask[:, None, :, None, None]  # [B, 1, S, 1, 1]
        elif attn_mask.ndim == 3:
            attn_mask = attn_mask[:, :, :, None, None]  # [B, T, S, 1, 1]
        else:
            raise ValueError(f"attn_mask must be rank-2 or rank-3, got {attn_mask.ndim}")
        attn_logits = jnp.where(attn_mask, attn_logits, jnp.finfo(attn_logits.dtype).min)

    attn_weights = jax.nn.softmax(attn_logits.astype(jnp.float32), axis=2).astype(attn_logits.dtype)
    attn_output = jnp.einsum("BTSKG,BSKH->BTKGH", attn_weights, v, out_sharding=out_sharding)

    return attn_output.astype(q.dtype)


class LlamaAttention(nnx.Module):
    """Multi-Head Self Attention"""

    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.config = cfg
        self.hidden_size = cfg.hidden_size
        self.head_dim = cfg.head_dim
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = ShardedLinear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            sharding=cfg.shd_cfg.q_proj,
            use_bias=False,
            dtype=cfg.dtype,
            rngs=rngs,
        )
        self.k_proj = ShardedLinear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            sharding=cfg.shd_cfg.k_proj,
            use_bias=False,
            dtype=cfg.dtype,
            rngs=rngs,
        )
        self.v_proj = ShardedLinear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            sharding=cfg.shd_cfg.v_proj,
            use_bias=False,
            dtype=cfg.dtype,
            rngs=rngs,
        )
        self.o_proj = ShardedLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            sharding=cfg.shd_cfg.o_proj,
            use_bias=False,
            dtype=cfg.dtype,
            rngs=rngs,
        )

    def _make_cache_mask(self, cache: LayerCache, t: int) -> Array:
        """Create attention mask for cached key-value states."""

        q_pos = cache.cur_ind[...] + jnp.arange(t, dtype=jnp.int32)[None, :] - cache.start_ind[...][:, None]
        ts = jnp.arange(cache.size, dtype=jnp.int32)
        kv_valid = (ts[None, :] >= cache.start_ind[...][:, None]) & (ts[None, :] < cache.cur_ind[...] + t)
        k_pos = ts[None, :] - cache.start_ind[...][:, None]
        causal_mask = k_pos[:, None, :] <= q_pos[:, :, None]
        return causal_mask & kv_valid[:, None, :]

    def _make_stateless_mask(self, segment_ids: Array, t: int) -> Array:
        """Create attention mask without cache."""
        q_pos = jnp.arange(t, dtype=jnp.int32)[None, :]
        k_pos = jnp.arange(t, dtype=jnp.int32)[None, :]
        causal_mask = k_pos[:, None, :] <= q_pos[:, :, None]
        segment_mask = segment_ids[:, :, None] == segment_ids[:, None, :]
        return causal_mask & segment_mask

    @jax.named_scope("attention")
    def __call__(
        self,
        x: Array,
        segment_ids: Array,
        attn_mask: Array | None,
        cache: LayerCache | None = None,
    ) -> Array:
        b, t, _ = x.shape

        # Project to Q, K, V and reshape to [B, T, N/K, H]
        act_shd = self.config.shd_cfg.activation
        q = self.q_proj(x, out_sharding=act_shd).reshape((b, t, self.num_heads, self.head_dim))
        k = self.k_proj(x, out_sharding=act_shd).reshape((b, t, self.num_kv_heads, self.head_dim))
        v = self.v_proj(x, out_sharding=act_shd).reshape((b, t, self.num_kv_heads, self.head_dim))

        # Apply RoPE
        position_ids = compute_positions_from_segment_ids(segment_ids)
        if cache is not None:
            left_pads = count_left_pads(segment_ids)
            cache.start_ind.set_value(jnp.where(cache.start_ind[...] < 0, left_pads, cache.start_ind[...]))
            position_ids = position_ids + cache.cur_ind[...]

        sin, cos = _generate_pos_embeddings(
            position_ids, self.head_dim, self.config.rope_theta, self.config.rope_scaling
        )
        q = apply_rope(q, sin, cos)
        k = apply_rope(k, sin, cos)

        if cache is not None:
            # Update cache with new keys and values
            slice_indices = (0, cache.cur_ind[...], 0, 0)
            cache_dtype = cache.k_cache[...].dtype
            k = k.astype(cache_dtype)
            v = v.astype(cache_dtype)
            cache_shd = self.config.shd_cfg.cache
            k = shard(k, cache_shd)
            v = shard(v, cache_shd)
            cache.k_cache.set_value(jax.lax.dynamic_update_slice(cache.k_cache[...], k, slice_indices))
            cache.v_cache.set_value(jax.lax.dynamic_update_slice(cache.v_cache[...], v, slice_indices))
            k = cache.k_cache[...]
            v = cache.v_cache[...]

        # Reshape query for GQA: [B, T, K, n_rep, H]
        query_proj_gqa = q.reshape((b, t, self.num_kv_heads, self.n_rep, self.head_dim))

        if attn_mask is None:
            if cache is not None:
                attn_mask = self._make_cache_mask(cache, t)
            else:
                attn_mask = self._make_stateless_mask(segment_ids, t)

        # Attention (with optional sharding) -> [B, T, K, n_rep, H]
        attn_output = sharded_attention(
            query_proj_gqa,
            k,
            v,
            attn_mask,
            self.scale,
            attn_logit_sharding=self.config.shd_cfg.attn_logits,
            out_sharding=self.config.shd_cfg.attn_out,
        )

        if cache is not None:
            cache.cur_ind.set_value(cache.cur_ind[...] + t)

        # Reshape back to [B, T, N * H] for the output projection.
        attn_output = attn_output.reshape((b, t, self.num_heads * self.head_dim))

        # Output projection: [B, T, D]
        return self.o_proj(attn_output, out_sharding=act_shd)


class LlamaDecoderLayer(nnx.Module):
    """Llama Decoder Layer"""

    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.self_attn = LlamaAttention(cfg, rngs=rngs)
        self.mlp = LlamaMLP(cfg, rngs=rngs)
        self.input_layernorm = LlamaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, rngs=rngs)
        self.post_attention_layernorm = LlamaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, rngs=rngs)

    @jax.named_scope("decoder_layer")
    def __call__(
        self,
        x: Array,
        segment_ids: Array,
        attn_mask: Array | None = None,
        cache: LayerCache | None = None,
    ) -> Array:
        # Self-Attention Block
        normed_x = self.input_layernorm(x)
        attn_output = self.self_attn(normed_x, segment_ids, attn_mask, cache=cache)
        x = x + attn_output

        # Feed-Forward Block
        normed_x = self.post_attention_layernorm(x)
        mlp_output = self.mlp(normed_x)
        x = x + mlp_output

        return x


class Llama(nnx.Module):
    """Llama Model"""

    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.config = cfg
        embed_init = partial(default_embed_init, out_sharding=cfg.shd_cfg.emb_weight)
        self.embedder = ShardedEmbedding(
            num_embeddings=cfg.vocab_size,
            features=cfg.hidden_size,
            dtype=cfg.dtype,
            embedding_init=embed_init,
            rngs=rngs,
        )
        self.layers = nnx.List([LlamaDecoderLayer(cfg, rngs=rngs) for _ in range(cfg.num_hidden_layers)])
        self.final_norm = LlamaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, rngs=rngs)
        if cfg.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = ShardedLinear(
                cfg.hidden_size,
                cfg.vocab_size,
                sharding=cfg.shd_cfg.lm_head,
                use_bias=False,
                dtype=cfg.dtype,
                rngs=rngs,
            )

    def init_cache(
        self,
        cfg: ModelConfig,
        batch_size: int,
        token_len: int,
        generate_steps: int,
        max_cache_len: int = 4096,
    ) -> Cache:
        target_len = min(max_cache_len, token_len + generate_steps)
        cache_size = 2 ** math.ceil(math.log2(max(target_len, 1)))
        cache_size = min(cache_size, max_cache_len)
        cache_dtype = self.layers[0].self_attn.k_proj.kernel[...].dtype
        return [LayerCache(cfg, batch_size, cache_size, cache_dtype) for _ in range(cfg.num_hidden_layers)]

    def __call__(
        self,
        tokens: Array,
        segment_ids: Array,
        cache: Cache | None,
        attn_mask: Array | None = None,
    ) -> Array:
        x = self.embedder(tokens, out_sharding=self.config.shd_cfg.activation)
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x = layer(x, segment_ids, attn_mask=attn_mask, cache=layer_cache)
        hidden = self.final_norm(x)
        if self.config.tie_word_embeddings:
            logits = self.embedder.decode(hidden, out_sharding=self.config.shd_cfg.logits)
        else:
            assert self.lm_head is not None
            logits = self.lm_head(hidden, out_sharding=self.config.shd_cfg.logits)
        return logits


@jax.jit
def forward(
    model: nnx.Module,
    cache: Cache,
    tokens: Array,
    pad_id: int,
    attention_mask: Array | None = None,
    segment_ids: Array | None = None,
) -> tuple[Array, Cache]:
    # Use attention_mask when available because pad_id can equal eos_id, which would
    # misclassify real tokens as padding if we rely on tokens != pad_id.
    if attention_mask is not None and segment_ids is not None:
        raise ValueError("Provide only one of attention_mask or segment_ids.")

    if segment_ids is None:
        if attention_mask is None:
            segment_ids = 1 * (tokens != pad_id)
            pad_mask = segment_ids
            num_right_pads = count_right_pads(tokens, pad_id)
        else:
            segment_ids = attention_mask.astype(jnp.int32)
            pad_mask = segment_ids
            num_right_pads = count_right_pads_from_mask(pad_mask)
    else:
        segment_ids = segment_ids.astype(jnp.int32)
        pad_mask = (segment_ids != 0).astype(jnp.int32)
        num_right_pads = count_right_pads_from_mask(pad_mask)

    logits = model(tokens, segment_ids, cache, attn_mask=None)
    target_ind = tokens.shape[-1] - num_right_pads - 1
    batch_idx = jnp.arange(tokens.shape[0])
    return logits[batch_idx, target_ind], cache
