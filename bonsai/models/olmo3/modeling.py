# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from enum import Enum
from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.nnx.nn.linear import default_embed_init
from jax import P
from typing import TypeAlias

from jax.sharding import PartitionSpec
from jaxtyping import Array, DTypeLike

LARGE_NEGATIVE = jnp.finfo(jnp.float32).min


class AttentionMode(Enum):
    FULL = "full_attention"
    SLIDE = "sliding_attention"


class ShardMode(Enum):
    FSDP = "fsdp"
    TP = "tp"


def _set_attention_modes(global_attn_freq: int, layers: int) -> list[AttentionMode]:
    """Returns a list of attention modes where every global_attn_freq layers uses global attention."""
    return [AttentionMode.FULL if i % global_attn_freq == 0 else AttentionMode.SLIDE for i in range(1, layers + 1)]


@dataclass(slots=True, frozen=True)
class ShardingConfig:
    attn_kernel: PartitionSpec
    attn_bias: PartitionSpec
    attn_qk_activation: PartitionSpec
    fc1_kernel: PartitionSpec
    fc1_bias: PartitionSpec
    fc2_kernel: PartitionSpec
    fc2_bias: PartitionSpec
    activation: PartitionSpec
    norm: PartitionSpec
    emb_kernel: PartitionSpec
    cache: PartitionSpec

    @staticmethod
    def no_sharding():
        return ShardingConfig.default(False, False)

    @staticmethod
    def default(use_fsdp: bool, use_tp: bool):
        fsdp = ShardMode.FSDP.value if use_fsdp else None
        tp = ShardMode.TP.value if use_tp else None
        return ShardingConfig(
            attn_kernel=P(tp, fsdp),
            attn_bias=P(tp),
            attn_qk_activation=P(fsdp, tp),
            fc1_kernel=P(fsdp, tp),
            fc1_bias=P(tp),
            fc2_kernel=P(tp, fsdp),
            fc2_bias=P(tp),
            activation=P(fsdp, None, tp),
            norm=P(tp),
            emb_kernel=P(None, tp),
            cache=P(fsdp, None, tp, None),
        )


@dataclass
class ModelConfig:
    dtype: DTypeLike
    vocab_size: int
    max_position_embeddings: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    use_cache: bool
    yarn_beta_fast: float
    yarn_beta_slow: float
    yarn_factor: float
    yarn_orig_max_pos_embs: int
    rope_theta: int
    attention_bias: bool
    attention_dropout: float
    rms_norm_eps: float
    sliding_window: int
    layer_types: list[str]
    shd_cfg: ShardingConfig
    pad_token_id: int = 100277

    @classmethod
    def olmo3_7b(cls, use_fsdp: bool, use_tp: bool, dtype: DTypeLike):  # TODO: Maybe update name
        layers = 32
        return cls(
            dtype=dtype,
            vocab_size=100278,
            max_position_embeddings=65536,
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=layers,
            num_attention_heads=32,
            num_key_value_heads=32,
            use_cache=True,
            yarn_beta_fast=32.0,
            yarn_beta_slow=1.0,
            yarn_factor=8.0,
            yarn_orig_max_pos_embs=8192,
            rope_theta=500000,
            attention_bias=False,
            attention_dropout=0.0,
            rms_norm_eps=1e-06,
            sliding_window=4096,
            layer_types=_set_attention_modes(4, layers),
            shd_cfg=ShardingConfig.default(use_fsdp, use_tp),
        )


class LayerCache(nnx.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int, batch_size: int, cache_size: int, dtype: jnp.dtype):
        head_dim = cfg.hidden_size // cfg.num_key_value_heads
        cache_shape = (batch_size, cache_size, cfg.num_key_value_heads, head_dim)
        kv_shd = cfg.shd_cfg.cache
        self.k_cache = nnx.Cache(jnp.zeros(cache_shape, dtype=dtype, out_sharding=kv_shd))
        self.v_cache = nnx.Cache(jnp.zeros(cache_shape, dtype=dtype, out_sharding=kv_shd))
        self.size = self.k_cache.shape[1]
        self.start_ind = nnx.Variable(-1 * jnp.ones((batch_size,), dtype=jnp.int32, out_sharding=P(kv_shd[0])))
        self.cur_ind = nnx.Variable(jnp.zeros((), dtype=jnp.int32))


Cache: TypeAlias = list[LayerCache]


# TODO: Update to have a memory efficient cache for sliding window.
def init_cache(
    cfg: ModelConfig, batch_size: int, token_len: int, generate_steps: int, dtype: jnp.dtype = jnp.bfloat16
) -> Cache:
    cache_size = 2 ** math.ceil(math.log2(max(token_len + generate_steps, 1)))  # Pad for a sharding-friendly size.
    return [LayerCache(cfg, i, batch_size, cache_size, dtype) for i in range(cfg.num_hidden_layers)]


# TODO: Replace with nnx.Linear once explicit sharding is supported.
class ShardedLinear(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        use_bias: bool = True,
        kernel_sharding,
        bias_sharding,
        dtype=None,
        rngs,
    ):
        kernel_initializer = jax.nn.initializers.lecun_normal()
        self.kernel = nnx.Param(
            kernel_initializer(rngs.params(), (in_dim, out_dim), dtype=dtype, out_sharding=kernel_sharding)
        )
        if use_bias:
            self.bias = nnx.Param(jnp.zeros((out_dim,), dtype=dtype, out_sharding=bias_sharding))
        else:
            self.bias = nnx.data(jnp.zeros((out_dim,), dtype=dtype, out_sharding=bias_sharding))

    def __call__(self, x, *, out_sharding):
        return jnp.matmul(x, self.kernel, out_sharding=out_sharding) + self.bias


# TODO: Replace with nnx.Embed once explicit sharding is supported.
class ShardedEmbedding(nnx.Embed):
    def __call__(self, inputs: Array, *, out_sharding) -> Array:
        # Modified from Flax NNX
        if not jnp.issubdtype(inputs.dtype, jnp.integer):
            raise ValueError("Input type must be an integer or unsigned integer.")
        (embedding,) = self.promote_dtype((self.embedding.get_value(),), dtype=self.dtype, inexact=False)
        if self.num_embeddings == 1:
            return jnp.broadcast_to(embedding, (*inputs.shape, self.features))
        return embedding.at[inputs].get(out_sharding=out_sharding)

    def attend(self, query: Array, *, out_sharding) -> Array:
        query, embedding = self.promote_dtype((query, self.embedding.get_value()), dtype=self.dtype)
        return jnp.dot(query, embedding.T, out_sharding=out_sharding)


# adapted from the jax.nn.dot_product_attention implementation
def sharded_attention(q, k, v, mask, scale=None, *, attn_logit_sharding: PartitionSpec, out_sharding: PartitionSpec):
    logits = jnp.einsum("BTNH,BSNH->BNTS", q, k, out_sharding=attn_logit_sharding)
    scale_val = (1.0 / np.sqrt(k.shape[-1])) if scale is None else scale
    logits *= jnp.array(scale_val, dtype=logits.dtype)

    padded_logits = jnp.where(mask[:, None, :, :], logits.astype(np.float32), LARGE_NEGATIVE)
    probs = jax.nn.softmax(padded_logits, axis=-1).astype(k.dtype)
    # TODO: Add dropout here

    attn_out = jnp.einsum("BNTS,BSNH->BTNH", probs, v, out_sharding=out_sharding)
    return attn_out


class Olmo3RMSNorm(nnx.Module):
    def __init__(self, dim, eps, *, dtype: DTypeLike, shd: PartitionSpec, rngs: nnx.Rngs):
        self.weight = nnx.Param(jnp.ones(dim, dtype, out_sharding=shd))
        self.eps = eps

    def __call__(self, x):
        xf32 = x.astype(jnp.float32)
        out = xf32 * jax.lax.rsqrt(jnp.square(xf32).mean(-1, keepdims=True) + self.eps)
        out = self.weight.get_value().astype(jnp.float32) * out
        return out.astype(x.dtype)


def _generate_pos_embeddings_rope(
    positions: jax.Array,
    head_dim: int,
    cfg: ModelConfig,
) -> tuple[jax.Array, jax.Array]:
    # Forked from: jax-llm-examples/qwen3/qwen3_jax/model.py;l=571
    fraction = jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim
    timescale = cfg.rope_theta**fraction
    rotational_frequency = 1.0 / timescale
    # Use high-precision einsum to prevent catastrophic bfloat16 rounding (ex: 257→256), as sin(257) differs from sin(256).
    sinusoid_inp = jnp.einsum("BT,k->BTk", positions, rotational_frequency, precision=jax.lax.Precision.HIGHEST)
    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


def _generate_pos_embeddings_yarn(positions: jax.Array, head_dim: int, cfg: ModelConfig) -> tuple[jax.Array, jax.Array]:
    # Adapted from: jax-llm-examples/gpt_oss/gpt_oss_jax/model.py;l=553
    base, factor = cfg.rope_theta, cfg.yarn_factor
    original_max_pos = cfg.yarn_orig_max_pos_embs
    low = (head_dim * math.log(original_max_pos / (cfg.yarn_beta_fast * 2 * math.pi))) / (2 * math.log(base))
    high = (head_dim * math.log(original_max_pos / (cfg.yarn_beta_slow * 2 * math.pi))) / (2 * math.log(base))
    # NOTE: We truncate by default
    low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, head_dim - 1)

    timescale = base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
    rot_freq_extra, rot_freq_inter = 1.0 / timescale, 1.0 / (factor * timescale)

    high = high if low != high else (high + 0.001)
    interp_factor = 1 - jnp.clip((jnp.arange(head_dim // 2, dtype=jnp.float32) - low) / (high - low), min=0, max=1)

    rotational_frequency = rot_freq_inter * (1 - interp_factor) + rot_freq_extra * interp_factor
    # Use high-precision einsum to prevent catastrophic bfloat16 rounding (ex: 257→256), as sin(257) differs from sin(256).
    sinusoid_inp = jnp.einsum("BT,k->BTk", positions, rotational_frequency, precision=jax.lax.Precision.HIGHEST)

    attention_scaling = 1.0 if factor <= 1 else (0.1 * math.log(factor) + 1.0)
    return jnp.sin(sinusoid_inp) * attention_scaling, jnp.cos(sinusoid_inp) * attention_scaling


def apply_rope(x: jax.Array, sin: jax.Array, cos: jax.Array) -> jax.Array:
    assert x.ndim == 4 and sin.ndim == 3 and cos.ndim == 3
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    sin, cos = sin[:, :, None, :], cos[:, :, None, :]
    out = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(x.dtype)
    return out


def count_left_pads(x: jax.Array) -> int:
    """Count left padding tokens."""
    return jnp.sum(jnp.cumsum(x != 0, axis=-1) == 0, -1)


def compute_positions_from_segment_ids(seg_ids: Array):
    return jax.vmap(lambda row: jnp.where(row != 0, jnp.arange(seg_ids.shape[1]) - jnp.argmax(row), 2**30))(seg_ids)


def repeat_kv(hidden_states: Array, n_rep: int):
    b, t, kv_heads, head_dim = hidden_states.shape
    hidden_states = jnp.expand_dims(hidden_states, axis=3)
    hidden_states = jnp.repeat(hidden_states, repeats=n_rep, axis=3)
    return hidden_states.reshape(b, t, kv_heads * n_rep, head_dim)


class Olmo3Attention(nnx.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int, *, rngs: nnx.Rngs):
        self.config = cfg
        self.layer_idx = layer_idx
        self.use_sliding = cfg.layer_types[layer_idx] == AttentionMode.SLIDE
        self.num_kv_heads = self.config.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        shd = cfg.shd_cfg
        self.q_proj = ShardedLinear(
            cfg.hidden_size,
            cfg.num_attention_heads * self.head_dim,
            use_bias=cfg.attention_bias,
            kernel_sharding=shd.attn_kernel,
            bias_sharding=shd.attn_bias,
            rngs=rngs,
        )
        self.k_proj = ShardedLinear(
            cfg.hidden_size,
            cfg.num_key_value_heads * self.head_dim,
            use_bias=cfg.attention_bias,
            kernel_sharding=shd.attn_kernel,
            bias_sharding=shd.attn_bias,
            rngs=rngs,
        )
        self.v_proj = ShardedLinear(
            cfg.hidden_size,
            cfg.num_key_value_heads * self.head_dim,
            use_bias=cfg.attention_bias,
            kernel_sharding=shd.attn_kernel,
            bias_sharding=shd.attn_bias,
            rngs=rngs,
        )
        self.o_proj = ShardedLinear(
            cfg.num_attention_heads * self.head_dim,
            cfg.hidden_size,
            use_bias=cfg.attention_bias,
            kernel_sharding=shd.attn_kernel,
            bias_sharding=shd.attn_bias,
            rngs=rngs,
        )
        norm_kwargs = dict(eps=cfg.rms_norm_eps, dtype=cfg.dtype, shd=P())
        self.q_norm = Olmo3RMSNorm(cfg.num_attention_heads * self.head_dim, rngs=rngs, **norm_kwargs)
        self.k_norm = Olmo3RMSNorm(cfg.num_key_value_heads * self.head_dim, rngs=rngs, **norm_kwargs)

        self.n_rep = cfg.num_attention_heads // cfg.num_key_value_heads
        self.scale = self.head_dim**-0.5

    def __call__(self, x: Array, cache: LayerCache | None, segment_ids: Array, mask: Array | None) -> Array:
        # Get projections
        new_shape = (*x.shape[:-1], -1, self.head_dim)
        shd = self.config.shd_cfg.activation
        q = self.q_norm(self.q_proj(x, out_sharding=shd)).reshape(new_shape)
        k = self.k_norm(self.k_proj(x, out_sharding=shd)).reshape(new_shape)
        v = self.v_proj(x, out_sharding=shd).reshape(new_shape)

        # Apply rope
        left_pads = count_left_pads(segment_ids)
        cache.start_ind[...] = jnp.where(cache.start_ind.get_value() < 0, left_pads, cache.start_ind.get_value())
        position_ids = compute_positions_from_segment_ids(segment_ids) + cache.cur_ind.get_value()
        if self.use_sliding:
            sin, cos = _generate_pos_embeddings_rope(position_ids, self.head_dim, self.config)
        else:
            sin, cos = _generate_pos_embeddings_yarn(position_ids, self.head_dim, self.config)
        q = apply_rope(q, sin, cos)
        k = apply_rope(k, sin, cos)

        # Update cache
        slice_indices = (0, cache.cur_ind.get_value(), 0, 0)
        cache.k_cache[...] = jax.lax.dynamic_update_slice(cache.k_cache.get_value(), k, slice_indices)
        cache.v_cache[...] = jax.lax.dynamic_update_slice(cache.v_cache.get_value(), v, slice_indices)

        k, v = repeat_kv(cache.k_cache.get_value(), self.n_rep), repeat_kv(cache.v_cache.get_value(), self.n_rep)
        intermediate_shd = self.config.shd_cfg.attn_qk_activation

        # update mask based on start_ind
        mask = (jnp.arange(cache.size) >= cache.start_ind[:, None, None]) & mask

        # attention
        qkv = sharded_attention(
            q, k, v, mask=mask, scale=self.scale, attn_logit_sharding=intermediate_shd, out_sharding=shd
        )
        t = x.shape[1]
        cache.cur_ind[...] = cache.cur_ind.get_value() + t
        return self.o_proj(qkv.reshape(*x.shape[:-1], -1), out_sharding=shd)


class Olmo3MLP(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.config = cfg
        shd = cfg.shd_cfg
        self.gate_proj = ShardedLinear(
            cfg.hidden_size,
            cfg.intermediate_size,
            use_bias=False,
            kernel_sharding=shd.fc1_kernel,
            bias_sharding=shd.fc1_bias,
            rngs=rngs,
        )
        self.up_proj = ShardedLinear(
            cfg.hidden_size,
            cfg.intermediate_size,
            use_bias=False,
            kernel_sharding=shd.fc1_kernel,
            bias_sharding=shd.fc1_bias,
            rngs=rngs,
        )
        self.down_proj = ShardedLinear(
            cfg.intermediate_size,
            cfg.hidden_size,
            use_bias=False,
            kernel_sharding=shd.fc2_kernel,
            bias_sharding=shd.fc2_bias,
            rngs=rngs,
        )

    def __call__(self, x):
        shd = self.config.shd_cfg.activation
        x = jax.nn.silu(self.gate_proj(x, out_sharding=shd)) * self.up_proj(x, out_sharding=shd)
        return self.down_proj(x, out_sharding=shd)


class Olmo3DecoderLayer(nnx.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int, *, rngs: nnx.Rngs):
        self.self_attn = Olmo3Attention(cfg, layer_idx, rngs=rngs)
        self.mlp = Olmo3MLP(cfg, rngs=rngs)

        norm_kwargs = dict(dim=cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=cfg.dtype, shd=cfg.shd_cfg.norm)
        self.post_attention_layernorm = Olmo3RMSNorm(**norm_kwargs, rngs=rngs)
        self.post_feedforward_layernorm = Olmo3RMSNorm(**norm_kwargs, rngs=rngs)

    def __call__(self, x: Array, cache: LayerCache | None, segment_ids: Array, mask: Array | None) -> Array:
        res = x
        x = self.self_attn(x=x, cache=cache, segment_ids=segment_ids, mask=mask)
        x = self.post_attention_layernorm(x)
        x = res + x

        res = x
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        return res + x


# TODO: use the out_sharding
def make_causal_mask(layer_cache: LayerCache, t: int, *, out_sharding: PartitionSpec):
    seq_arange = jnp.arange(t)
    cache_arange = jnp.arange(layer_cache.size)
    causal_mask = seq_arange[:, None] - cache_arange[None, :] >= -layer_cache.cur_ind
    causal_mask = causal_mask.astype(jnp.bool_)
    return causal_mask


def make_window_mask(layer_cache: LayerCache, t: int, slide_size: int, *, out_sharding: PartitionSpec):
    causal_mask = make_causal_mask(layer_cache, t, out_sharding=out_sharding)
    *_, t, c = causal_mask.shape
    seq_arange = jnp.arange(t)
    cache_arange = jnp.arange(c)
    slide = seq_arange[:, None] - cache_arange[None, :] < slide_size
    return causal_mask & slide


class Olmo3Model(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.config = cfg
        self.padding_idx = cfg.pad_token_id
        self.sliding_window_size = cfg.sliding_window
        ei = partial(default_embed_init, out_sharding=cfg.shd_cfg.emb_kernel)
        self.embed_tokens = ShardedEmbedding(cfg.vocab_size, cfg.hidden_size, embedding_init=ei, rngs=rngs)
        self.layers = nnx.List(
            [Olmo3DecoderLayer(cfg, layer_idx, rngs=rngs) for layer_idx in range(cfg.num_hidden_layers)]
        )
        self.norm = Olmo3RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=cfg.dtype, shd=cfg.shd_cfg.norm, rngs=rngs)

    def __call__(self, input_ids: Array, segment_ids: Array, cache: Cache):
        shd = self.config.shd_cfg.activation
        t = segment_ids.shape[1]
        causal_mask = make_causal_mask(cache[0], t, out_sharding=shd)
        sliding_mask = make_window_mask(cache[0], t, slide_size=self.sliding_window_size, out_sharding=shd)
        x = self.embed_tokens(input_ids, out_sharding=shd)

        for lt, c, layer in zip(self.config.layer_types, cache, self.layers):
            mask = sliding_mask if lt == AttentionMode.SLIDE else causal_mask
            x = layer(x, c, segment_ids, mask)
        x = self.norm(x)
        return x


class Olmo3ForCausalLM(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.config = cfg
        self.model = Olmo3Model(cfg, rngs=rngs)
        self.vocab_size = cfg.vocab_size
        self.lm_head = ShardedLinear(
            cfg.hidden_size,
            cfg.vocab_size,
            use_bias=False,
            kernel_sharding=self.config.shd_cfg.emb_kernel,
            bias_sharding=None,
            rngs=rngs,
        )

    def __call__(self, input_ids: Array, segment_ids: Array, cache: Cache):
        hiddens = self.model(input_ids, segment_ids, cache)
        return self.lm_head(hiddens, out_sharding=self.config.shd_cfg.activation)


@jax.jit
def forward(model: Olmo3ForCausalLM, cache: Cache, input_ids: Array, segment_ids: Array) -> Array:
    logits = model(input_ids, segment_ids, cache)
    return logits[:, -1, :], cache
