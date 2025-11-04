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

import dataclasses
from functools import partial
from typing import Any, Tuple, TypeAlias

import flax
import jax
from flax import nnx
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.sharding import auto_axes, get_abstract_mesh, reshard
from jaxtyping import Array, ArrayLike

from bonsai.utils import BaseSampler

_K_MASK = jnp.finfo(jnp.bfloat16).min
ShardingSpec = Tuple[str | None, ...]
ShardingType = ShardingSpec | None


@dataclasses.dataclass(slots=True, frozen=True)
class ShardingCfg:
    emb_vd: ShardingSpec
    emb_dv: ShardingSpec
    q_weight_ndh: ShardingSpec
    kv_weight_ndh: ShardingSpec
    o_weight_nhd: ShardingSpec
    ffw_weight_df: ShardingSpec
    ffw_weight_fd: ShardingSpec
    rms_norm: ShardingSpec
    act_btd: ShardingSpec
    act_btf: ShardingSpec
    act_btnh: ShardingSpec

    @staticmethod
    def default(is_sampling: bool = False):
        fsdp = "fsdp" if not is_sampling else None
        return ShardingCfg(
            emb_vd=("tp", fsdp),
            emb_dv=(fsdp, "tp"),
            q_weight_ndh=("tp", fsdp, None),
            kv_weight_ndh=("tp", fsdp, None),
            o_weight_nhd=("tp", None, fsdp),
            ffw_weight_df=(fsdp, "tp"),
            ffw_weight_fd=("tp", fsdp),
            rms_norm=("tp",),
            act_btd=("fsdp", None, None if is_sampling else "tp"),
            act_btf=("fsdp", None, "tp"),
            act_btnh=("fsdp", None, "tp", None),
        )


@dataclasses.dataclass(frozen=True)
class ModelCfg:
    num_layers: int
    vocab_size: int
    emb_dim: int
    mlp_dim: int
    num_heads: int
    head_dim: int
    num_kv_heads: int
    rope_theta: int
    rope_scaling_factor: int
    local_rope_theta: float
    norm_eps: float
    tie_word_embeddings: bool
    shd_cfg: ShardingCfg = ShardingCfg.default()

    @classmethod
    def qwen3_0_6b(cls):  # qwen3-0.6B
        return cls(
            num_layers=28,
            vocab_size=151936,
            emb_dim=1024,
            mlp_dim=3072,
            num_heads=16,
            head_dim=128,
            num_kv_heads=8,
            norm_eps=1e-06,
            rope_theta=1_000_000,
            rope_scaling_factor=8.0,
            local_rope_theta=1e4,
            tie_word_embeddings=True,
        )

    @classmethod
    def qwen3_1_7b(cls):  # qwen3-1.7B
        return cls(
            num_layers=28,
            vocab_size=151936,
            emb_dim=2048,
            mlp_dim=6144,
            num_heads=16,
            head_dim=128,
            num_kv_heads=8,
            norm_eps=1e-06,
            rope_theta=1_000_000,
            rope_scaling_factor=8.0,
            local_rope_theta=1e4,
            tie_word_embeddings=True,
        )

    @classmethod
    def qwen3_4b(cls):  # qwen3-4B
        return cls(
            num_layers=36,
            vocab_size=151936,
            emb_dim=2560,
            mlp_dim=9728,
            num_heads=32,
            head_dim=128,
            num_kv_heads=8,
            norm_eps=1e-06,
            rope_theta=1_000_000,
            rope_scaling_factor=8.0,
            local_rope_theta=1e4,
            tie_word_embeddings=True,
        )

    @classmethod
    def qwen3_8b(cls):  # qwen3-8B
        return cls(
            num_layers=36,
            vocab_size=151936,
            emb_dim=4096,
            mlp_dim=12288,
            num_heads=32,
            head_dim=128,
            num_kv_heads=8,
            norm_eps=1e-06,
            rope_theta=1_000_000,
            rope_scaling_factor=8.0,
            local_rope_theta=1e4,
            tie_word_embeddings=False,
        )

    @classmethod
    def qwen3_14b(cls):  # qwen3-14B
        return cls(
            num_layers=40,
            vocab_size=151936,
            emb_dim=5120,
            mlp_dim=17408,
            num_heads=40,
            head_dim=128,
            num_kv_heads=8,
            norm_eps=1e-06,
            rope_theta=1_000_000,
            rope_scaling_factor=8.0,
            local_rope_theta=1e4,
            tie_word_embeddings=False,
        )

    @classmethod
    def qwen3_32b(cls):  # qwen3-32B
        return cls(
            num_layers=64,
            vocab_size=151936,
            emb_dim=5120,
            mlp_dim=25600,
            num_heads=64,
            head_dim=128,
            num_kv_heads=8,
            norm_eps=1e-06,
            rope_theta=1_000_000,
            rope_scaling_factor=8.0,
            local_rope_theta=1e4,
            tie_word_embeddings=False,
        )


def shard(x: jnp.ndarray, s: ShardingType):
    mesh = get_abstract_mesh()
    if mesh and mesh.size > 1 and jax.devices()[0].platform != "cpu":
        return reshard(x, P(*s))
    return x


class LayerCache(nnx.Module):
    def __init__(self, batch_size, cache_size, num_kv_heads, head_dim, dtype):
        cache_shape = (batch_size, cache_size, num_kv_heads, head_dim)
        shd = ShardingCfg.default().act_btnh
        self.k_cache = nnx.Cache(shard(jnp.zeros(cache_shape, dtype=dtype), shd))
        self.v_cache = nnx.Cache(shard(jnp.zeros(cache_shape, dtype=dtype), shd))
        self.size = self.k_cache.shape[1]
        self.start_ind = nnx.Variable(-1 * jnp.ones((batch_size,), dtype=jnp.int32))  # first non-pad ind.
        self.cur_ind = nnx.Variable(jnp.zeros((), dtype=jnp.int32))  # scalar for compute efficiency.


Cache: TypeAlias = list[LayerCache]


def init_cache(
    num_layers: int,
    batch_size: int,
    cache_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: jnp.dtype = jnp.bfloat16,
) -> Cache:
    return [LayerCache(batch_size, cache_size, num_kv_heads, head_dim, dtype) for _ in range(num_layers)]


class Einsum(nnx.Module):
    def __init__(self, einsum_str: str, shape: flax.typing.Shape, *, shd: ShardingSpec, rngs: nnx.Rngs):
        self.einsum_str = einsum_str
        self.shape = shape
        self.w = shard(nnx.Param(nnx.initializers.normal()(rngs.params(), shape)), shd)

    @jax.named_scope("einsum")
    def __call__(self, x: ArrayLike) -> Array:
        return jnp.einsum(self.einsum_str, x, self.w.value)


# TODO(team): Revisit Embedder for proper sharding.
class Embedder(nnx.Module):
    def __init__(self, vocab_size: int, emb_dim: int, *, shd: ShardingSpec, rngs: nnx.Rngs):
        self.input_emb = nnx.Param(nnx.initializers.normal()(rngs.params(), (vocab_size, emb_dim)))
        self.shd = shd

    @jax.named_scope("embedder_encode")
    def encode(self, x: ArrayLike) -> Array:
        return self.input_emb.value.at[(x,)].get(out_sharding=P(*self.shd))

    @jax.named_scope("embedder_decode")
    def decode(self, x: ArrayLike) -> Array:
        return jnp.dot(x, self.input_emb.value.T)


def _generate_pos_embeddings(
    positions: jax.Array,
    head_dim: int,
    rope_theta: int = 1_000_000,
) -> tuple[jax.Array, jax.Array]:
    # Forked from: jax-llm-examples/qwen3/qwen3_jax/model.py;l=571
    fraction = jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim
    timescale = rope_theta**fraction
    rotational_frequency = 1.0 / timescale
    # Use high-precision einsum to prevent catastrophic bfloat16 rounding (ex: 257→256), as sin(257) differs from sin(256).
    sinusoid_inp = jnp.einsum("BT,k->BTk", positions, rotational_frequency, precision=jax.lax.Precision.HIGHEST)
    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


def apply_rope(x: jax.Array, sin: jax.Array, cos: jax.Array) -> jax.Array:
    assert x.ndim == 4 and sin.ndim == 3 and cos.ndim == 3
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    # [B, T, head_dim] -> [B, h, T, head_dim]
    sin, cos = sin[:, :, None, :], cos[:, :, None, :]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(x.dtype)


class RMSNorm(nnx.Module):
    def __init__(self, dim: int, *, norm_eps: float = 1e-06, shd: ShardingSpec, rngs: nnx.Rngs):
        self.scale = shard(nnx.Param(nnx.initializers.ones_init()(rngs.params(), dim)), shd)
        self.norm_eps = norm_eps

    @jax.named_scope("rms_norm")
    def __call__(self, x: Array) -> Array:
        dtype = x.dtype
        rms = jnp.sqrt(jnp.mean(jnp.astype(x, jnp.float32) ** 2, axis=-1, keepdims=True) + self.norm_eps)
        return jnp.astype(self.scale.value * x / rms, dtype)


@auto_axes
def count_left_pads(x: jax.Array):
    return jnp.sum(jnp.cumsum(x != 0, axis=-1) == 0, -1)


@auto_axes
def count_right_pads(x: jax.Array):
    return jnp.sum(jnp.cumsum(jnp.flip(x[0] != 0, axis=-1), axis=-1) == 0, -1)  # Same paddings, right aligned.


def compute_positions_from_segment_ids(seg_ids):
    return jax.vmap(lambda row: jnp.where(row != 0, jnp.arange(seg_ids.shape[1]) - jnp.argmax(row), 2**30))(seg_ids)


class Attention(nnx.Module):
    def __init__(self, cfg: ModelCfg, *, shd_cfg: ShardingCfg = ShardingCfg.default(), rngs: nnx.Rngs):
        self.shd_cfg = shd_cfg
        einsum_fn = partial(Einsum, rngs=rngs)
        self.q_proj = einsum_fn("BTD,DNH->BTNH", (cfg.emb_dim, cfg.num_heads, cfg.head_dim), shd=shd_cfg.q_weight_ndh)
        self.k_proj = einsum_fn(
            "BSD,DKH->BSKH", (cfg.emb_dim, cfg.num_kv_heads, cfg.head_dim), shd=shd_cfg.kv_weight_ndh
        )
        self.v_proj = einsum_fn(
            "BSD,DKH->BSKH", (cfg.emb_dim, cfg.num_kv_heads, cfg.head_dim), shd=shd_cfg.kv_weight_ndh
        )
        self.o_proj = einsum_fn("BTNH,NHD->BTD", (cfg.num_heads, cfg.head_dim, cfg.emb_dim), shd=shd_cfg.o_weight_nhd)

        self.q_norm = RMSNorm(cfg.head_dim, norm_eps=cfg.norm_eps, shd=shd_cfg.rms_norm, rngs=rngs)
        self.k_norm = RMSNorm(cfg.head_dim, norm_eps=cfg.norm_eps, shd=shd_cfg.rms_norm, rngs=rngs)
        self.n_rep = cfg.num_heads // cfg.num_kv_heads
        self.scale = self.head_dim**-0.5

    @jax.named_scope("attention")
    def __call__(self, x: Array, cache: LayerCache | None, segment_ids: Array, num_right_pads: int) -> Array:
        query_proj = shard(self.q_norm(self.q_proj(x)), self.shd_cfg.act_btnh)
        key_proj = shard(self.k_norm(self.k_proj(x)), self.shd_cfg.act_btnh)
        value_proj = shard(self.v_proj(x), self.shd_cfg.act_btnh)

        cache.start_ind.value = jnp.where(
            cache.start_ind.value < 0, count_left_pads(segment_ids, out_sharding=P(None)), cache.start_ind.value
        )
        position_ids = compute_positions_from_segment_ids(segment_ids) + cache.cur_ind.value
        sin, cos = _generate_pos_embeddings(position_ids, self.head_dim)
        query_proj = apply_rope(query_proj, sin, cos)
        key_proj = apply_rope(key_proj, sin, cos)

        slice_indices = (0, cache.cur_ind.value, 0, 0)
        cache.v_cache.value = jax.lax.dynamic_update_slice(cache.v_cache.value, value_proj, slice_indices)
        cache.k_cache.value = jax.lax.dynamic_update_slice(cache.k_cache.value, key_proj, slice_indices)
        b, t, qh, d = query_proj.shape

        # GQA
        query_proj = query_proj.reshape((b, t, self.num_kv_heads, qh // self.num_kv_heads, d))
        attn_logits = jnp.einsum("BTHGD,BSHD->BHGTS", query_proj, cache.k_cache.value) * self.scale
        q_pos = cache.cur_ind.value + jnp.arange(t, dtype=jnp.int32)[None, :] - cache.start_ind.value[:, None]
        ts = jnp.arange(cache.size, dtype=jnp.int32)  # (b, cache.size)

        kv_segment_ids = (ts[None, :] >= cache.start_ind.value[:, None]) & (ts[None, :] < cache.cur_ind.value + t)
        k_pos = ts[None, :] - cache.start_ind.value[:, None]  # (b, cache.size)
        causal_mask = k_pos[:, None, :] <= q_pos[:, :, None]
        segment_mask = kv_segment_ids[:, None, :] == segment_ids[:, :, None]
        final_mask = causal_mask & segment_mask  # (b, q_len, k_len)
        attn_logits = jnp.where(final_mask[:, None, None, :, :], attn_logits, _K_MASK)
        attn_weights = jax.nn.softmax(attn_logits.astype(jnp.float32), axis=-1).astype(attn_logits.dtype)
        qkv = jnp.einsum("BHGTS,BSHD->BTHGD", attn_weights, cache.v_cache.value).reshape((b, t, qh, d))

        cache.cur_ind.value = cache.cur_ind.value + t - num_right_pads
        return shard(self.o_proj(qkv), self.shd_cfg.act_btd)

    @property
    def head_dim(self):
        return self.o_proj.shape[1]

    @property
    def num_heads(self):
        return self.q_proj.shape[1]

    @property
    def num_kv_heads(self):
        return self.k_proj.shape[1]


class MLP(nnx.Module):
    def __init__(self, cfg: ModelCfg, *, shd_cfg: ShardingCfg = ShardingCfg.default(), rngs: nnx.Rngs):
        self.shd_cfg = shd_cfg
        linear = partial(nnx.Linear, use_bias=False, rngs=rngs)
        self.gate_proj = shard(linear(cfg.emb_dim, cfg.mlp_dim), shd_cfg.ffw_weight_df)
        self.up_proj = shard(linear(cfg.emb_dim, cfg.mlp_dim), shd_cfg.ffw_weight_df)
        self.down_proj = shard(linear(cfg.mlp_dim, cfg.emb_dim), shd_cfg.ffw_weight_fd)

    @jax.named_scope("feed_forward")
    def __call__(self, x: ArrayLike) -> Array:
        activations = nnx.silu(self.gate_proj(x)) * self.up_proj(x)
        activations = shard(activations, self.shd_cfg.act_btf)
        outputs = self.down_proj(activations)
        return outputs


class DecoderLayer(nnx.Module):
    def __init__(self, cfg: ModelCfg, *, shd_cfg: ShardingCfg = ShardingCfg.default(), rngs: nnx.Rngs):
        self.input_layernorm = RMSNorm(cfg.emb_dim, norm_eps=cfg.norm_eps, shd=shd_cfg.rms_norm, rngs=rngs)
        self.attn = Attention(cfg=cfg, shd_cfg=shd_cfg, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(cfg.emb_dim, norm_eps=cfg.norm_eps, shd=shd_cfg.rms_norm, rngs=rngs)
        self.mlp = MLP(cfg=cfg, shd_cfg=shd_cfg, rngs=rngs)

    def __call__(self, x: Array, cache: LayerCache | None, segment_ids: Array, num_right_pads: int) -> Array:
        inputs_normalized = self.input_layernorm(x)
        attn_output = x + self.attn(inputs_normalized, cache, segment_ids, num_right_pads)
        outputs = attn_output + self.mlp(self.post_attention_layernorm(attn_output))
        return outputs


class Qwen3(nnx.Module):
    def __init__(self, cfg: ModelCfg, *, shd_cfg: ShardingCfg = ShardingCfg.default(), rngs: nnx.Rngs):
        self.embedder = Embedder(vocab_size=cfg.vocab_size, emb_dim=cfg.emb_dim, shd=shd_cfg.act_btd, rngs=rngs)
        self.layers = nnx.List([DecoderLayer(cfg=cfg, shd_cfg=shd_cfg, rngs=rngs) for _ in range(cfg.num_layers)])
        self.final_norm = RMSNorm(cfg.emb_dim, norm_eps=cfg.norm_eps, shd=shd_cfg.rms_norm, rngs=rngs)
        self.lm_head = Einsum(
            einsum_str="BTD,DV->BTV", shape=(cfg.emb_dim, cfg.vocab_size), shd=shd_cfg.emb_dv, rngs=rngs
        )

    def __call__(self, tokens, segment_ids, num_right_pads, cache):
        x = self.embedder.encode(tokens)
        for i, layer in enumerate(self.layers):
            x = layer(x, cache[i], segment_ids, num_right_pads)
        logits = self.lm_head(self.final_norm(x))
        return logits


@partial(jax.jit, donate_argnums=(1), static_argnums=(4))
def forward(
    graphdef: nnx.GraphDef[tuple[nnx.Module, Cache]],
    state: nnx.State,
    tokens: Array,
    pad_id: int,
    sampler: BaseSampler,
    key: jax.random.PRNGKey,
) -> tuple[Array, nnx.State]:
    model, cache = nnx.merge(graphdef, state)
    segment_ids = 1 * (tokens != pad_id)
    num_right_pads = count_right_pads(segment_ids, out_sharding=P(None))
    logits = model(tokens, segment_ids, num_right_pads, cache)
    next_tokens = sampler(logits[:, -num_right_pads - 1], key=key)
    state = jax.tree.leaves(nnx.state((model, cache)))
    return next_tokens, state
