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
import math
from enum import Enum
from typing import TypeAlias

import jax
from flax import nnx
from jax import P
from jax import numpy as jnp
from jax.sharding import PartitionSpec, get_abstract_mesh, reshard
from jaxtyping import Array, ArrayLike

LARGE_NEGATIVE = jnp.finfo(jnp.bfloat16).min


class ShardMode(Enum):
    FSDP = "fsdp"
    TP = "tp"


@dataclasses.dataclass(slots=True, frozen=True)
class ShardConfig:
    emb_vd: PartitionSpec | None = None
    emb_dv: PartitionSpec | None = None
    q_weight_ndh: PartitionSpec | None = None
    kv_weight_ndh: PartitionSpec | None = None
    o_weight_nhd: PartitionSpec | None = None
    ffw_weight_df: PartitionSpec | None = None
    ffw_weight_fd: PartitionSpec | None = None
    decoder_norm: PartitionSpec | None = None
    act_btd: PartitionSpec | None = None
    act_btf: PartitionSpec | None = None
    act_btnh: PartitionSpec | None = None

    @staticmethod
    def no_sharding():
        return ShardConfig()

    @staticmethod
    def default(use_fsdp: bool, use_tp: bool):
        fsdp = ShardMode.FSDP.value if use_fsdp else None
        tp = ShardMode.TP.value if use_tp else None
        return ShardConfig(
            emb_vd=P(tp, fsdp),
            emb_dv=P(fsdp, tp),
            q_weight_ndh=P(tp, fsdp, None),
            kv_weight_ndh=P(tp, fsdp, None),
            o_weight_nhd=P(tp, None, fsdp),
            ffw_weight_df=P(fsdp, tp),
            ffw_weight_fd=P(tp, fsdp),
            decoder_norm=P(tp),
            act_btd=P(fsdp, None, tp),
            act_btf=P(fsdp, None, tp),
            act_btnh=P(fsdp, None, tp, None),
        )


@dataclasses.dataclass(frozen=True)
class ModelConfig:
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
    shd_cfg: ShardConfig = ShardConfig.no_sharding()

    @classmethod
    def _from_param(cls, use_fsdp: bool = False, use_tp: bool = False, **kwargs):
        if use_fsdp or use_tp:
            kwargs["shd_cfg"] = ShardConfig.default(use_fsdp=use_fsdp, use_tp=use_tp)
        return cls(**kwargs)

    @classmethod
    def qwen3_0_6b(cls, use_fsdp: bool = False, use_tp: bool = False):  # qwen3-0.6B
        return cls._from_param(
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
            use_fsdp=use_fsdp,
            use_tp=use_tp,
        )

    @classmethod
    def qwen3_1_7b(cls, use_fsdp: bool = False, use_tp: bool = False):  # qwen3-1.7B
        return cls._from_param(
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
            use_fsdp=use_fsdp,
            use_tp=use_tp,
        )

    @classmethod
    def qwen3_4b(cls, use_fsdp: bool = False, use_tp: bool = False):  # qwen3-4B
        return cls._from_param(
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
            use_fsdp=use_fsdp,
            use_tp=use_tp,
        )

    @classmethod
    def qwen3_8b(cls, use_fsdp: bool = False, use_tp: bool = False):  # qwen3-8B
        return cls._from_param(
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
            use_fsdp=use_fsdp,
            use_tp=use_tp,
        )

    @classmethod
    def qwen3_14b(cls, use_fsdp: bool = False, use_tp: bool = False):  # qwen3-14B
        return cls._from_param(
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
            use_fsdp=use_fsdp,
            use_tp=use_tp,
        )


def shard(x: jnp.ndarray, s: PartitionSpec | None):
    mesh = get_abstract_mesh()
    if not mesh.empty and len(mesh.axis_names) > 0 and s is not None:
        return reshard(x, s)
    return x


class LayerCache(nnx.Module):
    def __init__(self, cfg: ModelConfig, batch_size: int, cache_size: int, dtype: jnp.dtype):
        cache_shape = (batch_size, cache_size, cfg.num_kv_heads, cfg.head_dim)
        kv_shd = cfg.shd_cfg.act_btnh
        self.k_cache = nnx.Cache(jnp.zeros(cache_shape, dtype=dtype, out_sharding=kv_shd))
        self.v_cache = nnx.Cache(jnp.zeros(cache_shape, dtype=dtype, out_sharding=kv_shd))
        self.size = self.k_cache.shape[1]
        start_ind_shd = None if kv_shd is None else P(kv_shd[0])
        self.start_ind = nnx.Variable(-1 * jnp.ones((batch_size,), dtype=jnp.int32, out_sharding=start_ind_shd))
        self.cur_ind = nnx.Variable(jnp.zeros((), dtype=jnp.int32))


Cache: TypeAlias = list[LayerCache]


def _generate_pos_embeddings(
    positions: jax.Array, head_dim: int, rope_theta: int = 1_000_000
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


def count_left_pads(x: jax.Array) -> int:
    """Count left padding tokens."""
    return jnp.sum(jnp.cumsum(x != 0, axis=-1) == 0, -1)


def count_right_pads(x: jax.Array, pad_id) -> int:
    result = jnp.where(
        jnp.all(x == pad_id, axis=1), x.shape[1], jnp.argmin(jnp.flip(x == pad_id, axis=1).astype(jnp.int32), axis=1)
    )
    return jnp.max(result)


def compute_positions_from_segment_ids(seg_ids):
    return jax.vmap(lambda row: jnp.where(row != 0, jnp.arange(seg_ids.shape[1]) - jnp.argmax(row), 2**30))(seg_ids)


class Attention(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.shd_cfg = cfg.shd_cfg
        self.q_proj = nnx.Einsum(
            "BTD,DNH->BTNH",
            (cfg.emb_dim, cfg.num_heads, cfg.head_dim),
            kernel_metadata={"out_sharding": self.shd_cfg.q_weight_ndh},
            rngs=rngs,
        )
        self.k_proj = nnx.Einsum(
            "BSD,DKH->BSKH",
            (cfg.emb_dim, cfg.num_kv_heads, cfg.head_dim),
            kernel_metadata={"out_sharding": self.shd_cfg.kv_weight_ndh},
            rngs=rngs,
        )
        self.v_proj = nnx.Einsum(
            "BSD,DKH->BSKH",
            (cfg.emb_dim, cfg.num_kv_heads, cfg.head_dim),
            kernel_metadata={"out_sharding": self.shd_cfg.kv_weight_ndh},
            rngs=rngs,
        )
        self.o_proj = nnx.Einsum(
            "BTNH,NHD->BTD",
            (cfg.num_heads, cfg.head_dim, cfg.emb_dim),
            kernel_metadata={"out_sharding": self.shd_cfg.o_weight_nhd},
            rngs=rngs,
        )

        scale_metadata = {"out_sharding": P() if self.shd_cfg.q_weight_ndh is not None else None}
        self.q_norm = nnx.RMSNorm(cfg.head_dim, epsilon=cfg.norm_eps, scale_metadata=scale_metadata, rngs=rngs)
        self.k_norm = nnx.RMSNorm(cfg.head_dim, epsilon=cfg.norm_eps, scale_metadata=scale_metadata, rngs=rngs)
        self.n_rep = cfg.num_heads // cfg.num_kv_heads
        self.scale = cfg.head_dim**-0.5

    @jax.named_scope("attention")
    def __call__(self, x: Array, cache: LayerCache | None, segment_ids: Array) -> Array:
        shd = self.shd_cfg.act_btnh
        query_proj = self.q_norm(self.q_proj(x, out_sharding=shd))  # [B, T, N, H]
        key_proj = self.k_norm(self.k_proj(x, out_sharding=shd))  # [B, T, K, H]
        value_proj = self.v_proj(x, out_sharding=shd)  # [B, T, K, H]

        # RoPE and Cache Logic
        left_pads = count_left_pads(segment_ids)
        if self.shd_cfg.act_btnh is not None:
            left_pads = shard(left_pads, P(self.shd_cfg.act_btnh[0]))
        cache.start_ind.set_value(jnp.where(cache.start_ind[...] < 0, left_pads, cache.start_ind[...]))
        position_ids = compute_positions_from_segment_ids(segment_ids) + cache.cur_ind[...]
        sin, cos = _generate_pos_embeddings(position_ids, self.head_dim)
        query_proj = apply_rope(query_proj, sin, cos)
        key_proj = apply_rope(key_proj, sin, cos)

        # Update K/V cache [B, S, K, H]
        slice_indices = (0, cache.cur_ind[...], 0, 0)
        cache.v_cache.set_value(jax.lax.dynamic_update_slice(cache.v_cache[...], value_proj, slice_indices))
        cache.k_cache.set_value(jax.lax.dynamic_update_slice(cache.k_cache[...], key_proj, slice_indices))

        b, t, n, h = query_proj.shape

        # GQA reshape and attention logits
        query_proj_gqa = query_proj.reshape((b, t, self.num_kv_heads, self.n_rep, h))
        attn_logits = jnp.einsum("BTKGH,BSKH->BTSKG", query_proj_gqa, cache.k_cache[...]) * self.scale

        # Masking and Softmax
        q_pos = cache.cur_ind[...] + jnp.arange(t, dtype=jnp.int32)[None, :] - cache.start_ind[:, None]
        ts = jnp.arange(cache.size, dtype=jnp.int32)  # (cache.size,)
        kv_segment_ids = (ts[None, :] >= cache.start_ind[:, None]) & (ts[None, :] < cache.cur_ind[...] + t)
        k_pos = ts[None, :] - cache.start_ind[:, None]  # (b, cache.size)
        causal_mask = k_pos[:, None, :] <= q_pos[:, :, None]
        segment_mask = kv_segment_ids[:, None, :] == segment_ids[:, :, None]
        final_mask = causal_mask & segment_mask  # (B, T, S)
        attn_mask = final_mask[:, :, :, None, None]
        attn_logits = jnp.where(attn_mask, attn_logits, LARGE_NEGATIVE)

        # Softmax
        attn_weights = jax.nn.softmax(attn_logits.astype(jnp.float32), axis=2).astype(attn_logits.dtype)
        qkv = jnp.einsum("BTSKG,BSKH->BTKGH", attn_weights, cache.v_cache[...])
        qkv = qkv.reshape((b, t, n, h))

        cache.cur_ind.set_value(cache.cur_ind[...] + t)
        return self.o_proj(qkv, out_sharding=self.shd_cfg.act_btd)

    @property
    def head_dim(self):
        return self.o_proj.kernel.shape[1]

    @property
    def num_heads(self):
        return self.q_proj.kernel.shape[1]

    @property
    def num_kv_heads(self):
        return self.k_proj.kernel.shape[1]


class MLP(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.shd_cfg = cfg.shd_cfg
        kernel_metadata = {"out_sharding": self.shd_cfg.ffw_weight_df}
        self.gate_proj = nnx.Linear(
            cfg.emb_dim, cfg.mlp_dim, kernel_metadata=kernel_metadata, use_bias=False, rngs=rngs
        )
        self.up_proj = nnx.Linear(cfg.emb_dim, cfg.mlp_dim, kernel_metadata=kernel_metadata, use_bias=False, rngs=rngs)
        kernel_metadata = {"out_sharding": self.shd_cfg.ffw_weight_fd}
        self.down_proj = nnx.Linear(
            cfg.mlp_dim, cfg.emb_dim, kernel_metadata=kernel_metadata, use_bias=False, rngs=rngs
        )

    @jax.named_scope("feed_forward")
    def __call__(self, x: ArrayLike) -> Array:
        ux = self.up_proj(x, out_sharding=self.shd_cfg.act_btf)
        gx = nnx.silu(self.gate_proj(x, out_sharding=self.shd_cfg.act_btf))
        outputs = self.down_proj(gx * ux, out_sharding=self.shd_cfg.act_btf)
        return outputs


class DecoderLayer(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        scale_metadata = {"out_sharding": cfg.shd_cfg.decoder_norm}
        norm_kwargs = dict(num_features=cfg.emb_dim, epsilon=cfg.norm_eps, scale_metadata=scale_metadata, rngs=rngs)
        self.input_layernorm = nnx.RMSNorm(**norm_kwargs)
        self.attn = Attention(cfg=cfg, rngs=rngs)
        self.post_attention_layernorm = nnx.RMSNorm(**norm_kwargs)
        self.mlp = MLP(cfg=cfg, rngs=rngs)

    def __call__(self, x: Array, cache: LayerCache | None, segment_ids: Array) -> Array:
        inputs_normalized = self.input_layernorm(x)
        attn_output = x + self.attn(inputs_normalized, cache, segment_ids)
        outputs = attn_output + self.mlp(self.post_attention_layernorm(attn_output))
        return outputs


class Qwen3(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        embedding_metadata = {"out_sharding": cfg.shd_cfg.emb_vd}
        self.embedder = nnx.Embed(
            num_embeddings=cfg.vocab_size,
            features=cfg.emb_dim,
            dtype=jnp.bfloat16,
            embedding_metadata=embedding_metadata,
            rngs=rngs,
        )
        self.out_emb_shd = cfg.shd_cfg.act_btd
        self.layers = nnx.List([DecoderLayer(cfg=cfg, rngs=rngs) for _ in range(cfg.num_layers)])
        scale_metadata = {"out_sharding": cfg.shd_cfg.decoder_norm}
        self.final_norm = nnx.RMSNorm(cfg.emb_dim, epsilon=cfg.norm_eps, scale_metadata=scale_metadata, rngs=rngs)
        self.lm_head = nnx.Einsum(
            "BTD,DV->BTV",
            (cfg.emb_dim, cfg.vocab_size),
            kernel_metadata={"out_sharding": cfg.shd_cfg.emb_dv},
            rngs=rngs,
        )

    def __call__(self, tokens, *, cache, segment_ids):
        x = self.embedder.embedding[...].at[(tokens,)].get(out_sharding=self.out_emb_shd)
        for i, layer in enumerate(self.layers):
            x = layer(x, cache[i], segment_ids)
        logits = self.lm_head(self.final_norm(x), out_sharding=self.out_emb_shd)
        return logits


def init_cache(
    cfg: ModelConfig, batch_size: int, token_len: int, generate_steps: int, dtype: jnp.dtype = jnp.bfloat16
) -> Cache:
    cache_size = 2 ** math.ceil(math.log2(max(token_len + generate_steps, 1)))  # Pad for a sharding-friendly size.
    return [LayerCache(cfg, batch_size, cache_size, dtype) for _ in range(cfg.num_layers)]


@jax.jit
def forward(model: nnx.Module, cache: Cache, tokens: Array, pad_id: int) -> tuple[Array, nnx.Cache]:
    segment_ids = 1 * (tokens != pad_id)
    num_right_pads = count_right_pads(tokens, pad_id)
    logits = model(tokens, cache=cache, segment_ids=segment_ids)
    target_ind = tokens.shape[-1] - num_right_pads - 1
    return logits[:, target_ind], cache
