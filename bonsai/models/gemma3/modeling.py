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

import math
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.nnx.nn.linear import default_embed_init
from jax import P

# TODO: Would be better to rely on something not in jax._src
from jax._src.nn.functions import _apply_masks
from jax.sharding import PartitionSpec
from jaxtyping import Array


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
class VisionShardingCfg:
    attn_kernel: PartitionSpec
    attn_bias: PartitionSpec
    attn_qk_activation: PartitionSpec
    fc1_kernel: PartitionSpec
    fc1_bias: PartitionSpec
    fc2_kernel: PartitionSpec
    fc2_bias: PartitionSpec
    activation: PartitionSpec
    layer_norm: PartitionSpec
    emb_patch_kernel: PartitionSpec
    emb_patch_bias: PartitionSpec
    emb_pos_kernel: PartitionSpec

    @staticmethod
    def no_sharding():
        return VisionShardingCfg.default(False, False)

    @staticmethod
    def default(use_fsdp: bool, use_tp: bool):
        fsdp = ShardMode.FSDP.value if use_fsdp else None
        tp = ShardMode.TP.value if use_tp else None
        return VisionShardingCfg(
            attn_kernel=P(tp, fsdp),
            attn_bias=P(tp),
            attn_qk_activation=P(fsdp, tp),
            fc1_kernel=P(fsdp, tp),
            fc1_bias=P(tp),
            fc2_kernel=P(tp, fsdp),
            fc2_bias=P(tp),
            activation=P(fsdp, None, tp),
            layer_norm=P(tp),
            emb_patch_kernel=P(None, None, None, tp),
            emb_patch_bias=P(tp),
            emb_pos_kernel=P(None, tp),
        )


@dataclass(slots=True, frozen=True)
class TextShardingCfg:
    attn_kernel: PartitionSpec
    attn_bias: PartitionSpec
    attn_qk_activation: PartitionSpec
    down_kernel: PartitionSpec
    down_bias: PartitionSpec
    up_gate_kernel: PartitionSpec
    up_gate_bias: PartitionSpec
    activation: PartitionSpec
    decoder_norm: PartitionSpec
    cache: PartitionSpec
    emb_kernel: PartitionSpec

    @staticmethod
    def no_sharding():
        return TextShardingCfg.default(False, False)

    @staticmethod
    def default(use_fsdp: bool, use_tp: bool):
        fsdp = ShardMode.FSDP.value if use_fsdp else None
        tp = ShardMode.TP.value if use_tp else None
        return TextShardingCfg(
            attn_kernel=P(tp, fsdp),
            attn_bias=P(tp),
            attn_qk_activation=P(fsdp, tp),
            down_kernel=P(tp, fsdp),
            down_bias=P(tp),
            up_gate_kernel=P(fsdp, tp),
            up_gate_bias=P(tp),
            activation=P(fsdp, None, tp),
            decoder_norm=P(tp),
            cache=P(fsdp, None, tp, None),
            emb_kernel=P(None, tp),
        )


@dataclass(slots=True, frozen=True)
class ShardingCfg:
    mmp_norm: PartitionSpec
    mmp_weight: PartitionSpec

    @staticmethod
    def default(use_tp: bool):
        tp = ShardMode.TP.value if use_tp else None
        return ShardingCfg(mmp_norm=P(tp), mmp_weight=P(tp))


@dataclass(frozen=True)
class VisionConfig:
    attention_dropout: float  # TODO: unused
    hidden_size: int
    image_size: int
    intermediate_size: int
    layer_norm_eps: float
    num_attention_heads: int
    num_channels: int
    num_hidden_layers: int
    patch_size: int
    vision_use_head: bool
    shd_cfg: VisionShardingCfg

    @classmethod
    def gemma3_4b_it(cls, use_fsdp: bool, use_tp: bool):
        return cls(
            attention_dropout=0.0,
            hidden_size=1152,
            image_size=896,
            intermediate_size=4304,
            layer_norm_eps=1e-6,
            num_attention_heads=16,
            num_channels=3,
            num_hidden_layers=27,
            patch_size=14,
            vision_use_head=False,
            shd_cfg=VisionShardingCfg.default(use_fsdp, use_tp),
        )


@dataclass(frozen=True)
class TextConfig:
    attention_bias: bool
    attention_dropout: float  # TODO: unused
    head_dim: int
    hidden_size: int
    intermediate_size: int
    layer_types: list[AttentionMode]
    max_position_embeddings: int  # TODO: unused
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_full_factor: float
    rope_full_theta: float
    rope_slide_factor: float
    rope_slide_theta: float
    sliding_window: int
    vocab_size: int
    norm_dtype: jnp.dtype
    shd_cfg: TextShardingCfg

    @classmethod
    def gemma3_4b_it(cls, use_fsdp: bool, use_tp: bool, *, norm_dtype: jnp.dtype):
        num_hidden_layers = 34
        return cls(
            attention_bias=False,
            attention_dropout=0.0,  # TODO: unused
            head_dim=256,
            hidden_size=2560,
            intermediate_size=10240,
            layer_types=_set_attention_modes(6, num_hidden_layers),
            max_position_embeddings=131072,  # TODO: unused
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            rms_norm_eps=1e-6,
            rope_full_factor=8.0,
            rope_full_theta=1000000.0,
            rope_slide_factor=1.0,
            rope_slide_theta=10000.0,
            sliding_window=1024,
            vocab_size=262208,
            norm_dtype=norm_dtype,
            shd_cfg=TextShardingCfg.default(use_fsdp, use_tp),
        )


@dataclass(frozen=True)
class ModelConfig:
    vision_config: VisionConfig
    text_config: TextConfig
    mm_tokens_per_image: int
    dtype: str  # TODO: unused
    final_logit_softcapping: float | None
    shd_cfg: ShardingCfg

    @classmethod
    def gemma3_4b_it(cls, use_fsdp: bool = False, use_tp: bool = False, *, norm_dtype: jnp.dtype):
        return cls(
            vision_config=VisionConfig.gemma3_4b_it(use_fsdp, use_tp),
            text_config=TextConfig.gemma3_4b_it(use_fsdp, use_tp, norm_dtype=norm_dtype),
            mm_tokens_per_image=256,
            dtype="bfloat16",  # TODO: unused
            final_logit_softcapping=None,
            shd_cfg=ShardingCfg.default(use_tp),
        )


# General Components


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
            # Use take because fancy indexing numpy arrays with JAX indices does not
            # work correctly.
        (embedding,) = self.promote_dtype((self.embedding.value,), dtype=self.dtype, inexact=False)
        if self.num_embeddings == 1:
            return jnp.broadcast_to(embedding, (*inputs.shape, self.features))
        return embedding.at[inputs].get(out_sharding=out_sharding)

    def attend(self, query: Array, *, out_sharding) -> Array:
        query, embedding = self.promote_dtype((query, self.embedding.value), dtype=self.dtype)
        return jnp.dot(query, embedding.T, out_sharding=out_sharding)


# adapted from the jax.nn.dot_product_attention implementation
def sharded_attention(q, k, v, mask, scale=None, *, attn_logit_sharding: PartitionSpec, out_sharding: PartitionSpec):
    logits = jnp.einsum("BTNH,BSNH->BNTS", q, k, out_sharding=attn_logit_sharding)
    scale_val = (1.0 / np.sqrt(k.shape[-1])) if scale is None else scale
    logits *= jnp.array(scale_val, dtype=logits.dtype)

    is_causal = False
    local_window_size, q_seqlen, kv_seqlen = None, None, None
    padded_logits = _apply_masks(logits, mask, is_causal, q_seqlen, kv_seqlen, local_window_size)

    padded_logits = padded_logits.astype(np.float32)
    probs = jax.nn.softmax(padded_logits, axis=-1).astype(k.dtype)
    # TODO: Add dropout here

    attn_out = jnp.einsum("BNTS,BSNH->BTNH", probs, v, out_sharding=out_sharding)
    return attn_out


# Vision Components


# TODO: update to include interpolate_pos_encoding
class SiglipVisionEmbeddings(nnx.Module):
    def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.num_patches = (config.image_size // config.patch_size) ** 2

        ki = partial(jax.nn.initializers.lecun_normal(), out_sharding=config.shd_cfg.emb_patch_kernel)
        bi = partial(jax.nn.initializers.zeros, out_sharding=config.shd_cfg.emb_patch_bias)
        self.patch_embedding = nnx.Conv(
            config.num_channels,
            config.hidden_size,
            kernel_size=(config.patch_size,) * 2,
            strides=(config.patch_size,) * 2,
            padding="valid",
            kernel_init=ki,
            bias_init=bi,
            rngs=rngs,
        )

        ei = partial(default_embed_init, out_sharding=config.shd_cfg.emb_pos_kernel)
        self.position_embedding = ShardedEmbedding(self.num_patches, config.hidden_size, embedding_init=ei, rngs=rngs)

        shd = P(config.shd_cfg.activation[0])
        self.position_ids = jax.device_put(jnp.expand_dims(jnp.arange(self.num_patches), 0), shd)

    def __call__(self, pixel_values: Array):
        patch_embeds = self.patch_embedding(pixel_values)
        b, h, w, c = patch_embeds.shape
        embeddings = patch_embeds.reshape((b, h * w, c))
        shd = self.config.shd_cfg.activation
        out = embeddings + self.position_embedding(self.position_ids, out_sharding=shd)
        return out


class SiglipAttention(nnx.Module):
    def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        hs, shd = config.hidden_size, config.shd_cfg
        self.k_proj = ShardedLinear(hs, hs, kernel_sharding=shd.attn_kernel, bias_sharding=shd.attn_bias, rngs=rngs)
        self.v_proj = ShardedLinear(hs, hs, kernel_sharding=shd.attn_kernel, bias_sharding=shd.attn_bias, rngs=rngs)
        self.q_proj = ShardedLinear(hs, hs, kernel_sharding=shd.attn_kernel, bias_sharding=shd.attn_bias, rngs=rngs)
        self.out_proj = ShardedLinear(hs, hs, kernel_sharding=shd.attn_kernel, bias_sharding=shd.attn_bias, rngs=rngs)

    def __call__(self, x: Array, attn_mask: Array | None):
        batch_size, seq_length, _ = x.shape
        shape = (batch_size, seq_length, self.num_heads, self.head_dim)
        shd = self.config.shd_cfg.activation

        q = self.q_proj(x, out_sharding=shd).reshape(shape)
        k = self.k_proj(x, out_sharding=shd).reshape(shape)
        v = self.v_proj(x, out_sharding=shd).reshape(shape)

        intermediate_shd = self.config.shd_cfg.attn_qk_activation
        attn = sharded_attention(
            q, k, v, mask=attn_mask, attn_logit_sharding=intermediate_shd, out_sharding=shd
        ).reshape(x.shape)
        return self.out_proj(attn, out_sharding=shd)


class SiglipMLP(nnx.Module):
    def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs):
        self.config = config
        shd = config.shd_cfg
        self.fc1 = ShardedLinear(
            config.hidden_size,
            config.intermediate_size,
            kernel_sharding=shd.fc1_kernel,
            bias_sharding=shd.fc1_bias,
            rngs=rngs,
        )
        self.fc2 = ShardedLinear(
            config.intermediate_size,
            config.hidden_size,
            kernel_sharding=shd.fc2_kernel,
            bias_sharding=shd.fc2_bias,
            rngs=rngs,
        )

    def __call__(self, x: Array):
        x = self.fc1(x, out_sharding=self.config.shd_cfg.activation)
        x = jax.nn.gelu(x)
        x = self.fc2(x, out_sharding=self.config.shd_cfg.activation)
        return x


class SiglipEncoderLayer(nnx.Module):
    def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs):
        self.config = config
        shd = config.shd_cfg.layer_norm
        si = partial(jax.nn.initializers.ones, out_sharding=shd)
        bi = partial(jax.nn.initializers.zeros, out_sharding=shd)
        self.layer_norm1 = nnx.LayerNorm(
            config.hidden_size, epsilon=config.layer_norm_eps, scale_init=si, bias_init=bi, rngs=rngs
        )
        self.layer_norm2 = nnx.LayerNorm(
            config.hidden_size, epsilon=config.layer_norm_eps, scale_init=si, bias_init=bi, rngs=rngs
        )
        self.self_attn = SiglipAttention(config, rngs=rngs)
        self.mlp = SiglipMLP(config, rngs=rngs)

    def __call__(self, x: Array, attn_mask: Array | None):
        hidden = self.layer_norm1(x)
        hidden = self.self_attn(hidden, attn_mask)
        hidden = x + hidden
        x = hidden
        hidden = self.layer_norm2(hidden)
        hidden = self.mlp(hidden)
        return hidden + x


class SiglipEncoder(nnx.Module):
    def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.layers = nnx.List([SiglipEncoderLayer(config, rngs=rngs) for _ in range(config.num_hidden_layers)])

    def __call__(self, x: Array, attn_mask: Array | None):
        for l in self.layers:
            x = l(x, attn_mask)
        return x


# TODO: Skip for now since not in 4b, but test later
class SiglipMultiheadAttentionPoolingHead(nnx.Module):
    def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.probe = nnx.Param(nnx.initializers.normal(stddev=0.02)(rngs.params(), (1, 1, config.hidden_size)))
        self.attention = nnx.MultiHeadAttention(config.num_attention_heads, config.hidden_size, rngs=rngs)
        self.layernorm = nnx.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps, rngs=rngs)
        self.mlp = SiglipMLP(config, rngs=rngs)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Not yet implemented")


class SiglipVisionTransformer(nnx.Module):
    def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.embeddings = SiglipVisionEmbeddings(config, rngs=rngs)
        self.encoder = SiglipEncoder(config, rngs=rngs)
        shd = config.shd_cfg.layer_norm
        si = partial(jax.nn.initializers.ones, out_sharding=shd)
        bi = partial(jax.nn.initializers.zeros, out_sharding=shd)
        self.post_layernorm = nnx.LayerNorm(
            config.hidden_size, epsilon=config.layer_norm_eps, scale_init=si, bias_init=bi, rngs=rngs
        )

        self.use_head = True if not hasattr(config, "vision_use_head") else config.vision_use_head
        if self.use_head:
            self.head = SiglipMultiheadAttentionPoolingHead(config)

    def __call__(self, pixel_values: Array):
        x = self.embeddings(pixel_values)
        x = self.encoder(x, attn_mask=None)
        x = self.post_layernorm(x)
        if self.use_head:
            x = self.head(x)
        return x


# Language components


class LayerCache(nnx.Module):
    def __init__(self, cfg: TextConfig, batch_size: int, cache_size: int, dtype: jnp.dtype):
        cache_shape = (batch_size, cache_size, cfg.num_key_value_heads, cfg.head_dim)
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
    return [
        LayerCache(cfg.text_config, batch_size, cache_size, dtype) for _ in range(cfg.text_config.num_hidden_layers)
    ]


class Gemma3RMSNorm(nnx.Module):
    def __init__(self, dim: int, eps: float, *, dtype: jnp.dtype, shd: PartitionSpec, rngs: nnx.Rngs):
        self.scale = nnx.Param(jax.nn.initializers.zeros(rngs.params(), dim, dtype=dtype, out_sharding=shd))
        self.eps = eps

    @jax.named_scope("rms_norm")
    def __call__(self, x: Array) -> Array:
        dtype = x.dtype
        xf32 = x.astype(jnp.float32)
        out = xf32 * jax.lax.rsqrt(jnp.square(xf32).mean(-1, keepdims=True) + self.eps)
        out = out * (1.0 + self.scale.value.astype(jnp.float32))
        return out.astype(dtype)


class Gemma3TextScaledWordEmbedding(nnx.Module):
    def __init__(self, cfg: TextConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        ei = partial(default_embed_init, out_sharding=cfg.shd_cfg.emb_kernel)
        self.weight = ShardedEmbedding(cfg.vocab_size, cfg.hidden_size, embedding_init=ei, rngs=rngs)
        self.embed_scale = jnp.array(cfg.hidden_size**0.5, dtype=jnp.bfloat16).astype(jnp.float32)

    def __call__(self, input_ids: Array):
        shd = self.cfg.shd_cfg.activation
        x = self.weight(input_ids, out_sharding=shd) * self.embed_scale
        return x


def _generate_pos_embeddings(
    positions: jax.Array,
    head_dim: int,
    rope_theta: int = 1_000_000,
    factor: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    # Forked from: jax-llm-examples/qwen3/qwen3_jax/model.py;l=571
    fraction = jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim
    timescale = rope_theta**fraction
    rotational_frequency = 1.0 / timescale
    rotational_frequency /= factor
    # Use high-precision einsum to prevent catastrophic bfloat16 rounding (ex: 257→256), as sin(257) differs from sin(256).
    sinusoid_inp = jnp.einsum("BT,k->BTk", positions, rotational_frequency, precision=jax.lax.Precision.HIGHEST)
    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


def apply_rope(x: jax.Array, sin: jax.Array, cos: jax.Array) -> jax.Array:
    assert x.ndim == 4 and sin.ndim == 3 and cos.ndim == 3
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    # [B, T, head_dim] -> [B, h, T, head_dim]
    sin, cos = sin[:, :, None, :], cos[:, :, None, :]
    out = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(x.dtype)
    return out


def count_left_pads(x: jax.Array) -> int:
    """Count left padding tokens."""
    return jnp.sum(jnp.cumsum(x != 0, axis=-1) == 0, -1)


# TODO: Not used right now
def count_right_pads(x: jax.Array, pad_id) -> int:
    result = jnp.where(
        jnp.all(x == pad_id, axis=1), x.shape[1], jnp.argmin(jnp.flip(x == pad_id, axis=1).astype(jnp.int32), axis=1)
    )
    return jnp.max(result)


def compute_positions_from_segment_ids(seg_ids: Array):
    return jax.vmap(lambda row: jnp.where(row != 0, jnp.arange(seg_ids.shape[1]) - jnp.argmax(row), 2**30))(seg_ids)


def repeat_kv(hidden_states: Array, n_rep: int):
    b, t, kv_heads, head_dim = hidden_states.shape
    hidden_states = jnp.expand_dims(hidden_states, axis=3)
    hidden_states = jnp.repeat(hidden_states, repeats=n_rep, axis=3)
    return hidden_states.reshape(b, t, kv_heads * n_rep, head_dim)


class Gemma3Attention(nnx.Module):
    def __init__(self, config: TextConfig, layer_idx: int, *, rngs: nnx.Rngs):
        self.config = config
        self.layer_idx = layer_idx
        self.use_sliding = config.layer_types[layer_idx] == AttentionMode.SLIDE
        self.num_kv_heads = self.config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        shd = config.shd_cfg
        self.q_proj = ShardedLinear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            use_bias=config.attention_bias,
            kernel_sharding=shd.attn_kernel,
            bias_sharding=shd.attn_bias,
            rngs=rngs,
        )
        self.k_proj = ShardedLinear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            use_bias=config.attention_bias,
            kernel_sharding=shd.attn_kernel,
            bias_sharding=shd.attn_bias,
            rngs=rngs,
        )
        self.v_proj = ShardedLinear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            use_bias=config.attention_bias,
            kernel_sharding=shd.attn_kernel,
            bias_sharding=shd.attn_bias,
            rngs=rngs,
        )
        self.o_proj = ShardedLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            use_bias=config.attention_bias,
            kernel_sharding=shd.attn_kernel,
            bias_sharding=shd.attn_bias,
            rngs=rngs,
        )
        self.q_norm = Gemma3RMSNorm(config.head_dim, config.rms_norm_eps, dtype=config.norm_dtype, shd=P(), rngs=rngs)
        self.k_norm = Gemma3RMSNorm(config.head_dim, config.rms_norm_eps, dtype=config.norm_dtype, shd=P(), rngs=rngs)

        self.rope_theta = config.rope_slide_theta if self.use_sliding else config.rope_full_theta
        self.factor = config.rope_slide_factor if self.use_sliding else config.rope_full_factor

        self.n_rep = config.num_attention_heads // config.num_key_value_heads
        self.scale = config.head_dim**-0.5

    def __call__(self, x: Array, cache: LayerCache | None, segment_ids: Array, mask: Array | None) -> Array:
        # Get projections
        new_shape = (*x.shape[:-1], -1, self.head_dim)
        shd = self.config.shd_cfg.activation
        q = self.q_norm(self.q_proj(x, out_sharding=shd).reshape(new_shape))
        k = self.k_norm(self.k_proj(x, out_sharding=shd).reshape(new_shape))
        v = self.v_proj(x, out_sharding=shd).reshape(new_shape)

        # Apply rope
        left_pads = count_left_pads(segment_ids)
        cache.start_ind.value = jnp.where(cache.start_ind.value < 0, left_pads, cache.start_ind.value)
        position_ids = compute_positions_from_segment_ids(segment_ids) + cache.cur_ind.value
        sin, cos = _generate_pos_embeddings(position_ids, self.head_dim, self.rope_theta, factor=self.factor)
        q = apply_rope(q, sin, cos)
        k = apply_rope(k, sin, cos)

        # Update cache
        slice_indices = (0, cache.cur_ind.value, 0, 0)
        cache.k_cache.value = jax.lax.dynamic_update_slice(cache.k_cache.value, k, slice_indices)
        cache.v_cache.value = jax.lax.dynamic_update_slice(cache.v_cache.value, v, slice_indices)

        k, v = repeat_kv(cache.k_cache.value, self.n_rep), repeat_kv(cache.v_cache.value, self.n_rep)
        intermediate_shd = self.config.shd_cfg.attn_qk_activation
        qkv = sharded_attention(
            q, k, v, mask=mask, scale=self.scale, attn_logit_sharding=intermediate_shd, out_sharding=shd
        )
        t = x.shape[1]
        cache.cur_ind.value = cache.cur_ind.value + t
        return self.o_proj(qkv.reshape(*x.shape[:-1], -1), out_sharding=shd)


class Gemma3MLP(nnx.Module):
    def __init__(self, config: TextConfig, *, rngs: nnx.Rngs):
        self.config = config
        hsize, isize, shd = config.hidden_size, config.intermediate_size, config.shd_cfg
        self.gate_proj = ShardedLinear(
            hsize, isize, use_bias=False, kernel_sharding=shd.up_gate_kernel, bias_sharding=shd.up_gate_bias, rngs=rngs
        )
        self.up_proj = ShardedLinear(
            hsize, isize, use_bias=False, kernel_sharding=shd.up_gate_kernel, bias_sharding=shd.up_gate_bias, rngs=rngs
        )
        self.down_proj = ShardedLinear(
            isize, hsize, use_bias=False, kernel_sharding=shd.down_kernel, bias_sharding=shd.down_bias, rngs=rngs
        )

    def __call__(self, x: Array):
        ux = self.up_proj(x, out_sharding=self.config.shd_cfg.activation)
        gx = jax.nn.gelu(self.gate_proj(x, out_sharding=self.config.shd_cfg.activation))
        out = self.down_proj(gx * ux, out_sharding=self.config.shd_cfg.activation)
        return out


class Gemma3DecoderLayer(nnx.Module):
    def __init__(self, config: TextConfig, layer_idx: int, *, rngs: nnx.Rngs):
        super().__init__()
        self.self_attn = Gemma3Attention(config=config, layer_idx=layer_idx, rngs=rngs)
        self.mlp = Gemma3MLP(config, rngs=rngs)

        norm_shd = config.shd_cfg.decoder_norm
        norm_kwargs = dict(dim=config.hidden_size, eps=config.rms_norm_eps, dtype=config.norm_dtype, shd=norm_shd)
        self.input_layernorm = Gemma3RMSNorm(**norm_kwargs, rngs=rngs)
        self.post_attention_layernorm = Gemma3RMSNorm(**norm_kwargs, rngs=rngs)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(**norm_kwargs, rngs=rngs)
        self.post_feedforward_layernorm = Gemma3RMSNorm(**norm_kwargs, rngs=rngs)

    def __call__(self, x: Array, cache: LayerCache | None, segment_ids: Array, mask: Array | None) -> Array:
        res = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, cache, segment_ids, mask=mask)
        x = self.post_attention_layernorm(x)
        x = res + x
        res = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        return x + res

    @property
    def head_dim(self):
        return self.o_proj.shape[1]


class Gemma3TextModel(nnx.Module):
    def __init__(self, config: TextConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.layers = nnx.List(
            [Gemma3DecoderLayer(config, layer_idx, rngs=rngs) for layer_idx in range(config.num_hidden_layers)]
        )
        norm_shd = config.shd_cfg.decoder_norm
        self.norm = Gemma3RMSNorm(
            config.hidden_size, config.rms_norm_eps, dtype=config.norm_dtype, shd=norm_shd, rngs=rngs
        )

    def __call__(self, x, cache: Cache, segment_ids: Array, sliding_mask: Array | None, causal_mask: Array | None):
        for lt, c, layer in zip(self.config.layer_types, cache, self.layers):
            mask = sliding_mask if lt == AttentionMode.SLIDE else causal_mask
            x = layer(x, c, segment_ids, mask)
        x = self.norm(x)
        return x


class Gemma3MultiModalProjector(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.config = config
        vhs, ths = config.vision_config.hidden_size, config.text_config.hidden_size
        eps = config.vision_config.layer_norm_eps
        self.patches_per_img = int(config.vision_config.image_size // config.vision_config.patch_size)
        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_img // self.tokens_per_side

        mmp_w_shd, mmp_norm_shd = config.shd_cfg.mmp_weight, config.shd_cfg.mmp_norm
        self.mm_input_projection_weight = nnx.Param(jnp.zeros((vhs, ths), out_sharding=mmp_w_shd), rngs=rngs)
        self.mm_soft_emb_norm = Gemma3RMSNorm(
            vhs, eps=eps, dtype=config.text_config.norm_dtype, shd=mmp_norm_shd, rngs=rngs
        )

    def __call__(self, vision_outputs: Array) -> Array:
        b, _, t = vision_outputs.shape
        vision_outputs = vision_outputs.swapaxes(1, 2).reshape(b, t, self.patches_per_img, self.patches_per_img)

        x = nnx.avg_pool(
            vision_outputs[:, :, :, :, None],
            window_shape=(1, 1, self.kernel_size, self.kernel_size),
            strides=(1, 1, self.kernel_size, self.kernel_size),
        )[:, :, :, :, 0]
        x = x.reshape(b, t, -1).swapaxes(1, 2)
        x = self.mm_soft_emb_norm(x)
        x = jnp.matmul(
            x, self.mm_input_projection_weight.value, out_sharding=self.config.vision_config.shd_cfg.activation
        )
        return x.astype(vision_outputs.dtype)


def make_causal_mask(layer_cache: LayerCache, token_type_ids: Array, *, out_sharding):
    b, t = token_type_ids.shape
    c = layer_cache.size
    seq_arange = jnp.arange(t)
    cache_arange = jnp.arange(c)
    causal_mask = seq_arange[:, None] - cache_arange[None, :] >= -layer_cache.cur_ind
    tti = token_type_ids.astype(jnp.bool_)
    cache_padded_tti = jnp.concat([tti, jnp.zeros((b, c - t), dtype=jnp.bool_, out_sharding=out_sharding)], axis=-1)
    image_or_mask = tti[:, None, :, None] & cache_padded_tti[:, None, None, :]
    causal_mask = causal_mask.astype(jnp.bool_) | image_or_mask
    return causal_mask


def make_window_mask(layer_cache, token_type_ids, slide_size: int, *, out_sharding):
    causal_mask = make_causal_mask(layer_cache, token_type_ids, out_sharding=out_sharding)
    *_, t, c = causal_mask.shape
    seq_arange = jnp.arange(t)
    cache_arange = jnp.arange(c)
    slide = seq_arange[:, None] - cache_arange[None, :] < slide_size
    return causal_mask & slide


def merge_modalities(img_emb: Array, text_emb: Array, token_mask: Array) -> Array:
    # This function fills the image tokens into the text_emb sequence
    # The token_mask tells us where the image tokens are (0 for text, 1 for image)
    # image_emb is (Li, D)
    # text_emb is (Lt, D)
    # token_mask is (Lt)
    # We have Li < Lt
    img_indices = jnp.cumsum(token_mask) - 1
    safe_indices = jnp.clip(img_indices, 0, img_emb.shape[0] - 1)
    aligned_images = img_emb[safe_indices]
    return jnp.where(token_mask[:, None], aligned_images, text_emb)


def batched_merge_modalities(img_emb: Array, text_emb: Array, token_mask: Array) -> Array:
    # image_emb is (B, Li, D)
    # text_emb is (B, Lt, D)
    # token_mask is (B, Lt)
    # We have Li < Lt
    return jax.vmap(merge_modalities)(img_emb, text_emb, token_mask)


class Gemma3Model(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.config = cfg
        self.sliding_window_size = cfg.text_config.sliding_window
        self.embed_tokens = Gemma3TextScaledWordEmbedding(cfg.text_config, rngs=rngs)
        self.vision_tower = SiglipVisionTransformer(cfg.vision_config, rngs=rngs)
        self.multi_modal_projector = Gemma3MultiModalProjector(cfg, rngs=rngs)
        self.language_model = Gemma3TextModel(cfg.text_config, rngs=rngs)
        self.final_logit_softcapping = cfg.final_logit_softcapping

    def __call__(
        self, input_ids: Array, pixel_values: Array, cache: Cache, segment_ids: Array, token_type_ids: Array
    ) -> Array:
        assert input_ids.shape == token_type_ids.shape
        shd = P(self.config.text_config.shd_cfg.activation[0])
        causal_mask = make_causal_mask(cache[0], token_type_ids, out_sharding=shd)
        sliding_mask = make_window_mask(cache[0], token_type_ids, slide_size=self.sliding_window_size, out_sharding=shd)
        inputs_embeds = self.embed_tokens(input_ids)

        # Merge text and images
        if pixel_values is not None:
            vision_outputs = self.vision_tower(pixel_values)
            image_features = self.multi_modal_projector(vision_outputs).astype(inputs_embeds.dtype)
            inputs_embeds = batched_merge_modalities(image_features, inputs_embeds, token_type_ids)

        out = self.language_model(inputs_embeds, cache, segment_ids, sliding_mask, causal_mask)
        shd = P(self.config.text_config.shd_cfg.activation[0])
        out = self.embed_tokens.weight.attend(out, out_sharding=shd)

        if self.config.final_logit_softcapping is not None:
            out = out / self.final_logit_softcapping
            out = jax.nn.tanh(out)
            out = out * self.final_logit_softcapping

        return out


@jax.jit
def forward(
    model: nnx.Module, cache: Cache, input_ids: Array, pixel_values: Array, segment_ids: Array, token_type_ids
) -> tuple[Array, nnx.Cache]:
    logits = model(input_ids, pixel_values, cache, segment_ids, token_type_ids)
    return logits[:, -1, :], cache
