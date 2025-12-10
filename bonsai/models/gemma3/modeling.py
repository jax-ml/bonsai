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
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Any, Optional, Tuple, TypeAlias

import flax
import jax
import jax.numpy as jnp
import jax.sharding as shd
from flax import nnx
from jax.interpreters import pxla
from jaxtyping import Array, Float

_K_MASK = jax._src.nn.functions._get_large_negative(jax.numpy.float32).item()


class AttentionType(Enum):
    FULL = "full_attention"
    SLIDE = "sliding_attention"


def _make_attn_types():
    # Fix this (5x slide 1x full) x 5 + (4x slide)
    return [AttentionType.FULL if i % 6 == 5 else AttentionType.SLIDE for i in range(34)]


@dataclass
class VisionConfig:
    attention_dropout: float = 0.0
    hidden_act: str = "gelu_pytorch_tanh"
    hidden_size: int = 1152
    image_size: int = 896
    intermediate_size: int = 4304
    layer_norm_eps: float = 1e-6
    num_attention_heads: int = 16
    num_channels: int = 3
    num_hidden_layers: int = 27
    patch_size: int = 14
    vision_use_head: bool = False


@dataclass
class TextConfig:
    _sliding_window_pattern: int = 6
    attention_bias: bool = False
    attention_dropout: float = 0.0
    attn_logit_softcapping: Optional[float] = None
    final_logit_softcapping: Optional[float] = None
    head_dim: int = 256
    hidden_activation: str = "gelu_pytorch_tanh"
    hidden_size: int = 2560
    initializer_range: float = 0.02
    intermediate_size: int = 10240
    layer_types: list[AttentionType] = field(default_factory=lambda: _make_attn_types())
    max_position_embeddings: int = 131072
    num_attention_heads: int = 8
    num_hidden_layers: int = 34
    num_key_value_heads: int = 4
    query_pre_attn_scalar: int = 256
    rms_norm_eps: float = 1e-6
    rope_local_base_freq: float = 10000.0
    rope_scaling: dict[str, Any] = field(default_factory=lambda: {"factor": 8.0, "rope_type": "linear"})
    rope_theta: float = 1000000.0
    sliding_window: int = 1024
    use_cache: bool = True
    vocab_size: int = 262208


@dataclass
class ModelConfig:
    vision_config: VisionConfig = field(default_factory=lambda: VisionConfig())
    text_config: TextConfig = field(default_factory=lambda: TextConfig())
    mm_tokens_per_image: int = 256
    boi_token_index: int = 255999
    dtype: str = "bfloat16"
    eoi_token_index: int = 256000
    eos_token_id: list[int] = field(default_factory=lambda: [1, 106])
    image_token_index: int = 262144
    initializer_range: float = 0.02
    mm_tokens_per_image: int = 256


## GENERAL


## VISION


# TODO: update to include interpolate_pos_encoding
class SiglipVisionEmbeddings(nnx.Module):
    def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.patch_embedding = nnx.Conv(
            config.num_channels,
            config.hidden_size,
            kernel_size=(config.patch_size,) * 2,
            strides=(config.patch_size,) * 2,
            padding="valid",
            rngs=rngs,
        )
        self.position_embedding = nnx.Embed(self.num_patches, config.hidden_size, rngs=rngs)
        self.position_ids = jnp.expand_dims(jnp.arange(self.num_patches), 0)

    def __call__(self, pixel_values: Array):
        patch_embeds = self.patch_embedding(pixel_values)
        b, h, w, c = patch_embeds.shape
        embeddings = patch_embeds.reshape((b, h * w, c))
        return embeddings + self.position_embedding(self.position_ids)


class SiglipAttention(nnx.Module):
    def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.k_proj = nnx.Linear(config.hidden_size, config.hidden_size, rngs=rngs)
        self.v_proj = nnx.Linear(config.hidden_size, config.hidden_size, rngs=rngs)
        self.q_proj = nnx.Linear(config.hidden_size, config.hidden_size, rngs=rngs)
        self.out_proj = nnx.Linear(config.hidden_size, config.hidden_size, rngs=rngs)

    def __call__(self, x: Array, attn_mask: Array | None):
        batch_size, seq_length, _ = x.shape
        shape = (batch_size, seq_length, self.num_heads, self.head_dim)
        q = self.q_proj(x).reshape(shape)
        k = self.k_proj(x).reshape(shape)
        v = self.v_proj(x).reshape(shape)

        attn = jax.nn.dot_product_attention(q, k, v, mask=attn_mask).reshape(x.shape)
        return self.out_proj(attn)


class SiglipMLP(nnx.Module):
    def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.act = jax.nn.gelu
        self.fc1 = nnx.Linear(config.hidden_size, config.intermediate_size, rngs=rngs)
        self.fc2 = nnx.Linear(config.intermediate_size, config.hidden_size, rngs=rngs)

    def __call__(self, x: Array):
        return self.fc2(self.act(self.fc1(x)))


class SiglipEncoderLayer(nnx.Module):
    def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.layer_norm1 = nnx.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps, rngs=rngs)
        self.self_attn = SiglipAttention(config, rngs=rngs)
        self.layer_norm2 = nnx.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps, rngs=rngs)
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
        self.post_layernorm = nnx.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps, rngs=rngs)
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


## LANGUAGE


# from qwen3
class LayerCache(nnx.Module):
    def __init__(self, cfg: ModelConfig, batch_size: int, cache_size: int, dtype: jnp.dtype):
        cache_shape = (batch_size, cache_size, cfg.num_key_value_heads, cfg.head_dim)
        self.k_cache = nnx.Cache(jnp.zeros(cache_shape, dtype=dtype))
        self.v_cache = nnx.Cache(jnp.zeros(cache_shape, dtype=dtype))
        self.size = self.k_cache.shape[1]
        self.start_ind = nnx.Variable(-1 * jnp.ones((batch_size,), dtype=jnp.int32))
        self.cur_ind = nnx.Variable(jnp.zeros((), dtype=jnp.int32))  # scalar for compute efficiency.


def shard(x: jnp.ndarray, s: Tuple[str, ...]):
    mesh = pxla.thread_resources.env.physical_mesh
    if mesh.empty or jax.devices()[0].platform == "cpu":
        return x
    return jax.lax.with_sharding_constraint(x, shd.NamedSharding(mesh, shd.PartitionSpec(*s)))


Cache: TypeAlias = list[LayerCache]


def init_cache(
    cfg: ModelConfig, batch_size: int, token_len: int, generate_steps: int, dtype: jnp.dtype = jnp.bfloat16
) -> Cache:
    cache_size = 2 ** math.ceil(math.log2(max(token_len + generate_steps, 1)))  # Pad for a sharding-friendly size.
    return [
        LayerCache(cfg.text_config, batch_size, cache_size, dtype) for _ in range(cfg.text_config.num_hidden_layers)
    ]


class Gemma3RMSNorm(nnx.Module):
    def __init__(self, dim: int, eps: float, *, rngs: nnx.Rngs):
        self.scale = nnx.Param(nnx.initializers.zeros_init()(rngs.params(), dim))
        self.eps = eps

    @jax.named_scope("rms_norm")
    def __call__(self, x: Array) -> Array:
        dtype = x.dtype
        xf32 = x.astype(jnp.float32)
        out = xf32 * jax.lax.rsqrt(jnp.square(xf32).mean(-1, keepdims=True) + self.eps)
        out = out * (1.0 + self.scale.value.astype(jnp.float32))
        return out.astype(dtype)


class Gemma3TextScaledWordEmbedding(nnx.Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: float = 1.0, *, rngs: nnx.Rngs
    ):
        self.weight = nnx.Embed(num_embeddings, embedding_dim, rngs=rngs)
        self.embed_scale = jnp.array(embed_scale, dtype=jnp.bfloat16).astype(jnp.float32)

    def __call__(self, input_ids: Array):
        return self.weight(input_ids) * self.embed_scale


# below is from qwen3


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
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(x.dtype)


def count_left_pads(x: jax.Array) -> int:
    """Count left padding tokens."""
    return jnp.sum(jnp.cumsum(x != 0, axis=-1) == 0, -1)


def count_right_pads(x: jax.Array, pad_id) -> int:
    result = jnp.where(
        jnp.all(x == pad_id, axis=1), x.shape[1], jnp.argmin(jnp.flip(x == pad_id, axis=1).astype(jnp.int32), axis=1)
    )
    return jnp.max(result)


def compute_positions_from_segment_ids(seg_ids: Array):
    return jax.vmap(lambda row: jnp.where(row != 0, jnp.arange(seg_ids.shape[1]) - jnp.argmax(row), 2**30))(seg_ids)


## Above is from qwen3


def repeat_kv(hidden_states: Array, n_rep: int):
    b, t, kv_heads, head_dim = hidden_states.shape
    hidden_states = jnp.expand_dims(hidden_states, axis=3)
    hidden_states = jnp.repeat(hidden_states, repeats=n_rep, axis=3)
    return hidden_states.reshape(b, t, kv_heads * n_rep, head_dim)


class Gemma3Attention(nnx.Module):
    def __init__(self, config: TextConfig, layer_idx: int, *, rngs: nnx.Rngs):
        self.config = config
        self.layer_idx = layer_idx
        self.use_sliding = config.layer_types[layer_idx] == AttentionType.SLIDE
        self.num_kv_heads = self.config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.q_proj = nnx.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, use_bias=config.attention_bias, rngs=rngs
        )
        self.k_proj = nnx.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, use_bias=config.attention_bias, rngs=rngs
        )
        self.v_proj = nnx.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, use_bias=config.attention_bias, rngs=rngs
        )
        self.o_proj = nnx.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, use_bias=config.attention_bias, rngs=rngs
        )
        self.q_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps, rngs=rngs)
        self.k_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps, rngs=rngs)

        self.rope_theta = config.rope_local_base_freq if self.use_sliding else config.rope_theta
        self.factor = 1.0 if self.use_sliding else 8.0

        self.n_rep = config.num_attention_heads // config.num_key_value_heads
        self.scale = config.head_dim**-0.5

    def __call__(self, x: Array, cache: LayerCache | None, segment_ids: Array, mask: Array | None) -> Array:
        # get projections
        new_shape = (*x.shape[:-1], -1, self.head_dim)
        q = self.q_norm(self.q_proj(x).reshape(new_shape))
        k = self.k_norm(self.k_proj(x).reshape(new_shape))
        v = self.v_proj(x).reshape(new_shape)

        # Apply rope
        left_pads = count_left_pads(segment_ids)
        cache.start_ind.value = jnp.where(cache.start_ind.value < 0, left_pads, cache.start_ind.value)
        position_ids = compute_positions_from_segment_ids(segment_ids) + cache.cur_ind.value
        sin, cos = _generate_pos_embeddings(position_ids, self.head_dim, self.rope_theta, factor=self.factor)
        q = apply_rope(q, sin, cos)
        k = apply_rope(k, sin, cos)

        # Update cache
        slice_indices = (0, cache.cur_ind.value, 0, 0)
        cache.v_cache.value = jax.lax.dynamic_update_slice(cache.v_cache.value, v, slice_indices)
        cache.k_cache.value = jax.lax.dynamic_update_slice(cache.k_cache.value, k, slice_indices)
        t = q.shape[1]
        cache.cur_ind.value += x.shape[1]

        # TODO: Need to do this with the kv cache next
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)
        qkv = jax.nn.dot_product_attention(q, k, v, is_causal=False, mask=mask[:, :, :, :t], scale=self.scale)
        # k, v = repeat_kv(cache.k_cache.value, self.n_rep), repeat_kv(cache.v_cache.value, self.n_rep)
        # qkv = jax.nn.dot_product_attention(q, k, v, is_causal=False, mask=mask[:, :, :, :t], scale=self.scale)

        cache.cur_ind.value = cache.cur_ind.value + t
        return self.o_proj(qkv.reshape(*x.shape[:-1], -1))


class Gemma3MLP(nnx.Module):
    def __init__(self, config: TextConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.gate_proj = nnx.Linear(config.hidden_size, config.intermediate_size, use_bias=False, rngs=rngs)
        self.up_proj = nnx.Linear(config.hidden_size, config.intermediate_size, use_bias=False, rngs=rngs)
        self.down_proj = nnx.Linear(config.intermediate_size, config.hidden_size, use_bias=False, rngs=rngs)

    def __call__(self, x: Array):
        return self.down_proj(jax.nn.gelu(self.gate_proj(x)) * self.up_proj(x))


class Gemma3DecoderLayer(nnx.Module):
    def __init__(self, config: TextConfig, layer_idx: int, *, rngs: nnx.Rngs):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = Gemma3Attention(config=config, layer_idx=layer_idx, rngs=rngs)
        self.mlp = Gemma3MLP(config, rngs=rngs)
        self.input_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps, rngs=rngs)
        self.post_attention_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps, rngs=rngs)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps, rngs=rngs)
        self.post_feedforward_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps, rngs=rngs)

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
        # TODO: Move this out of this class into the larger class
        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            "self.padding_idx",
            embed_scale=self.config.hidden_size**0.5,
            rngs=rngs,
        )
        self.layers = nnx.List(
            [Gemma3DecoderLayer(config, layer_idx, rngs=rngs) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps, rngs=rngs)

    def __call__(self, x, cache: Cache, segment_ids: Array, sliding_mask: Array | None, causal_mask: Array | None):
        # x = self.embed_tokens(x) # done in higher layer now
        for lt, c, layer in zip(self.config.layer_types, cache, self.layers):
            mask = sliding_mask if lt == AttentionType.SLIDE else causal_mask
            x = layer(x, c, segment_ids, mask)
        return self.norm(x)


class Gemma3MultiModalProjector(nnx.Module):
    def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs):
        self.config = config
        vhs = config.vision_config.hidden_size
        ths = config.text_config.hidden_size
        eps = config.vision_config.layer_norm_eps
        self.mm_input_projection_weight = nnx.Param(jnp.zeros((vhs, ths)), rngs=rngs)
        self.mm_soft_emb_norm = Gemma3RMSNorm(vhs, eps=eps, rngs=rngs)
        self.patches_per_img = int(config.vision_config.image_size // config.vision_config.patch_size)
        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_img // self.tokens_per_side

    def __call__(self, vision_outputs: Array) -> Array:
        b, _, t = vision_outputs.shape
        vision_outputs = vision_outputs.swapaxes(1, 2).reshape(b, t, self.patches_per_img, self.patches_per_img)
        # TODO: update this to get rid of the None and 0.
        # Might have to write my own avg pool.
        x = flax.linen.avg_pool(
            vision_outputs[:, :, :, :, None],
            window_shape=(1, 1, self.kernel_size, self.kernel_size),
            strides=(1, 1, self.kernel_size, self.kernel_size),
        )[:, :, :, :, 0]
        x = x.reshape(b, t, -1)
        x = x.swapaxes(1, 2)
        x = self.mm_soft_emb_norm(x)
        x = jnp.matmul(x, self.mm_input_projection_weight.value)
        return x.astype(vision_outputs.dtype)


# def make_causal_mask(cache_layer: LayerCache, token_type_ids):
#     pass


def make_causal_mask(b: int, t: int, token_type_ids: Array):
    my_mask = nnx.make_causal_mask(jnp.ones((b, t)))
    tti = token_type_ids.astype(jnp.bool_)
    or_mask = tti[:, None, None, :] & tti[:, None, :, None]
    my_mask = my_mask.astype(jnp.bool_) | or_mask
    return my_mask


def make_window_mask(b: int, t: int, token_type_ids: Array, slide_size: int):
    my_mask = make_causal_mask(b, t, token_type_ids)
    tmp = jnp.arange(my_mask.shape[-1])
    slide = tmp[:, None] - tmp[None, :] < slide_size
    return my_mask & slide


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
        self.vision_tower = SiglipVisionTransformer(cfg.vision_config, rngs=rngs)
        self.multi_modal_projector = Gemma3MultiModalProjector(cfg, rngs=rngs)
        self.language_model = Gemma3TextModel(cfg.text_config, rngs=rngs)

    def __call__(
        self, input_ids: Array, pixel_values: Array, cache: Cache, segment_ids: Array, token_type_ids: Array
    ) -> Array:
        causal_mask = make_causal_mask(input_ids.shape[0], input_ids.shape[1], token_type_ids)
        sliding_mask = make_causal_mask(input_ids.shape[0], input_ids.shape[1], token_type_ids)

        inputs_embeds = self.language_model.embed_tokens(input_ids)

        # Merge text and images
        if pixel_values is not None:
            vision_outputs = self.vision_tower(pixel_values)
            image_features = self.multi_modal_projector(vision_outputs)

            image_features = image_features.astype(inputs_embeds.dtype)
            inputs_embeds = batched_merge_modalities(image_features, inputs_embeds, token_type_ids)

        out = self.language_model(inputs_embeds, cache, segment_ids, sliding_mask, causal_mask)
        return out


# TODO: Implement a jitted forward method
