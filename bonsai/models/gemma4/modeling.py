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

"""
Gemma 4 model implementation in JAX/Flax NNX.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec
from typing import TypeAlias
import math
from jaxtyping import Array
from jax import P

from bonsai.utils.rope import RoPE, apply_rope


import inspect

_linear_sig = inspect.signature(nnx.Linear.__init__)
_LINEAR_SUPPORTS_METADATA = "kernel_metadata" in _linear_sig.parameters or any(
    p.kind == inspect.Parameter.VAR_KEYWORD for p in _linear_sig.parameters.values()
)

_embed_sig = inspect.signature(nnx.Embed.__init__)
_EMBED_SUPPORTS_METADATA = "embedding_metadata" in _embed_sig.parameters or any(
    p.kind == inspect.Parameter.VAR_KEYWORD for p in _embed_sig.parameters.values()
)


def _make_linear(*args, kernel_metadata=None, bias_metadata=None, **kwargs):
    """Instantiates nnx.Linear and conditionally injects sharding metadata if supported."""
    if _LINEAR_SUPPORTS_METADATA:
        if kernel_metadata is not None:
            kwargs["kernel_metadata"] = kernel_metadata
        if bias_metadata is not None:
            kwargs["bias_metadata"] = bias_metadata
    return nnx.Linear(*args, **kwargs)


def _make_embed(*args, embedding_metadata=None, **kwargs):
    """Instantiates nnx.Embed and conditionally injects sharding metadata if supported."""
    if _EMBED_SUPPORTS_METADATA:
        if embedding_metadata is not None:
            kwargs["embedding_metadata"] = embedding_metadata
    return nnx.Embed(*args, **kwargs)


class Gemma4RMSNorm(nnx.Module):
    """RMSNorm layer for Gemma 4.

    Gemma 4 models typically use an offset scale (`1.0 + scale`) for normal layers,
    but MoE gate norms and v_norm require `with_scale=False` (no learned scale).

    Attributes:
        dim: The input dimension.
        eps: Epsilon to prevent division by zero.
        with_scale: Whether to include a learned scale parameter.
        dtype: The data type for computation.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        with_scale: bool = True,
        *,
        dtype: jnp.dtype = jnp.float32,
        shd: PartitionSpec | None = None,
        rngs: nnx.Rngs,
    ):
        self.eps = eps
        self.with_scale = with_scale
        self.dtype = dtype

        if self.with_scale:
            self.scale = nnx.Param(jax.nn.initializers.zeros(rngs.params(), dim, dtype=dtype), out_sharding=shd)
        else:
            self.scale = None

    @jax.named_scope("gemma4_rms_norm")
    def __call__(self, x: Array) -> Array:
        """Applies RMS normalization."""
        xf32 = x.astype(jnp.float32)
        normed = xf32 * jax.lax.rsqrt(jnp.square(xf32).mean(-1, keepdims=True) + self.eps)

        if self.with_scale:
            scale_val = jnp.asarray(self.scale[...], dtype=jnp.float32)
            out = normed * (1.0 + scale_val)
        else:
            out = normed

        return out.astype(self.dtype)


@dataclass(slots=True, frozen=True)
class VisionShardConfig:
    """Sharding configuration for Vision Transformer."""

    attn_kernel: PartitionSpec | None = None
    attn_bias: PartitionSpec | None = None
    attn_qk_activation: PartitionSpec | None = None
    fc1_kernel: PartitionSpec | None = None
    fc1_bias: PartitionSpec | None = None
    fc2_kernel: PartitionSpec | None = None
    fc2_bias: PartitionSpec | None = None
    activation: PartitionSpec | None = None
    layer_norm: PartitionSpec | None = None
    emb_patch_kernel: PartitionSpec | None = None
    emb_patch_bias: PartitionSpec | None = None
    emb_patch_activation: PartitionSpec | None = None
    emb_pos_kernel: PartitionSpec | None = None
    emb_pos_activation: PartitionSpec | None = None

    @staticmethod
    def no_sharding():
        """Returns an unpartitioned default VisionShardConfig."""
        return VisionShardConfig()


@dataclass(frozen=True)
class VisionConfig:
    """Configuration for the Vision Transformer in Gemma 4."""

    hidden_size: int = 1152
    image_size: int = 896
    intermediate_size: int = 4304
    layer_norm_eps: float = 1e-6
    num_attention_heads: int = 16
    num_channels: int = 3
    num_hidden_layers: int = 27
    patch_size: int = 14
    shd_cfg: VisionShardConfig = VisionShardConfig.no_sharding()


class SiglipVisionEmbeddings(nnx.Module):
    """Embeddings for the SigLIP vision model."""

    def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.num_patches = (config.image_size // config.patch_size) ** 2

        import functools

        ki = functools.partial(jax.nn.initializers.lecun_normal(), out_sharding=config.shd_cfg.emb_patch_kernel)
        bi = functools.partial(jax.nn.initializers.zeros, out_sharding=config.shd_cfg.emb_patch_bias)
        self.patch_embedding = nnx.Conv(
            config.num_channels,
            config.hidden_size,
            kernel_size=(config.patch_size, config.patch_size),
            strides=(config.patch_size, config.patch_size),
            padding="valid",
            kernel_init=ki,
            bias_init=bi,
            rngs=rngs,
        )

        self.position_embedding = _make_embed(
            self.num_patches,
            config.hidden_size,
            embedding_metadata={"out_sharding": config.shd_cfg.emb_pos_kernel},
            rngs=rngs,
        )
        self.position_ids = nnx.data(jnp.expand_dims(jnp.arange(self.num_patches), 0))

    def __call__(self, pixel_values: Array) -> Array:
        """Applies patch and position embeddings to pixel values."""
        patch_embeds = self.patch_embedding(pixel_values)
        b, h, w, c = patch_embeds.shape
        embeddings = patch_embeds.reshape((b, h * w, c))
        out = embeddings + self.position_embedding(self.position_ids)
        return out


class SiglipAttention(nnx.Module):
    """Attention block for SigLIP."""

    def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        hs, shd = config.hidden_size, config.shd_cfg
        km = {"out_sharding": shd.attn_kernel}
        bm = {"out_sharding": shd.attn_bias}
        self.q_proj = _make_linear(hs, hs, kernel_metadata=km, bias_metadata=bm, rngs=rngs)
        self.k_proj = _make_linear(hs, hs, kernel_metadata=km, bias_metadata=bm, rngs=rngs)
        self.v_proj = _make_linear(hs, hs, kernel_metadata=km, bias_metadata=bm, rngs=rngs)
        self.out_proj = _make_linear(hs, hs, kernel_metadata=km, bias_metadata=bm, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        """Applies multi-head attention."""
        b, t, _ = x.shape
        shd = self.config.shd_cfg.activation

        q = self.q_proj(x).reshape((b, t, self.num_heads, self.head_dim))
        k = self.k_proj(x).reshape((b, t, self.num_heads, self.head_dim))
        v = self.v_proj(x).reshape((b, t, self.num_heads, self.head_dim))

        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 3, 1))
        v = jnp.transpose(v, (0, 2, 1, 3))

        scores = jnp.matmul(q, k) / jnp.sqrt(self.head_dim)
        attn_weights = jax.nn.softmax(scores, axis=-1)
        out = jnp.matmul(attn_weights, v)
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape((b, t, -1))
        return self.out_proj(out)


class SiglipMLP(nnx.Module):
    """MLP for SigLIP.

    Uses the tanh-approximate GELU (`approximate=True`) to match the
    `gelu_pytorch_tanh` activation used in the HuggingFace reference.
    """

    def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs):
        self.config = config
        shd = config.shd_cfg
        self.fc1 = _make_linear(
            config.hidden_size,
            config.intermediate_size,
            kernel_metadata={"out_sharding": shd.fc1_kernel},
            bias_metadata={"out_sharding": shd.fc1_bias},
            rngs=rngs,
        )
        self.fc2 = _make_linear(
            config.intermediate_size,
            config.hidden_size,
            kernel_metadata={"out_sharding": shd.fc2_kernel},
            bias_metadata={"out_sharding": shd.fc2_bias},
            rngs=rngs,
        )

    def __call__(self, x: Array) -> Array:
        """Applies the MLP with tanh-approximate GELU activation."""
        x = self.fc1(x)
        x = jax.nn.gelu(x, approximate=True)
        return self.fc2(x)


class SiglipEncoderLayer(nnx.Module):
    """A single SigLIP encoder layer.

    Uses Gemma4RMSNorm (matching the HuggingFace reference) rather than LayerNorm.
    """

    def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs):
        self.config = config
        shd = config.shd_cfg.layer_norm
        self.layer_norm1 = Gemma4RMSNorm(config.hidden_size, eps=config.layer_norm_eps, shd=shd, rngs=rngs)
        self.layer_norm2 = Gemma4RMSNorm(config.hidden_size, eps=config.layer_norm_eps, shd=shd, rngs=rngs)
        self.self_attn = SiglipAttention(config, rngs=rngs)
        self.mlp = SiglipMLP(config, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        """Processes the encoder layer."""
        hidden = self.layer_norm1(x)
        hidden = self.self_attn(hidden)
        x = x + hidden
        hidden = self.layer_norm2(x)
        hidden = self.mlp(hidden)
        return x + hidden


class ConstVar(nnx.Variable):
    """Constant variable that should not be updated during training.

    This is used to store static tensors like inverse timescales for RoPE
    that need to be part of the model state but are not trainable parameters
    or mutable caches.
    """

    pass


class StatVar(nnx.Variable):
    """Statistical variable for tracking metrics like min/max values.

    This is used by layers like Gemma4ClippableLinear to track the bounds
    of activations for potential quantization or clipping purposes.
    """

    pass


class Gemma4ClippableLinear(nnx.Module):
    """Linear layer with optional input/output clipping."""

    def __init__(self, in_features: int, out_features: int, use_clipped_linears: bool = True, *, rngs: nnx.Rngs):
        self.use_clipped_linears = use_clipped_linears
        self.linear = nnx.Linear(in_features, out_features, use_bias=False, rngs=rngs)

        if self.use_clipped_linears:
            self.input_min = StatVar(jnp.array(-jnp.inf))
            self.input_max = StatVar(jnp.array(jnp.inf))
            self.output_min = StatVar(jnp.array(-jnp.inf))
            self.output_max = StatVar(jnp.array(jnp.inf))

    def __call__(self, x: jax.Array) -> jax.Array:
        """Applies a linear transformation, conditionally clipping the output."""
        if self.use_clipped_linears:
            x = jnp.clip(x, self.input_min[...], self.input_max[...])
        x = self.linear(x)
        if self.use_clipped_linears:
            x = jnp.clip(x, self.output_min[...], self.output_max[...])
        return x


class Gemma4AudioRelPositionalEncoding(nnx.Module):
    """Sinusoidal relative positional encoding for the audio encoder."""

    def __init__(self, config: AudioConfig):
        self.hidden_size = config.hidden_size
        self.context_size = (
            config.attention_chunk_size + config.attention_context_left - 1 + config.attention_context_right
        )
        min_timescale = 1.0
        max_timescale = 10000.0
        num_timescales = self.hidden_size // 2
        log_timescale_increment = math.log(max_timescale / min_timescale) / max(num_timescales - 1, 1)
        inv_timescales = min_timescale * jnp.exp(jnp.arange(num_timescales) * -log_timescale_increment)
        self.inv_timescales = ConstVar(inv_timescales[None, None, :])

    def __call__(self, x: jax.Array) -> jax.Array:
        """Applies relative positional encoding."""
        position_ids = jnp.arange(self.context_size // 2, -1, -1, dtype=x.dtype)
        position_ids = position_ids[..., None]
        scaled_time = position_ids * self.inv_timescales[...]
        pos_embed = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=-1)
        return pos_embed.astype(x.dtype)


class Gemma4AudioAttention(nnx.Module):
    """Chunked local attention with relative position bias for audio."""

    def __init__(self, config: AudioConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.q_scale = (self.head_dim**-0.5) / math.log(2)
        self.k_scale = math.log(1 + math.e) / math.log(2)
        self.chunk_size = config.attention_chunk_size
        self.max_past_horizon = config.attention_context_left - 1
        self.max_future_horizon = config.attention_context_right
        self.context_size = self.chunk_size + self.max_past_horizon + self.max_future_horizon
        self.softcap = config.attention_logit_cap
        self.invalid_logits_value = config.attention_invalid_logits_value

        hs = config.hidden_size
        self.q_proj = Gemma4ClippableLinear(hs, self.num_heads * self.head_dim, config.use_clipped_linears, rngs=rngs)
        self.k_proj = Gemma4ClippableLinear(hs, self.num_heads * self.head_dim, config.use_clipped_linears, rngs=rngs)
        self.v_proj = Gemma4ClippableLinear(hs, self.num_heads * self.head_dim, config.use_clipped_linears, rngs=rngs)
        self.post = Gemma4ClippableLinear(hs, hs, config.use_clipped_linears, rngs=rngs)

        self.relative_k_proj = nnx.Linear(hs, self.num_heads * self.head_dim, use_bias=False, rngs=rngs)
        self.per_dim_scale = nnx.Param(jnp.zeros((self.head_dim,)))

    def _convert_to_block(self, x: jax.Array) -> jax.Array:
        """Reshapes the input into chunks/blocks for block-wise attention."""
        batch_size, seq_len, num_heads, head_dim = x.shape
        num_blocks = (seq_len + self.chunk_size - 1) // self.chunk_size
        pad_len = num_blocks * self.chunk_size - seq_len
        x = jnp.pad(x, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        return x.reshape(batch_size, num_blocks, self.chunk_size, num_heads, head_dim)

    def _extract_block_context(self, x: jax.Array) -> jax.Array:
        """Extracts the left context block for block-wise attention."""
        batch_size, seq_len, num_heads, head_dim = x.shape
        x = jnp.pad(x, ((0, 0), (self.max_past_horizon, self.max_future_horizon + self.chunk_size - 1), (0, 0), (0, 0)))
        num_blocks = (seq_len + self.chunk_size - 1) // self.chunk_size
        blocks = []
        for i in range(num_blocks):
            start = i * self.chunk_size
            blocks.append(
                jax.lax.dynamic_slice(x, (0, start, 0, 0), (batch_size, self.context_size, num_heads, head_dim))
            )
        x = jnp.stack(blocks, axis=1)
        return x

    def _rel_shift(self, x: jax.Array) -> jax.Array:
        """Performs relative shift on attention scores."""
        batch_size, num_heads, num_blocks, block_size, position_length = x.shape
        x = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, 0), (0, self.context_size + 1 - position_length)))
        x = x.reshape((batch_size, num_heads, num_blocks, block_size * (self.context_size + 1)))
        x = x[..., : block_size * self.context_size]
        return x.reshape((batch_size, num_heads, num_blocks, block_size, self.context_size))

    def __call__(self, x: jax.Array, pos_emb: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        """Computes the multi-head attention for audio inputs."""
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).reshape((batch_size, seq_len, self.num_heads, self.head_dim))
        k = self.k_proj(x).reshape((batch_size, seq_len, self.num_heads, self.head_dim))
        v = self.v_proj(x).reshape((batch_size, seq_len, self.num_heads, self.head_dim))

        q = q * self.q_scale * jax.nn.softplus(self.per_dim_scale[...])
        k = k * self.k_scale

        q_block = self._convert_to_block(q)
        k_context = self._extract_block_context(k)
        v_context = self._extract_block_context(v)

        num_blocks = q_block.shape[1]
        rel_k = self.relative_k_proj(pos_emb).reshape((-1, self.num_heads, self.head_dim)).astype(q.dtype)

        queries = jnp.transpose(q_block, (0, 3, 1, 2, 4))
        keys = jnp.transpose(k_context, (0, 3, 1, 4, 2))
        matrix_ac = jnp.matmul(queries, keys)

        queries_flat = queries.reshape((batch_size, self.num_heads, -1, self.head_dim))
        rel_k_t = jnp.transpose(rel_k, (1, 2, 0))
        matrix_bd = jnp.matmul(queries_flat, rel_k_t)
        matrix_bd = matrix_bd.reshape((batch_size, self.num_heads, num_blocks, self.chunk_size, -1))
        matrix_bd = self._rel_shift(matrix_bd)

        attn_weights = matrix_ac + matrix_bd
        attn_weights = attn_weights / self.softcap
        attn_weights = jnp.tanh(attn_weights) * self.softcap

        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, self.invalid_logits_value)

        attn_weights = jax.nn.softmax(attn_weights, axis=-1).astype(v_context.dtype)
        values = jnp.transpose(v_context, (0, 3, 1, 2, 4))
        out = jnp.matmul(attn_weights, values)

        out = jnp.transpose(out, (0, 2, 3, 1, 4))
        out = out.reshape((batch_size, num_blocks * self.chunk_size, -1))
        out = out[:, :seq_len, :]

        return self.post(out)


class Gemma4AudioSubSampleConvProjectionLayer(nnx.Module):
    """A single convolutional projection layer for audio subsampling."""

    def __init__(self, in_channels: int, out_channels: int, norm_eps: float, *, rngs: nnx.Rngs):
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            rngs=rngs,
        )
        self.norm = nnx.LayerNorm(out_channels, epsilon=norm_eps, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array, mask: jax.Array | None = None) -> tuple[jax.Array, jax.Array | None]:
        """Applies the subsample convolution projection layer."""
        if mask is not None:
            x = x * mask[:, None, :, None]

        x = jnp.transpose(x, (0, 2, 3, 1))  # NCHW to NHWC for nnx.Conv
        x = self.conv(x)
        x = self.norm(x)
        x = jax.nn.relu(x)
        x = jnp.transpose(x, (0, 3, 1, 2))  # NHWC back to NCHW

        if mask is not None:
            mask = mask[:, ::2]

        return x, mask


class Gemma4AudioSubSampleConvProjection(nnx.Module):
    """Full convolutional projection module for audio subsampling."""

    def __init__(self, config: AudioConfig, *, rngs: nnx.Rngs):
        c0, c1 = config.subsampling_conv_channels
        self.layer0 = Gemma4AudioSubSampleConvProjectionLayer(1, c0, config.rms_norm_eps, rngs=rngs)
        self.layer1 = Gemma4AudioSubSampleConvProjectionLayer(c0, c1, config.rms_norm_eps, rngs=rngs)
        proj_input_dim = (c0 // 4) * c1
        self.input_proj_linear = nnx.Linear(proj_input_dim, config.hidden_size, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array, mask: jax.Array | None = None) -> tuple[jax.Array, jax.Array | None]:
        """Applies the full subsample convolution projection."""
        x = jnp.expand_dims(x, 1)  # Add channel dim
        x, mask = self.layer0(x, mask)
        x, mask = self.layer1(x, mask)

        batch_size, _, seq_len, _ = x.shape
        x = jnp.transpose(x, (0, 2, 3, 1)).reshape((batch_size, seq_len, -1))
        return self.input_proj_linear(x), mask


class Gemma4AudioFeedForward(nnx.Module):
    """Feed forward network used in the audio tower."""

    def __init__(self, config: AudioConfig, *, rngs: nnx.Rngs):
        self.ffw_layer_1 = Gemma4ClippableLinear(
            config.hidden_size, config.hidden_size * 4, config.use_clipped_linears, rngs=rngs
        )
        self.ffw_layer_2 = Gemma4ClippableLinear(
            config.hidden_size * 4, config.hidden_size, config.use_clipped_linears, rngs=rngs
        )
        self.pre_layer_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=jnp.float32, rngs=rngs)
        self.post_layer_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=jnp.float32, rngs=rngs)
        self.gradient_clipping = config.gradient_clipping
        self.post_layer_scale = config.residual_weight

    def __call__(self, x: jax.Array) -> jax.Array:
        """Applies the feed forward network."""
        residual = x
        x = jnp.clip(x, -self.gradient_clipping, self.gradient_clipping)
        x = self.pre_layer_norm(x)
        x = self.ffw_layer_1(x)
        x = jax.nn.silu(x)
        x = self.ffw_layer_2(x)
        x = jnp.clip(x, -self.gradient_clipping, self.gradient_clipping)
        x = self.post_layer_norm(x)
        x *= self.post_layer_scale
        return residual + x


class Gemma4AudioCausalConv1d(nnx.Module):
    """Causal 1D convolution layer for audio processing."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, groups: int, *, rngs: nnx.Rngs):
        self.kernel_size = kernel_size
        self.left_pad = kernel_size - 1
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(kernel_size,),
            feature_group_count=groups,
            use_bias=False,
            padding=0,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Applies causal 1D convolution."""
        x = jnp.pad(x, ((0, 0), (self.left_pad, 0), (0, 0)))  # Pad time dimension (batch, time, channels)
        return self.conv(x)


class Gemma4AudioLightConv1d(nnx.Module):
    """Lightweight 1D convolution module for audio."""

    def __init__(self, config: AudioConfig, *, rngs: nnx.Rngs):
        self.linear_start = Gemma4ClippableLinear(
            config.hidden_size, config.hidden_size * 2, config.use_clipped_linears, rngs=rngs
        )
        self.linear_end = Gemma4ClippableLinear(
            config.hidden_size, config.hidden_size, config.use_clipped_linears, rngs=rngs
        )
        self.depthwise_conv1d = Gemma4AudioCausalConv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=config.conv_kernel_size,
            groups=config.hidden_size,
            rngs=rngs,
        )
        self.pre_layer_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=jnp.float32, rngs=rngs)
        self.conv_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=jnp.float32, rngs=rngs)
        self.gradient_clipping = config.gradient_clipping

    def __call__(self, x: jax.Array) -> jax.Array:
        """Applies lightweight 1D convolution."""
        residual = x
        x = self.pre_layer_norm(x)
        x = self.linear_start(x)

        # GLU
        x, gate = jnp.split(x, 2, axis=-1)
        x = x * jax.nn.sigmoid(gate)

        x = self.depthwise_conv1d(x)

        x = jnp.clip(x, -self.gradient_clipping, self.gradient_clipping)
        x = self.conv_norm(x)
        x = jax.nn.silu(x)
        x = self.linear_end(x)
        return residual + x


class Gemma4AudioLayer(nnx.Module):
    """A single layer of the audio transformer model."""

    def __init__(self, config: AudioConfig, *, rngs: nnx.Rngs):
        self.feed_forward1 = Gemma4AudioFeedForward(config, rngs=rngs)
        self.feed_forward2 = Gemma4AudioFeedForward(config, rngs=rngs)
        self.self_attn = Gemma4AudioAttention(config, rngs=rngs)
        self.lconv1d = Gemma4AudioLightConv1d(config, rngs=rngs)
        self.norm_pre_attn = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=jnp.float32, rngs=rngs)
        self.norm_post_attn = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=jnp.float32, rngs=rngs)
        self.norm_out = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=jnp.float32, rngs=rngs)
        self.gradient_clipping = config.gradient_clipping

    def __call__(self, x: jax.Array, pos_emb: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        """Applies a single audio transformer layer."""
        x = self.feed_forward1(x)
        residual = x

        x = jnp.clip(x, -self.gradient_clipping, self.gradient_clipping)
        x = self.norm_pre_attn(x)

        x = self.self_attn(x, pos_emb, mask)

        x = jnp.clip(x, -self.gradient_clipping, self.gradient_clipping)
        x = self.norm_post_attn(x)
        x += residual

        x = self.lconv1d(x)
        x = self.feed_forward2(x)

        x = jnp.clip(x, -self.gradient_clipping, self.gradient_clipping)
        x = self.norm_out(x)
        return x


class Gemma4AudioModel(nnx.Module):
    """An audio encoder based on the Universal Speech Model architecture."""

    def __init__(self, config: AudioConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.subsample_conv_projection = Gemma4AudioSubSampleConvProjection(config, rngs=rngs)
        self.rel_pos_enc = Gemma4AudioRelPositionalEncoding(config)
        self.layers = nnx.List([Gemma4AudioLayer(config, rngs=rngs) for _ in range(config.num_hidden_layers)])
        self.output_proj = nnx.Linear(config.hidden_size, config.output_proj_dims, rngs=rngs)

    def _convert_4d_mask_to_blocked_5d(self, mask_4d: jax.Array) -> jax.Array:
        """Converts a 4D attention mask to a 5D blocked format."""
        batch_size, _, seq_len, _ = mask_4d.shape
        chunk_size = self.config.attention_chunk_size
        max_past_horizon = self.config.attention_context_left - 1
        max_future_horizon = self.config.attention_context_right

        num_blocks = (seq_len + chunk_size - 1) // chunk_size
        padded_seq_len = num_blocks * chunk_size
        pad_amount = padded_seq_len - seq_len

        mask_4d = jnp.pad(mask_4d, ((0, 0), (0, pad_amount), (0, 0), (0, pad_amount)))
        mask_5d = mask_4d.reshape(batch_size, 1, num_blocks, chunk_size, padded_seq_len)
        mask_5d = jnp.pad(mask_5d, ((0, 0), (0, 0), (0, 0), (0, 0), (max_past_horizon, max_future_horizon)))

        # Emulate gather
        block_starts = jnp.arange(num_blocks) * chunk_size
        offsets = jnp.arange(chunk_size + max_past_horizon + max_future_horizon)
        kv_indices = block_starts[:, None] + offsets[None, :]
        kv_indices = jnp.broadcast_to(
            kv_indices[None, None, :, None, :],
            (batch_size, 1, num_blocks, chunk_size, chunk_size + max_past_horizon + max_future_horizon),
        )

        return jnp.take_along_axis(mask_5d, kv_indices, axis=-1)

    def __call__(self, input_features: jax.Array, attention_mask: jax.Array | None = None) -> jax.Array:
        """Forward pass for the Gemma 4 Audio model."""
        x, mask = self.subsample_conv_projection(input_features, attention_mask)
        pos_emb = self.rel_pos_enc(x)

        if mask is not None:
            # Gemma4 audio attention mask requires 5D conversion
            mask_4d = mask[:, None, :, None] * mask[:, None, None, :]
            mask_5d = self._convert_4d_mask_to_blocked_5d(mask_4d)
        else:
            mask_5d = None

        for layer in self.layers:
            x = layer(x, pos_emb, mask_5d)

        return self.output_proj(x)


class Gemma4MultimodalEmbedder(nnx.Module):
    """Embeds multimodal soft tokens (e.g., from audio) into language model space."""

    def __init__(self, multimodal_hidden_size: int, text_hidden_size: int, eps: float, *, rngs: nnx.Rngs):
        self.embedding_projection = nnx.Linear(multimodal_hidden_size, text_hidden_size, use_bias=False, rngs=rngs)
        self.embedding_pre_projection_norm = Gemma4RMSNorm(multimodal_hidden_size, eps=eps, with_scale=False, rngs=rngs)

    def __call__(self, inputs_embeds: jax.Array) -> jax.Array:
        """Embeds multimodal inputs."""
        embs_normed = self.embedding_pre_projection_norm(inputs_embeds)
        return self.embedding_projection(embs_normed)


class SiglipVisionTransformer(nnx.Module):
    """The SigLIP Vision Transformer.

    Uses Gemma4RMSNorm throughout (matching the HuggingFace reference) rather
    than LayerNorm.
    """

    def __init__(self, config: VisionConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.embeddings = SiglipVisionEmbeddings(config, rngs=rngs)
        self.layers = nnx.List([SiglipEncoderLayer(config, rngs=rngs) for _ in range(config.num_hidden_layers)])
        shd = config.shd_cfg.layer_norm
        self.post_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.layer_norm_eps, shd=shd, rngs=rngs)

    def __call__(self, pixel_values: Array) -> Array:
        """Applies the vision transformer to pixel values."""
        x = self.embeddings(pixel_values)
        for layer in self.layers:
            x = layer(x)
        return self.post_layernorm(x)


class Gemma4MultiModalProjector(nnx.Module):
    """Projects vision features into the language model's hidden dimension.

    Pools patch tokens using position-based weighted averaging (matching the
    HuggingFace reference), then projects into the text model's hidden space.

    Attributes:
        mm_input_projection_weight: Weight matrix (vision_hidden, text_hidden).
        mm_soft_emb_norm: RMSNorm applied to pooled patch embeddings.
        patches_per_img: Number of patches along one spatial dimension.
        tokens_per_side: Number of output tokens along one spatial dimension.
        kernel_size: Pooling kernel size (patches_per_img // tokens_per_side).
        num_output_tokens: Total output tokens per image (tokens_per_side ** 2).
    """

    def __init__(
        self, text_config: ModelConfig, vision_config: VisionConfig, mm_tokens_per_image: int, *, rngs: nnx.Rngs
    ):
        self.text_config = text_config
        self.vision_config = vision_config
        vhs, ths = vision_config.hidden_size, text_config.hidden_size

        self.patches_per_img = int(vision_config.image_size // vision_config.patch_size)
        self.tokens_per_side = int(mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_img // self.tokens_per_side
        self.num_output_tokens = self.tokens_per_side * self.tokens_per_side

        self.mm_input_projection_weight = nnx.Param(jnp.zeros((vhs, ths)), rngs=rngs)
        self.mm_soft_emb_norm = Gemma4RMSNorm(vhs, eps=vision_config.layer_norm_eps, dtype=text_config.dtype, rngs=rngs)

    def _avg_pool_by_positions(self, x: Array) -> Array:
        """Pools patch tokens into a fixed grid using position-based averaging.

        Each patch is assigned to a kernel bin via floor(position / kernel_size).
        Averaging is done with one-hot weights divided by kernel_size^2, matching
        the HuggingFace reference implementation exactly.

        Args:
            x: Patch embeddings (B, num_patches, hidden_size).

        Returns:
            Pooled embeddings (B, num_output_tokens, hidden_size).
        """
        b, num_patches, hidden = x.shape
        k = self.kernel_size
        length = self.num_output_tokens
        k_sq = k * k

        positions = jnp.arange(num_patches)
        # 2D patch index → 1D output token index
        row = positions // self.patches_per_img
        col = positions % self.patches_per_img
        kernel_idxs = (row // k) * self.tokens_per_side + (col // k)

        # One-hot weights: (num_patches, num_output_tokens) / k^2
        weights = jax.nn.one_hot(kernel_idxs, length, dtype=jnp.float32) / k_sq  # (P, L)
        # (B, L, P) @ (B, P, D) → (B, L, D)
        pooled = jnp.matmul(weights.T[None], x.astype(jnp.float32))
        return pooled

    def __call__(self, vision_outputs: Array) -> Array:
        """Projects and pools the vision outputs.

        Args:
            vision_outputs: Patch embeddings from the vision encoder (B, num_patches, hidden_size).

        Returns:
            Projected image tokens (B, num_output_tokens, text_hidden_size).
        """
        pooled = self._avg_pool_by_positions(vision_outputs)
        # Scale by sqrt(hidden_size) to match the reference post-pooling normalization.
        pooled = pooled * math.sqrt(self.vision_config.hidden_size)
        pooled = pooled.astype(self.text_config.dtype)
        pooled = self.mm_soft_emb_norm(pooled)
        return jnp.matmul(pooled, self.mm_input_projection_weight[...])


def batched_merge_modalities(img_emb: Array, text_emb: Array, token_mask: Array) -> Array:
    """Merges image and text embeddings based on a token mask.

    Args:
        img_emb: Image embeddings (B, Li, D)
        text_emb: Text embeddings (B, Lt, D)
        token_mask: Boolean mask indicating image token positions (B, Lt)

    Returns:
        Merged embeddings (B, Lt, D)
    """

    def merge_modalities(i_emb, t_emb, mask):
        """Merges image and text embeddings using the provided token mask."""
        img_indices = jnp.cumsum(mask) - 1
        safe_indices = jnp.clip(img_indices, 0, i_emb.shape[0] - 1)
        aligned_images = i_emb[safe_indices]
        return jnp.where(mask[:, None], aligned_images, t_emb)

    return jax.vmap(merge_modalities)(img_emb, text_emb, token_mask)


class AttentionType(Enum):
    """Types of attention layers in Gemma 4."""

    LOCAL_SLIDING = "local_sliding"
    GLOBAL = "global"


class ShardMode(Enum):
    """Sharding mode choices."""

    FSDP = "fsdp"
    TP = "tp"


@dataclass(slots=True, frozen=True)
class ShardConfig:
    """Sharding configuration mappings."""

    attn_kernel: PartitionSpec | None = None
    attn_bias: PartitionSpec | None = None
    attn_qk_activation: PartitionSpec | None = None
    fc1_kernel: PartitionSpec | None = None
    fc1_bias: PartitionSpec | None = None
    fc2_kernel: PartitionSpec | None = None
    fc2_bias: PartitionSpec | None = None
    moe_fc1_kernel: PartitionSpec | None = None
    moe_fc2_kernel: PartitionSpec | None = None
    activation: PartitionSpec | None = None
    norm: PartitionSpec | None = None
    emb_kernel: PartitionSpec | None = None
    cache: PartitionSpec | None = None

    @staticmethod
    def no_sharding():
        """Returns empty sharding config."""
        return ShardConfig()

    @staticmethod
    def default(use_fsdp: bool, use_tp: bool):
        """Returns standard sharding patterns."""
        fsdp = ShardMode.FSDP.value if use_fsdp else None
        tp = ShardMode.TP.value if use_tp else None
        return ShardConfig(
            attn_kernel=P(tp, fsdp),
            attn_bias=P(tp),
            attn_qk_activation=P(fsdp, tp),
            fc1_kernel=P(fsdp, tp),
            fc1_bias=P(tp),
            fc2_kernel=P(tp, fsdp),
            fc2_bias=P(tp),
            moe_fc1_kernel=P(fsdp, None, tp),
            moe_fc2_kernel=P(fsdp, tp, None),
            activation=P(fsdp, None, tp),
            norm=P(tp),
            emb_kernel=P(None, tp),
            cache=P(fsdp, None, tp, None),
        )


class LayerCache(nnx.Module):
    """KV Cache for a single decoder layer.

    Attributes:
        k_cache: The key cache tensor.
        v_cache: The value cache tensor.
        cur_ind: The current sequence index being written to.
        size: The maximum sequence length the cache can hold.
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: jnp.dtype,
        shd: PartitionSpec | None = None,
    ):
        cache_shape = (batch_size, max_seq_len, num_kv_heads, head_dim)
        self.k_cache = nnx.Cache(jnp.zeros(cache_shape, dtype=dtype, out_sharding=shd))
        self.v_cache = nnx.Cache(jnp.zeros(cache_shape, dtype=dtype, out_sharding=shd))
        self.cur_ind = nnx.Cache(jnp.zeros((), dtype=jnp.int32))
        self.size = max_seq_len


Cache: TypeAlias = list[LayerCache]


def init_cache(config: ModelConfig, batch_size: int, max_seq_len: int) -> Cache:
    """Initializes the KV cache for all layers.

    Args:
        config: The model configuration.
        batch_size: The batch size for generation.
        max_seq_len: The maximum sequence length to cache.

    Returns:
        A list of LayerCache objects, one for each hidden layer.
    """
    cache_size = 2 ** math.ceil(math.log2(max(max_seq_len, 1)))
    caches = []
    for i in range(config.num_hidden_layers):
        attn_type = GEMMA4_ATTENTION_PATTERN[i % len(GEMMA4_ATTENTION_PATTERN)]
        if attn_type == AttentionType.GLOBAL:
            num_kv = (
                config.num_global_key_value_heads
                if config.num_global_key_value_heads is not None
                else config.num_key_value_heads
            )
            hd = config.global_head_dim if config.global_head_dim is not None else config.head_dim
        else:
            num_kv = config.num_key_value_heads
            hd = config.head_dim
        caches.append(LayerCache(batch_size, cache_size, num_kv, hd, config.dtype, config.shd_cfg.cache))
    return caches


# Default hybrid attention pattern for Gemma 4
GEMMA4_ATTENTION_PATTERN = (
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.GLOBAL,
)


@dataclass(frozen=True)
class AudioConfig:
    """Configuration for the Audio Encoder in Gemma 4."""

    hidden_size: int = 1024
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    hidden_act: str = "silu"

    subsampling_conv_channels: tuple[int, int] = (128, 32)
    conv_kernel_size: int = 5
    residual_weight: float = 0.5
    attention_chunk_size: int = 12
    attention_context_left: int = 13
    attention_context_right: int = 0
    attention_logit_cap: float = 50.0
    attention_invalid_logits_value: float = 1e-9
    use_clipped_linears: bool = True
    gradient_clipping: float = 1e10
    output_proj_dims: int = 1536
    rms_norm_eps: float = 1e-6


@dataclass
class ModelConfig:
    """Configuration for Gemma 4.

    Attributes:
        vocab_size: Vocabulary size.
        hidden_size: Dimension of the hidden representations.
        intermediate_size: Dimension of the MLP / expert representations.
        num_hidden_layers: Number of decoder layers.
        num_attention_heads: Number of query heads.
        num_key_value_heads: Number of key/value heads for GQA.
        head_dim: Dimension of each attention head.
        rms_norm_eps: Epsilon value for RMSNorm.
        sliding_window_size: Window size for local sliding attention.
        num_experts: Total number of routed experts.
        num_experts_per_tok: Number of experts activated per token.
        num_shared_experts: Multiplier for shared experts capacity.
        dtype: Data type for activations.
        weight_dtype: Data type for weights.
        rope_max_timescale: Default max timescale for RoPE (local layers).
        global_rope_max_timescale: Max timescale for global attention RoPE (default 1,000,000).
        local_rope_max_timescale: Override max timescale for local attention RoPE.
        local_rope_proportion: Fraction of head_dim rotated in local attention (1.0 = full RoPE).
        global_rope_proportion: Fraction of head_dim rotated in global attention (0.25 = partial RoPE).
        float32_gate_logits: Whether to compute gate logits in float32.
        final_logit_softcapping: Final logit soft-capping value. None disables softcapping (default).
    """

    vocab_size: int = 256000
    vocab_size_per_layer_input: int | None = None
    hidden_size: int = 2048
    hidden_size_per_layer_input: int | None = None
    intermediate_size: int = 8192
    moe_intermediate_size: int | None = None
    num_hidden_layers: int = 24
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    num_global_key_value_heads: int | None = None
    head_dim: int = 256
    global_head_dim: int | None = None
    rms_norm_eps: float = 1e-6
    sliding_window_size: int = 512

    share_kv_projections: bool = False

    num_experts: int = 4
    num_experts_per_tok: int = 2
    num_shared_experts: int = 1

    dtype: jnp.dtype = jnp.float32
    weight_dtype: jnp.dtype = jnp.float32

    rope_max_timescale: int = 10000
    global_rope_max_timescale: int = 1_000_000
    local_rope_max_timescale: int | None = None
    local_rope_proportion: float = 1.0
    global_rope_proportion: float = 0.25

    float32_gate_logits: bool = True
    final_logit_softcapping: float | None = None
    attn_logits_soft_cap: float | None = 50.0
    shd_cfg: ShardConfig = ShardConfig.no_sharding()
    vision_config: VisionConfig | None = None
    audio_config: AudioConfig | None = None
    mm_tokens_per_image: int = 256
    audio_token_id: int | None = None

    @classmethod
    def gemma4_base(cls, use_fsdp: bool = False, use_tp: bool = False):
        """Preset configuration for a base Gemma 4 model."""
        kwargs = {}
        if use_fsdp or use_tp:
            kwargs["shd_cfg"] = ShardConfig.default(use_fsdp, use_tp)
        return cls(**kwargs)

    @classmethod
    def gemma4_e2b(cls, use_fsdp: bool = False, use_tp: bool = False):
        """Preset configuration for Gemma 4 E2B."""
        kwargs = {}
        if use_fsdp or use_tp:
            kwargs["shd_cfg"] = ShardConfig.default(use_fsdp, use_tp)
        return cls(
            num_hidden_layers=35,
            hidden_size=1024,
            intermediate_size=4096,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=256,
            global_head_dim=512,
            num_experts=1,
            vocab_size=262144,
            **kwargs,
        )

    @classmethod
    def gemma4_e4b(cls, use_fsdp: bool = False, use_tp: bool = False):
        """Preset configuration for Gemma 4 E4B."""
        kwargs = {}
        if use_fsdp or use_tp:
            kwargs["shd_cfg"] = ShardConfig.default(use_fsdp, use_tp)
        return cls(
            num_hidden_layers=42,
            hidden_size=2560,
            intermediate_size=10240,
            num_attention_heads=10,
            num_key_value_heads=1,
            head_dim=256,
            global_head_dim=512,
            num_experts=1,
            vocab_size=262144,
            **kwargs,
        )

    @classmethod
    def gemma4_26b_a4b(cls, use_fsdp: bool = False, use_tp: bool = False):
        """Preset configuration for Gemma 4 26B A4B (MoE)."""
        kwargs = {}
        if use_fsdp or use_tp:
            kwargs["shd_cfg"] = ShardConfig.default(use_fsdp, use_tp)
        return cls(
            num_hidden_layers=30,
            hidden_size=2816,
            intermediate_size=2112,
            moe_intermediate_size=704,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=256,
            global_head_dim=512,
            num_experts=128,
            num_experts_per_tok=2,
            vocab_size=262144,
            **kwargs,
        )

    @classmethod
    def gemma4_31b(cls, use_fsdp: bool = False, use_tp: bool = False):
        """Preset configuration for Gemma 4 31B."""
        kwargs = {}
        if use_fsdp or use_tp:
            kwargs["shd_cfg"] = ShardConfig.default(use_fsdp, use_tp)
        return cls(
            num_hidden_layers=60,
            hidden_size=5376,
            intermediate_size=21504,
            num_attention_heads=32,
            num_key_value_heads=16,
            head_dim=256,
            global_head_dim=512,
            num_experts=1,
            vocab_size=262144,
            **kwargs,
        )


class Gemma4MLP(nnx.Module):
    """Standard SwiGLU MLP used for both shared and routed experts."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        dtype: jnp.dtype,
        shd: ShardConfig = ShardConfig.no_sharding(),
        rngs: nnx.Rngs,
    ):
        self.gate_proj = _make_linear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            kernel_metadata={"out_sharding": shd.fc1_kernel},
            bias_metadata={"out_sharding": shd.fc1_bias},
            rngs=rngs,
        )
        self.up_proj = _make_linear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            kernel_metadata={"out_sharding": shd.fc1_kernel},
            bias_metadata={"out_sharding": shd.fc1_bias},
            rngs=rngs,
        )
        self.down_proj = _make_linear(
            intermediate_size,
            hidden_size,
            use_bias=False,
            kernel_metadata={"out_sharding": shd.fc2_kernel},
            bias_metadata={"out_sharding": shd.fc2_bias},
            rngs=rngs,
        )
        self.dtype = dtype

    @jax.named_scope("gemma4_mlp")
    def __call__(self, x: Array) -> Array:
        """Applies SwiGLU MLP transformation."""
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        activated = jax.nn.silu(gate) * up
        out = self.down_proj(activated)
        return out.astype(self.dtype)


class Gemma4RoutedExperts(nnx.Module):
    """Monolithic MoE expert module vectorizing all routed experts."""

    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.config = config
        E = config.num_experts
        H = config.hidden_size
        I = config.moe_intermediate_size if config.moe_intermediate_size is not None else config.intermediate_size
        self.dtype = config.dtype
        shd = config.shd_cfg

        import functools

        ki1 = functools.partial(
            jax.nn.initializers.normal(stddev=config.hidden_size**-0.5), out_sharding=shd.moe_fc1_kernel
        )
        ki2 = functools.partial(
            jax.nn.initializers.normal(stddev=config.hidden_size**-0.5), out_sharding=shd.moe_fc2_kernel
        )
        self.gate_proj_kernel = nnx.Param(ki1(rngs.params(), (E, H, I)))
        self.up_proj_kernel = nnx.Param(ki1(rngs.params(), (E, H, I)))
        self.down_proj_kernel = nnx.Param(ki2(rngs.params(), (E, I, H)))

    def __call__(self, x: Array, topk_indices: Array, topk_weights: Array) -> Array:
        """Applies the selected experts efficiently.

        Args:
            x: Input sequence (B, T, H)
            topk_indices: Indices of selected experts (B, T, K)
            topk_weights: Weights for selected experts (B, T, K)

        Returns:
            Output from the routed experts (B, T, H)
        """
        B, T, H = x.shape
        K = topk_indices.shape[-1]

        # Flatten batch and sequence to simplify routing
        x_flat = x.reshape(B * T, H)
        idx_flat = topk_indices.reshape(B * T, K)
        w_flat = topk_weights.reshape(B * T, K)

        # (B*T, K, 1, H)
        x_expanded = jnp.expand_dims(jnp.expand_dims(x_flat, 1), 1)

        # Fetch weights for the selected experts
        gate_w = jnp.take(self.gate_proj_kernel[...], idx_flat, axis=0)  # (B*T, K, H, I)
        up_w = jnp.take(self.up_proj_kernel[...], idx_flat, axis=0)  # (B*T, K, H, I)
        down_w = jnp.take(self.down_proj_kernel[...], idx_flat, axis=0)  # (B*T, K, I, H)

        # Compute activations
        gate_out = jnp.matmul(x_expanded, gate_w)  # (B*T, K, 1, I)
        up_out = jnp.matmul(x_expanded, up_w)  # (B*T, K, 1, I)
        act = jax.nn.silu(gate_out) * up_out

        # Compute output
        out = jnp.matmul(act, down_w).squeeze(2)  # (B*T, K, H)

        # Apply routing weights
        out = out * jnp.expand_dims(w_flat, 2)

        # Sum across experts
        out = jnp.sum(out, axis=1)  # (B*T, H)

        return out.reshape((B, T, H)).astype(self.dtype)


class Gemma4MoE(nnx.Module):
    """Gemma 4 Mixture of Experts combining routed and shared experts.

    Implements a Top-K routing mechanism for multiple parallel MLPs alongside
    a shared MLP that is always executed.

    Attributes:
        shared_experts: Dense MLP applied to every token.
        pre_forward_scale_2: Learned per-dimension scale applied before gating.
        gate_norm: Scale-free RMSNorm applied to routing inputs.
        gate: Linear projection from hidden_size to num_experts.
        per_expert_scale: Learned per-expert scalar applied after top-k renormalization.
        routed_experts: Monolithic weight tensor for all routed experts.
        pre_feedforward_layernorm_2: Pre-norm for routed expert inputs.
        post_feedforward_layernorm_1: Post-norm for shared expert output.
        post_feedforward_layernorm_2: Post-norm for routed expert output.
    """

    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.dtype = config.dtype
        shd = config.shd_cfg

        # Shared expert (just a wider MLP)
        shared_dim = config.intermediate_size * config.num_shared_experts
        self.shared_experts = Gemma4MLP(config.hidden_size, shared_dim, dtype=config.dtype, shd=shd, rngs=rngs)

        # Routing and gating
        self.pre_forward_scale_2 = nnx.Param(
            jnp.ones((config.hidden_size,), dtype=config.weight_dtype), out_sharding=shd.norm
        )
        self.gate_norm = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, with_scale=False, dtype=config.dtype, shd=shd.norm, rngs=rngs
        )
        gate_dtype = jnp.float32 if config.float32_gate_logits else config.dtype
        self.gate = _make_linear(
            config.hidden_size,
            config.num_experts,
            use_bias=False,
            dtype=gate_dtype,
            kernel_metadata={"out_sharding": shd.fc1_kernel},
            bias_metadata={"out_sharding": shd.fc1_bias},
            rngs=rngs,
        )
        # Per-expert learned scale applied after top-k renormalization.
        self.per_expert_scale = nnx.Param(jnp.ones((config.num_experts,), dtype=config.weight_dtype))

        # Routed experts utilizing a monolithic weight tensor
        self.routed_experts = Gemma4RoutedExperts(config, rngs=rngs)

        # Normalizations
        self.pre_feedforward_layernorm_2 = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=config.dtype, shd=shd.norm, rngs=rngs
        )
        self.post_feedforward_layernorm_1 = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=config.dtype, shd=shd.norm, rngs=rngs
        )
        self.post_feedforward_layernorm_2 = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=config.dtype, shd=shd.norm, rngs=rngs
        )

    @jax.named_scope("gemma4_moe")
    def __call__(self, x: Array, original_x: Array) -> Array:
        """Applies Mixture of Experts with shared and routed execution paths."""
        # 1. Shared Expert Path
        shared_out = self.shared_experts(x)
        shared_out = self.post_feedforward_layernorm_1(shared_out)

        # 2. Routed Experts Path
        routed_inputs = self.pre_feedforward_layernorm_2(original_x)

        # Gating logic
        unscaled_norm = self.gate_norm(original_x)
        root_size = self.config.hidden_size**-0.5
        router_scale = jnp.asarray(self.pre_forward_scale_2[...], dtype=unscaled_norm.dtype)
        gate_inputs = unscaled_norm * root_size * router_scale

        # Compute routing weights
        router_logits = self.gate(gate_inputs)
        routing_weights = jax.nn.softmax(router_logits, axis=-1)

        # Top-K selection and renormalization
        topk_weights, topk_indices = jax.lax.top_k(routing_weights, k=self.config.num_experts_per_tok)
        topk_weights = topk_weights / jnp.sum(topk_weights, axis=-1, keepdims=True)

        # Apply per-expert learned scale
        per_expert = jnp.asarray(self.per_expert_scale[...], dtype=topk_weights.dtype)
        topk_weights = topk_weights * per_expert[topk_indices]
        topk_weights = topk_weights.astype(self.dtype)

        # Compute routed expert outputs using vectorized computation
        routed_out = self.routed_experts(routed_inputs, topk_indices, topk_weights)
        routed_out = self.post_feedforward_layernorm_2(routed_out)

        return shared_out + routed_out


class Gemma4Attention(nnx.Module):
    """Multi-Head / Grouped-Query Attention for Gemma 4.

    Incorporates Q/K/V normalization and RoPE.
    """

    def __init__(
        self,
        config: ModelConfig,
        attention_type: AttentionType,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.attention_type = attention_type
        self.num_heads = config.num_attention_heads

        if attention_type == AttentionType.GLOBAL:
            self.num_kv_heads = (
                config.num_global_key_value_heads
                if config.num_global_key_value_heads is not None
                else config.num_key_value_heads
            )
            self.head_dim = config.global_head_dim if config.global_head_dim is not None else config.head_dim
            self.share_kv = config.share_kv_projections
        else:
            self.num_kv_heads = config.num_key_value_heads
            self.head_dim = config.head_dim
            self.share_kv = False

        self.hidden_size = config.hidden_size
        self.dtype = config.dtype
        shd = config.shd_cfg

        self.q_proj = _make_linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            use_bias=False,
            kernel_metadata={"out_sharding": shd.attn_kernel},
            bias_metadata={"out_sharding": shd.attn_bias},
            rngs=rngs,
        )
        self.k_proj = _make_linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            use_bias=False,
            kernel_metadata={"out_sharding": shd.attn_kernel},
            bias_metadata={"out_sharding": shd.attn_bias},
            rngs=rngs,
        )
        if not self.share_kv:
            self.v_proj = _make_linear(
                self.hidden_size,
                self.num_kv_heads * self.head_dim,
                use_bias=False,
                kernel_metadata={"out_sharding": shd.attn_kernel},
                bias_metadata={"out_sharding": shd.attn_bias},
                rngs=rngs,
            )
        else:
            self.v_proj = None
        self.o_proj = _make_linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            use_bias=False,
            kernel_metadata={"out_sharding": shd.attn_kernel},
            bias_metadata={"out_sharding": shd.attn_bias},
            rngs=rngs,
        )

        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=config.dtype, shd=shd.norm, rngs=rngs)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=config.dtype, shd=shd.norm, rngs=rngs)
        # v_norm has no learned scale — matches the reference implementation.
        self.v_norm = Gemma4RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, with_scale=False, dtype=config.dtype, shd=shd.norm, rngs=rngs
        )

        if attention_type == AttentionType.GLOBAL:
            rope_factor = config.global_rope_proportion
            rope_theta = (
                config.global_rope_max_timescale
                if config.global_rope_max_timescale is not None
                else config.rope_max_timescale
            )
        else:
            rope_factor = config.local_rope_proportion
            rope_theta = (
                config.local_rope_max_timescale
                if config.local_rope_max_timescale is not None
                else config.rope_max_timescale
            )

        self.rope = RoPE(
            rope_type="default",
            head_dim=self.head_dim,
            rope_theta=rope_theta,
            factor=rope_factor,
        )

    @jax.named_scope("gemma4_attention")
    def __call__(
        self,
        x: Array,
        positions: Array,
        cache: LayerCache | None = None,
        attention_mask: Array | None = None,
    ) -> Array:
        """Applies attention over the input sequences.

        Args:
            x: Input sequence tensor.
            positions: Position indices for RoPE and masking.
            cache: Optional KV cache for this layer.
            attention_mask: Optional custom attention mask. Generated automatically if None.

        Returns:
            The attention output tensor.
        """
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).reshape((batch_size, seq_len, self.num_heads, self.head_dim))
        k = self.k_proj(x).reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))
        if self.share_kv:
            v = k
        else:
            v = self.v_proj(x).reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))

        # Apply normalization per head
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        # RoPE
        sin, cos = self.rope(positions)
        q = apply_rope(q, sin, cos)
        k = apply_rope(k, sin, cos)

        if cache is not None:
            slice_indices = (0, cache.cur_ind[...], 0, 0)
            cache.k_cache[...] = jax.lax.dynamic_update_slice(cache.k_cache[...], k, slice_indices)
            cache.v_cache[...] = jax.lax.dynamic_update_slice(cache.v_cache[...], v, slice_indices)
            k = cache.k_cache[...]
            v = cache.v_cache[...]

        if cache is not None:
            k_len = k.shape[1]
            k_pos = jnp.arange(k_len)[None, :]
            q_pos = positions
            mask = q_pos[:, :, None] >= k_pos[:, None, :]
            if self.attention_type == AttentionType.LOCAL_SLIDING:
                window = q_pos[:, :, None] - k_pos[:, None, :]
                mask = mask & (window < self.config.sliding_window_size)
            structural_mask = jnp.where(mask, 0.0, -1e4).astype(q.dtype)[:, None, :, :]
        else:
            q_pos = positions[:, :, None]
            k_pos = positions[:, None, :]
            mask = q_pos >= k_pos
            if self.attention_type == AttentionType.LOCAL_SLIDING:
                window = q_pos - k_pos
                mask = mask & (window < self.config.sliding_window_size)
            structural_mask = jnp.where(mask, 0.0, -1e4).astype(q.dtype)[:, None, :, :]

        if attention_mask is None:
            attention_mask = structural_mask
        else:
            attention_mask = attention_mask + structural_mask

        # GQA: repeat K and V heads
        if self.num_kv_heads != self.num_heads:
            repeats = self.num_heads // self.num_kv_heads
            k = jnp.repeat(k, repeats, axis=2)
            v = jnp.repeat(v, repeats, axis=2)

        # Attention scores
        # q: [B, T, H, D], k: [B, S, H, D]
        q = jnp.transpose(q, (0, 2, 1, 3))  # [B, H, T, D]
        k = jnp.transpose(k, (0, 2, 3, 1))  # [B, H, D, S]
        v = jnp.transpose(v, (0, 2, 1, 3))  # [B, H, S, D]

        scores = jnp.matmul(q, k) / jnp.sqrt(self.head_dim)

        if self.config.attn_logits_soft_cap is not None:
            scores = scores / self.config.attn_logits_soft_cap
            scores = jnp.tanh(scores)
            scores = scores * self.config.attn_logits_soft_cap

        if attention_mask is not None:
            # Expand mask to [B, H, T, S] if needed
            scores = scores + attention_mask

        attn_weights = jax.nn.softmax(scores, axis=-1)
        out = jnp.matmul(attn_weights, v)  # [B, H, T, D]

        if cache is not None:
            cache.cur_ind[...] = cache.cur_ind[...] + seq_len

        out = jnp.transpose(out, (0, 2, 1, 3)).reshape((batch_size, seq_len, -1))
        return self.o_proj(out).astype(self.dtype)


class Gemma4DecoderLayer(nnx.Module):
    """A single decoder layer combining Attention, MoE, and Normalization."""

    def __init__(self, config: ModelConfig, attention_type: AttentionType, *, rngs: nnx.Rngs):
        self.config = config
        shd = config.shd_cfg

        self.pre_self_attention_norm = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=config.dtype, shd=shd.norm, rngs=rngs
        )
        self.self_attention = Gemma4Attention(config, attention_type, rngs=rngs)

        # In Gemma4, post_attn_norm is optional, let's include it for completeness
        # based on maxtext config `use_post_attn_norm` (default might be False, but we add it to mirror maxtext)
        self.post_self_attention_norm = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=config.dtype, shd=shd.norm, rngs=rngs
        )

        self.pre_ffw_norm = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=config.dtype, shd=shd.norm, rngs=rngs
        )

        if config.num_experts > 1:
            self.mlp = Gemma4MoE(config, rngs=rngs)
        else:
            self.mlp = Gemma4MLP(config.hidden_size, config.intermediate_size, dtype=config.dtype, shd=shd, rngs=rngs)

        self.post_ffw_norm = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=config.dtype, shd=shd.norm, rngs=rngs
        )

        if config.hidden_size_per_layer_input:
            self.per_layer_input_gate = _make_linear(
                config.hidden_size,
                config.hidden_size_per_layer_input,
                use_bias=False,
                kernel_metadata={"out_sharding": shd.fc1_kernel},
                bias_metadata={"out_sharding": shd.fc1_bias},
                rngs=rngs,
            )
            self.per_layer_projection = _make_linear(
                config.hidden_size_per_layer_input,
                config.hidden_size,
                use_bias=False,
                kernel_metadata={"out_sharding": shd.fc2_kernel},
                bias_metadata={"out_sharding": shd.fc2_bias},
                rngs=rngs,
            )
            self.post_per_layer_input_norm = Gemma4RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps, dtype=config.dtype, shd=shd.norm, rngs=rngs
            )

        self.layer_scalar = nnx.Param(jnp.ones((1,), dtype=config.weight_dtype), out_sharding=None)

    @jax.named_scope("gemma4_decoder_layer")
    def __call__(
        self,
        x: Array,
        positions: Array,
        cache: LayerCache | None = None,
        attention_mask: Array | None = None,
        per_layer_input: Array | None = None,
    ) -> Array:
        """Processes a single layer of attention and MLP/MoE.

        Args:
            x: Input tensor.
            positions: Position sequence for RoPE.
            cache: Optional KV cache for the layer.
            attention_mask: Optional attention mask.
            per_layer_input: Optional Per-Layer Embedding for this layer.

        Returns:
            The layer output.
        """
        # Self-attention block
        lnx = self.pre_self_attention_norm(x)
        attn_out = self.self_attention(lnx, positions, cache, attention_mask)
        attn_out = self.post_self_attention_norm(attn_out)

        # Residual
        x = x + attn_out

        # MLP / MoE block
        lnx2 = self.pre_ffw_norm(x)

        if isinstance(self.mlp, Gemma4MoE):
            mlp_out = self.mlp(lnx2, original_x=x)
        else:
            mlp_out = self.mlp(lnx2)

        mlp_out = self.post_ffw_norm(mlp_out)

        # Residual
        x = x + mlp_out

        # Per-Layer Embedding logic
        if self.config.hidden_size_per_layer_input and per_layer_input is not None:
            residual = x
            x_ple = self.per_layer_input_gate(x)
            x_ple = jax.nn.gelu(x_ple, approximate=True)  # gelu_pytorch_tanh
            x_ple = x_ple * per_layer_input
            x_ple = self.per_layer_projection(x_ple)
            x_ple = self.post_per_layer_input_norm(x_ple)
            x = residual + x_ple

        # Scale layer output
        layer_scale = jnp.asarray(self.layer_scalar[...], dtype=self.config.dtype)
        return x * layer_scale


class Gemma4Model(nnx.Module):
    """The base Gemma 4 trunk consisting of embeddings and a stack of decoder layers."""

    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.config = config
        shd = config.shd_cfg

        self.embed_tokens = _make_embed(
            config.vocab_size, config.hidden_size, embedding_metadata={"out_sharding": shd.emb_kernel}, rngs=rngs
        )

        # Scaling embedding by sqrt(hidden_size) as standard in Gemma
        self.embed_scale = jnp.sqrt(config.hidden_size)

        if config.hidden_size_per_layer_input:
            vocab_size_per_layer = (
                config.vocab_size_per_layer_input
                if config.vocab_size_per_layer_input is not None
                else config.vocab_size
            )
            self.embed_tokens_per_layer = _make_embed(
                vocab_size_per_layer,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                embedding_metadata={"out_sharding": shd.emb_kernel},
                rngs=rngs,
            )
            self.per_layer_input_scale = 2.0**-0.5
            self.per_layer_model_projection = _make_linear(
                config.hidden_size,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                use_bias=False,
                kernel_metadata={"out_sharding": shd.fc1_kernel},
                bias_metadata={"out_sharding": shd.fc1_bias},
                rngs=rngs,
            )
            self.per_layer_model_projection_scale = config.hidden_size**-0.5
            self.per_layer_projection_norm = Gemma4RMSNorm(
                config.hidden_size_per_layer_input, eps=config.rms_norm_eps, dtype=config.dtype, shd=shd.norm, rngs=rngs
            )

        self.layers = nnx.List()
        for i in range(config.num_hidden_layers):
            attn_type = GEMMA4_ATTENTION_PATTERN[i % len(GEMMA4_ATTENTION_PATTERN)]
            self.layers.append(Gemma4DecoderLayer(config, attn_type, rngs=rngs))

        self.norm = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=config.dtype, shd=shd.norm, rngs=rngs
        )

    def get_per_layer_inputs(self, input_ids: Array) -> Array:
        """Compute the token-identity component of Per-Layer Embeddings (PLE)."""
        ple = self.embed_tokens_per_layer(input_ids) * (self.config.hidden_size_per_layer_input**0.5)
        batch_size, seq_len, _ = ple.shape
        return ple.reshape(batch_size, seq_len, self.config.num_hidden_layers, self.config.hidden_size_per_layer_input)

    def project_per_layer_inputs(self, inputs_embeds: Array, per_layer_inputs: Array | None = None) -> Array:
        """Projects `inputs_embeds` and combines with token-identity `per_layer_inputs`."""
        batch_size, seq_len, _ = inputs_embeds.shape
        proj = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
        proj = proj.reshape(batch_size, seq_len, self.config.num_hidden_layers, self.config.hidden_size_per_layer_input)
        proj = self.per_layer_projection_norm(proj)
        if per_layer_inputs is not None:
            proj = (proj + per_layer_inputs) * self.per_layer_input_scale
        return proj

    @jax.named_scope("gemma4_model")
    def __call__(
        self,
        input_ids: Array,
        positions: Array,
        cache: Cache | None = None,
        attention_mask: Array | None = None,
        per_layer_inputs: Array | None = None,
    ) -> Array:
        """Applies embeddings and runs the forward pass through all decoder layers.

        Args:
            input_ids: Token IDs.
            positions: Sequence positions.
            cache: Optional list of KV caches (one per layer).
            attention_mask: Optional attention mask.
            per_layer_inputs: Optional computed per layer inputs for PLE.

        Returns:
            Hidden states output.
        """
        x = self.embed_tokens(input_ids) * self.embed_scale

        if self.config.hidden_size_per_layer_input:
            if per_layer_inputs is None:
                per_layer_inputs = self.get_per_layer_inputs(input_ids)
            per_layer_inputs = self.project_per_layer_inputs(x, per_layer_inputs)

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            layer_ple = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            x = layer(x, positions, layer_cache, attention_mask, per_layer_input=layer_ple)

        return self.norm(x)


class Gemma4ForCausalLM(nnx.Module):
    """Gemma 4 model with a language modeling head."""

    @classmethod
    def from_pretrained(cls, model_name: str, config: ModelConfig | None = None):
        """model_name the *model id* of a pretrained model hosted inside
        a model repo on huggingface.co. For example, "google/gemma-4-E2B-it".
        Note that access to the model is restricted and you need to be authorized to access it.
        """
        from huggingface_hub import snapshot_download
        from bonsai.models.gemma4 import params

        if config is None:
            config_map = {
                "google/gemma-4-E2B": ModelConfig.gemma4_e2b,
                "google/gemma-4-E2B-it": ModelConfig.gemma4_e2b,
                "google/gemma-4-E4B": ModelConfig.gemma4_e4b,
                "google/gemma-4-E4B-it": ModelConfig.gemma4_e4b,
                "google/gemma-4-26B-A4B": ModelConfig.gemma4_26b_a4b,
                "google/gemma-4-26B-A4B-it": ModelConfig.gemma4_26b_a4b,
                "google/gemma-4-31B": ModelConfig.gemma4_31b,
                "google/gemma-4-31B-it": ModelConfig.gemma4_31b,
            }
            if model_name not in config_map:
                raise ValueError(f"Model name '{model_name}' is unknown, please provide config argument")
            config = config_map[model_name]()

        model_ckpt_path = snapshot_download(repo_id=model_name, allow_patterns="*.safetensors")
        return params.create_gemma4_from_pretrained(model_ckpt_path, config)

    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.config = config
        shd = config.shd_cfg
        self.model = Gemma4Model(config, rngs=rngs)
        # Note: Usually LM head shares weights with embedding or has its own sharding.
        # Gemma traditionally shares embeddings, but we'll use a separate linear layer with matching sharding here if not explicitly tied.
        self.lm_head = _make_linear(
            config.hidden_size,
            config.vocab_size,
            use_bias=False,
            kernel_metadata={"out_sharding": shd.fc2_kernel},
            bias_metadata={"out_sharding": shd.fc2_bias},
            rngs=rngs,
        )
        self.vision_tower = SiglipVisionTransformer(config.vision_config, rngs=rngs) if config.vision_config else None
        self.multi_modal_projector = (
            Gemma4MultiModalProjector(config, config.vision_config, config.mm_tokens_per_image, rngs=rngs)
            if config.vision_config
            else None
        )
        self.audio_tower = Gemma4AudioModel(config.audio_config, rngs=rngs) if config.audio_config else None
        if config.audio_config:
            multimodal_hidden_size = getattr(config.audio_config, "output_proj_dims", config.audio_config.hidden_size)
            self.embed_audio = Gemma4MultimodalEmbedder(
                multimodal_hidden_size, config.hidden_size, config.audio_config.rms_norm_eps, rngs=rngs
            )
        else:
            self.embed_audio = None

    @jax.named_scope("gemma4_causal_lm")
    def __call__(
        self,
        input_ids: Array,
        positions: Array,
        cache: Cache | None = None,
        attention_mask: Array | None = None,
        pixel_values: Array | None = None,
        image_token_mask: Array | None = None,
        input_features: Array | None = None,
        input_features_mask: Array | None = None,
        audio_token_mask: Array | None = None,
    ) -> Array:
        """Computes logits for the given sequence, optionally applying soft-capping.

        Args:
            input_ids: Token IDs.
            positions: Sequence positions.
            cache: Optional list of KV caches.
            attention_mask: Optional attention mask.
            pixel_values: Optional image pixel values (B, H, W, C).
            image_token_mask: Optional boolean mask for image tokens (B, T).
            input_features: Optional audio features (e.g., log-mel spectrograms).
            input_features_mask: Optional mask for audio features.
            audio_token_mask: Optional boolean mask for audio tokens (B, T).

        Returns:
            Output logits.
        """
        has_vision = pixel_values is not None and self.vision_tower is not None
        has_audio = input_features is not None and self.audio_tower is not None

        if has_vision or has_audio:
            # Embed text first
            inputs_embeds = self.model.embed_tokens(input_ids) * self.model.embed_scale

            if has_vision:
                vision_outputs = self.vision_tower(pixel_values)
                image_features = self.multi_modal_projector(vision_outputs)
                if image_token_mask is not None:
                    inputs_embeds = batched_merge_modalities(image_features, inputs_embeds, image_token_mask)

            if has_audio:
                audio_outputs = self.audio_tower(input_features, input_features_mask)
                audio_features = self.embed_audio(audio_outputs)
                if audio_token_mask is not None:
                    inputs_embeds = batched_merge_modalities(audio_features, inputs_embeds, audio_token_mask)

            # Forward layers
            hidden_states = inputs_embeds

            if self.config.hidden_size_per_layer_input:
                # Transformers masks the vision tokens to padding when looking up the token identity
                # Here we just use the raw input_ids (where image placeholders usually reside)
                # and project the merged inputs_embeds
                per_layer_inputs_id = self.model.get_per_layer_inputs(input_ids)
                per_layer_inputs = self.model.project_per_layer_inputs(hidden_states, per_layer_inputs_id)
            else:
                per_layer_inputs = None

            for i, layer in enumerate(self.model.layers):
                layer_cache = cache[i] if cache is not None else None
                layer_ple = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
                hidden_states = layer(hidden_states, positions, layer_cache, attention_mask, per_layer_input=layer_ple)
            hidden_states = self.model.norm(hidden_states)
        else:
            hidden_states = self.model(input_ids, positions, cache, attention_mask)
        logits = self.lm_head(hidden_states)

        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = jnp.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        return logits.astype(jnp.float32)


@jax.jit
def forward(
    model: nnx.Module,
    cache: Cache,
    input_ids: Array,
    positions: Array,
    pixel_values: Array | None = None,
    image_token_mask: Array | None = None,
    input_features: Array | None = None,
    input_features_mask: Array | None = None,
    audio_token_mask: Array | None = None,
) -> tuple[Array, Cache]:
    """Executes a standard forward pass returning logits and updated cache."""
    logits = model(
        input_ids=input_ids,
        positions=positions,
        cache=cache,
        pixel_values=pixel_values,
        image_token_mask=image_token_mask,
        input_features=input_features,
        input_features_mask=input_features_mask,
        audio_token_mask=audio_token_mask,
    )
    return logits[:, -1, :], cache
