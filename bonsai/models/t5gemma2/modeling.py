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
from __future__ import annotations
import functools
from dataclasses import dataclass, field
from enum import Enum

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.module import first_from
from jaxtyping import Array, Bool, Float, Int


# =============================================================================
# Configuration Classes
# =============================================================================


class AttentionType(Enum):
    GLOBAL = "global"
    LOCAL_SLIDING = "local_sliding"


# Attention pattern: 5 local sliding + 1 global
_ATTN_PATTERN = (
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.GLOBAL,
)


def _make_layer_types(num_layers: int) -> tuple[AttentionType, ...]:
    """Generate attention types for all layers using the standard pattern."""
    n = len(_ATTN_PATTERN)
    return _ATTN_PATTERN * (num_layers // n) + _ATTN_PATTERN[: num_layers % n]


@dataclass(frozen=True)
class RoPEParameters:
    rope_type: str = "default"
    rope_theta: float = 10_000.0
    factor: float = 1.0


# Default RoPE parameters used across all model sizes
_DEFAULT_ROPE_PARAMS: dict[str, RoPEParameters | None] = {
    "full_attention": RoPEParameters(rope_type="linear", rope_theta=1_000_000.0, factor=8.0),
    "sliding_attention": RoPEParameters(rope_type="default", rope_theta=10_000.0, factor=1.0),
}


@dataclass(frozen=True)
class T5Gemma2VisionConfig:
    width: int = 1152
    image_size: int = 896
    patch_size: int = 14
    depth: int = 27
    mlp_dim: int = 4304
    num_heads: int = 16
    posemb: str = "learn"
    dropout: float = 0.0


@dataclass(frozen=True)
class T5Gemma2TextConfig:
    num_hidden_layers: int
    embed_dim: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    sliding_window: int
    layer_types: tuple[AttentionType, ...] = ()
    vocab_size: int = 262144
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    rope_parameters: dict[str, RoPEParameters | None] = field(default_factory=lambda: _DEFAULT_ROPE_PARAMS)
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 1
    attn_logit_softcapping: float | None = None

    @functools.cached_property
    def query_pre_attn_scalar(self) -> float:
        return self.head_dim**-0.5


@dataclass(frozen=True)
class T5Gemma2EncoderConfig:
    text_config: T5Gemma2TextConfig
    vision_config: T5Gemma2VisionConfig | None = None
    mm_tokens_per_image: int = 256
    image_token_id: int = 256001
    pad_token_id: int = 0


@dataclass(frozen=True)
class T5Gemma2DecoderConfig(T5Gemma2TextConfig):
    """Decoder config - inherits all fields from T5Gemma2TextConfig."""


@dataclass(frozen=True)
class T5Gemma2Config:
    encoder: T5Gemma2EncoderConfig
    decoder: T5Gemma2DecoderConfig
    eoi_token_index: int = 256000
    pad_token_id: int = 0

    @classmethod
    def _from_params(
        cls,
        num_layers: int,
        embed_dim: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        sliding_window: int,
        with_vision: bool = True,
    ) -> "T5Gemma2Config":
        layer_types = _make_layer_types(num_layers)
        text_cfg = T5Gemma2TextConfig(
            num_hidden_layers=num_layers,
            embed_dim=embed_dim,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            sliding_window=sliding_window,
            layer_types=layer_types,
        )
        vision_cfg = T5Gemma2VisionConfig() if with_vision else None
        return cls(
            encoder=T5Gemma2EncoderConfig(text_config=text_cfg, vision_config=vision_cfg),
            decoder=T5Gemma2DecoderConfig(
                num_hidden_layers=num_layers,
                embed_dim=embed_dim,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                sliding_window=sliding_window,
                layer_types=layer_types,
            ),
        )

    @classmethod
    def t5gemma2_270m_270m(cls, with_vision: bool = True) -> "T5Gemma2Config":
        return cls._from_params(18, 640, 2048, 4, 1, 512, with_vision)

    @classmethod
    def t5gemma2_1b_1b(cls, with_vision: bool = True) -> "T5Gemma2Config":
        return cls._from_params(26, 1152, 6912, 4, 1, 512, with_vision)

    @classmethod
    def t5gemma2_4b_4b(cls, with_vision: bool = True) -> "T5Gemma2Config":
        return cls._from_params(34, 2560, 10240, 8, 4, 1024, with_vision)


# =============================================================================
# Constants
# =============================================================================

# Large negative number for masking in attention
K_MASK = -2.3819763e38

# Special tokens
BOS_TOKEN = 2
EOS_TOKEN = 1
NEW_LINE_TOKEN = 108
START_OF_IMAGE_TOKEN = 255999
END_OF_IMAGE_TOKEN = 256000
IMAGE_PLACEHOLDER_IN_PROMPT = "<start_of_image>"
IMAGE_PLACEHOLDER_TOKEN = 256001  # Placeholder for image. Different from Gemma3.
NUM_PLACEHOLDER_TOKENS_PER_IMAGE = 256

# Default initializers
NORMAL_INIT = nnx.initializers.normal()
ZEROS_INIT = nnx.initializers.zeros_init()


# =============================================================================
# Helper Functions
# =============================================================================


def _get_rope_base_frequency(
    text_config: T5Gemma2TextConfig,
    attn_type: AttentionType,
) -> int:
    """Get RoPE base frequency for a given attention type.

    Args:
        text_config: Text configuration containing rope_parameters.
        attn_type: The attention type (GLOBAL or LOCAL_SLIDING).

    Returns:
        The base frequency for RoPE.
    """
    # Map attention type to rope parameter key
    if attn_type == AttentionType.GLOBAL:
        key = "full_attention"
    else:
        key = "sliding_attention"

    rope_params = text_config.rope_parameters.get(key)
    if rope_params is not None:
        return int(rope_params.rope_theta)
    return 10_000  # Default


def _get_rope_scale_factor(
    text_config: T5Gemma2TextConfig,
    attn_type: AttentionType,
) -> float:
    """Get RoPE scale factor for a given attention type.

    For "linear" rope_type, this returns the factor from config.
    For "default" rope_type, returns 1.0 (no scaling).

    Args:
        text_config: Text configuration containing rope_parameters.
        attn_type: The attention type (GLOBAL or LOCAL_SLIDING).

    Returns:
        The scale factor for RoPE.
    """
    # Map attention type to rope parameter key
    if attn_type == AttentionType.GLOBAL:
        key = "full_attention"
    else:
        key = "sliding_attention"

    rope_params = text_config.rope_parameters.get(key)
    if rope_params is not None:
        # "linear" rope_type uses factor, "default" uses 1.0
        if rope_params.rope_type == "linear":
            return rope_params.factor
    return 1.0  # Default (no scaling)


def apply_rope(
    inputs: Float[Array, "B L N H"],  # noqa: F722
    positions: Float[Array, "B L"],  # noqa: F722
    *,
    base_frequency: int,
    scale_factor: float = 1.0,
    rope_proportion: float = 1.0,
) -> Float[Array, "B L N H"]:  # noqa: F722
    """Applies Rotary Position Embeddings (RoPE).

    Args:
        inputs: Array of shape [B, L, N, H].
        positions: Array of shape [B, L].
        base_frequency: Base frequency used to compute rotations.
        scale_factor: Scale factor for positional interpolation.
        rope_proportion: Proportion of head dimension to apply RoPE to.

    Returns:
        Array of shape [B, L, N, H].
    """
    head_dim = inputs.shape[-1]
    rope_angles = int(rope_proportion * head_dim // 2)
    nope_angles = head_dim // 2 - rope_angles
    freq_exponents = (2.0 / head_dim) * jnp.arange(0, rope_angles, dtype=jnp.float32)
    timescale = jnp.pad(
        base_frequency**freq_exponents,
        (0, nope_angles),
        mode="constant",
        constant_values=(0, jnp.inf),
    )

    sinusoid_inp = positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
    sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
    if scale_factor < 1.0:
        raise ValueError(f"scale_factor must be >= 1.0, got {scale_factor}")
    sinusoid_inp /= scale_factor

    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)

    first_half, second_half = jnp.split(inputs, 2, axis=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    out = jnp.concatenate([first_part, second_part], axis=-1)
    return out.astype(inputs.dtype)


# =============================================================================
# Base Layers
# =============================================================================


class T5Gemma2Einsum(nnx.Module):
    """Parameterized einsum layer."""

    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        kernel_init: nnx.Initializer = NORMAL_INIT,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.w = nnx.Param(kernel_init(rngs.params(), shape, dtype))

    def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
        return jnp.einsum(eqn, x, self.w.value)


class T5Gemma2RMSNorm(nnx.Module):
    """RMS Normalization layer."""

    def __init__(
        self,
        features: int,
        *,
        epsilon: float = 1e-6,
        scale_init: nnx.Initializer = ZEROS_INIT,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.epsilon = epsilon
        self.scale = nnx.Param(scale_init(rngs.params(), (features,), dtype))

    def __call__(self, x: Float[Array, "B L D"]) -> Float[Array, "B L D"]:  # noqa: F722
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)

        # Jax.lax.rsqrt is used because it returns different floats than
        # jnp.reciprocal(jnp.sqrt(var + 1e-06))
        normed_inputs = x * jax.lax.rsqrt(var + self.epsilon)

        # normed_inputs is a rank-K tensor, K > 1 (K is typically 2 or 3). scale is
        # a rank-1 tensor. To avoid implicit rank-promotion, reshape scale to
        # a (1, ..., 1, D) tensor, so the rank of scale matches normed_inputs.
        scale = jnp.expand_dims(self.scale.value, axis=range(len(x.shape) - 1))
        return normed_inputs * (1 + scale)


class T5Gemma2FeedForward(nnx.Module):
    """Feed-forward module with gated activation."""

    def __init__(
        self,
        features: int,
        hidden_dim: int,
        *,
        transpose_gating_einsum: bool,
        kernel_init: nnx.Initializer = NORMAL_INIT,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.features = features
        self.hidden_dim = hidden_dim
        self.transpose_gating_einsum = transpose_gating_einsum

        if transpose_gating_einsum:
            gating_shape = (2, hidden_dim, features)
        else:
            gating_shape = (2, features, hidden_dim)

        self.gating_einsum = T5Gemma2Einsum(
            shape=gating_shape,
            kernel_init=kernel_init,
            dtype=dtype,
            rngs=rngs,
        )

        self.linear = T5Gemma2Einsum(
            shape=(hidden_dim, features),
            kernel_init=kernel_init,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, x: Float[Array, "B L D"]) -> Float[Array, "B L D"]:  # noqa: F722
        """Applies feed-forward transformation.

        Args:
            x: Input of shape [batch_size, seq_len, features].

        Returns:
            Output of shape [batch_size, seq_len, features].
        """
        eq = "...F,NHF->...NH" if self.transpose_gating_einsum else "...F,NFH->...NH"
        gate = self.gating_einsum(eq, x)
        activations = jax.nn.gelu(gate[..., 0, :]) * gate[..., 1, :]
        return self.linear("...H,HF->...F", activations)


# =============================================================================
# Attention Masks
# =============================================================================


def make_bidirectional_mask(
    input_mask: Bool[Array, "B L"],  # noqa: F722
) -> Bool[Array, "B 1 L L"]:  # noqa: F722
    """Creates a bidirectional attention mask for encoder.

    Args:
        input_mask: Boolean mask where True indicates valid tokens.

    Returns:
        Attention mask of shape [B, 1, L, L].
    """
    # [B, L] -> [B, 1, 1, L]
    mask = input_mask[:, None, None, :]
    return mask


def make_causal_mask(
    input_mask: Bool[Array, "B L"],  # noqa: F722
) -> Bool[Array, "B 1 L L"]:  # noqa: F722
    """Creates a causal attention mask for decoder.

    Args:
        input_mask: Boolean mask where True indicates valid tokens.

    Returns:
        Causal attention mask of shape [B, 1, L, L].
    """
    seq_len = input_mask.shape[-1]
    # Create causal mask: [L, L]
    causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    # Combine with input mask: [B, 1, L, L]
    mask = input_mask[:, None, None, :] & causal[None, None, :, :]
    return mask


def make_sliding_window_mask(
    positions: Int[Array, "B L"],  # noqa: F722
    sliding_window: int,
    *,
    bidirectional: bool = False,
) -> Bool[Array, "B 1 L L"]:  # noqa: F722
    """Creates a sliding window mask.

    Args:
        positions: Position indices of shape [B, L].
        sliding_window: Size of the sliding window.
        bidirectional: If True, creates a symmetric bidirectional window where
            the total window size equals `sliding_window` (split evenly on both
            sides). If False, creates a causal window of size `sliding_window`.

    Returns:
        Sliding window mask of shape [B, 1, L, L].
    """
    # [B, L, 1] and [B, 1, L]
    q_pos = positions[:, :, None]
    k_pos = positions[:, None, :]

    if bidirectional:
        # Bidirectional: symmetric window centered on query position
        # Total window size = sliding_window, split evenly on both sides
        # Matches PyTorch: left_window = (sw + 1) // 2, right_window = sw // 2 + 1
        left_window = (sliding_window + 1) // 2
        right_window = sliding_window // 2 + 1
        dist = q_pos - k_pos
        left_mask = (dist >= 0) & (dist < left_window)
        right_mask = (dist < 0) & (-dist < right_window)
        mask = left_mask | right_mask
    else:
        # Causal: can only attend to past tokens within window
        dist = jnp.abs(q_pos - k_pos)
        mask = dist < sliding_window

    return mask[:, None, :, :]


def make_sliding_window_causal_mask(
    input_mask: Bool[Array, "B L"],  # noqa: F722
    positions: Int[Array, "B L"],  # noqa: F722
    sliding_window: int,
) -> Bool[Array, "B 1 L L"]:  # noqa: F722
    """Creates a causal mask with sliding window for decoder.

    Combines causal mask (lower triangular) with sliding window constraint.

    Args:
        input_mask: Boolean mask where True indicates valid tokens.
        positions: Position indices of shape [B, L].
        sliding_window: Size of the sliding window.

    Returns:
        Sliding window causal mask of shape [B, 1, L, L].
    """
    # Start with causal mask
    causal_mask = make_causal_mask(input_mask)

    # Create sliding window mask
    sliding_mask = make_sliding_window_mask(positions, sliding_window)

    # Combine: must satisfy both causal AND sliding window
    return causal_mask & sliding_mask


def make_merged_attention_mask(
    decoder_mask: Bool[Array, "B 1 L_dec L_dec"],  # noqa: F722
    encoder_mask: Bool[Array, "B 1 1 L_enc"],  # noqa: F722
) -> Bool[Array, "B 1 L_dec L_combined"]:  # noqa: F722
    """Creates merged attention mask for decoder's merged attention.

    Concatenates the decoder self-attention mask with the cross-attention
    mask to form a single mask for the merged attention operation.

    Args:
        decoder_mask: Causal mask for decoder self-attention [B, 1, L_dec, L_dec].
        encoder_mask: Mask for encoder hidden states [B, 1, 1, L_enc].

    Returns:
        Merged mask of shape [B, 1, L_dec, L_dec + L_enc].
    """
    batch_size, _, seq_len, _ = decoder_mask.shape
    enc_len = encoder_mask.shape[-1]

    # Broadcast encoder mask to [B, 1, L_dec, L_enc]
    cross_mask = jnp.broadcast_to(encoder_mask, (batch_size, 1, seq_len, enc_len))

    # Concatenate along key dimension
    return jnp.concatenate([decoder_mask, cross_mask], axis=-1)


def make_decode_mode_self_mask(
    batch_size: int,
    query_len: int,
    max_cache_len: int,
    current_cache_len: int,
) -> Bool[Array, "B 1 Q K"]:  # noqa: F722
    """Creates self-attention mask for decode mode (with KV cache).

    During decode mode, the query attends to all previously cached keys.
    We mask out positions that haven't been written to yet (>= current_cache_len).

    Args:
        batch_size: Batch size.
        query_len: Number of query positions (typically 1 for decode mode).
        max_cache_len: Total allocated size of the cache.
        current_cache_len: Current valid length of the cache.

    Returns:
        Mask of shape [B, 1, query_len, max_cache_len].
    """
    k_pos = jnp.arange(max_cache_len)
    mask = k_pos < current_cache_len
    return jnp.broadcast_to(mask[None, None, None, :], (batch_size, 1, query_len, max_cache_len))


def make_decode_mode_sliding_mask(
    batch_size: int,
    query_pos: int,
    max_cache_len: int,
    sliding_window: int,
) -> Bool[Array, "B 1 1 K"]:  # noqa: F722
    """Creates sliding window mask for decode mode (with KV cache).

    During decode mode with sliding window, the query can only attend to
    positions within the sliding window range.

    Args:
        batch_size: Batch size.
        query_pos: Position of the current query (typically cache_index).
        max_cache_len: Total allocated size of the cache.
        sliding_window: Size of the sliding window.

    Returns:
        Mask of shape [B, 1, 1, max_cache_len].
    """
    # Key positions: 0, 1, 2, ..., max_cache_len-1
    k_positions = jnp.arange(max_cache_len)
    # Within sliding window: |query_pos - k_pos| < sliding_window
    in_window = jnp.abs(query_pos - k_positions) < sliding_window
    # Also must be causal (k_pos <= query_pos)
    is_causal = k_positions <= query_pos
    mask = in_window & is_causal
    return jnp.broadcast_to(mask[None, None, None, :], (batch_size, 1, 1, max_cache_len))


# =============================================================================
# Text Embeddings
# =============================================================================


class T5Gemma2ScaledWordEmbedding(nnx.Module):
    """Scaled word embedding with special EOI (end-of-image) token.

    The embeddings are scaled by the provided embed_scale, and the EOI token
    has a separate learnable embedding.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        *,
        padding_idx: int = 0,
        embed_scale: float | None = None,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        # If embed_scale is not provided, default to sqrt(embed_dim)
        if embed_scale is None:
            embed_scale = embed_dim**0.5
        self.embed_scale = jnp.array(embed_scale, dtype=dtype)

        self.embedding = nnx.Param(
            NORMAL_INIT(rngs.params(), (vocab_size, embed_dim), dtype),
        )
        self.eoi_embedding = nnx.Param(
            ZEROS_INIT(rngs.params(), (embed_dim,), dtype),
        )

    def __call__(self, input_ids: Int[Array, "B L"]) -> Float[Array, "B L D"]:  # noqa: F722
        """Embed input tokens with scaling.

        Args:
            input_ids: Input token IDs of shape [B, L].

        Returns:
            Embeddings of shape [B, L, hidden_size].
        """
        embeddings = self.embedding[input_ids] * self.embed_scale

        # Replace EOI token embeddings
        eoi_mask = input_ids == END_OF_IMAGE_TOKEN

        return jnp.where(eoi_mask[..., None], self.eoi_embedding[...], embeddings)


# =============================================================================
# Vision Components
# =============================================================================


def _posemb_sincos_2d(
    h: int,
    w: int,
    *,
    width: int,
    temperature: float = 10_000.0,
    dtype: jnp.dtype = jnp.float32,
) -> Float[Array, "1 M D"]:  # noqa: F722
    """Sinusoidal 2D position embeddings (MoCo v3 style)."""
    y, x = jnp.mgrid[:h, :w]

    assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
    omega = jnp.arange(width // 4) / (width // 4 - 1)
    omega = 1.0 / (temperature**omega)
    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)
    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
    return jnp.asarray(pe, dtype)[None, :, :]


class T5Gemma2VisionMLP(nnx.Module):
    """MLP for Vision Transformer."""

    def __init__(
        self,
        *,
        width: int,
        mlp_dim: int,
        dropout: float = 0.0,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.dense1 = nnx.Linear(
            in_features=width,
            out_features=mlp_dim,
            dtype=dtype,
            kernel_init=nnx.initializers.xavier_uniform(),
            bias_init=nnx.initializers.normal(stddev=1e-6),
            rngs=rngs,
        )
        self.dense2 = nnx.Linear(
            in_features=mlp_dim,
            out_features=width,
            dtype=dtype,
            kernel_init=nnx.initializers.xavier_uniform(),
            bias_init=nnx.initializers.normal(stddev=1e-6),
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(
        self,
        x: Float[Array, "B L D"],  # noqa: F722
        *,
        deterministic: bool | None = None,
    ) -> Float[Array, "B L D"]:  # noqa: F722
        x = self.dense1(x)
        x = jax.nn.gelu(x)
        x = self.dropout(x, deterministic=deterministic)
        return self.dense2(x)


class T5Gemma2VisionEncoderBlock(nnx.Module):
    """Transformer encoder block for Vision Transformer."""

    def __init__(
        self,
        *,
        width: int,
        mlp_dim: int,
        num_heads: int = 12,
        dropout: float = 0.0,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.layer_norm1 = nnx.LayerNorm(
            num_features=width,
            scale_init=nnx.initializers.ones_init(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )
        self.layer_norm2 = nnx.LayerNorm(
            num_features=width,
            scale_init=nnx.initializers.ones_init(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )

        self.dropout1 = nnx.Dropout(rate=dropout, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=dropout, rngs=rngs)

        self.mha = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=width,
            dtype=dtype,
            decode=False,
            kernel_init=nnx.initializers.xavier_uniform(),
            out_kernel_init=nnx.initializers.xavier_uniform(),
            bias_init=nnx.initializers.zeros_init(),
            out_bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )

        self.mlp = T5Gemma2VisionMLP(
            width=width,
            mlp_dim=mlp_dim,
            dropout=dropout,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        x: Float[Array, "B L D"],  # noqa: F722
        *,
        deterministic: bool | None = None,
    ) -> Float[Array, "B L D"]:  # noqa: F722
        y = self.layer_norm1(x)
        y = self.mha(y, deterministic=deterministic)
        y = self.dropout1(y, deterministic=deterministic)
        x = x + y

        y = self.layer_norm2(x)
        y = self.mlp(y, deterministic=deterministic)
        y = self.dropout2(y, deterministic=deterministic)
        return x + y


class T5Gemma2VisionEncoder(nnx.Module):
    """Vision Transformer Encoder."""

    def __init__(
        self,
        *,
        width: int,
        depth: int,
        mlp_dim: int,
        num_heads: int = 12,
        dropout: float = 0.0,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.depth = depth

        self.blocks = nnx.List(
            [
                T5Gemma2VisionEncoderBlock(
                    width=width,
                    mlp_dim=mlp_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    dtype=dtype,
                    rngs=rngs,
                )
                for _ in range(depth)
            ]
        )

        self.encoder_norm = nnx.LayerNorm(
            num_features=width,
            scale_init=nnx.initializers.ones_init(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool | None = None,
    ) -> jax.Array:
        for i in range(self.depth):
            x = self.blocks[i](x, deterministic=deterministic)
        return self.encoder_norm(x)


class T5Gemma2VisionExit(nnx.Module):
    """Vision exit layer - spatially pools soft tokens to output length."""

    def __init__(self, output_length: int = 256, *, rngs: nnx.Rngs):
        self.output_length = output_length

    def __call__(self, x: jax.Array) -> jax.Array:
        cur_length = x.shape[1]
        if cur_length == self.output_length:
            return x

        cur_width = int(cur_length**0.5)
        assert cur_width**2 == cur_length
        output_width = int(self.output_length**0.5)
        assert output_width**2 == self.output_length, f"Cannot pool {x.shape=} to {self.output_length}=!"

        batch_size = x.shape[0]
        embed_dim = x.shape[-1]
        x = jnp.reshape(x, (batch_size, cur_width, cur_width, embed_dim))
        assert not cur_width % output_width, f"{cur_width=} {output_width=}"
        window = cur_width // output_width
        window_shape = (window, window)
        x = nnx.avg_pool(x, window_shape, strides=window_shape, padding="VALID")
        batch_size, height, width, embed_dim = x.shape
        return jnp.reshape(x, (batch_size, height * width, embed_dim))


class T5Gemma2VisionSoftTokenizer(nnx.Module):
    """Vision soft tokenizer (ViT trained with SigLiP).

    Transforms images into soft tokens that can be embedded into the
    text embedding space.
    """

    def __init__(
        self,
        config: T5Gemma2VisionConfig,
        *,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.rngs = rngs

        self.embedding = nnx.Conv(
            in_features=3,
            out_features=config.width,
            kernel_size=(config.patch_size, config.patch_size),
            strides=(config.patch_size, config.patch_size),
            padding="VALID",
            kernel_init=nnx.initializers.xavier_uniform(),
            bias_init=nnx.initializers.zeros_init(),
            dtype=dtype,
            rngs=rngs,
        )

        self.pos_embedding = self._get_posemb(
            config.posemb,
            seqshape=(
                config.image_size // config.patch_size,
                config.image_size // config.patch_size,
            ),
            width=config.width,
            dtype=dtype,
        )

        self.dropout = nnx.Dropout(rate=config.dropout, rngs=rngs)

        self.transformer = T5Gemma2VisionEncoder(
            width=config.width,
            depth=config.depth,
            mlp_dim=config.mlp_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            dtype=dtype,
            rngs=rngs,
        )

        self.vision_exit = T5Gemma2VisionExit(output_length=256, rngs=rngs)

    def __call__(
        self,
        images: Float[Array, "B N H W C"],  # noqa: F722
        *,
        deterministic: bool | None = None,
    ) -> Float[Array, "B N P D"]:  # noqa: F722
        if len(images.shape) == 4:
            images = images[:, None, :]
        b, n, h, w, c = images.shape
        x = jnp.reshape(images, [b * n, h, w, c])

        x = self.embedding(x)
        bn, h, w, c = x.shape
        x = jnp.reshape(x, [bn, h * w, c])

        x = x + self.pos_embedding.value
        x = self.dropout(x, deterministic=deterministic)
        x = self.transformer(x, deterministic=deterministic)
        x = self.vision_exit(x)

        bn, s, d = x.shape
        return jnp.reshape(x, [b, n, s, d])

    def _get_posemb(
        self,
        typ: str,
        *,
        seqshape: tuple[int, int],
        width: int,
        dtype: jnp.dtype = jnp.float32,
    ) -> nnx.Param:
        """Returns the position embedding."""
        if typ == "learn":
            shape = (1, seqshape[0] * seqshape[1], width)
            initializer = nnx.initializers.normal(stddev=1 / (width**0.5))
            return nnx.Param(initializer(self.rngs.params(), shape, dtype))
        if typ == "sincos2d":
            return nnx.Param(
                _posemb_sincos_2d(*seqshape, width=width, dtype=dtype),
            )
        raise ValueError(f"Unknown posemb type: {typ}")


class T5Gemma2VisionSoftTokensEmbedder(nnx.Module):
    """Embeds vision soft tokens into the text embedding space."""

    def __init__(
        self,
        embed_dim: int,
        *,
        soft_tokens_dim: int,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.mm_soft_embedding_norm = T5Gemma2RMSNorm(
            soft_tokens_dim,
            scale_init=nnx.initializers.zeros_init(),
            dtype=dtype,
            rngs=rngs,
        )
        self.mm_input_projection = T5Gemma2Einsum(
            (soft_tokens_dim, embed_dim),
            kernel_init=nnx.initializers.normal(),
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, x: Float[Array, "B N P Dv"]) -> Float[Array, "B N P De"]:  # noqa: F722
        x = self.mm_soft_embedding_norm(x)
        return self.mm_input_projection("...tm,md->...td", x)


class T5Gemma2VisionEmbedder(nnx.Module):
    """Vision embedder - tokenizes images and embeds into text space."""

    def __init__(
        self,
        *,
        vision_config: T5Gemma2VisionConfig,
        embed_dim: int,
        freeze_params: bool = True,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.embed_dim = embed_dim
        self.freeze_params = freeze_params

        self.soft_tokenizer = T5Gemma2VisionSoftTokenizer(
            vision_config,
            dtype=dtype,
            rngs=rngs,
        )
        self.soft_tokens_embedder = T5Gemma2VisionSoftTokensEmbedder(
            embed_dim,
            soft_tokens_dim=vision_config.width,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        images: Float[Array, "B N H W C"],  # noqa: F722
        *,
        deterministic: bool | None = None,
    ) -> Float[Array, "B N P De"]:  # noqa: F722
        soft_tokens = self.soft_tokenizer(images, deterministic=deterministic)

        if self.freeze_params:
            soft_tokens = jax.lax.stop_gradient(soft_tokens)

        return self.soft_tokens_embedder(soft_tokens)


# =============================================================================
# Multimodal Utilities
# =============================================================================
def merge_mm_embeddings(
    text_embeddings: jnp.ndarray,
    vision_embeddings: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Merge text and multimodal (vision) embeddings."""
    return jax.vmap(_merge_mm_embeddings_inner, in_axes=(0, 0, 0))(
        text_embeddings,
        vision_embeddings,
        mask,
    )


def _merge_mm_embeddings_inner(
    text_embeddings: jnp.ndarray,
    vision_embeddings: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Merge embeddings without batch dimension."""
    num_images, num_toks_per_image, d = vision_embeddings.shape
    vision_embeddings = jnp.reshape(
        vision_embeddings,
        (num_images * num_toks_per_image, d),
    )

    target_pos = jnp.nonzero(mask, size=len(vision_embeddings))
    first_pos = text_embeddings[0]
    merged = text_embeddings.at[target_pos, :].set(vision_embeddings)
    return merged.at[0].set(first_pos)


# =============================================================================
# Encoder Components
# =============================================================================


class T5Gemma2EncoderAttention(nnx.Module):
    """Bidirectional self-attention for T5Gemma2 encoder.

    This is separate from Gemma3Attention because the encoder needs
    bidirectional sliding window attention, not causal sliding window.
    """

    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        features: int,
        head_dim: int,
        attn_type: AttentionType,
        query_pre_attn_scalar: float,
        *,
        rope_base_frequency: int = 10_000,
        rope_scale_factor: float = 1.0,
        rms_norm_eps: float = 1e-6,
        attn_logits_soft_cap: float | None = None,
        sliding_window_size: int | None = None,
        use_qk_norm: bool = False,
        kernel_init: nnx.Initializer = NORMAL_INIT,
        scale_init: nnx.Initializer = ZEROS_INIT,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.features = features
        self.head_dim = head_dim
        self.attn_type = attn_type
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.rope_base_frequency = rope_base_frequency
        self.rope_scale_factor = rope_scale_factor
        self.rms_norm_eps = rms_norm_eps
        self.attn_logits_soft_cap = attn_logits_soft_cap
        self.sliding_window_size = sliding_window_size
        self.use_qk_norm = use_qk_norm

        self.attn_vec_einsum = T5Gemma2Einsum(
            shape=(num_q_heads, head_dim, features),
            kernel_init=kernel_init,
            dtype=dtype,
            rngs=rngs,
        )

        # Check if we can use combined QKV projection
        if num_kv_heads == num_q_heads:
            self.qkv_einsum = T5Gemma2Einsum(
                shape=(3, num_q_heads, features, head_dim),
                kernel_init=kernel_init,
                dtype=dtype,
                rngs=rngs,
            )
            self.q_einsum = None
            self.kv_einsum = None
        else:
            self.qkv_einsum = None
            self.q_einsum = T5Gemma2Einsum(
                shape=(num_q_heads, features, head_dim),
                kernel_init=kernel_init,
                dtype=dtype,
                rngs=rngs,
            )
            self.kv_einsum = T5Gemma2Einsum(
                shape=(2, num_kv_heads, features, head_dim),
                kernel_init=kernel_init,
                dtype=dtype,
                rngs=rngs,
            )

        if use_qk_norm:
            self._query_norm = T5Gemma2RMSNorm(
                head_dim,
                epsilon=rms_norm_eps,
                dtype=dtype,
                scale_init=scale_init,
                rngs=rngs,
            )
            self._key_norm = T5Gemma2RMSNorm(
                head_dim,
                epsilon=rms_norm_eps,
                dtype=dtype,
                scale_init=scale_init,
                rngs=rngs,
            )

    @property
    def use_qkv_einsum(self) -> bool:
        return self.qkv_einsum is not None

    @property
    def use_gqa(self) -> bool:
        return self.num_kv_heads != self.num_q_heads and self.num_kv_heads > 1

    def __call__(
        self,
        x: Float[Array, "B L D"],  # noqa: F722
        segment_pos: Float[Array, "B L"],  # noqa: F722
        attn_mask: Float[Array, "B L L"],  # noqa: F722
    ) -> Float[Array, "B L D"]:  # noqa: F722
        """Applies bidirectional multi-head attention.

        Args:
            x: Input sequence of shape [batch_size, seq_len, embed_dim].
            segment_pos: Absolute positions of shape [batch_size, seq_len].
            attn_mask: Attention mask of shape [batch_size, seq_len, seq_len].

        Returns:
            Output sequence of shape [batch_size, seq_len, embed_dim].
        """
        # Project Q, K, V
        if self.use_qkv_einsum:
            query_proj, key_proj, value_proj = self.qkv_einsum("BTD,SNDH->SBTNH", x)
        else:
            query_proj = self.q_einsum("BTD,NDH->BTNH", x)
            key_proj, value_proj = self.kv_einsum("BSD,CKDH->CBSKH", x)

        # Apply QK normalization
        if self.use_qk_norm:
            query_proj = self._query_norm(query_proj)
            key_proj = self._key_norm(key_proj)

        # Apply RoPE
        query_proj = apply_rope(
            query_proj,
            segment_pos,
            base_frequency=self.rope_base_frequency,
            scale_factor=self.rope_scale_factor,
        )
        query_scaled = query_proj * self.query_pre_attn_scalar

        key_proj = apply_rope(
            key_proj,
            segment_pos,
            base_frequency=self.rope_base_frequency,
            scale_factor=self.rope_scale_factor,
        )

        # Compute attention logits
        if self.use_gqa:
            b, t, kg, h = query_scaled.shape
            query_scaled = query_scaled.reshape(
                (b, t, self.num_kv_heads, int(kg / self.num_kv_heads), h),
            )
            logits = jnp.einsum("BTKGH,BSKH->BTKGS", query_scaled, key_proj)
            b, t, k, g, s = logits.shape
            logits = logits.reshape((b, t, k * g, s))
        else:
            logits = jnp.einsum("BTNH,BSNH->BTNS", query_scaled, key_proj)

        # Apply soft capping
        if self.attn_logits_soft_cap is not None:
            logits = jnp.tanh(logits / self.attn_logits_soft_cap)
            logits = logits * self.attn_logits_soft_cap

        # Apply bidirectional sliding window mask for LOCAL_SLIDING layers
        if self.attn_type == AttentionType.LOCAL_SLIDING:
            if self.sliding_window_size is None:
                raise ValueError(
                    "sliding_window_size must be set for LOCAL_SLIDING attention",
                )
            # Use make_sliding_window_mask with bidirectional=True, squeeze to [B, L, L]
            sliding_mask = make_sliding_window_mask(
                segment_pos,
                self.sliding_window_size,
                bidirectional=True,
            )[:, 0, :, :]
            attn_mask = attn_mask * sliding_mask

        # Apply mask and softmax
        padded_logits = jnp.where(jnp.expand_dims(attn_mask, -2), logits, K_MASK)
        probs = jax.nn.softmax(padded_logits, axis=-1).astype(key_proj.dtype)

        # Compute attention output
        if self.use_gqa:
            b, t, kg, s = probs.shape
            probs = probs.reshape(
                (b, t, self.num_kv_heads, int(kg / self.num_kv_heads), s),
            )
            encoded = jnp.einsum("BTKGS,BSKH->BTKGH", probs, value_proj)
            b, t, k, g, h = encoded.shape
            encoded = encoded.reshape((b, t, k * g, h))
        else:
            encoded = jnp.einsum("BTNS,BSNH->BTNH", probs, value_proj)

        return self.attn_vec_einsum("BTNH,NHD->BTD", encoded)


class T5Gemma2EncoderBlock(nnx.Module):
    """Encoder transformer block with bidirectional attention.

    Uses T5Gemma2EncoderAttention which handles bidirectional sliding window
    correctly (unlike Gemma3Attention which uses causal sliding window).
    """

    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        embed_dim: int,
        head_dim: int,
        hidden_dim: int,
        *,
        use_post_attn_norm: bool,
        use_post_ffw_norm: bool,
        attn_type: AttentionType,
        query_pre_attn_scalar: float,
        transpose_gating_einsum: bool,
        rope_base_frequency: int = 10_000,
        rope_scale_factor: float = 1.0,
        rms_norm_eps: float = 1e-6,
        attn_logits_soft_cap: float | None = None,
        sliding_window_size: int | None = None,
        use_qk_norm: bool = False,
        kernel_init: nnx.Initializer = NORMAL_INIT,
        scale_init: nnx.Initializer = ZEROS_INIT,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.use_post_attn_norm = use_post_attn_norm
        self.use_post_ffw_norm = use_post_ffw_norm

        self.pre_attention_norm = T5Gemma2RMSNorm(
            embed_dim,
            epsilon=rms_norm_eps,
            dtype=dtype,
            scale_init=scale_init,
            rngs=rngs,
        )

        self.attn = T5Gemma2EncoderAttention(
            num_q_heads=num_q_heads,
            features=embed_dim,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            attn_type=attn_type,
            query_pre_attn_scalar=query_pre_attn_scalar,
            rope_base_frequency=rope_base_frequency,
            rope_scale_factor=rope_scale_factor,
            rms_norm_eps=rms_norm_eps,
            attn_logits_soft_cap=attn_logits_soft_cap,
            sliding_window_size=sliding_window_size,
            use_qk_norm=use_qk_norm,
            scale_init=scale_init,
            dtype=dtype,
            rngs=rngs,
        )

        if use_post_attn_norm:
            self.post_attention_norm = T5Gemma2RMSNorm(
                embed_dim,
                epsilon=rms_norm_eps,
                dtype=dtype,
                scale_init=scale_init,
                rngs=rngs,
            )

        self.pre_ffw_norm = T5Gemma2RMSNorm(
            embed_dim,
            epsilon=rms_norm_eps,
            dtype=dtype,
            scale_init=scale_init,
            rngs=rngs,
        )

        self.mlp = T5Gemma2FeedForward(
            features=embed_dim,
            hidden_dim=hidden_dim,
            transpose_gating_einsum=transpose_gating_einsum,
            kernel_init=kernel_init,
            dtype=dtype,
            rngs=rngs,
        )

        if use_post_ffw_norm:
            self.post_ffw_norm = T5Gemma2RMSNorm(
                embed_dim,
                epsilon=rms_norm_eps,
                dtype=dtype,
                scale_init=scale_init,
                rngs=rngs,
            )

    def __call__(
        self,
        x: Float[Array, "B L D"],  # noqa: F722
        segment_pos: Int[Array, "B L"],  # noqa: F722
        attn_mask: Bool[Array, "B L L"],  # noqa: F722
    ) -> Float[Array, "B L D"]:  # noqa: F722
        """Apply encoder layer.

        Args:
            x: Input hidden states [B, L, D].
            segment_pos: Position indices [B, L].
            attn_mask: Bidirectional attention mask [B, L, L].

        Returns:
            Output hidden states [B, L, D].
        """
        # Attention block
        inputs_normalized = self.pre_attention_norm(x)
        attn_output = self.attn(inputs_normalized, segment_pos, attn_mask)

        if self.use_post_attn_norm:
            attn_output = self.post_attention_norm(attn_output)

        attn_output = attn_output + x

        # Feed-forward block
        outputs = self.pre_ffw_norm(attn_output)
        outputs = self.mlp(outputs)

        if self.use_post_ffw_norm:
            outputs = self.post_ffw_norm(outputs)

        return outputs + attn_output


class T5Gemma2Encoder(nnx.Module):
    """T5Gemma2 Encoder with optional vision support.

    Processes text (and optional images) using bidirectional self-attention.
    """

    def __init__(
        self,
        config: T5Gemma2EncoderConfig,
        *,
        embedder: nnx.Module | None = None,
        dtype: jnp.dtype = jnp.float32,
        dtype_mm: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.config = config
        text_config = config.text_config

        # Text embeddings
        self.embedder = embedder or T5Gemma2ScaledWordEmbedding(
            vocab_size=text_config.vocab_size,
            embed_dim=text_config.embed_dim,
            padding_idx=text_config.pad_token_id,
            dtype=dtype,
            rngs=rngs,
        )

        # Vision encoder (optional) - uses Gemma3 vision component
        self.vision_embedder = nnx.data(None)
        if config.vision_config is not None:
            self.vision_embedder = T5Gemma2VisionEmbedder(
                vision_config=config.vision_config,
                embed_dim=text_config.embed_dim,
                freeze_params=True,
                dtype=dtype_mm,
                rngs=rngs,
            )

        # Determine attention types per layer
        attention_types = text_config.layer_types
        if not attention_types:
            attention_types = tuple(AttentionType.GLOBAL for _ in range(text_config.num_hidden_layers))

        # Encoder layers (use Gemma3Block via alias)
        self.blocks = nnx.List(
            [
                T5Gemma2EncoderBlock(
                    num_q_heads=text_config.num_attention_heads,
                    num_kv_heads=text_config.num_key_value_heads,
                    embed_dim=text_config.embed_dim,
                    head_dim=text_config.head_dim,
                    hidden_dim=text_config.intermediate_size,
                    use_post_attn_norm=True,
                    use_post_ffw_norm=True,
                    attn_type=attention_types[i],
                    query_pre_attn_scalar=text_config.query_pre_attn_scalar,
                    transpose_gating_einsum=True,
                    rope_base_frequency=_get_rope_base_frequency(
                        text_config,
                        attention_types[i],
                    ),
                    rope_scale_factor=_get_rope_scale_factor(
                        text_config,
                        attention_types[i],
                    ),
                    rms_norm_eps=text_config.rms_norm_eps,
                    attn_logits_soft_cap=text_config.attn_logit_softcapping,
                    sliding_window_size=(
                        text_config.sliding_window if attention_types[i] == AttentionType.LOCAL_SLIDING else None
                    ),
                    use_qk_norm=True,
                    dtype=dtype,
                    rngs=rngs,
                )
                for i in range(text_config.num_hidden_layers)
            ]
        )

        # Final normalization
        self.norm = T5Gemma2RMSNorm(
            text_config.embed_dim,
            epsilon=text_config.rms_norm_eps,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        tokens: Int[Array, "B L"],  # noqa: F722
        attention_mask: Bool[Array, "B L"] | None = None,  # noqa: F722
        position_ids: Int[Array, "B L"] | None = None,  # noqa: F722
        images: Float[Array, "B N H W C"] | None = None,  # noqa: F722
        *,
        deterministic: bool = True,
    ) -> Float[Array, "B L D"]:  # noqa: F722
        """Forward pass of the encoder.

        Args:
            tokens: Input token IDs [B, L].
            attention_mask: Attention mask [B, L].
            position_ids: Position indices [B, L].
            images: Optional input images [B, N, H, W, C] where N is images per batch.
            deterministic: Whether to run in deterministic mode.

        Returns:
            Encoder hidden states [B, L, D].
        """
        batch_size, seq_len = tokens.shape

        # Create position IDs if not provided
        if position_ids is None:
            position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.bool_)

        # Embed tokens
        x = self.embedder(tokens)

        # Process images and merge with text embeddings
        if images is not None and self.vision_embedder is not None:
            image_embeddings = self.vision_embedder(images, deterministic=deterministic)
            x = merge_mm_embeddings(
                x,
                image_embeddings,
                tokens == IMAGE_PLACEHOLDER_TOKEN,
            )

        # Create bidirectional attention mask for encoder [B, L, L]
        # Each query can attend to all valid (non-padded) key positions
        attn_mask = make_bidirectional_mask(attention_mask)
        # Squeeze to [B, L, L] if it's [B, 1, L, L]
        if attn_mask.ndim == 4:
            attn_mask = attn_mask[:, 0, :, :]

        # Ensure mask is [B, L, L] by broadcasting if needed
        if attn_mask.shape[-2] == 1:
            # [B, 1, L] -> [B, L, L]
            attn_mask = jnp.broadcast_to(attn_mask, (batch_size, seq_len, seq_len))

        # Apply encoder layers
        # T5Gemma2EncoderBlock handles bidirectional sliding window internally
        for block in self.blocks:
            x = block(x, position_ids, attn_mask)

        # Final normalization
        return self.norm(x)


# =============================================================================
# Decoder Components
# =============================================================================


class T5Gemma2MergedAttention(nnx.Module):
    """Merged self-attention and cross-attention for decoder.

    Combines self-attention and cross-attention in a single operation:
    - Query comes from decoder hidden states
    - Keys/Values are concatenation of [self_kv, cross_kv]
    - Mask is concatenation of [causal_mask, encoder_mask]

    This is more efficient than separate attention operations.

    Uses nnx.Cache for autoregressive decoding. Call init_cache() before
    using decode=True.
    """

    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        features: int,
        head_dim: int,
        query_pre_attn_scalar: float,
        *,
        rope_base_frequency: int = 10_000,
        rope_scale_factor: float = 1.0,
        rms_norm_eps: float = 1e-6,
        attn_logits_soft_cap: float | None = None,
        use_qk_norm: bool = True,
        kernel_init: nnx.Initializer = NORMAL_INIT,
        scale_init: nnx.Initializer = ZEROS_INIT,
        dtype: jnp.dtype = jnp.float32,
        decode: bool | None = None,
        rngs: nnx.Rngs,
    ):
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.features = features
        self.head_dim = head_dim
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.rope_base_frequency = rope_base_frequency
        self.rope_scale_factor = rope_scale_factor
        self.rms_norm_eps = rms_norm_eps
        self.attn_logits_soft_cap = attn_logits_soft_cap
        self.use_qk_norm = use_qk_norm
        self.decode = decode

        self.num_kv_groups = num_q_heads // num_kv_heads

        # Projections
        self.q_proj = T5Gemma2Einsum(
            shape=(num_q_heads, features, head_dim),
            kernel_init=kernel_init,
            dtype=dtype,
            rngs=rngs,
        )
        self.k_proj = T5Gemma2Einsum(
            shape=(num_kv_heads, features, head_dim),
            kernel_init=kernel_init,
            dtype=dtype,
            rngs=rngs,
        )
        self.v_proj = T5Gemma2Einsum(
            shape=(num_kv_heads, features, head_dim),
            kernel_init=kernel_init,
            dtype=dtype,
            rngs=rngs,
        )
        self.o_proj = T5Gemma2Einsum(
            shape=(num_q_heads, head_dim, features),
            kernel_init=kernel_init,
            dtype=dtype,
            rngs=rngs,
        )

        # QK normalization
        if use_qk_norm:
            self.q_norm = T5Gemma2RMSNorm(
                head_dim,
                epsilon=rms_norm_eps,
                dtype=dtype,
                scale_init=scale_init,
                rngs=rngs,
            )
            self.k_norm = T5Gemma2RMSNorm(
                head_dim,
                epsilon=rms_norm_eps,
                dtype=dtype,
                scale_init=scale_init,
                rngs=rngs,
            )

        # Cache for autoregressive decoding (nnx.Cache pattern like Gemma3)
        # Self-attention cache
        self.cached_self_key: nnx.Cache[Array] | None = nnx.data(None)
        self.cached_self_value: nnx.Cache[Array] | None = nnx.data(None)
        # Cross-attention cache (computed once from encoder outputs)
        self.cached_cross_key: nnx.Cache[Array] | None = nnx.data(None)
        self.cached_cross_value: nnx.Cache[Array] | None = nnx.data(None)
        # Cache index for self-attention
        self.cache_index: nnx.Cache[Array] | None = nnx.data(None)

    def init_cache(
        self,
        *,
        batch_size: int,
        max_decode_length: int,
        encoder_seq_length: int,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        """Initialize KV caches for autoregressive decoding.

        Must be called before using decode=True.

        Args:
            batch_size: Number of sequences in the batch.
            max_decode_length: Maximum decoder sequence length.
            encoder_seq_length: Encoder sequence length (for cross-attention).
            dtype: Data type for cache arrays.
        """
        # Self-attention cache shape: [B, max_decode_len, num_kv_heads, head_dim]
        self_cache_shape = (
            batch_size,
            max_decode_length,
            self.num_kv_heads,
            self.head_dim,
        )
        self.cached_self_key = nnx.Cache(jnp.zeros(self_cache_shape, dtype))
        self.cached_self_value = nnx.Cache(jnp.zeros(self_cache_shape, dtype))

        # Cross-attention cache shape: [B, enc_seq_len, num_kv_heads, head_dim]
        cross_cache_shape = (
            batch_size,
            encoder_seq_length,
            self.num_kv_heads,
            self.head_dim,
        )
        self.cached_cross_key = nnx.Cache(jnp.zeros(cross_cache_shape, dtype))
        self.cached_cross_value = nnx.Cache(jnp.zeros(cross_cache_shape, dtype))

        # Cache index tracks current position in self-attention cache
        self.cache_index = nnx.Cache(jnp.array(0, dtype=jnp.int32))

    def __call__(
        self,
        hidden_states: Float[Array, "B L_dec D"],  # noqa: F722
        encoder_hidden_states: Float[Array, "B L_enc D"],  # noqa: F722
        position_ids: Int[Array, "B L_dec"],  # noqa: F722
        merged_attention_mask: Bool[Array, "B 1 L_dec L_combined"] | None = None,  # noqa: F722
        *,
        decode: bool | None = None,
    ) -> Float[Array, "B L_dec D"]:  # noqa: F722
        """Apply merged self-attention and cross-attention.

        Args:
            hidden_states: Decoder hidden states [B, L_dec, D].
            encoder_hidden_states: Encoder outputs [B, L_enc, D].
            position_ids: Decoder position indices [B, L_dec].
            merged_attention_mask: Merged mask [B, 1, L_dec, L_combined].
            decode: Whether to use KV cache for autoregressive decoding.

        Returns:
            Output hidden states [B, L_dec, D].
        """
        seq_len = hidden_states.shape[1]

        # Project decoder Q, K, V (self-attention)
        query = self.q_proj("BTD,NDH->BTNH", hidden_states)
        self_key = self.k_proj("BTD,NDH->BTNH", hidden_states)
        self_value = self.v_proj("BTD,NDH->BTNH", hidden_states)

        # QK normalization for self-attention
        if self.use_qk_norm:
            query = self.q_norm(query)
            self_key = self.k_norm(self_key)

        # Apply RoPE to query and self-attention key
        query = apply_rope(
            query,
            position_ids,
            base_frequency=self.rope_base_frequency,
            scale_factor=self.rope_scale_factor,
        )
        self_key = apply_rope(
            self_key,
            position_ids,
            base_frequency=self.rope_base_frequency,
            scale_factor=self.rope_scale_factor,
        )

        # Determine decode mode
        decode = first_from(
            decode,
            self.decode,
            error_msg="No decode argument provided to T5Gemma2MergedAttention",
        )

        # Handle caching for autoregressive decoding
        if decode:
            if (
                self.cached_self_key is None
                or self.cached_self_value is None
                or self.cached_cross_key is None
                or self.cached_cross_value is None
                or self.cache_index is None
            ):
                raise ValueError(
                    "Autoregressive cache not initialized. Call init_cache() first.",
                )

            cur_index = self.cache_index[...]
            slice_indices = (0, cur_index, 0, 0)

            # Update self-attention cache using dynamic_update_slice
            self_key_updated = jax.lax.dynamic_update_slice(
                self.cached_self_key[...],
                self_key,
                slice_indices,
            )
            self_value_updated = jax.lax.dynamic_update_slice(
                self.cached_self_value[...],
                self_value,
                slice_indices,
            )

            self.cached_self_key[...] = self_key_updated
            self.cached_self_value[...] = self_value_updated

            # Use the full updated cache (static shape) for JIT compatibility
            self_key = self_key_updated
            self_value = self_value_updated

            # Cross-attention: compute and cache on first call (when index is 0)
            # After that, reuse cached values
            def compute_cross_kv():
                cross_k = self.k_proj("BTD,NDH->BTNH", encoder_hidden_states)
                cross_v = self.v_proj("BTD,NDH->BTNH", encoder_hidden_states)
                if self.use_qk_norm:
                    cross_k = self.k_norm(cross_k)
                return cross_k, cross_v

            # Use lax.cond to conditionally compute cross-attention KV
            cross_key, cross_value = jax.lax.cond(
                cur_index == 0,
                compute_cross_kv,
                lambda: (self.cached_cross_key[...], self.cached_cross_value[...]),
            )

            # Update cross cache (will be no-op after first call due to cond)
            self.cached_cross_key[...] = cross_key
            self.cached_cross_value[...] = cross_value

            # Update cache index
            self.cache_index[...] = cur_index + seq_len
        else:
            # No caching - compute cross-attention KV directly
            cross_key = self.k_proj("BTD,NDH->BTNH", encoder_hidden_states)
            cross_value = self.v_proj("BTD,NDH->BTNH", encoder_hidden_states)
            if self.use_qk_norm:
                cross_key = self.k_norm(cross_key)

        # Scale query
        query = query * self.query_pre_attn_scalar

        # Concatenate self and cross KV
        key = jnp.concatenate([self_key, cross_key], axis=1)
        value = jnp.concatenate([self_value, cross_value], axis=1)

        # Transpose to [B, N, T, H] for standard attention computation
        query = jnp.transpose(query, (0, 2, 1, 3))  # [B, N, T, H]
        key = jnp.transpose(key, (0, 2, 1, 3))  # [B, K, S, H]
        value = jnp.transpose(value, (0, 2, 1, 3))  # [B, K, S, H]

        # Expand KV heads for GQA
        if self.num_kv_groups > 1:
            key = jnp.repeat(key, self.num_kv_groups, axis=1)
            value = jnp.repeat(value, self.num_kv_groups, axis=1)

        # Compute attention scores: [B, N, T, S]
        attn_weights = jnp.einsum("BNTH,BNSH->BNTS", query, key)

        # Apply softcapping
        if self.attn_logits_soft_cap is not None:
            attn_weights = jnp.tanh(attn_weights / self.attn_logits_soft_cap)
            attn_weights = attn_weights * self.attn_logits_soft_cap

        # Apply attention mask
        if merged_attention_mask is not None:
            attn_weights = jnp.where(merged_attention_mask, attn_weights, K_MASK)

        # Softmax
        attn_weights = jax.nn.softmax(attn_weights, axis=-1).astype(value.dtype)

        # Apply attention to values: [B, N, T, H]
        attn_output = jnp.einsum("BNTS,BNSH->BNTH", attn_weights, value)

        # Transpose back to [B, T, N, H]
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))

        # Output projection
        output = self.o_proj("BTNH,NHD->BTD", attn_output)

        return output


class T5Gemma2DecoderBlock(nnx.Module):
    """Decoder transformer block with merged self/cross attention."""

    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        embed_dim: int,
        head_dim: int,
        hidden_dim: int,
        *,
        attn_type: AttentionType = AttentionType.GLOBAL,
        use_post_attn_norm: bool = True,
        use_post_ffw_norm: bool = True,
        query_pre_attn_scalar: float,
        rope_base_frequency: int = 10_000,
        rope_scale_factor: float = 1.0,
        rms_norm_eps: float = 1e-6,
        attn_logits_soft_cap: float | None = None,
        use_qk_norm: bool = True,
        kernel_init: nnx.Initializer = NORMAL_INIT,
        scale_init: nnx.Initializer = ZEROS_INIT,
        decode: bool | None = None,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.attn_type = attn_type
        self.use_post_attn_norm = use_post_attn_norm
        self.use_post_ffw_norm = use_post_ffw_norm

        # Pre-attention norm
        self.pre_attention_norm = T5Gemma2RMSNorm(
            embed_dim,
            epsilon=rms_norm_eps,
            dtype=dtype,
            scale_init=scale_init,
            rngs=rngs,
        )

        # Merged attention (self + cross)
        self.attn = T5Gemma2MergedAttention(
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            features=embed_dim,
            head_dim=head_dim,
            query_pre_attn_scalar=query_pre_attn_scalar,
            rope_base_frequency=rope_base_frequency,
            rope_scale_factor=rope_scale_factor,
            rms_norm_eps=rms_norm_eps,
            attn_logits_soft_cap=attn_logits_soft_cap,
            use_qk_norm=use_qk_norm,
            kernel_init=kernel_init,
            scale_init=scale_init,
            decode=decode,
            dtype=dtype,
            rngs=rngs,
        )

        # Post-attention norm
        if use_post_attn_norm:
            self.post_attention_norm = T5Gemma2RMSNorm(
                embed_dim,
                epsilon=rms_norm_eps,
                dtype=dtype,
                scale_init=scale_init,
                rngs=rngs,
            )

        # Pre-FFW norm
        self.pre_ffw_norm = T5Gemma2RMSNorm(
            embed_dim,
            epsilon=rms_norm_eps,
            dtype=dtype,
            scale_init=scale_init,
            rngs=rngs,
        )

        # Feed-forward
        self.mlp = T5Gemma2FeedForward(
            features=embed_dim,
            hidden_dim=hidden_dim,
            transpose_gating_einsum=True,
            kernel_init=kernel_init,
            dtype=dtype,
            rngs=rngs,
        )

        # Post-FFW norm
        if use_post_ffw_norm:
            self.post_ffw_norm = T5Gemma2RMSNorm(
                embed_dim,
                epsilon=rms_norm_eps,
                dtype=dtype,
                scale_init=scale_init,
                rngs=rngs,
            )

    def __call__(
        self,
        x: Float[Array, "B L_dec D"],  # noqa: F722
        encoder_hidden_states: Float[Array, "B L_enc D"],  # noqa: F722
        segment_pos: Int[Array, "B L_dec"],  # noqa: F722
        attn_mask: Bool[Array, "B 1 L_dec L_combined"] | None = None,  # noqa: F722
        *,
        decode: bool | None = None,
    ) -> Float[Array, "B L_dec D"]:  # noqa: F722
        """Apply decoder layer.

        Args:
            x: Decoder hidden states [B, L_dec, D].
            encoder_hidden_states: Encoder outputs [B, L_enc, D].
            segment_pos: Decoder position indices [B, L_dec].
            attn_mask: Merged attention mask [B, 1, L_dec, L_combined].
            decode: Whether to use KV cache.

        Returns:
            Output hidden states [B, L_dec, D].
        """
        # Merged attention block
        inputs_normalized = self.pre_attention_norm(x)
        attn_output = self.attn(
            inputs_normalized,
            encoder_hidden_states,
            segment_pos,
            attn_mask,
            decode=decode,
        )

        if self.use_post_attn_norm:
            attn_output = self.post_attention_norm(attn_output)

        attn_output += x

        # Feed-forward block
        outputs = self.pre_ffw_norm(attn_output)
        outputs = self.mlp(outputs)

        if self.use_post_ffw_norm:
            outputs = self.post_ffw_norm(outputs)

        outputs += attn_output

        return outputs


class T5Gemma2Decoder(nnx.Module):
    """T5Gemma2 Decoder with merged self/cross attention.

    Uses nnx.Cache for autoregressive decoding. Call init_cache() before
    using decode=True.
    """

    def __init__(
        self,
        config: T5Gemma2DecoderConfig,
        *,
        embedder: nnx.Module | None = None,
        dtype: jnp.dtype = jnp.float32,
        decode: bool | None = None,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.dtype = dtype

        # Text embeddings
        self.embedder = embedder or T5Gemma2ScaledWordEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            padding_idx=config.pad_token_id,
            dtype=dtype,
            rngs=rngs,
        )

        # Determine attention types per layer
        attention_types = config.layer_types
        if not attention_types:
            attention_types = tuple(AttentionType.GLOBAL for _ in range(config.num_hidden_layers))

        # Decoder layers
        self.blocks = nnx.List(
            [
                T5Gemma2DecoderBlock(
                    num_q_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    embed_dim=config.embed_dim,
                    head_dim=config.head_dim,
                    hidden_dim=config.intermediate_size,
                    attn_type=attention_types[i],
                    use_post_attn_norm=True,
                    use_post_ffw_norm=True,
                    query_pre_attn_scalar=config.query_pre_attn_scalar,
                    rope_base_frequency=_get_rope_base_frequency(config, attention_types[i]),
                    rope_scale_factor=_get_rope_scale_factor(config, attention_types[i]),
                    rms_norm_eps=config.rms_norm_eps,
                    attn_logits_soft_cap=config.attn_logit_softcapping,
                    use_qk_norm=True,
                    decode=decode,
                    dtype=dtype,
                    rngs=rngs,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        # Final normalization
        self.norm = T5Gemma2RMSNorm(
            config.embed_dim,
            epsilon=config.rms_norm_eps,
            dtype=dtype,
            rngs=rngs,
        )

    def init_cache(
        self,
        *,
        batch_size: int,
        max_decode_length: int,
        encoder_seq_length: int,
        dtype: jnp.dtype | None = None,
    ) -> None:
        """Initialize KV caches for all decoder layers.

        Must be called before using decode=True.

        Args:
            batch_size: Number of sequences in the batch.
            max_decode_length: Maximum decoder sequence length.
            encoder_seq_length: Encoder sequence length (for cross-attention).
            dtype: Data type for cache arrays. Defaults to model dtype.
        """
        if dtype is None:
            dtype = self.dtype

        for block in self.blocks:
            block.attn.init_cache(
                batch_size=batch_size,
                max_decode_length=max_decode_length,
                encoder_seq_length=encoder_seq_length,
                dtype=dtype,
            )

    def get_cache_index(self) -> int:
        """Get current cache index from the first layer."""
        if len(self.blocks) > 0 and self.blocks[0].attn.cache_index is not None:
            return int(self.blocks[0].attn.cache_index[...])
        return 0

    def __call__(
        self,
        input_ids: Int[Array, "B L_dec"],  # noqa: F722
        encoder_hidden_states: Float[Array, "B L_enc D"],  # noqa: F722
        attention_mask: Bool[Array, "B L_dec"] | None = None,  # noqa: F722
        encoder_attention_mask: Bool[Array, "B L_enc"] | None = None,  # noqa: F722
        position_ids: Int[Array, "B L_dec"] | None = None,  # noqa: F722
        *,
        decode: bool | None = None,
    ) -> Float[Array, "B L_dec D"]:  # noqa: F722
        """Forward pass of the decoder.

        Args:
            input_ids: Decoder input token IDs [B, L_dec].
            encoder_hidden_states: Encoder outputs [B, L_enc, D].
            attention_mask: Decoder attention mask [B, L_dec].
            encoder_attention_mask: Encoder attention mask [B, L_enc].
            position_ids: Decoder position indices [B, L_dec].
            decode: Whether to use KV cache.

        Returns:
            Decoder hidden states [B, L_dec, D].
        """
        batch_size, dec_seq_len = input_ids.shape
        enc_seq_len = encoder_hidden_states.shape[1]

        # Create position IDs if not provided
        if position_ids is None:
            cache_index = self.get_cache_index()
            if decode and cache_index > 0:
                # During autoregressive decoding, position is cache_index
                position_ids = jnp.full(
                    (batch_size, dec_seq_len),
                    cache_index,
                    dtype=jnp.int32,
                )
            else:
                position_ids = jnp.arange(dec_seq_len)[None, :].repeat(
                    batch_size,
                    axis=0,
                )

        # Create attention masks if not provided
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, dec_seq_len), dtype=jnp.bool_)
        if encoder_attention_mask is None:
            encoder_attention_mask = jnp.ones(
                (batch_size, enc_seq_len),
                dtype=jnp.bool_,
            )

        # Create encoder mask for cross-attention (bidirectional)
        encoder_mask = encoder_attention_mask[:, None, None, :]

        if decode:
            # Decode mode: create masks accounting for cached sequence length
            cache_index = self.get_cache_index()
            # Get max_decode_length (static) from the first block's cache
            max_decode_len = self.blocks[0].attn.cached_self_key.shape[1]
            # Current valid length (dynamic)
            current_len = cache_index + dec_seq_len

            # Full attention: new tokens can attend to all cached + encoder
            full_decoder_mask = make_decode_mode_self_mask(
                batch_size,
                dec_seq_len,
                max_decode_len,
                current_len,
            )
            # Sliding attention: limited to sliding window
            sliding_decoder_mask = make_decode_mode_sliding_mask(
                batch_size,
                cache_index,  # Current query position
                max_decode_len,
                self.config.sliding_window,
            )

            # Merged masks with encoder (broadcast encoder mask for query length)
            enc_mask_broadcast = jnp.broadcast_to(encoder_mask, (batch_size, 1, dec_seq_len, enc_seq_len))
            merged_mask_full = jnp.concatenate([full_decoder_mask, enc_mask_broadcast], axis=-1)
            merged_mask_sliding = jnp.concatenate([sliding_decoder_mask, enc_mask_broadcast], axis=-1)
        else:
            # Non-decode mode: standard causal masks
            # Full attention: standard causal mask
            full_decoder_mask = make_causal_mask(attention_mask)
            # Sliding attention: causal + sliding window
            sliding_decoder_mask = make_sliding_window_causal_mask(
                attention_mask,
                position_ids,
                self.config.sliding_window,
            )

            # Create merged masks for each attention type
            merged_mask_full = make_merged_attention_mask(full_decoder_mask, encoder_mask)
            merged_mask_sliding = make_merged_attention_mask(sliding_decoder_mask, encoder_mask)

        # Embed tokens
        hidden_states = self.embedder(input_ids)

        # Apply decoder layers with per-layer masks
        for block in self.blocks:
            # Select mask based on layer's attention type
            if block.attn_type == AttentionType.LOCAL_SLIDING:
                block_mask = merged_mask_sliding
            else:
                block_mask = merged_mask_full

            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                position_ids,
                block_mask,
                decode=decode,
            )

        # Final normalization
        hidden_states = self.norm(hidden_states)

        return hidden_states


# =============================================================================
# Full Model
# =============================================================================


class T5Gemma2(nnx.Module):
    """T5Gemma2 Encoder-Decoder Model.

    Combines the encoder and decoder into a full sequence-to-sequence model.
    Uses nnx.Cache for autoregressive decoding. Call init_cache() before
    using decode=True.
    """

    def __init__(
        self,
        config: T5Gemma2Config,
        *,
        dtype: jnp.dtype = jnp.float32,
        dtype_mm: jnp.dtype = jnp.float32,
        decode: bool | None = None,
        rngs: nnx.Rngs,
    ):
        self.config = config

        # Use tied embedding for encoder and decoder
        self.embedder = T5Gemma2ScaledWordEmbedding(
            vocab_size=config.encoder.text_config.vocab_size,
            embed_dim=config.encoder.text_config.embed_dim,
            padding_idx=config.encoder.text_config.pad_token_id,
            dtype=dtype,
            rngs=rngs,
        )

        # Encoder
        self.encoder = T5Gemma2Encoder(
            config.encoder,
            embedder=self.embedder,
            dtype=dtype,
            dtype_mm=dtype_mm,
            rngs=rngs,
        )

        # Decoder
        self.decoder = T5Gemma2Decoder(
            config.decoder,
            embedder=self.embedder,
            dtype=dtype,
            decode=decode,
            rngs=rngs,
        )

    def init_cache(
        self,
        *,
        batch_size: int,
        max_decode_length: int,
        encoder_seq_length: int,
        dtype: jnp.dtype | None = None,
    ) -> None:
        """Initialize decoder KV caches for autoregressive decoding.

        Args:
            batch_size: Number of sequences in the batch.
            max_decode_length: Maximum decoder sequence length.
            encoder_seq_length: Encoder sequence length.
            dtype: Data type for cache arrays.
        """
        self.decoder.init_cache(
            batch_size=batch_size,
            max_decode_length=max_decode_length,
            encoder_seq_length=encoder_seq_length,
            dtype=dtype,
        )

    def __call__(
        self,
        input_ids: Int[Array, "B L_enc"],  # noqa: F722
        decoder_input_ids: Int[Array, "B L_dec"],  # noqa: F722
        attention_mask: Bool[Array, "B L_enc"] | None = None,  # noqa: F722
        decoder_attention_mask: Bool[Array, "B L_dec"] | None = None,  # noqa: F722
        position_ids: Int[Array, "B L_enc"] | None = None,  # noqa: F722
        decoder_position_ids: Int[Array, "B L_dec"] | None = None,  # noqa: F722
        pixel_values: Float[Array, "B N H W C"] | None = None,  # noqa: F722
        encoder_outputs: Float[Array, "B L_enc D"] | None = None,  # noqa: F722
        *,
        decode: bool | None = None,
        deterministic: bool = True,
    ) -> tuple[Float[Array, "B L_dec D"], Float[Array, "B L_enc D"]]:  # noqa: F722
        """Forward pass of the full model.

        Args:
            input_ids: Encoder input token IDs [B, L_enc].
            decoder_input_ids: Decoder input token IDs [B, L_dec].
            attention_mask: Encoder attention mask [B, L_enc].
            decoder_attention_mask: Decoder attention mask [B, L_dec].
            position_ids: Encoder position indices [B, L_enc].
            decoder_position_ids: Decoder position indices [B, L_dec].
            pixel_values: Optional input images [B, N, H, W, C] where N is images per batch.
            encoder_outputs: Pre-computed encoder outputs (for generation).
            decode: Whether to use KV cache.
            deterministic: Whether to run in deterministic mode.

        Returns:
            Tuple of (decoder hidden states, encoder hidden states).
        """
        # Encode if not provided
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                images=pixel_values,  # Note: encoder uses 'images' parameter
                deterministic=deterministic,
            )

        # Decode
        decoder_outputs = self.decoder(
            decoder_input_ids,
            encoder_hidden_states=encoder_outputs,
            attention_mask=decoder_attention_mask,
            encoder_attention_mask=attention_mask,
            position_ids=decoder_position_ids,
            decode=decode,
        )

        return decoder_outputs, encoder_outputs
