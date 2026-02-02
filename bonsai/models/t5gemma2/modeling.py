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

import functools
from dataclasses import dataclass, field
from enum import Enum

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.module import first_from
from jax import P
from jax.sharding import PartitionSpec
from jaxtyping import Array, Bool, Float, Int


class AttentionType(Enum):
    GLOBAL = "global"
    LOCAL_SLIDING = "local_sliding"


class ShardMode(Enum):
    FSDP = "fsdp"
    TP = "tp"


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
    n = len(_ATTN_PATTERN)
    return _ATTN_PATTERN * (num_layers // n) + _ATTN_PATTERN[: num_layers % n]


@dataclass(slots=True, frozen=True)
class VisionShardingCfg:
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
    emb_pos_kernel: PartitionSpec | None = None

    @staticmethod
    def no_sharding():
        return VisionShardingCfg()

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
    attn_kernel: PartitionSpec | None = None
    attn_bias: PartitionSpec | None = None
    attn_qk_activation: PartitionSpec | None = None
    down_kernel: PartitionSpec | None = None
    down_bias: PartitionSpec | None = None
    up_gate_kernel: PartitionSpec | None = None
    up_gate_bias: PartitionSpec | None = None
    activation: PartitionSpec | None = None
    norm: PartitionSpec | None = None
    cache: PartitionSpec | None = None
    emb_kernel: PartitionSpec | None = None

    @staticmethod
    def no_sharding():
        return TextShardingCfg()

    @staticmethod
    def default(use_fsdp: bool, use_tp: bool):
        fsdp = ShardMode.FSDP.value if use_fsdp else None
        tp = ShardMode.TP.value if use_tp else None
        return TextShardingCfg(
            attn_kernel=P(tp, fsdp),
            attn_bias=P(tp),
            attn_qk_activation=P(fsdp, None, tp),
            down_kernel=P(tp, fsdp),
            down_bias=P(tp),
            up_gate_kernel=P(fsdp, tp),
            up_gate_bias=P(tp),
            activation=P(fsdp, None, tp),
            norm=P(tp),
            cache=P(fsdp, None, tp, None),
            emb_kernel=P(None, tp),
        )


@dataclass(slots=True, frozen=True)
class MMShardingCfg:
    mmp_norm: PartitionSpec | None = None
    mmp_weight: PartitionSpec | None = None

    @staticmethod
    def no_sharding():
        return MMShardingCfg()

    @staticmethod
    def default(use_tp: bool):
        tp = ShardMode.TP.value if use_tp else None
        return MMShardingCfg(mmp_norm=P(tp), mmp_weight=P(tp))


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
    shd_cfg: VisionShardingCfg = field(default_factory=VisionShardingCfg)


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
    shd_cfg: TextShardingCfg = field(default_factory=TextShardingCfg)

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
    pass


@dataclass(frozen=True)
class T5Gemma2Config:
    encoder: T5Gemma2EncoderConfig
    decoder: T5Gemma2DecoderConfig
    eoi_token_index: int = 256000
    pad_token_id: int = 0
    shd_cfg: MMShardingCfg = field(default_factory=MMShardingCfg)

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
        use_fsdp: bool = False,
        use_tp: bool = False,
    ) -> "T5Gemma2Config":
        layer_types = _make_layer_types(num_layers)

        if use_fsdp or use_tp:
            text_shd = TextShardingCfg.default(use_fsdp, use_tp)
            vision_shd = VisionShardingCfg.default(use_fsdp, use_tp)
            mm_shd = MMShardingCfg.default(use_tp)
        else:
            text_shd = TextShardingCfg.no_sharding()
            vision_shd = VisionShardingCfg.no_sharding()
            mm_shd = MMShardingCfg.no_sharding()

        text_cfg = T5Gemma2TextConfig(
            num_hidden_layers=num_layers,
            embed_dim=embed_dim,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            sliding_window=sliding_window,
            layer_types=layer_types,
            shd_cfg=text_shd,
        )
        vision_cfg = T5Gemma2VisionConfig(shd_cfg=vision_shd) if with_vision else None
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
                shd_cfg=text_shd,
            ),
            shd_cfg=mm_shd,
        )

    @classmethod
    def t5gemma2_270m_270m(
        cls, with_vision: bool = True, use_fsdp: bool = False, use_tp: bool = False
    ) -> "T5Gemma2Config":
        return cls._from_params(18, 640, 2048, 4, 1, 512, with_vision, use_fsdp, use_tp)

    @classmethod
    def t5gemma2_1b_1b(cls, with_vision: bool = True, use_fsdp: bool = False, use_tp: bool = False) -> "T5Gemma2Config":
        return cls._from_params(26, 1152, 6912, 4, 1, 512, with_vision, use_fsdp, use_tp)

    @classmethod
    def t5gemma2_4b_4b(cls, with_vision: bool = True, use_fsdp: bool = False, use_tp: bool = False) -> "T5Gemma2Config":
        return cls._from_params(34, 2560, 10240, 8, 4, 1024, with_vision, use_fsdp, use_tp)


_K_MASK = jnp.finfo(jnp.bfloat16).min

BOS_TOKEN = 2
EOS_TOKEN = 1
NEW_LINE_TOKEN = 108
START_OF_IMAGE_TOKEN = 255999
END_OF_IMAGE_TOKEN = 256000
IMAGE_PLACEHOLDER_IN_PROMPT = "<start_of_image>"
IMAGE_PLACEHOLDER_TOKEN = 256001  # Placeholder for image. Different from Gemma3.
NUM_PLACEHOLDER_TOKENS_PER_IMAGE = 256

NORMAL_INIT = nnx.initializers.normal()
ZEROS_INIT = nnx.initializers.zeros_init()


def _get_rope_base_frequency(
    text_config: T5Gemma2TextConfig,
    attn_type: AttentionType,
) -> int:
    if attn_type == AttentionType.GLOBAL:
        key = "full_attention"
    else:
        key = "sliding_attention"

    rope_params = text_config.rope_parameters.get(key)
    if rope_params is not None:
        return int(rope_params.rope_theta)
    return 10_000


def _get_rope_scale_factor(
    text_config: T5Gemma2TextConfig,
    attn_type: AttentionType,
) -> float:
    if attn_type == AttentionType.GLOBAL:
        key = "full_attention"
    else:
        key = "sliding_attention"

    rope_params = text_config.rope_parameters.get(key)
    if rope_params is not None:
        if rope_params.rope_type == "linear":
            return rope_params.factor
    return 1.0


def apply_rope(
    inputs: Float[Array, "B L N H"],  # noqa: F722
    positions: Float[Array, "B L"],  # noqa: F722
    *,
    base_frequency: int,
    scale_factor: float = 1.0,
    rope_proportion: float = 1.0,
) -> Float[Array, "B L N H"]:  # noqa: F722
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


class T5Gemma2Einsum(nnx.Module):
    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        kernel_init: nnx.Initializer = NORMAL_INIT,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.w = nnx.Param(kernel_init(rngs.params(), shape, dtype))

    @jax.named_scope("einsum")
    def __call__(self, eqn: str, x: jax.Array, *, out_sharding=None) -> jax.Array:
        return jnp.einsum(eqn, x, self.w[...], out_sharding=out_sharding)


class T5Gemma2RMSNorm(nnx.Module):
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

    @jax.named_scope("rms_norm")
    def __call__(self, x: Float[Array, "B L D"]) -> Float[Array, "B L D"]:  # noqa: F722
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed_inputs = x * jax.lax.rsqrt(var + self.epsilon)
        scale = jnp.expand_dims(self.scale[...], axis=range(len(x.shape) - 1))
        return normed_inputs * (1 + scale)


class T5Gemma2FeedForward(nnx.Module):
    def __init__(
        self,
        features: int,
        hidden_dim: int,
        *,
        transpose_gating_einsum: bool,
        shd_cfg: TextShardingCfg | None = None,
        kernel_init: nnx.Initializer = NORMAL_INIT,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.features = features
        self.hidden_dim = hidden_dim
        self.transpose_gating_einsum = transpose_gating_einsum
        self.shd_cfg = shd_cfg

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

    @jax.named_scope("feed_forward")
    def __call__(self, x: Float[Array, "B L D"]) -> Float[Array, "B L D"]:  # noqa: F722
        eq = "...F,NHF->...NH" if self.transpose_gating_einsum else "...F,NFH->...NH"
        gate = self.gating_einsum(eq, x)
        activations = jax.nn.gelu(gate[..., 0, :]) * gate[..., 1, :]
        shd = self.shd_cfg.activation if self.shd_cfg is not None else None
        return self.linear("...H,HF->...F", activations, out_sharding=shd)


def make_bidirectional_mask(
    input_mask: Bool[Array, "B L"],  # noqa: F722
) -> Bool[Array, "B 1 L L"]:  # noqa: F722
    mask = input_mask[:, None, None, :]
    return mask


def make_causal_mask(
    input_mask: Bool[Array, "B L"],  # noqa: F722
) -> Bool[Array, "B 1 L L"]:  # noqa: F722
    seq_len = input_mask.shape[-1]
    causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    mask = input_mask[:, None, None, :] & causal[None, None, :, :]
    return mask


def make_sliding_window_mask(
    positions: Int[Array, "B L"],  # noqa: F722
    sliding_window: int,
    *,
    bidirectional: bool = False,
) -> Bool[Array, "B 1 L L"]:  # noqa: F722
    q_pos = positions[:, :, None]
    k_pos = positions[:, None, :]

    if bidirectional:
        left_window = (sliding_window + 1) // 2
        right_window = sliding_window // 2 + 1
        dist = q_pos - k_pos
        left_mask = (dist >= 0) & (dist < left_window)
        right_mask = (dist < 0) & (-dist < right_window)
        mask = left_mask | right_mask
    else:
        dist = jnp.abs(q_pos - k_pos)
        mask = dist < sliding_window

    return mask[:, None, :, :]


def make_sliding_window_causal_mask(
    input_mask: Bool[Array, "B L"],  # noqa: F722
    positions: Int[Array, "B L"],  # noqa: F722
    sliding_window: int,
) -> Bool[Array, "B 1 L L"]:  # noqa: F722
    causal_mask = make_causal_mask(input_mask)
    sliding_mask = make_sliding_window_mask(positions, sliding_window)
    return causal_mask & sliding_mask


def make_merged_attention_mask(
    decoder_mask: Bool[Array, "B 1 L_dec L_dec"],  # noqa: F722
    encoder_mask: Bool[Array, "B 1 1 L_enc"],  # noqa: F722
) -> Bool[Array, "B 1 L_dec L_combined"]:  # noqa: F722
    batch_size, _, seq_len, _ = decoder_mask.shape
    enc_len = encoder_mask.shape[-1]
    cross_mask = jnp.broadcast_to(encoder_mask, (batch_size, 1, seq_len, enc_len))
    return jnp.concatenate([decoder_mask, cross_mask], axis=-1)


def make_decode_mode_self_mask(
    batch_size: int,
    query_len: int,
    max_cache_len: int,
    current_cache_len: int,
) -> Bool[Array, "B 1 Q K"]:  # noqa: F722
    k_pos = jnp.arange(max_cache_len)
    mask = k_pos < current_cache_len
    return jnp.broadcast_to(mask[None, None, None, :], (batch_size, 1, query_len, max_cache_len))


def make_decode_mode_sliding_mask(
    batch_size: int,
    query_pos: int,
    max_cache_len: int,
    sliding_window: int,
) -> Bool[Array, "B 1 1 K"]:  # noqa: F722
    k_positions = jnp.arange(max_cache_len)
    in_window = jnp.abs(query_pos - k_positions) < sliding_window
    is_causal = k_positions <= query_pos
    mask = in_window & is_causal
    return jnp.broadcast_to(mask[None, None, None, :], (batch_size, 1, 1, max_cache_len))


class T5Gemma2ScaledWordEmbedding(nnx.Module):
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
        embeddings = self.embedding[input_ids] * self.embed_scale
        eoi_mask = input_ids == END_OF_IMAGE_TOKEN
        return jnp.where(eoi_mask[..., None], self.eoi_embedding[...], embeddings)


def _posemb_sincos_2d(
    h: int,
    w: int,
    *,
    width: int,
    temperature: float = 10_000.0,
    dtype: jnp.dtype = jnp.float32,
) -> Float[Array, "1 M D"]:  # noqa: F722
    y, x = jnp.mgrid[:h, :w]

    assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
    omega = jnp.arange(width // 4) / (width // 4 - 1)
    omega = 1.0 / (temperature**omega)
    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)
    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
    return jnp.asarray(pe, dtype)[None, :, :]


class T5Gemma2VisionMLP(nnx.Module):
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
    """Spatially pools soft tokens to output length."""

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
    """Vision soft tokenizer (ViT trained with SigLiP)."""

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

        x = x + self.pos_embedding[...]
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
        mm_shd_cfg: MMShardingCfg | None = None,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.mm_shd_cfg = mm_shd_cfg
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
        shd = self.mm_shd_cfg.mmp_weight if self.mm_shd_cfg is not None else None
        return self.mm_input_projection("...tm,md->...td", x, out_sharding=shd)


class T5Gemma2VisionEmbedder(nnx.Module):
    """Vision embedder - tokenizes images and embeds into text space."""

    def __init__(
        self,
        *,
        vision_config: T5Gemma2VisionConfig,
        embed_dim: int,
        mm_shd_cfg: MMShardingCfg | None = None,
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
            mm_shd_cfg=mm_shd_cfg,
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


def merge_mm_embeddings(
    text_embeddings: jnp.ndarray,
    vision_embeddings: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
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
    num_images, num_toks_per_image, d = vision_embeddings.shape
    vision_embeddings = jnp.reshape(
        vision_embeddings,
        (num_images * num_toks_per_image, d),
    )

    target_pos = jnp.nonzero(mask, size=len(vision_embeddings))
    first_pos = text_embeddings[0]
    merged = text_embeddings.at[target_pos, :].set(vision_embeddings)
    return merged.at[0].set(first_pos)


class T5Gemma2EncoderAttention(nnx.Module):
    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        features: int,
        head_dim: int,
        attn_type: AttentionType,
        query_pre_attn_scalar: float,
        *,
        shd_cfg: TextShardingCfg | None = None,
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
        self.shd_cfg = shd_cfg

        self.attn_vec_einsum = T5Gemma2Einsum(
            shape=(num_q_heads, head_dim, features),
            kernel_init=kernel_init,
            dtype=dtype,
            rngs=rngs,
        )

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

    @jax.named_scope("encoder_attention")
    def __call__(
        self,
        x: Float[Array, "B L D"],  # noqa: F722
        segment_pos: Float[Array, "B L"],  # noqa: F722
        attn_mask: Float[Array, "B L L"],  # noqa: F722
    ) -> Float[Array, "B L D"]:  # noqa: F722
        shd = self.shd_cfg.activation if self.shd_cfg is not None else None

        if self.use_qkv_einsum:
            query_proj, key_proj, value_proj = self.qkv_einsum("BTD,SNDH->SBTNH", x)
        else:
            query_proj = self.q_einsum("BTD,NDH->BTNH", x, out_sharding=shd)
            key_proj, value_proj = self.kv_einsum("BSD,CKDH->CBSKH", x)

        if self.use_qk_norm:
            query_proj = self._query_norm(query_proj)
            key_proj = self._key_norm(key_proj)

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

        if self.attn_logits_soft_cap is not None:
            logits = jnp.tanh(logits / self.attn_logits_soft_cap)
            logits = logits * self.attn_logits_soft_cap

        if self.attn_type == AttentionType.LOCAL_SLIDING:
            if self.sliding_window_size is None:
                raise ValueError(
                    "sliding_window_size must be set for LOCAL_SLIDING attention",
                )
            sliding_mask = make_sliding_window_mask(
                segment_pos,
                self.sliding_window_size,
                bidirectional=True,
            )[:, 0, :, :]
            attn_mask = attn_mask * sliding_mask

        padded_logits = jnp.where(jnp.expand_dims(attn_mask, -2), logits, _K_MASK)
        probs = jax.nn.softmax(padded_logits, axis=-1).astype(key_proj.dtype)

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

        return self.attn_vec_einsum("BTNH,NHD->BTD", encoded, out_sharding=shd)


class T5Gemma2EncoderBlock(nnx.Module):
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
        shd_cfg: TextShardingCfg | None = None,
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
            shd_cfg=shd_cfg,
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
            shd_cfg=shd_cfg,
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

    @jax.named_scope("encoder_block")
    def __call__(
        self,
        x: Float[Array, "B L D"],  # noqa: F722
        segment_pos: Int[Array, "B L"],  # noqa: F722
        attn_mask: Bool[Array, "B L L"],  # noqa: F722
    ) -> Float[Array, "B L D"]:  # noqa: F722
        inputs_normalized = self.pre_attention_norm(x)
        attn_output = self.attn(inputs_normalized, segment_pos, attn_mask)

        if self.use_post_attn_norm:
            attn_output = self.post_attention_norm(attn_output)

        attn_output = attn_output + x

        outputs = self.pre_ffw_norm(attn_output)
        outputs = self.mlp(outputs)

        if self.use_post_ffw_norm:
            outputs = self.post_ffw_norm(outputs)

        return outputs + attn_output


class T5Gemma2Encoder(nnx.Module):
    def __init__(
        self,
        config: T5Gemma2EncoderConfig,
        *,
        mm_shd_cfg: MMShardingCfg | None = None,
        embedder: nnx.Module | None = None,
        dtype: jnp.dtype = jnp.float32,
        dtype_mm: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.config = config
        text_config = config.text_config
        shd_cfg = text_config.shd_cfg if text_config.shd_cfg.activation is not None else None

        self.embedder = embedder or T5Gemma2ScaledWordEmbedding(
            vocab_size=text_config.vocab_size,
            embed_dim=text_config.embed_dim,
            padding_idx=text_config.pad_token_id,
            dtype=dtype,
            rngs=rngs,
        )

        self.vision_embedder = nnx.data(None)
        if config.vision_config is not None:
            self.vision_embedder = T5Gemma2VisionEmbedder(
                vision_config=config.vision_config,
                embed_dim=text_config.embed_dim,
                mm_shd_cfg=mm_shd_cfg,
                freeze_params=True,
                dtype=dtype_mm,
                rngs=rngs,
            )

        attention_types = text_config.layer_types
        if not attention_types:
            attention_types = tuple(AttentionType.GLOBAL for _ in range(text_config.num_hidden_layers))

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
                    shd_cfg=shd_cfg,
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
        batch_size, seq_len = tokens.shape

        if position_ids is None:
            position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.bool_)

        x = self.embedder(tokens)

        if images is not None and self.vision_embedder is not None:
            image_embeddings = self.vision_embedder(images, deterministic=deterministic)
            x = merge_mm_embeddings(
                x,
                image_embeddings,
                tokens == IMAGE_PLACEHOLDER_TOKEN,
            )

        attn_mask = make_bidirectional_mask(attention_mask)
        if attn_mask.ndim == 4:
            attn_mask = attn_mask[:, 0, :, :]

        if attn_mask.shape[-2] == 1:
            attn_mask = jnp.broadcast_to(attn_mask, (batch_size, seq_len, seq_len))

        for block in self.blocks:
            x = block(x, position_ids, attn_mask)

        return self.norm(x)


class T5Gemma2MergedAttention(nnx.Module):
    """Merged self-attention and cross-attention for decoder.

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
        shd_cfg: TextShardingCfg | None = None,
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
        self.shd_cfg = shd_cfg

        self.num_kv_groups = num_q_heads // num_kv_heads

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

        self.cached_self_key: nnx.Cache[Array] | None = nnx.data(None)
        self.cached_self_value: nnx.Cache[Array] | None = nnx.data(None)
        self.cached_cross_key: nnx.Cache[Array] | None = nnx.data(None)
        self.cached_cross_value: nnx.Cache[Array] | None = nnx.data(None)
        self.cache_index: nnx.Cache[Array] | None = nnx.data(None)

    def init_cache(
        self,
        *,
        batch_size: int,
        max_decode_length: int,
        encoder_seq_length: int,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        cache_shd = self.shd_cfg.cache if self.shd_cfg is not None else None

        self_cache_shape = (
            batch_size,
            max_decode_length,
            self.num_kv_heads,
            self.head_dim,
        )
        self.cached_self_key = nnx.Cache(jnp.zeros(self_cache_shape, dtype, out_sharding=cache_shd))
        self.cached_self_value = nnx.Cache(jnp.zeros(self_cache_shape, dtype, out_sharding=cache_shd))

        cross_cache_shape = (
            batch_size,
            encoder_seq_length,
            self.num_kv_heads,
            self.head_dim,
        )
        self.cached_cross_key = nnx.Cache(jnp.zeros(cross_cache_shape, dtype, out_sharding=cache_shd))
        self.cached_cross_value = nnx.Cache(jnp.zeros(cross_cache_shape, dtype, out_sharding=cache_shd))

        self.cache_index = nnx.Cache(jnp.array(0, dtype=jnp.int32))

    @jax.named_scope("merged_attention")
    def __call__(
        self,
        hidden_states: Float[Array, "B L_dec D"],  # noqa: F722
        encoder_hidden_states: Float[Array, "B L_enc D"],  # noqa: F722
        position_ids: Int[Array, "B L_dec"],  # noqa: F722
        merged_attention_mask: Bool[Array, "B 1 L_dec L_combined"] | None = None,  # noqa: F722
        *,
        decode: bool | None = None,
    ) -> Float[Array, "B L_dec D"]:  # noqa: F722
        seq_len = hidden_states.shape[1]
        shd = self.shd_cfg.activation if self.shd_cfg is not None else None

        query = self.q_proj("BTD,NDH->BTNH", hidden_states, out_sharding=shd)
        self_key = self.k_proj("BTD,NDH->BTNH", hidden_states, out_sharding=shd)
        self_value = self.v_proj("BTD,NDH->BTNH", hidden_states, out_sharding=shd)

        if self.use_qk_norm:
            query = self.q_norm(query)
            self_key = self.k_norm(self_key)

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

        decode = first_from(
            decode,
            self.decode,
            error_msg="No decode argument provided to T5Gemma2MergedAttention",
        )

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

            self_key = self_key_updated
            self_value = self_value_updated

            def compute_cross_kv():
                cross_k = self.k_proj("BTD,NDH->BTNH", encoder_hidden_states, out_sharding=shd)
                cross_v = self.v_proj("BTD,NDH->BTNH", encoder_hidden_states, out_sharding=shd)
                if self.use_qk_norm:
                    cross_k = self.k_norm(cross_k)
                return cross_k, cross_v

            cross_key, cross_value = jax.lax.cond(
                cur_index == 0,
                compute_cross_kv,
                lambda: (self.cached_cross_key[...], self.cached_cross_value[...]),
            )

            self.cached_cross_key[...] = cross_key
            self.cached_cross_value[...] = cross_value

            self.cache_index[...] = cur_index + seq_len
        else:
            cross_key = self.k_proj("BTD,NDH->BTNH", encoder_hidden_states, out_sharding=shd)
            cross_value = self.v_proj("BTD,NDH->BTNH", encoder_hidden_states, out_sharding=shd)
            if self.use_qk_norm:
                cross_key = self.k_norm(cross_key)

        query = query * self.query_pre_attn_scalar

        key = jnp.concatenate([self_key, cross_key], axis=1)
        value = jnp.concatenate([self_value, cross_value], axis=1)

        query = jnp.transpose(query, (0, 2, 1, 3))  # [B, N, T, H]
        key = jnp.transpose(key, (0, 2, 1, 3))  # [B, K, S, H]
        value = jnp.transpose(value, (0, 2, 1, 3))  # [B, K, S, H]

        if self.num_kv_groups > 1:
            key = jnp.repeat(key, self.num_kv_groups, axis=1)
            value = jnp.repeat(value, self.num_kv_groups, axis=1)

        attn_weights = jnp.einsum("BNTH,BNSH->BNTS", query, key)

        if self.attn_logits_soft_cap is not None:
            attn_weights = jnp.tanh(attn_weights / self.attn_logits_soft_cap)
            attn_weights = attn_weights * self.attn_logits_soft_cap

        if merged_attention_mask is not None:
            attn_weights = jnp.where(merged_attention_mask, attn_weights, _K_MASK)

        attn_weights = jax.nn.softmax(attn_weights, axis=-1).astype(value.dtype)

        attn_output = jnp.einsum("BNTS,BNSH->BNTH", attn_weights, value)

        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))

        output = self.o_proj("BTNH,NHD->BTD", attn_output, out_sharding=shd)

        return output


class T5Gemma2DecoderBlock(nnx.Module):
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
        shd_cfg: TextShardingCfg | None = None,
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

        self.pre_attention_norm = T5Gemma2RMSNorm(
            embed_dim,
            epsilon=rms_norm_eps,
            dtype=dtype,
            scale_init=scale_init,
            rngs=rngs,
        )

        self.attn = T5Gemma2MergedAttention(
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            features=embed_dim,
            head_dim=head_dim,
            query_pre_attn_scalar=query_pre_attn_scalar,
            shd_cfg=shd_cfg,
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
            transpose_gating_einsum=True,
            shd_cfg=shd_cfg,
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

    @jax.named_scope("decoder_block")
    def __call__(
        self,
        x: Float[Array, "B L_dec D"],  # noqa: F722
        encoder_hidden_states: Float[Array, "B L_enc D"],  # noqa: F722
        segment_pos: Int[Array, "B L_dec"],  # noqa: F722
        attn_mask: Bool[Array, "B 1 L_dec L_combined"] | None = None,  # noqa: F722
        *,
        decode: bool | None = None,
    ) -> Float[Array, "B L_dec D"]:  # noqa: F722
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

        outputs = self.pre_ffw_norm(attn_output)
        outputs = self.mlp(outputs)

        if self.use_post_ffw_norm:
            outputs = self.post_ffw_norm(outputs)

        outputs += attn_output

        return outputs


class T5Gemma2Decoder(nnx.Module):
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
        shd_cfg = config.shd_cfg if config.shd_cfg.activation is not None else None

        self.embedder = embedder or T5Gemma2ScaledWordEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            padding_idx=config.pad_token_id,
            dtype=dtype,
            rngs=rngs,
        )

        attention_types = config.layer_types
        if not attention_types:
            attention_types = tuple(AttentionType.GLOBAL for _ in range(config.num_hidden_layers))

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
                    shd_cfg=shd_cfg,
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
        batch_size, dec_seq_len = input_ids.shape
        enc_seq_len = encoder_hidden_states.shape[1]

        if position_ids is None:
            cache_index = self.get_cache_index()
            if decode and cache_index > 0:
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

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, dec_seq_len), dtype=jnp.bool_)
        if encoder_attention_mask is None:
            encoder_attention_mask = jnp.ones(
                (batch_size, enc_seq_len),
                dtype=jnp.bool_,
            )

        encoder_mask = encoder_attention_mask[:, None, None, :]

        if decode:
            cache_index = self.get_cache_index()
            max_decode_len = self.blocks[0].attn.cached_self_key.shape[1]
            current_len = cache_index + dec_seq_len

            full_decoder_mask = make_decode_mode_self_mask(
                batch_size,
                dec_seq_len,
                max_decode_len,
                current_len,
            )
            sliding_decoder_mask = make_decode_mode_sliding_mask(
                batch_size,
                cache_index,
                max_decode_len,
                self.config.sliding_window,
            )

            enc_mask_broadcast = jnp.broadcast_to(encoder_mask, (batch_size, 1, dec_seq_len, enc_seq_len))
            merged_mask_full = jnp.concatenate([full_decoder_mask, enc_mask_broadcast], axis=-1)
            merged_mask_sliding = jnp.concatenate([sliding_decoder_mask, enc_mask_broadcast], axis=-1)
        else:
            full_decoder_mask = make_causal_mask(attention_mask)
            sliding_decoder_mask = make_sliding_window_causal_mask(
                attention_mask,
                position_ids,
                self.config.sliding_window,
            )

            merged_mask_full = make_merged_attention_mask(full_decoder_mask, encoder_mask)
            merged_mask_sliding = make_merged_attention_mask(sliding_decoder_mask, encoder_mask)

        hidden_states = self.embedder(input_ids)

        for block in self.blocks:
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

        hidden_states = self.norm(hidden_states)

        return hidden_states


class T5Gemma2(nnx.Module):
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

        self.embedder = T5Gemma2ScaledWordEmbedding(
            vocab_size=config.encoder.text_config.vocab_size,
            embed_dim=config.encoder.text_config.embed_dim,
            padding_idx=config.encoder.text_config.pad_token_id,
            dtype=dtype,
            rngs=rngs,
        )

        self.encoder = T5Gemma2Encoder(
            config.encoder,
            mm_shd_cfg=config.shd_cfg if config.shd_cfg.mmp_weight is not None else None,
            embedder=self.embedder,
            dtype=dtype,
            dtype_mm=dtype_mm,
            rngs=rngs,
        )

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
    ) -> tuple[Float[Array, "B L_dec V"], Float[Array, "B L_enc D"]]:  # noqa: F722
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                images=pixel_values,
                deterministic=deterministic,
            )

        decoder_outputs = self.decoder(
            decoder_input_ids,
            encoder_hidden_states=encoder_outputs,
            attention_mask=decoder_attention_mask,
            encoder_attention_mask=attention_mask,
            position_ids=decoder_position_ids,
            decode=decode,
        )

        embed_table = self.embedder.embedding[...]
        logits = jnp.einsum("btd,vd->btv", decoder_outputs, embed_table)
        return logits, encoder_outputs


@jax.jit
def forward(
    model: T5Gemma2,
    encoder_input_ids: Array,
    decoder_input_ids: Array,
    pixel_values: Array | None = None,
) -> tuple[Array, Array]:
    return model(encoder_input_ids, decoder_input_ids, pixel_values=pixel_values)
