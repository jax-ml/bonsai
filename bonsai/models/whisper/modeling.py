# whisper modeling

import dataclasses
import math

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, DTypeLike

# TODO: Properly use these to match reference implementation.
SUPPRESS_TOKENS: list[int] = [
    1,
    2,
    7,
    8,
    9,
    10,
    14,
    25,
    26,
    27,
    28,
    29,
    31,
    58,
    59,
    60,
    61,
    62,
    63,
    90,
    91,
    92,
    93,
    359,
    503,
    522,
    542,
    873,
    893,
    902,
    918,
    922,
    931,
    1350,
    1853,
    1982,
    2460,
    2627,
    3246,
    3253,
    3268,
    3536,
    3846,
    3961,
    4183,
    4667,
    6585,
    6647,
    7273,
    9061,
    9383,
    10428,
    10929,
    11938,
    12033,
    12331,
    12562,
    13793,
    14157,
    14635,
    15265,
    15618,
    16553,
    16604,
    18362,
    18956,
    20075,
    21675,
    22520,
    26130,
    26161,
    26435,
    28279,
    29464,
    31650,
    32302,
    32470,
    36865,
    42863,
    47425,
    49870,
    50254,
    50258,
    50358,
    50359,
    50360,
    50361,
    50362,
]


# TODO: Double check all the numbers
# So far, tests are all passing. Should still double check before final version.
@dataclasses.dataclass(frozen=True)
class ModelCfg:
    activation_dropout: float
    activation_function: str
    apply_spec_augment: bool
    architectures: list[str]
    attention_dropout: float
    begin_suppress_tokens: list[int]
    bos_token_id: int
    classifier_proj_size: int
    d_model: int
    decoder_attention_heads: int
    decoder_ffn_dim: int
    decoder_layerdrop: float
    decoder_layers: int
    decoder_start_token_id: int
    dropout: float
    dtype: str
    encoder_attention_heads: int
    encoder_ffn_dim: int
    encoder_layerdrop: float
    encoder_layers: int
    eos_token_id: int
    forced_decoder_ids: list[list[int]]
    init_std: float
    is_encoder_decoder: bool
    mask_feature_length: int
    mask_feature_min_masks: int
    mask_feature_prob: float
    mask_time_length: int
    mask_time_min_masks: int
    mask_time_prob: float
    max_length: int
    max_source_positions: int
    max_target_positions: int
    median_filter_width: int
    model_type: str
    num_hidden_layers: int
    num_mel_bins: int
    pad_token_id: int
    scale_embedding: bool
    use_cache: bool
    use_weighted_layer_sum: bool
    vocab_size: int

    @classmethod
    def whisper_tiny(cls):
        return cls(
            activation_dropout=0.0,
            activation_function="gelu",
            apply_spec_augment=False,
            architectures=["WhisperForConditionalGeneration"],
            attention_dropout=0.0,
            begin_suppress_tokens=[220, 50257],
            bos_token_id=50257,
            classifier_proj_size=256,
            d_model=384,
            decoder_attention_heads=6,
            decoder_ffn_dim=1536,
            decoder_layerdrop=0.0,
            decoder_layers=4,
            decoder_start_token_id=50258,
            dropout=0.0,
            dtype="float32",
            encoder_attention_heads=6,
            encoder_ffn_dim=1536,
            encoder_layerdrop=0.0,
            encoder_layers=4,
            eos_token_id=50257,
            forced_decoder_ids=[
                [1, 50259],
                [2, 50359],
                [3, 50363],
            ],
            init_std=0.02,
            is_encoder_decoder=True,
            mask_feature_length=10,
            mask_feature_min_masks=0,
            mask_feature_prob=0.0,
            mask_time_length=10,
            mask_time_min_masks=2,
            mask_time_prob=0.05,
            max_length=448,
            max_source_positions=1500,
            max_target_positions=448,
            median_filter_width=7,
            model_type="whisper",
            num_hidden_layers=4,
            num_mel_bins=80,
            pad_token_id=50257,
            scale_embedding=False,
            use_cache=True,
            use_weighted_layer_sum=False,
            vocab_size=51865,
        )


# this is only used in the EncoderLayer and DecoderLayer classes
# Slightly improves agreement between models.
# HF reference implementation uses multiple gelu implementations.
def custom_gelu(x: Array) -> Array:
    return x * 0.5 * (1.0 + jax.lax.erf(x / math.sqrt(2.0)))


class WhisperEncoderLayer(nnx.Module):
    def __init__(self, config: ModelCfg, *, rngs: nnx.Rngs):
        # TODO: Need to update the bias. This will work for now since 0 initialized
        self.self_attn = nnx.MultiHeadAttention(
            num_heads=config.encoder_attention_heads,
            in_features=config.d_model,
            out_features=config.d_model,
            use_bias=True,
            dropout_rate=config.dropout,
            rngs=rngs,
        )
        self.self_attn_layer_norm = nnx.LayerNorm(config.d_model, epsilon=1e-5, use_fast_variance=False, rngs=rngs)
        self.activation_fn = custom_gelu
        self.fc1 = nnx.Linear(config.d_model, config.encoder_ffn_dim, rngs=rngs)
        self.fc2 = nnx.Linear(config.encoder_ffn_dim, config.d_model, rngs=rngs)
        self.final_layer_norm = nnx.LayerNorm(config.d_model, epsilon=1e-5, use_fast_variance=False, rngs=rngs)

    def __call__(self, hidden_states: Array, attn_mask: Array | None) -> Array:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            mask=attn_mask,
            decode=False,
            deterministic=True,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class WhisperEncoder(nnx.Module):
    def __init__(self, config: ModelCfg, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(config.num_mel_bins, config.d_model, kernel_size=3, strides=1, padding=1, rngs=rngs)
        self.conv2 = nnx.Conv(config.d_model, config.d_model, kernel_size=3, strides=2, padding=1, rngs=rngs)
        self.embed_positions = nnx.Embed(config.max_source_positions, config.d_model, rngs=rngs)
        self.layers = nnx.List([WhisperEncoderLayer(config, rngs=rngs) for _ in range(config.encoder_layers)])
        self.layer_norm = nnx.LayerNorm(config.d_model, epsilon=1e-5, use_fast_variance=False, rngs=rngs)

    def __call__(self, input_features: Array, attention_mask: Array | None = None) -> Array:
        inputs_embeds = jax.nn.gelu(self.conv1(input_features))
        inputs_embeds = jax.nn.gelu(self.conv2(inputs_embeds))

        all_positions = jnp.arange(self.embed_positions.num_embeddings, device=inputs_embeds.device)
        hidden_states = inputs_embeds + self.embed_positions(all_positions)

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, None)

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class WhisperDecoderLayer(nnx.Module):
    def __init__(self, config: ModelCfg, *, rngs: nnx.Rngs):
        self.self_attn = nnx.MultiHeadAttention(
            num_heads=config.encoder_attention_heads,
            in_features=config.d_model,
            out_features=config.d_model,
            use_bias=True,
            dropout_rate=config.dropout,
            rngs=rngs,
        )
        self.activation_fn = custom_gelu
        self.self_attn_layer_norm = nnx.LayerNorm(config.d_model, epsilon=1e-5, use_fast_variance=False, rngs=rngs)
        self.encoder_attn = nnx.MultiHeadAttention(
            num_heads=config.encoder_attention_heads,
            in_features=config.d_model,
            out_features=config.d_model,
            use_bias=True,
            dropout_rate=config.dropout,
            rngs=rngs,
        )
        self.encoder_attn_layer_norm = nnx.LayerNorm(config.d_model, epsilon=1e-5, use_fast_variance=False, rngs=rngs)
        self.fc1 = nnx.Linear(config.d_model, config.decoder_ffn_dim, rngs=rngs)
        self.fc2 = nnx.Linear(config.decoder_ffn_dim, config.d_model, rngs=rngs)
        self.final_layer_norm = nnx.LayerNorm(config.d_model, epsilon=1e-5, use_fast_variance=False, rngs=rngs)

    def __call__(
        self,
        hidden_states: Array,
        self_mask: Array | None,
        encoder_hidden_states: Array,
        cross_mask: Array | None,
        decode: bool,
    ) -> Array:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(hidden_states, mask=self_mask, decode=decode, deterministic=True)
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            hidden_states = self.encoder_attn(
                inputs_q=hidden_states,
                inputs_k=encoder_hidden_states,
                inputs_v=encoder_hidden_states,
                mask=cross_mask,
                decode=decode,
                deterministic=True,
            )
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def init_cache(self, input_shape: tuple, dtype: DTypeLike):
        self.self_attn.init_cache(input_shape, dtype)
        self.encoder_attn.init_cache(input_shape, dtype)


class WhisperDecoder(nnx.Module):
    def __init__(self, config: ModelCfg, *, rngs: nnx.Rngs):
        self.embed_tokens = nnx.Embed(config.vocab_size, config.d_model, rngs=rngs)
        self.embed_positions = nnx.Embed(config.max_length, config.d_model, rngs=rngs)
        self.layers = nnx.List([WhisperDecoderLayer(config, rngs=rngs) for _ in range(config.decoder_layers)])
        self.layer_norm = nnx.LayerNorm(config.d_model, epsilon=1e-5, use_fast_variance=False, rngs=rngs)

    def __call__(
        self, input_ids: Array, self_mask: Array | None, encoder_hiddens: Array, cross_mask: Array | None, decode: bool
    ) -> Array:
        input_embeds = self.embed_tokens(input_ids)
        s = input_ids.shape[-1]
        pos_embeds = self.embed_positions(jnp.arange(s))

        hidden_state = input_embeds + pos_embeds
        for i, layer in enumerate(self.layers):
            hidden_state = layer(
                hidden_state,
                self_mask,
                encoder_hiddens,
                cross_mask,
                decode=decode,
            )

        hidden_state = self.layer_norm(hidden_state)
        return hidden_state


class Whisper(nnx.Module):
    def __init__(self, config: ModelCfg, *, rngs: nnx.Rngs):
        self.encoder = WhisperEncoder(config, rngs=rngs)
        self.decoder = WhisperDecoder(config, rngs=rngs)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Implement Whisper")
