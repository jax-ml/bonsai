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

import copy
import dataclasses
import math
from typing import Union

import jax
import jax.numpy as jnp
from flax import nnx
from jax._src.typing import DTypeLike
from jax.lax import Precision

ACT_FN = {
    "gelu": nnx.gelu,
    "relu": nnx.relu,
}


def fp16_clamp(x: jax.Array):
    if x.dtype == jnp.float16 and jnp.isinf(x).any():
        clamp = jnp.finfo(x.dtype).max - 1000
        x = jax.lax.clamp(x=x, min=-clamp, max=clamp)
    return x


@dataclasses.dataclass
class UMT5Config:
    """Configuration for UMT5 model."""

    vocab_size: int = 250112
    d_model: int = 512
    d_kv: int = 64
    d_ff: int = 1024
    num_layers: int = 8
    num_decoder_layers: int = None
    num_heads: int = 6
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    initializer_factor: float = 1.0
    feed_forward_proj: str = "gated-gelu"
    is_encoder_decoder: bool = True
    use_cache: bool = True
    tokenizer_class: str = "T5Tokenizer"
    tie_word_embeddings: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 1
    decoder_start_token_id: int = 0
    is_decoder: bool = False
    dtype: DTypeLike = jnp.float32

    def __post_init__(self):
        self.num_decoder_layers = (
            self.num_decoder_layers if self.num_decoder_layers is not None else self.num_layers
        )  # default = symmetry

        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"

        if (len(act_info) > 1 and act_info[0] != "gated") or len(act_info) > 2:
            raise ValueError(
                f"`feed_forward_proj`: {self.feed_forward_proj} is not a valid activation function of the dense layer. "
                "Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
                "'gated-gelu' or 'relu'"
            )

        if self.dense_act_fn not in ACT_FN:
            raise ValueError(
                f"`feed_forward_proj`: {self.feed_forward_proj} is not a valid activation function of the dense layer. "
                f"Supported activation functions are: {', '.join(ACT_FN.keys())}"
            )


class T5LayerNorm(nnx.Module):
    def __init__(
        self,
        dim: int,
        *,
        eps=1e-6,
        param_dtype: jnp.dtype | None = jnp.float32,
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.scale = nnx.Param(jnp.ones(dim), dtype=param_dtype)

    def __call__(self, hidden_states: jax.Array):
        # RMS normalization: hidden_states / sqrt(mean(hidden_states^2))
        variance = jnp.mean(hidden_states.astype(jnp.float32) ** 2, axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.eps)
        weight_dtype = self.scale.dtype
        if weight_dtype in [jnp.float16, jnp.bfloat16]:
            hidden_states = hidden_states.astype(weight_dtype)
        return self.scale * hidden_states


class UMT5DenseActDense(nnx.Module):
    def __init__(
        self,
        config: UMT5Config,
        *,
        is_gated_act: bool = True,
        param_dtype: jnp.dtype | None = jnp.float32,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.param_dtype = param_dtype
        self.is_gated_act = is_gated_act
        if self.is_gated_act:
            self.wi_0 = nnx.Linear(
                config.d_model,
                config.d_ff,
                precision=Precision.HIGHEST,
                param_dtype=param_dtype,
                use_bias=False,
                rngs=rngs,
            )
            self.wi_1 = nnx.Linear(
                config.d_model,
                config.d_ff,
                precision=Precision.HIGHEST,
                param_dtype=param_dtype,
                use_bias=False,
                rngs=rngs,
            )

        else:
            self.wi = nnx.Linear(
                config.d_model,
                config.d_ff,
                precision=Precision.HIGHEST,
                param_dtype=param_dtype,
                use_bias=False,
                rngs=rngs,
            )
        self.wo = nnx.Linear(
            config.d_ff, config.d_model, precision=Precision.HIGHEST, param_dtype=param_dtype, use_bias=False, rngs=rngs
        )
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)
        self.act = ACT_FN[config.dense_act_fn]

    def __call__(self, hidden_states: jax.Array):
        if self.is_gated_act:
            hidden_states = self.act(self.wi_0(hidden_states)) * self.wi_1(hidden_states)
        else:
            hidden_states = self.act(self.wi(hidden_states))

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class UMT5LayerFF(nnx.Module):
    def __init__(self, config: UMT5Config, *, param_dtype: jnp.dtype | None = jnp.float32, rngs: nnx.Rngs):
        super().__init__()
        self.DenseReluDense = UMT5DenseActDense(
            config, is_gated_act=config.is_gated_act, param_dtype=param_dtype, rngs=rngs
        )
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon, param_dtype=param_dtype)
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)

    def __call__(self, hidden_states: jax.Array):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class UMT5Attention(nnx.Module):
    """
    T5's attention using relative_attention_bias.
    """

    def __init__(
        self,
        config: UMT5Config,
        *,
        has_relative_attention_bias=False,
        layer_idx: int | None = None,
        param_dtype: jnp.dtype | None = jnp.float32,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        self.q = nnx.Linear(
            self.d_model,
            self.inner_dim,
            precision=Precision.HIGHEST,
            param_dtype=param_dtype,
            use_bias=False,
            rngs=rngs,
        )
        self.k = nnx.Linear(
            self.d_model,
            self.inner_dim,
            precision=Precision.HIGHEST,
            param_dtype=param_dtype,
            use_bias=False,
            rngs=rngs,
        )
        self.v = nnx.Linear(
            self.d_model,
            self.inner_dim,
            precision=Precision.HIGHEST,
            param_dtype=param_dtype,
            use_bias=False,
            rngs=rngs,
        )
        self.o = nnx.Linear(
            self.inner_dim,
            self.d_model,
            precision=Precision.HIGHEST,
            param_dtype=param_dtype,
            use_bias=False,
            rngs=rngs,
        )

        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nnx.Embed(
                self.relative_attention_num_buckets, self.n_heads, param_dtype=param_dtype, rngs=rngs
            )

    def _relative_position_bucket(self, rel_pos):
        """Convert relative positions to bucket indices."""
        if not self.is_decoder:
            num_buckets = self.relative_attention_num_buckets // 2
            rel_buckets = (rel_pos > 0).astype(jnp.int32) * num_buckets
            rel_pos = jnp.abs(rel_pos)
        else:
            num_buckets = self.relative_attention_num_buckets
            rel_buckets = 0
            rel_pos = -jnp.minimum(rel_pos, jnp.zeros_like(rel_pos))

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = rel_pos < max_exact

        # Logarithmic bucketing for large positions
        rel_pos_large = max_exact + (
            jnp.log(rel_pos.astype(jnp.float32) / max_exact)
            / math.log(self.relative_attention_max_distance / max_exact)
            * (num_buckets - max_exact)
        ).astype(jnp.int32)
        rel_pos_large = jnp.minimum(rel_pos_large, num_buckets - 1)

        rel_buckets = rel_buckets + jnp.where(is_small, rel_pos, rel_pos_large)
        return rel_buckets

    def compute_bias(self, q_len, k_len):
        """Compute binned relative position bias"""
        ctx_pos = jnp.arange(q_len, dtype=jnp.int32)[:, None]
        mem_pos = jnp.arange(k_len, dtype=jnp.int32)[None, :]
        rel_pos = mem_pos - ctx_pos  # shape (query_length, key_length)
        rel_pos_bkt = self._relative_position_bucket(rel_pos)
        values = self.relative_attention_bias(rel_pos_bkt)  # shape (query_length, key_length, num_heads)
        # shape (1, num_heads, query_length, key_length)
        values = values.transpose(2, 0, 1)[None, :, :, :]
        return values

    def __call__(
        self,
        hidden_states: jax.Array,
        encoder_hidden_states: jax.Array | None = None,
        attention_mask: jax.Array | None = None,
    ):
        b, n, c = hidden_states.shape[0], self.n_heads, self.key_value_proj_dim

        # if encoder_hidden_states are provided this layer is used as a cross-attention layer for the decoder
        is_cross_attention = encoder_hidden_states is not None
        current_states = encoder_hidden_states if is_cross_attention else hidden_states

        q = self.q(hidden_states).reshape(b, -1, n, c)
        k = self.k(current_states).reshape(b, -1, n, c)
        v = self.v(current_states).reshape(b, -1, n, c)

        # Attention bias
        q_len, k_len = q.shape[1], k.shape[1]
        if not self.has_relative_attention_bias:
            position_bias = jnp.zeros((1, n, q_len, k_len), dtype=q.dtype)
        else:
            position_bias = self.compute_bias(q_len, k_len)
            position_bias = position_bias[:, :, -q_len:, :]

        if attention_mask is not None:
            position_bias = position_bias + attention_mask

        attn = (
            jnp.einsum(
                "binc,bjnc->bnij",
                q,
                k,
                precision=Precision.HIGHEST,
            )
            + position_bias
        )

        attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(attn.dtype)

        attn = self.dropout(attn)

        o_attn = jnp.einsum(
            "bnij,bjnc->binc",
            attn,
            v,
            precision=Precision.HIGHEST,
        )

        o_attn = o_attn.reshape(b, -1, n * c)
        o_attn = self.o(o_attn)
        o_attn = self.dropout(o_attn)
        return o_attn


class UMT5LayerSelfAttention(nnx.Module):
    def __init__(
        self,
        config: UMT5Config,
        *,
        layer_idx: int | None = None,
        param_dtype: jnp.dtype | None = jnp.float32,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.SelfAttention = UMT5Attention(
            config,
            has_relative_attention_bias=True,
            layer_idx=layer_idx,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon, param_dtype=param_dtype)

    def __call__(
        self,
        hidden_states: jax.Array,
        attention_mask=None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            attention_mask=attention_mask,
        )
        outputs = hidden_states + attention_output
        return outputs


class UMT5LayerCrossAttention(nnx.Module):
    def __init__(
        self,
        config: UMT5Config,
        *,
        layer_idx: int | None = None,
        param_dtype: jnp.dtype | None = jnp.float32,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.EncDecAttention = UMT5Attention(
            config, has_relative_attention_bias=False, layer_idx=layer_idx, param_dtype=param_dtype, rngs=rngs
        )
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon, param_dtype=param_dtype)
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)

    def __call__(
        self,
        hidden_states: jax.Array,
        encoder_hidden_states: jax.Array = None,
        attention_mask: jax.Array = None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )
        return hidden_states + self.dropout(attention_output)


class UMT5Block(nnx.Module):
    def __init__(
        self,
        config: UMT5Config,
        *,
        layer_idx: int | None = None,
        param_dtype: jnp.dtype | None = jnp.float32,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nnx.List()
        self.layer.append(UMT5LayerSelfAttention(config, layer_idx=layer_idx, param_dtype=param_dtype, rngs=rngs))
        if self.is_decoder:
            self.layer.append(UMT5LayerCrossAttention(config, layer_idx=layer_idx, param_dtype=param_dtype, rngs=rngs))

        self.layer.append(UMT5LayerFF(config, param_dtype=param_dtype, rngs=rngs))

    def __call__(
        self,
        hidden_states: jax.Array,
        attention_mask: jax.Array = None,
        encoder_hidden_states: jax.Array = None,
        encoder_attention_mask: jax.Array = None,
    ):
        # Apply self-attention layer
        hidden_states = fp16_clamp(
            self.layer[0](
                hidden_states,
                attention_mask=attention_mask,
            )
        )
        # Cross-Attention Block
        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            hidden_states = fp16_clamp(
                self.layer[1](
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                )
            )

        # Apply Feed Forward layer
        hidden_states = fp16_clamp(self.layer[-1](hidden_states))

        return (hidden_states,)


class UMT5Stack(nnx.Module):
    def __init__(self, config: UMT5Config, *, param_dtype: jnp.dtype | None = jnp.float32, rngs: nnx.Rngs):
        super().__init__()
        self.embed_tokens = nnx.Embed(config.vocab_size, config.d_model, param_dtype=param_dtype, rngs=rngs)
        self.is_decoder = config.is_decoder
        self.block = nnx.List(
            [UMT5Block(config, layer_idx=i, param_dtype=param_dtype, rngs=rngs) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon, param_dtype=param_dtype)
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)

    def _prepare_4d_causal_attention_mask_for_decoder(
        self,
        attention_mask: jax.Array,
        batch_size: int,
        q_len: int,
        dtype: DTypeLike,
    ):
        if attention_mask is None:
            causal_mask = jnp.tril(jnp.ones((batch_size, 1, q_len, q_len)))
        elif attention_mask.ndim == 4:
            causal_mask = attention_mask
        elif attention_mask.ndim == 2:
            # shape of causal_mask is (batch_size, q_len), expand to (batch_size, 1, q_len, q_len)
            causal_mask = jnp.tril(attention_mask[:, None, None, :].repeat(q_len, axis=2))
        else:
            raise ValueError(f"Invalid attention mask ndim, expected ndim: 2, actual ndim: {attention_mask.ndim}")
        causal_mask = (1.0 - causal_mask) * jnp.finfo(dtype).min
        return causal_mask

    def _prepare_padding_mask(
        self,
        padding_mask: jax.Array,
        dtype: DTypeLike,
    ):
        """
        For decoder, padding mask is encoder attention mask. For encoder, padding mask is input mask from tokenizer.
        """
        assert padding_mask.ndim in [2, 3], f"Invalid padding mask ndim: {padding_mask.ndim}"

        # expand dim if needed
        if padding_mask.ndim == 3:
            # shape of encoder_attention_mask is (batch_size, seq_len, seq_len), expand to (batch_size, 1, seq_len, seq_len)
            padding_mask = padding_mask[:, None, :, :]
        elif padding_mask.ndim == 2:
            # shape of encoder_attention_mask is (batch_size, seq_len), expand to (batch_size, 1, 1, seq_len)
            padding_mask = padding_mask[:, None, None, :]

        # convert to additive biases
        padding_mask = (1.0 - padding_mask) * jnp.finfo(dtype).min
        return padding_mask

    def _prepare_attention_mask(
        self,
        input_ids,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        dtype: DTypeLike,
    ):
        """
        Prepare attention mask. Only SelfAttention mask is needed in encoder. Both SelfAttention mask and CrossAttention mask are needed in decoder.
        Args:
            input_ids: Indices of input sequence tokens in the vocabulary. shape: (batch_size, seq_len).
            attention_mask: causal mask of input_ids.
            encoder_hidden_states: The last hidden states of encoder output.
            encoder_attention_mask: The causal mask of encoder_hidden_states
        """
        if self.is_decoder:
            b, s = input_ids.shape
            # prepare self-attention causal mask for decoder
            causal_mask = self._prepare_4d_causal_attention_mask_for_decoder(attention_mask, b, s, dtype)
            # prepare cross-attention causal mask for decoder
            if encoder_hidden_states is not None:
                b, s, _ = encoder_hidden_states.shape
                # new mask if not provided
                if encoder_attention_mask is None:
                    encoder_attention_mask = jnp.ones((b, s), dtype=jnp.int32)
                encoder_attention_mask = self._prepare_padding_mask(encoder_attention_mask, dtype)
            else:
                encoder_attention_mask = None
        elif attention_mask is not None:
            # prepare padding mask for encoder
            causal_mask = self._prepare_padding_mask(attention_mask, dtype)
        else:
            causal_mask = None

        return causal_mask, encoder_attention_mask

    def __call__(
        self,
        input_ids: jax.Array = None,
        attention_mask: jax.Array = None,
        encoder_hidden_states: jax.Array = None,
        encoder_attention_mask: jax.Array = None,
    ):
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = self.dropout(inputs_embeds)

        # prepare attention mask for encoder and decoder
        causal_mask, encoder_attention_mask = self._prepare_attention_mask(
            input_ids, attention_mask, encoder_hidden_states, encoder_attention_mask, inputs_embeds.dtype
        )

        for _, layer_module in enumerate(self.block):
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=causal_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )
            hidden_states = layer_outputs[0]
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class UMT5EncoderModel(nnx.Module):
    def __init__(
        self,
        config: UMT5Config,
        *,
        param_dtype: jnp.dtype | None = jnp.float32,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        config.is_decoder = False
        self.encoder = UMT5Stack(config, param_dtype=param_dtype, rngs=rngs)

    def __call__(
        self,
        input_ids: jax.Array = None,
        attention_mask: jax.Array = None,
    ) -> Union[jax.Array]:
        r"""
        input_ids (`jax.Array` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. UMT5 is a model with relative position embeddings so you
            should be able to pad the inputs on both the right and the left.

            Indices can be obtained using [`AutoTokenizer`].

            To know more on how to prepare `input_ids` for pretraining take a look a [UMT5 Training](./umt5#training).
        ```"""
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return encoder_outputs


class UMT5Model(nnx.Module):
    def __init__(
        self,
        config: UMT5Config,
        *,
        param_dtype: jnp.dtype | None = jnp.float32,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.config = config
        self.model_dim = config.d_model

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        self.encoder = UMT5Stack(encoder_config, param_dtype=param_dtype, rngs=rngs)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = UMT5Stack(decoder_config, param_dtype=param_dtype, rngs=rngs)

        self.lm_head = nnx.Linear(
            config.d_model,
            config.vocab_size,
            use_bias=False,
            precision=Precision.HIGHEST,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: jax.Array | None = None,
        attention_mask: jax.Array | None = None,
        decoder_input_ids: jax.Array | None = None,
        decoder_attention_mask: jax.Array | None = None,
        encoder_outputs: jax.Array | None = None,
    ) -> jax.Array:
        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        hidden_states = encoder_outputs

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
        )

        return decoder_outputs

    # TODO(#96): Implement KV Cache for efficient inference
    def generate(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array = None,
        max_tokens: int | None = None,
        max_new_tokens: int | None = None,
    ) -> jax.Array:
        """Generate sequences using greedy decoding.

        Args:
            input_ids: Encoder input ids from tokenizer, shape (batch_size, seq_length)
            attention_mask: Encoder attention mask, shape (batch_size, seq_length)
            max_tokens: Maximum total length of decoder sequence (including start token).
                       Takes precedence over max_new_tokens if both are provided.
            max_new_tokens: Maximum number of new tokens to generate (excluding start token)

        Returns:
            Generated token ids, shape (batch_size, generated_length)
        """
        # Determine maximum generation length
        if max_tokens is not None:
            max_length = max_tokens
        elif max_new_tokens is not None:
            max_length = max_new_tokens + 1  # +1 for decoder_start_token
        else:
            max_length = 512  # default value

        # Encode input
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Initialize decoder input with start token
        batch_size = input_ids.shape[0]
        decoder_input_ids = jnp.full((batch_size, 1), self.config.decoder_start_token_id, dtype=jnp.int32)

        # Autoregressive generation loop
        for _ in range(max_length - 1):
            # Decoder forward pass
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs,
            )

            # Get logits and select next token (greedy)
            logits = self.lm_head(decoder_outputs)
            # here use simple greedy, but beem search is recommended
            next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)

            # Append to decoder input
            decoder_input_ids = jnp.concatenate([decoder_input_ids, next_token], axis=1)

            # Stop if all sequences generated EOS
            if jnp.all(next_token == self.config.eos_token_id):
                break

        return decoder_input_ids


@jax.jit
def forward(
    graphdef: nnx.GraphDef,
    state: nnx.State,
    *,
    input_ids: jax.Array | None = None,
    attention_mask: jax.Array | None = None,
    decoder_input_ids: jax.Array | None = None,
    decoder_attention_mask: jax.Array | None = None,
    encoder_outputs: jax.Array | None = None,
) -> jax.Array:
    model = nnx.merge(graphdef, state)
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        encoder_outputs=encoder_outputs,
    )


__all__ = ["UMT5EncoderModel", "UMT5Model"]
