import copy
import logging
import math
from typing import Optional, Union

import jax
import jax.numpy as jnp
from flax import nnx
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


class UMT5Config:
    """Configuration for UMT5 model."""

    def __init__(
        self,
        vocab_size=250112,
        d_model=512,
        d_kv=64,
        d_ff=1024,
        num_layers=8,
        num_decoder_layers=None,
        num_heads=6,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="gated-gelu",
        is_encoder_decoder=True,
        use_cache=True,
        tokenizer_class="T5Tokenizer",
        tie_word_embeddings=True,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
        is_decoder=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = (
            num_decoder_layers if num_decoder_layers is not None else self.num_layers
        )  # default = symmetry
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.use_cache = use_cache
        self.is_decoder = is_decoder

        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"

        if (len(act_info) > 1 and act_info[0] != "gated") or len(act_info) > 2:
            raise ValueError(
                f"`feed_forward_proj`: {feed_forward_proj} is not a valid activation function of the dense layer. "
                "Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
                "'gated-gelu' or 'relu'"
            )

        if self.dense_act_fn not in ACT_FN:
            raise ValueError(
                f"`feed_forward_proj`: {feed_forward_proj} is not a valid activation function of the dense layer. "
                f"Supported activation functions are: {', '.join(ACT_FN.keys())}"
            )


class UMT5DenseActDense(nnx.Module):
    def __init__(self, config: UMT5Config, *, param_dtype: jnp.dtype | None = jnp.float32, rngs: nnx.Rngs):
        super().__init__()
        self.param_dtype = param_dtype
        self.wi = nnx.Linear(
            config.d_model, config.d_ff, precision=Precision.HIGHEST, param_dtype=param_dtype, use_bias=False, rngs=rngs
        )
        self.wo = nnx.Linear(
            config.d_ff, config.d_model, precision=Precision.HIGHEST, param_dtype=param_dtype, use_bias=False, rngs=rngs
        )
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)
        self.act = ACT_FN[config.dense_act_fn]

    def __call___(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class UMT5DenseGatedActDense(nnx.Module):
    def __init__(self, config: UMT5Config, *, param_dtype: jnp.dtype | None = jnp.float32, rngs: nnx.Rngs):
        super().__init__()
        self.param_dtype = param_dtype
        self.wi_0 = nnx.Linear(
            config.d_model, config.d_ff, precision=Precision.HIGHEST, param_dtype=param_dtype, use_bias=False, rngs=rngs
        )
        self.wi_1 = nnx.Linear(
            config.d_model, config.d_ff, precision=Precision.HIGHEST, param_dtype=param_dtype, use_bias=False, rngs=rngs
        )
        self.wo = nnx.Linear(
            config.d_ff, config.d_model, precision=Precision.HIGHEST, param_dtype=param_dtype, use_bias=False, rngs=rngs
        )
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)
        self.act = ACT_FN[config.dense_act_fn]

    def __call__(self, hidden_states: jax.Array):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class UMT5LayerFF(nnx.Module):
    def __init__(self, config: UMT5Config, *, param_dtype: jnp.dtype | None = jnp.float32, rngs: nnx.Rngs):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = UMT5DenseGatedActDense(config, param_dtype=param_dtype, rngs=rngs)
        else:
            self.DenseReluDense = UMT5DenseActDense(config, param_dtype=param_dtype, rngs=rngs)

        self.layer_norm = nnx.RMSNorm(
            config.d_model, epsilon=config.layer_norm_epsilon, param_dtype=param_dtype, rngs=rngs
        )
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
        layer_idx: Optional[int] = None,
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
        encoder_hidden_states: Optional[jax.Array] = None,
        attention_mask: Optional[jax.Array] = None,
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
            assert attention_mask.ndim in [2, 3]
            if attention_mask.ndim == 2:
                attention_mask = attention_mask[:, None, None, :]
            else:
                attention_mask = attention_mask[:, None, :, :]
            position_bias = jnp.where(attention_mask == 0, jnp.finfo(hidden_states.dtype).min, position_bias)

        attn = (
            jnp.einsum(
                "binc,bjnc->bnij",
                q,
                k,
                precision=Precision.HIGHEST,
            )
            + position_bias
        )

        attn = jax.nn.softmax(attn, axis=-1)

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
        layer_idx: Optional[int] = None,
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
        self.layer_norm = nnx.RMSNorm(
            config.d_model, epsilon=config.layer_norm_epsilon, param_dtype=param_dtype, rngs=rngs
        )

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


class UMT5Block(nnx.Module):
    def __init__(
        self,
        config: UMT5Config,
        *,
        layer_idx: Optional[int] = None,
        param_dtype: jnp.dtype | None = jnp.float32,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nnx.List()
        self.layer.append(UMT5LayerSelfAttention(config, layer_idx=layer_idx, param_dtype=param_dtype, rngs=rngs))
        self.layer.append(UMT5LayerFF(config, param_dtype=param_dtype, rngs=rngs))

    def __call__(
        self,
        hidden_states: jax.Array,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        # Apply self-attention layer
        hidden_states = fp16_clamp(
            self.layer[0](
                hidden_states,
                attention_mask=attention_mask,
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
        self.final_layer_norm = nnx.RMSNorm(
            config.d_model, epsilon=config.layer_norm_epsilon, param_dtype=param_dtype, rngs=rngs
        )
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)

    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
    ):
        inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = self.dropout(inputs_embeds)

        for _, layer_module in enumerate(self.block):
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
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


__all__ = ["UMT5EncoderModel"]
