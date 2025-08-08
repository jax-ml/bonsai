# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import Callable, NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
from flax import nnx


class ActivationType(Enum):
    gelu = auto()
    relu = auto()
    silu = auto()
    swiglu = auto()


class ActSpec(NamedTuple):
    fn: Callable[[jax.Array], jax.Array]
    output_multiplier: float


def get_activation(act_type: ActivationType) -> ActSpec:
    """Return (callable, multiplier) matching LLaDA conventions."""
    if act_type == ActivationType.gelu:
        return ActSpec(nnx.gelu, 1.0)

    if act_type == ActivationType.relu:
        return ActSpec(nnx.relu, 1.0)

    if act_type == ActivationType.silu:
        return ActSpec(nnx.silu, 1.0)

    if act_type == ActivationType.swiglu:

        def swiglu(x: jax.Array) -> jax.Array:
            a, b = jnp.split(x, 2, axis=-1)  # split last dim
            return nnx.silu(b) * a

        return ActSpec(swiglu, 0.5)
    raise ValueError(f"Unknown activation: {act_type}")


class BlockType(str, Enum):
    """
    Which transformer-block wiring to use.

    1. SEQUENTIAL  - fused Q / K / V projection + vanilla GELU FFN
    2. LLAMA       - split Q / K / V + SwiGLU FFN (Llama-style)
    """

    SEQUENTIAL = "sequential"
    LLAMA = "llama"


class LayerNormType(Enum):
    DEFAULT = auto()
    RMS = auto()
    GEMMA_RMS = auto()


class GemmaRMSNorm(nnx.Module):
    def __init__(self, dim: int, epsilon: float = 1e-5, use_bias: bool = False, *, rngs: nnx.Rngs):
        self.eps = epsilon
        # initialise weight at zero so overall scale starts at 1
        self.scale = nnx.Param(jnp.zeros((dim,)), rngs=rngs)
        if use_bias:
            self.bias = nnx.Param(jnp.zeros((dim,)), rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        y = x * jax.lax.rsqrt(var + self.eps)
        if hasattr(self, "weight"):
            y = y * (1.0 + self.scale.value)
        if hasattr(self, "bias"):
            y = y + self.bias.value
        return y.astype(x.dtype)


def get_layer_norm(cfg, dim: int | None = None, *, rngs: nnx.Rngs):
    """Return an nnx.Module that implements the requested norm."""
    dim = dim or cfg.d_model
    eps = getattr(cfg, "rms_norm_eps", 1e-6)

    ln_type = cfg.layer_norm_type
    if ln_type == LayerNormType.DEFAULT:
        return nnx.LayerNorm(dim, epsilon=eps, rngs=rngs)

    if ln_type == LayerNormType.RMS:
        return nnx.RMSNorm(dim, epsilon=eps, rngs=rngs)

    if ln_type == LayerNormType.GEMMA_RMS:
        return GemmaRMSNorm(
            dim,
            epsilon=eps,
            use_bias=getattr(cfg, "bias_for_layer_norm", False),
            rngs=rngs,
        )

    raise ValueError(f"Unknown LayerNorm type: {ln_type}")


@dataclass
class ModelConfig:
    """
    Configuration for a LLaDA-style transformer.

    The defaults reproduce the original GPT-2 base model hyper-parameters.
    """

    # core dimensions
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int | None = None  # None ⇒ same as n_heads
    n_layers: int = 12

    # feed-forward
    mlp_ratio: int = 4
    mlp_hidden_size: int | None = None  # fagraphdef, state, tokens, pad_idlls back to mlp_ratio * d_model
    activation_type: ActivationType = ActivationType.swiglu

    # block wiring / attention
    block_type: BlockType = BlockType.SEQUENTIAL
    block_group_size: int = 1
    alibi: bool = False
    alibi_bias_max: float = 8.0
    rope: bool = False
    rope_full_precision: bool = True
    rope_theta: float = 10_000.0
    flash_attention: bool = False
    attention_dropout: float = 0.1
    multi_query_attention: bool | None = None  # auto-derived if None
    attention_layer_norm: bool = False
    attention_layer_norm_with_affine: bool = True

    # dropout / normalisation
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1
    input_emb_norm: bool = False
    layer_norm_type: LayerNormType = LayerNormType.DEFAULT
    layer_norm_with_affine: bool = True
    bias_for_layer_norm: bool | None = None
    rms_norm_eps: float = 1e-5

    # sequence / vocab
    max_sequence_length: int = 1_024
    vocab_size: int = 50_257
    embedding_size: int | None = 50_304

    # tokens
    eos_token_id: int = 50_256
    pad_token_id: int = 50_256
    mask_token_id: int | None = 50_256

    include_bias: bool = False
    include_qkv_bias: bool = False
    weight_tying: bool = True
    scale_logits: bool = False
    use_cache: bool = False  # can be wired in later

    # helpers
    @property
    def effective_n_kv_heads(self) -> int:
        """Derive number of K/V heads based on MQA / GQA flags."""
        if self.n_kv_heads is None:
            return 1 if self.multi_query_attention else self.n_heads
        if self.multi_query_attention is None:
            return self.n_kv_heads
        # explicit sanity-check
        expected = 1 if self.multi_query_attention else self.n_heads
        if self.n_kv_heads != expected:
            raise ValueError("Cannot set both `multi_query_attention` and `n_kv_heads` inconsistently.")
        return self.n_kv_heads

    @classmethod
    def llada_8b_instruct(cls):
        """Return the configuration that matches the 8-B-parameter LLaDA-Instruct checkpoint."""
        return cls(
            # architecture
            activation_type=ActivationType.silu,
            block_type=BlockType.LLAMA,
            d_model=4_096,
            n_layers=32,
            n_heads=32,
            n_kv_heads=32,
            mlp_hidden_size=12_288,
            rope=True,
            rope_theta=500_000.0,
            max_sequence_length=4_096,
            # dropouts / norms
            embedding_dropout=0.0,
            residual_dropout=0.0,
            attention_dropout=0.0,
            layer_norm_type=LayerNormType.RMS,
            rms_norm_eps=1e-5,
            attention_layer_norm=False,
            attention_layer_norm_with_affine=True,
            # embeddings & vocab
            vocab_size=126_464,
            embedding_size=126_464,
            # special tokens
            eos_token_id=126_081,
            pad_token_id=126_081,
            mask_token_id=126_336,
            # other flags
            flash_attention=False,
            include_bias=False,
            include_qkv_bias=False,
            weight_tying=False,
            alibi=False,
            scale_logits=False,
        )


class RotaryEmbedding(nnx.Module):
    """
    Rotary positional embeddings (RoPE) - JAX/nnx version.
    Keeps a sin/cos cache in nnx.Cache so it can grow on demand.
    """

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        dim = cfg.d_model // cfg.n_heads
        sin, cos = self._make_table(cfg.max_sequence_length, dim, cfg.rope_theta)
        # store as non-trainable cache
        self.pos_sin = nnx.Cache(sin)  # (1,L,1,dim)
        self.pos_cos = nnx.Cache(cos)

    def __call__(self, q: jax.Array, k: jax.Array) -> tuple[jax.Array, jax.Array]:
        """
        Args:
            q, k : (B, T, H, Dh)   - query / key after head reshape
        Returns:
            q_rot, k_rot : same shapes, after RoPE applied
        """
        T_q, T_k = q.shape[-3], k.shape[-3]

        qf, kf = q, k

        # ensure cache large enough
        sin, cos = self._get_tables(T_k, qf.dtype)
        # slice to match lengths
        sin_q, cos_q = sin[:, T_k - T_q : T_k, :, :], cos[:, T_k - T_q : T_k, :, :]

        q_out = self._apply_rotary(qf, sin_q, cos_q)
        k_out = self._apply_rotary(kf, sin, cos)

        return q_out.astype(q.dtype), k_out.astype(k.dtype)

    def _get_tables(self, seq_len: int, dtype) -> tuple[jax.Array, jax.Array]:
        sin, cos = self.pos_sin.value, self.pos_cos.value
        if sin.shape[-2] < seq_len:
            # grow cache
            dim = sin.shape[-1]
            sin, cos = self._make_table(seq_len, dim, self.cfg.rope_theta, dtype=dtype)
            self.pos_sin.value, self.pos_cos.value = sin, cos
        return sin[..., :seq_len, :], cos[..., :seq_len, :]

    @staticmethod
    def _make_table(L: int, dim: int, theta: float, dtype=jnp.float32):
        inv_freq = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=dtype) / dim))
        t = jnp.arange(L, dtype=dtype)
        freqs = jnp.einsum("i,j->ij", t, inv_freq)
        emb = jnp.concatenate([freqs, freqs], axis=-1)  # (L,dim)
        sin, cos = jnp.sin(emb)[None, :, None, :], jnp.cos(emb)[None, :, None, :]
        return sin, cos  # (1,L,1,dim)

    @staticmethod
    def _rotate_half(x):
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([-x2, x1], axis=-1)

    def _apply_rotary(self, t, sin, cos):
        # sin/cos broadcast on (B,T,H,Dh)
        return (t * cos) + (self._rotate_half(t) * sin)


KVCache: TypeAlias = tuple[jax.Array, ...]


class LLaDABlock(nnx.Module):
    """
    nnx implementation that covers the two projection/MLP variants.

    * Sequential  → fused QKV, plain FFN
    * Llama       → split Q K V, SwiGLU FFN
    Behaviour is chosen by `cfg.block_type`.
    """

    def __init__(self, layer_id: int, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.layer_id = layer_id
        hd = cfg.d_model // cfg.n_heads
        kv_hd = hd

        self.hidden_size = cfg.mlp_hidden_size or cfg.mlp_ratio * cfg.d_model

        # Dropout
        self.dropout = nnx.Dropout(self.cfg.residual_dropout)

        # Layer norms
        self.q_norm = None
        self.k_norm = None
        if cfg.attention_layer_norm:
            self.q_norm = get_layer_norm(
                cfg,
                dim=(cfg.d_model // cfg.n_heads) * cfg.effective_n_kv_heads,
                rngs=rngs,
            )
            self.k_norm = get_layer_norm(cfg, rngs=rngs)

        self.attn_norm = get_layer_norm(cfg, rngs=rngs)
        self.ff_norm = get_layer_norm(cfg, rngs=rngs)

        # Activation function
        self.act, self.act_output_mult = get_activation(cfg.activation_type)
        assert (self.act_output_mult * self.hidden_size) % 1 == 0

        # Feed forward input projection
        self.ff_proj = nnx.Linear(cfg.d_model, self.hidden_size, use_bias=cfg.include_bias, rngs=rngs)

        # Feed-forward output projection
        self.ff_out = nnx.Linear(int(self.act_output_mult * self.hidden_size), cfg.d_model, rngs=rngs)

        # Block specifics
        if cfg.block_type == BlockType.SEQUENTIAL:
            fused_out = cfg.d_model + 2 * cfg.effective_n_kv_heads * kv_hd
            self.att_proj = nnx.Linear(
                cfg.d_model,
                fused_out,
                use_bias=cfg.include_bias | cfg.include_qkv_bias,
                rngs=rngs,
            )
        else:  # llama style
            self.q_proj = nnx.Linear(
                cfg.d_model,
                cfg.d_model,
                use_bias=cfg.include_bias | cfg.include_qkv_bias,
                rngs=rngs,
            )
            self.k_proj = nnx.Linear(
                cfg.d_model,
                cfg.effective_n_kv_heads * kv_hd,
                use_bias=cfg.include_bias | cfg.include_qkv_bias,
                rngs=rngs,
            )
            self.v_proj = nnx.Linear(
                cfg.d_model,
                cfg.effective_n_kv_heads * kv_hd,
                use_bias=cfg.include_bias | cfg.include_qkv_bias,
                rngs=rngs,
            )

        self.attn_out = nnx.Linear(cfg.d_model, cfg.d_model, use_bias=cfg.include_bias, rngs=rngs)

        if cfg.block_type is BlockType.LLAMA:
            self.up_proj = nnx.Linear(cfg.d_model, self.hidden_size, use_bias=False, rngs=rngs)

        # rotary embeddings (optional)
        if cfg.rope:
            self.rotary = RotaryEmbedding(cfg)

    def attention(
        self,
        q: jax.Array,  # (B,T,D)
        k: jax.Array,  # (B,T,Dk)
        v: jax.Array,  # (B,T,Dk)
        attention_bias: jax.Array | None = None,
        layer_past: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array] | None]:
        B, T, C = q.shape
        H, Kh, Dh = self.cfg.n_heads, self.cfg.effective_n_kv_heads, C // self.cfg.n_heads

        # Optional per-head RMSNorm on Q, K
        if self.cfg.attention_layer_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # reshape to (B,T,H,Dh)
        q = q.reshape(B, T, H, Dh)
        k = k.reshape(B, T, Kh, Dh)
        v = v.reshape(B, T, Kh, Dh)

        # append past KV if supplied
        if layer_past is not None:
            past_k, past_v = layer_past
            k = jnp.concatenate([past_k, k], axis=1)
            v = jnp.concatenate([past_v, v], axis=1)
        present_kv = (k, v) if use_cache else None

        # rotary
        if self.cfg.rope:
            q, k = self.rotary(q, k)  # RotaryEmbedding module handles dtype

        # broadcast KV for GQA / MQA
        if H != Kh:
            repeat = H // Kh
            k = jnp.repeat(k, repeat, axis=2)
            v = jnp.repeat(v, repeat, axis=2)

        # slice & cast attention bias exactly
        if attention_bias is not None:
            q_len, k_len = q.shape[1], k.shape[1]
            attention_bias = attention_bias[:, :, k_len - q_len : k_len, :k_len]
            attention_bias = attention_bias.astype(q.dtype)

        # scaled dot-product
        # QUESTION: Why does original use no bias or mask here?
        attn_out = nnx.dot_product_attention(
            q,
            k,
            v,
            mask=None,
            bias=attention_bias,
            dropout_rate=self.cfg.attention_dropout,
        )  # (B,T,H,Dh)

        # Merge heads (B,T,H,Dh) -> (B,T,C)
        attn_out = attn_out.reshape(B, T, C)
        attn_out = self.attn_out(attn_out)
        return attn_out, present_kv

    def __call__(
        self,
        x: jax.Array,  # (B,T,D)
        attention_bias=None,
        layer_past: KVCache | None = None,
        use_cache: bool = False,  # KV-cache tuple or None
    ):
        Kh, Dh = self.cfg.effective_n_kv_heads, self.cfg.d_model // self.cfg.n_heads

        # Attention
        h = self.attn_norm(x)
        if self.cfg.block_type is BlockType.SEQUENTIAL:
            qkv = self.att_proj(h)
            q, k, v = jnp.split(
                qkv,
                [self.cfg.d_model, self.cfg.d_model + Kh * Dh],
                axis=-1,
            )
        else:
            q = self.q_proj(h)
            k = self.k_proj(h)
            v = self.v_proj(h)

        att, cache = self.attention(
            q,
            k,
            v,
            attention_bias=attention_bias,
            layer_past=layer_past,
            use_cache=use_cache,
        )
        x = x + self.dropout(att)

        # Add feed-forward projection
        og_x = x
        x = self.ff_norm(x)
        if self.cfg.block_type is BlockType.LLAMA:
            x, x_up = self.ff_proj(x), self.up_proj(x)
            x = self.act(x)
            x = x * x_up
        else:
            x = self.ff_proj(x)
            x = self.act(x)
        x = self.ff_out(x)
        x = og_x + self.dropout(x)
        return x, cache


_NEG_INF_F32 = jnp.finfo(jnp.float32).min


def ensure_finite(x: jax.Array, check_neg_inf: bool = True, check_pos_inf: bool = False) -> jax.Array:
    """
    Return a copy of *x* where any occurrences of +/-infty have been
    replaced by the floating-point limits of the array's dtype.

    Parameters
    ----------
    x : jax.Array
        Input tensor.
    check_neg_inf : bool, default ``True``
        If ``True`` replace ``-inf`` with ``jnp.finfo(dtype).min``.
    check_pos_inf : bool, default ``False``
        If ``True`` replace ``+inf`` with ``jnp.finfo(dtype).max``.
    """
    dtype_limits = jnp.finfo(x.dtype)

    if check_neg_inf:
        x = jnp.where(jnp.isneginf(x), dtype_limits.min, x)
    if check_pos_inf:
        x = jnp.where(jnp.isposinf(x), dtype_limits.max, x)

    return x


def causal_attention_bias(seq_len: int, dtype=jnp.float32) -> jax.Array:
    """
    Return a bias of shape (1, 1, L, L) whose i j entry is 0 when
    j <= i and -infty when j > i.  (Used to mask out future tokens.)
    """
    # upper-triangular (excluding diagonal) → 1 where j>i else 0
    mask = jnp.triu(jnp.ones((seq_len, seq_len), dtype=dtype), k=1)
    bias = jnp.where(mask == 1, _NEG_INF_F32, 0.0)
    return bias[None, None, :, :]  # (1,1,L,L)


def get_causal_attention_bias(cache: dict[str, jax.Array], seq_len: int, dtype=jnp.float32) -> jax.Array:
    """
    Retrieve (or create) a causal-mask bias from a simple Python dict.
    """
    bias = cache.get("causal_attention_bias")
    if bias is not None and bias.shape[-1] >= seq_len:
        # reuse existing tensor - slice if we requested a shorter length
        return bias[..., :seq_len, :seq_len]

    # need to (re)build
    bias = causal_attention_bias(seq_len, dtype)
    cache["causal_attention_bias"] = bias
    return bias


def alibi_attention_bias(seq_len: int, cfg, dtype=jnp.float32) -> jax.Array:
    """
    Construct the ALiBi (Attention with Linear Biases) bias tensor.
    """
    # Base vector: …, -2, -1, 0   (length L)
    base = jnp.arange(1 - seq_len, 1, dtype=dtype)

    # Pairwise differences |j - i| → broadcast to (1,1,L,L)
    diff = jnp.abs(base[None, None, :, None] - base[None, None, None, :])

    # Negative slope (so later attention scores favour recent tokens)
    alibi = -diff  # (1,1,L,L)

    # Per-head slope factors   a_h = (max / n_heads) · h
    slopes = (cfg.alibi_bias_max / cfg.n_heads) * jnp.arange(1, cfg.n_heads + 1, dtype=dtype)  # (n_heads,)

    scale = (1.0 / jnp.power(2.0, slopes)).reshape(1, cfg.n_heads, 1, 1)

    return alibi * scale  # (1,n_heads,L,L)


def get_alibi_attention_bias(cache: dict[str, jax.Array], seq_len: int, cfg, dtype=jnp.float32) -> jax.Array:
    """
    Retrieve (or create) the ALiBi (Attention with Linear Biases) bias tensor.
    """
    bias = cache.get("alibi_attention_bias")
    if bias is not None and bias.shape[-1] >= seq_len:
        # reuse existing tensor - slice if we requested a shorter length
        return bias[..., :seq_len, :seq_len]

    # need to (re)build
    bias = alibi_attention_bias(seq_len, cfg, dtype)
    cache["alibi_attention_bias"] = bias
    return bias


class LLaDAOutput(NamedTuple):
    logits: jax.Array
    """
    A tensor of shape `(batch_size, seq_len, vocab_size)` representing the log probabilities
    for the next token *before* normalization via (log) softmax.
    """

    attn_key_values: list[tuple[jax.Array, jax.Array]] | None
    """
    Attention keys and values from each block.
    """

    hidden_states: tuple[jax.Array, ...] | None
    """
    Hidden states from each block.
    """


class LLaDAModel(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs):
        self.cfg = cfg
        # Token embedding
        self.wte = nnx.Embed(self.cfg.vocab_size, self.cfg.d_model, rngs=rngs)
        self.emb_drop = nnx.Dropout(self.cfg.embedding_dropout, rngs=rngs)
        self.res_drop = nnx.Dropout(self.cfg.residual_dropout, rngs=rngs)

        if not (self.cfg.alibi or self.cfg.rope):
            self.wpe = nnx.Embed(self.cfg.max_sequence_length, self.cfg.d_model, rngs=rngs)

        # Transformer blocks
        self.blocks = [LLaDABlock(i, self.cfg, rngs=rngs) for i in range(self.cfg.n_layers)]

        # Final layer norm
        self.ln_f = nnx.LayerNorm(self.cfg.d_model, epsilon=1e-5, rngs=rngs)

        # Output projection
        if not self.cfg.weight_tying:
            self.ff_out = nnx.Linear(self.cfg.d_model, self.cfg.vocab_size, rngs=rngs)

        self.__cache = {}

    def __call__(
        self,
        input_ids: jax.Array,  # (B, T)
        input_embeddings: jax.Array | None = None,
        attention_mask: jax.Array | None = None,
        attention_bias: jax.Array | None = None,
        past_key_values: list[KVCache] | None = None,
        use_cache: bool = True,
        output_hidden_states: bool | None = None,
        last_logits_only: bool = False,
    ) -> LLaDAOutput:  # returns logits  (B, T or 1, V)
        batch_size, seq_len = input_ids.shape if input_embeddings is None else input_embeddings.shape[:2]
        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].shape[-2]

        # (B, T, D)
        x = self.wte(input_ids) if input_embeddings is None else input_embeddings

        if self.cfg.input_emb_norm:
            x = x * math.sqrt(self.cfg.d_model)

        if not (self.cfg.alibi or self.cfg.rope):
            # Get positional embeddings.
            pos = jnp.arange(past_length, past_length + seq_len)[None, :]
            pos_emb = self.wpe(pos)
            x = x + pos_emb

        # Add positional embeddings to input and dropout
        x = self.emb_drop(x)

        # Prepare attention masks
        if attention_mask is not None and jnp.any(attention_mask == 0):
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * jnp.finfo(attention_mask.dtype).min
        else:
            attention_mask = None

        # Merge attention mask with attention bias
        if attention_bias is not None or attention_mask is not None or self.cfg.alibi or past_key_values is not None:
            if attention_bias is None and self.cfg.alibi:
                attention_bias = get_causal_attention_bias(
                    self.__cache, past_length + seq_len
                ) + get_alibi_attention_bias(self.__cache, past_length + seq_len, self.cfg)
            elif attention_bias is None:
                attention_bias = get_causal_attention_bias(self.__cache, past_length + seq_len)

            # Transform to the right shape and data type.
            mask_len = seq_len
            if attention_mask is not None:
                mask_len = attention_mask.shape[-1]
            elif past_key_values is not None:
                mask_len = past_key_values[0][0].shape[-2] + seq_len
            attention_bias = attention_bias[:, :, :mask_len, :mask_len]

            # Add in the masking bias.
            if attention_mask is not None:
                attention_bias = attention_bias + attention_mask
                # Might get -infs after adding attention mask, since dtype.min + dtype.min = -inf.
                ensure_finite(attention_bias, check_neg_inf=True, check_pos_inf=False)

        attn_key_values = [] if use_cache else None

        # Decoder layers
        all_hidden_states = []

        # Apply blocks
        for blk_idx, blk in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states.append(x)
            layer_past = None if past_key_values is None else past_key_values[blk_idx]
            x, cache = blk(
                x,
                attention_bias=attention_bias,
                layer_past=layer_past,
                use_cache=use_cache,
            )  # (B, T, D)
            if attn_key_values is not None:
                assert cache is not None
                attn_key_values.append(cache)

        # Prepare logits (B, T | 1, V)
        if last_logits_only:
            x = x[:, -1:, :]  # keep dims for broadcasting

        # Apply final layer norm
        x = self.ln_f(x)
        if output_hidden_states:
            all_hidden_states.append(x)

        if self.cfg.weight_tying:
            logits = self.wte.attend(x)
        else:
            logits = self.ff_out(x)

        if self.cfg.scale_logits:
            logits = logits / math.sqrt(self.cfg.d_model)

        return LLaDAOutput(
            logits=logits,
            attn_key_values=attn_key_values,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
        )


def num_right_pad(x: jax.Array):
    return jnp.sum(jnp.cumsum(jnp.flip(x != 0, axis=-1), axis=-1) == 0, -1)


@partial(jax.jit, donate_argnums=(1))
def forward(graphdef: nnx.GraphDef[tuple[nnx.Module, list[KVCache]]], state, tokens, pad_id):
    model, cache = nnx.merge(graphdef, state)
    output = model(tokens, past_key_values=cache)

    # Predict next missing token
    right_paddings = num_right_pad(tokens != pad_id)
    next_tokens = jnp.argmax(output.logits[:, -right_paddings - 1], axis=-1, keepdims=True)

    # Update state
    state = jax.tree.leaves(nnx.state((model, output.attn_key_values)))
    return next_tokens, state
