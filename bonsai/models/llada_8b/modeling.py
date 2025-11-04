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
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax
from jax.scipy.special import logsumexp


class ActivationType(Enum):
    GELU = auto()
    RELU = auto()
    SILU = auto()
    SWIGLU = auto()


class ActSpec(NamedTuple):
    fn: Callable[[jax.Array], jax.Array]
    output_multiplier: float


def get_activation(act_type: ActivationType) -> ActSpec:
    """Return (callable, multiplier) matching LLaDA conventions."""

    def swiglu(x: jax.Array) -> jax.Array:
        a, b = jnp.split(x, 2, axis=-1)  # split last dim
        return nnx.silu(b) * a

    activations = {
        ActivationType.GELU: ActSpec(nnx.gelu, 1.0),
        ActivationType.RELU: ActSpec(nnx.relu, 1.0),
        ActivationType.SILU: ActSpec(nnx.silu, 1.0),
        ActivationType.SWIGLU: ActSpec(swiglu, 0.5),
    }
    if act_type in activations:
        return activations[act_type]
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
    mlp_hidden_size: int | None = None  # graphdef, state, tokens, pad_id fallback to mlp_ratio * d_model
    activation_type: ActivationType = ActivationType.SWIGLU

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

    # dropout / normalization
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
            activation_type=ActivationType.SILU,
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


class LayerNormBase(nnx.Module):
    def __init__(
        self, config, *, dim: int | None, elementwise_affine: bool | None = None, eps: float = 1e-05, rngs: nnx.Rngs
    ):
        self.config = config
        self.eps = eps
        self.normalized_shape = (dim or config.d_model,)
        if elementwise_affine or (elementwise_affine is None and self.config.layer_norm_with_affine):
            self.scale = nnx.Param(jnp.ones(self.normalized_shape), rngs=rngs)
            use_bias = self.config.bias_for_layer_norm
            if use_bias is None:
                use_bias = self.config.include_bias
            if use_bias:
                self.bias = nnx.Param(jnp.zeros(self.normalized_shape))
            else:
                self.bias = None
        else:
            self.bias = None
            self.weight = None

    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError

    @classmethod
    def build(cls, config, dim: int | None = None, *, rngs: nnx.Rngs, **kwargs):
        if config.layer_norm_type == LayerNormType.DEFAULT:
            return LayerNorm(config, dim=dim, rngs=rngs, **kwargs)
        if config.layer_norm_type == LayerNormType.RMS:
            return RMSNorm(config, dim=dim, rngs=rngs, **kwargs)
        if config.layer_norm_type == LayerNormType.GEMMA_RMS:
            return GemmaRMSNorm(config, dim=dim, rngs=rngs, **kwargs)


class LayerNorm(LayerNormBase):
    def __init__(self, config, dim: int | None, elementwise_affine: bool | None = None, eps: float = 1e-05, *, rngs):
        super().__init__(config, dim=dim, elementwise_affine=elementwise_affine, eps=eps, rngs=rngs)

    def __call__(self, x: jax.Array):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.eps)
        if self.scale is not None:
            if self.bias is not None:
                x = x * self.scale.value + self.bias.value
            else:
                x = x * self.scale.value
        return x


class RMSNorm(LayerNormBase):
    def __init__(
        self, config, dim: int | None, elementwise_affine: bool | None = None, eps: float = 1e-5, *, rngs: nnx.Rngs
    ):
        super().__init__(config, dim=dim, elementwise_affine=elementwise_affine, eps=eps, rngs=rngs)

    def __call__(self, x: jax.Array):
        og_dtype = x.dtype
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x.astype(jnp.float32)
        x = x * jax.lax.rsqrt(var + self.eps)
        if self.scale is not None:
            if self.bias is not None:
                x = x * self.scale.value + self.bias.value
            else:
                x = x * self.scale.value
        return x.astype(og_dtype)


class GemmaRMSNorm(LayerNormBase):
    def __init__(
        self, config, dim: int | None, elementwise_affine: bool | None = None, eps: float = 1e-5, *, rngs: nnx.Rngs
    ):
        super().__init__(config, dim=dim, elementwise_affine=elementwise_affine, eps=eps, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        og_dtype = x.dtype
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x.astype(jnp.float32)
        y = x * jax.lax.rsqrt(var + self.eps)
        if self.scale is not None:
            if self.bias is not None:
                y = y * (1.0 + self.scale.value) + self.bias.value
            else:
                y = y * (1.0 + self.scale.value)
        return y.astype(og_dtype)


class RotaryEmbedding(nnx.Module):
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.pos_sin = None
        self.pos_cos = None
        self.dh = cfg.d_model // cfg.n_heads

    def __call__(self, q: jax.Array, k: jax.Array) -> tuple[jax.Array, jax.Array]:
        """
        q, k: (B, T, H, Dh)
        returns rotated q, k in the original dtypes
        """
        if self.cfg.rope_full_precision:
            q_ = q.astype(jnp.float32)
            k_ = k.astype(jnp.float32)
        else:
            q_, k_ = q, k

        # Slice fp32 sin/cos tables to the current length
        T_q, T_k = q_.shape[-3], k_.shape[-3]
        self._ensure_table()
        sin = self.pos_sin.value[:, :T_k, :, :]
        cos = self.pos_cos.value[:, :T_k, :, :]

        sin_q, cos_q = sin[:, T_k - T_q : T_k, :, :], cos[:, T_k - T_q : T_k, :, :]

        sin = sin.astype(q_.dtype)
        cos = cos.astype(q_.dtype)
        sin_q = sin_q.astype(q_.dtype)
        cos_q = cos_q.astype(q_.dtype)

        # Apply rotation in the promoted dtype
        q_out = self._apply_rotary(q_, sin_q, cos_q)
        k_out = self._apply_rotary(k_, sin, cos)

        return q_out.astype(q.dtype), k_out.astype(k.dtype)

    @staticmethod
    def _make_table(L: int, dh: int, theta: float, dtype=jnp.float32):
        # Compute in fp32
        inv_freq = 1.0 / (theta ** (jnp.arange(0, dh, 2, dtype=dtype) / dh))
        t = jnp.arange(L, dtype=dtype)
        freqs = jnp.einsum("i,j->ij", t, inv_freq)  # (L, dh/2)
        emb = jnp.concatenate([freqs, freqs], axis=-1)  # (L, dh)
        sin = jnp.sin(emb)[None, :, None, :]  # (1, L, 1, dh)
        cos = jnp.cos(emb)[None, :, None, :]
        return sin, cos

    def _ensure_table(self):
        if self.pos_sin is not None:
            return
        self.pos_sin, self.pos_cos = self._make_table(self.cfg.max_sequence_length, self.dh, self.cfg.rope_theta)
        return

    @staticmethod
    def _rotate_half(x):
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([-x2, x1], axis=-1)

    def _apply_rotary(self, t, sin, cos):
        return (t * cos) + (self._rotate_half(t) * sin)


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

        self.hidden_size = cfg.mlp_hidden_size if cfg.mlp_hidden_size is not None else cfg.mlp_ratio * cfg.d_model

        # Dropout
        self.dropout = nnx.Dropout(self.cfg.residual_dropout)

        # Layer norms
        self.k_norm = None
        self.q_norm = None
        if cfg.attention_layer_norm:
            self.k_norm = LayerNormBase.build(
                cfg,
                dim=(cfg.d_model // cfg.n_heads) * cfg.effective_n_kv_heads,
                elementwise_affine=cfg.attention_layer_norm_with_affine,
                rngs=rngs,
            )
            self.q_norm = LayerNormBase.build(cfg, elementwise_affine=cfg.attention_layer_norm_with_affine, rngs=rngs)

        self.attn_norm = LayerNormBase.build(cfg, rngs=rngs)
        self.ff_norm = LayerNormBase.build(cfg, rngs=rngs)

        # Activation function
        self.act, self.act_output_mult = get_activation(cfg.activation_type)
        assert (self.act_output_mult * self.hidden_size) % 1 == 0

        # Feed forward input projection
        self.ff_proj = nnx.Linear(cfg.d_model, self.hidden_size, use_bias=cfg.include_bias, rngs=rngs)

        # Feed-forward output projection
        self.ff_out = nnx.Linear(
            int(self.act_output_mult * self.hidden_size), cfg.d_model, use_bias=cfg.include_bias, rngs=rngs
        )

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
    ) -> jax.Array:
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
        attn = nnx.dot_product_attention(
            q,
            k,
            v,
            mask=None,
            bias=None,
            dropout_rate=self.cfg.attention_dropout,
        )  # (B,T,H,Dh)

        # Merge heads (B,T,H,Dh) -> (B,T,C)
        attn = attn.reshape(B, T, C)
        return self.attn_out(attn)

    def __call__(
        self,
        x: jax.Array,  # (B,T,D)
        attention_bias=None,
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

        att = self.attention(
            q,
            k,
            v,
            attention_bias=attention_bias,
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
        return x


_NEG_INF_F32 = jnp.finfo(jnp.bfloat16).min


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


def causal_attention_bias(seq_len: int, dtype=jnp.bfloat16) -> jax.Array:
    """
    Return a bias of shape (1, 1, L, L) whose i j entry is 0 when
    j <= i and -infty when j > i.  (Used to mask out future tokens.)
    """
    # upper-triangular (excluding diagonal) → 1 where j>i else 0
    mask = jnp.triu(jnp.ones((seq_len, seq_len), dtype=dtype), k=1)
    bias = jnp.where(mask == 1, _NEG_INF_F32, 0.0)
    return bias[None, None, :, :]  # (1,1,L,L)


def get_causal_attention_bias(cache: dict[str, jax.Array], seq_len: int, dtype=jnp.bfloat16) -> jax.Array:
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


def alibi_attention_bias(seq_len: int, cfg, dtype=jnp.bfloat16) -> jax.Array:
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


def get_alibi_attention_bias(cache: dict[str, jax.Array], seq_len: int, cfg, dtype=jnp.bfloat16) -> jax.Array:
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
    hidden_states: tuple[jax.Array, ...] | None


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
        self.blocks = nnx.List([LLaDABlock(i, self.cfg, rngs=rngs) for i in range(self.cfg.n_layers)])

        # Final layer norm
        self.ln_f = LayerNormBase.build(self.cfg, rngs=rngs)

        # Output projection
        if not self.cfg.weight_tying:
            self.ff_out = nnx.Linear(
                self.cfg.d_model,
                self.cfg.embedding_size or self.cfg.vocab_size,
                use_bias=self.cfg.include_bias,
                rngs=rngs,
            )

        self.__cache = {}

    def __call__(
        self,
        input_ids: jax.Array,  # (B, T)
        input_embeddings: jax.Array | None = None,
        attention_mask: jax.Array | None = None,
        attention_bias: jax.Array | None = None,
        output_hidden_states: bool | None = None,
        last_logits_only: bool = False,
    ) -> LLaDAOutput:  # returns logits  (B, T or 1, V)
        _, seq_len = input_ids.shape if input_embeddings is None else input_embeddings.shape[:2]
        x = self.wte(input_ids) if input_embeddings is None else input_embeddings  # (B, T, D)

        if self.cfg.input_emb_norm:
            x = x * math.sqrt(self.cfg.d_model)

        if not (self.cfg.alibi or self.cfg.rope):
            # Get positional embeddings.
            pos = jnp.arange(0, seq_len)[None, :]
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
        if attention_bias is not None or attention_mask is not None or self.cfg.alibi:
            if attention_bias is None and self.cfg.alibi:
                attention_bias = get_causal_attention_bias(self.__cache, seq_len) + get_alibi_attention_bias(
                    self.__cache, seq_len, self.cfg
                )
            elif attention_bias is None:
                attention_bias = get_causal_attention_bias(self.__cache, seq_len)

            # Transform to the right shape and data type.
            mask_len = seq_len
            if attention_mask is not None:
                mask_len = attention_mask.shape[-1]
            attention_bias = attention_bias[:, :, :mask_len, :mask_len]

            # Add in the masking bias.
            if attention_mask is not None:
                attention_bias = attention_bias + attention_mask
                # Might get -infs after adding attention mask, since dtype.min + dtype.min = -inf.
                ensure_finite(attention_bias, check_neg_inf=True, check_pos_inf=False)

        # Decoder layers
        all_hidden_states = []

        # Apply blocks
        for blk_idx, blk in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states.append(x)
            x = blk(
                x,
                attention_bias=attention_bias,
            )  # (B,T,D)

        # Prepare logits (B,T|1,V)
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
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
        )


def add_gumbel_noise(logits: jax.Array, temperature: float, key: jax.Array) -> jax.Array:
    if temperature == 0.0:
        return logits
    logits64 = logits.astype(jnp.float64)
    u = jax.random.uniform(key, shape=logits.shape, dtype=jnp.float64, minval=1e-6, maxval=1 - 1e-6)
    g = jnp.power(-jnp.log(u), temperature)
    z = jnp.exp(logits64) / g
    return z.astype(logits.dtype)


def get_num_transfer_tokens(mask_index: jax.Array, steps: int) -> jax.Array:
    mask_num = jnp.sum(mask_index, axis=1, keepdims=True)  # (B,1)
    base = mask_num // steps  # (B,1)
    remainder = mask_num % steps  # (B,1)
    base = jnp.repeat(base, steps, axis=1)  # (B,steps)
    inc = (jnp.arange(steps)[None, :] < remainder).astype(jnp.int32)  # (B,steps)
    return (base + inc).astype(jnp.int32)


def _row_topk_mask(conf_row: jax.Array, k: jax.Array) -> jax.Array:
    L = conf_row.shape[0]
    idx_sorted = jnp.argsort(conf_row)[::-1]
    ranks = jnp.empty((L,), dtype=jnp.int32).at[idx_sorted].set(jnp.arange(L, dtype=jnp.int32))
    return ranks < k


row_topk_mask_vmapped = jax.vmap(_row_topk_mask, in_axes=(0, 0), out_axes=0)


@jax.jit
def forward(graphdef: nnx.GraphDef[nnx.Module], state: nnx.State, tokens: jax.Array):
    model = nnx.merge(graphdef, state)
    out = model(tokens)
    return out.logits


def generate_step(
    graphdef: nnx.GraphDef[nnx.Module],
    state: nnx.State,
    x: jax.Array,
    prompt_index: jax.Array,
    rng: jax.Array,
    step_idx: int,
    start: int,
    stop: int,
    num_transfer_tokens: jax.Array,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
) -> tuple[jax.Array, jax.Array]:
    """
    Execute a single generation step.

    Returns:
        (updated_x, updated_rng)
    """
    mask_index = x == mask_id

    # CFG: run both, keep the state from the *conditional* path
    if cfg_scale > 0.0:
        un_x = jnp.where(prompt_index, mask_id, x)
        x_stack = jnp.concatenate([x, un_x], axis=0)
        logits_both = forward(graphdef, state, x_stack)
        logits, logits_un = jnp.split(logits_both, 2, axis=0)
        logits = logits_un + (cfg_scale + 1.0) * (logits - logits_un)
    else:
        logits = forward(graphdef, state, x)

    # Noise + argmax
    rng, sub = jax.random.split(rng)
    if temperature > 0:
        logits_noisy = add_gumbel_noise(logits, temperature, sub)
    else:
        logits_noisy = logits
    x0 = jnp.argmax(logits_noisy, axis=-1).astype(jnp.int32)

    # Confidence
    if remasking == "low_confidence":
        # p = jax.nn.softmax(logits, axis=-1)
        # x0_p = jnp.squeeze(jnp.take_along_axis(p, x0[..., None], axis=-1), axis=-1)
        lse = logsumexp(logits, axis=-1)
        x0_logit = jnp.take_along_axis(logits, x0[..., None], axis=-1)[..., 0]
        x0_p = jnp.exp(x0_logit - lse)
    elif remasking == "random":
        rng, sub = jax.random.split(rng)
        x0_p = jax.random.uniform(sub, shape=x0.shape, dtype=logits.dtype)
    else:
        raise NotImplementedError(remasking)

    # Forbid beyond current block
    neg_inf = jnp.array(-jnp.inf, dtype=x0_p.dtype)
    pos = jnp.arange(x.shape[1])[None, :]
    x0_p = jnp.where(pos >= stop, neg_inf, x0_p)

    x0_sel = jnp.where(mask_index, x0, x)
    conf = jnp.where(mask_index, x0_p, neg_inf)

    # variable-k per row
    k_vec = num_transfer_tokens[:, step_idx]
    transfer_index = row_topk_mask_vmapped(conf, k_vec)

    x = jnp.where(transfer_index, x0_sel, x)
    return x, rng


def generate_block(
    graphdef: nnx.GraphDef[nnx.Module],
    state: nnx.State,
    x: jax.Array,
    prompt_index: jax.Array,
    rng: jax.Array,
    block_idx: int,
    prompt_len: int,
    block_length: int,
    steps_per_block: int,
    *,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
) -> tuple[jax.Array, jax.Array]:
    """
    Generate one block of tokens.

    Returns:
        (updated_x, updated_rng)
    """
    start = prompt_len + block_idx * block_length
    stop = prompt_len + (block_idx + 1) * block_length

    block_mask = lax.dynamic_slice_in_dim(x, start_index=start, slice_size=block_length, axis=1)
    block_mask_index = block_mask == mask_id
    num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

    for i in range(steps_per_block):
        x, rng = generate_step(
            graphdef,
            state,
            x,
            prompt_index,
            rng,
            step_idx=i,
            start=start,
            stop=stop,
            num_transfer_tokens=num_transfer_tokens,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking=remasking,
            mask_id=mask_id,
        )

    return x, rng


def generate(
    graphdef: nnx.GraphDef[nnx.Module],
    state: nnx.State,
    prompt: jax.Array,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    rng: jax.Array = jax.random.PRNGKey(0),
) -> jax.Array:
    """
    Generate tokens using masked iterative decoding.

    Returns:
        (B, prompt_len + gen_length) generated tokens
    """
    B, prompt_len = prompt.shape
    x = jnp.full((B, prompt_len + gen_length), mask_id, dtype=jnp.int32).at[:, :prompt_len].set(prompt)
    prompt_index = x != mask_id

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    for block_idx in range(num_blocks):
        x, rng = generate_block(
            graphdef,
            state,
            x,
            prompt_index,
            rng,
            block_idx=block_idx,
            prompt_len=prompt_len,
            block_length=block_length,
            steps_per_block=steps_per_block,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking=remasking,
            mask_id=mask_id,
        )

    return x
