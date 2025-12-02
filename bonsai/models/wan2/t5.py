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
import dataclasses
import math
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx
from jax.lax import Precision
from jaxtyping import Array


def gelu(x: Array) -> Array:
    return 0.5 * x * (1.0 + jnp.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * jnp.pow(x, 3.0))))


class RMSNorm(nnx.Module):
    def __init__(self, dim: int, eps: float = 1e-6, *, rngs: nnx.Rngs):
        self.dim = dim
        self.eps = eps
        self.weight = nnx.Param(jnp.ones(dim))

    def __call__(self, x: Array) -> Array:
        # RMS normalization: x / sqrt(mean(x^2))
        variance = jnp.mean(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.eps)
        return self.weight.value * x


class T5Attention(nnx.Module):
    """T5 Multi-head attention."""

    def __init__(self, dim: int, dim_attn: int, num_heads: int, dropout: float = 0.1, *, rngs: nnx.Rngs):
        assert dim_attn % num_heads == 0
        self.dim = dim
        self.dim_attn = dim_attn
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads

        # Linear projections
        self.q = nnx.Linear(dim, dim_attn, use_bias=False, precision=Precision.HIGHEST, rngs=rngs)
        self.k = nnx.Linear(dim, dim_attn, use_bias=False, precision=Precision.HIGHEST, rngs=rngs)
        self.v = nnx.Linear(dim, dim_attn, use_bias=False, precision=Precision.HIGHEST, rngs=rngs)
        self.o = nnx.Linear(dim_attn, dim, use_bias=False, precision=Precision.HIGHEST, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(
        self,
        x: Array,
        context: Optional[Array] = None,
        mask: Optional[Array] = None,
        pos_bias: Optional[Array] = None,
        deterministic: bool = True,
    ) -> Array:
        """
        Args:
            x: [B, L1, C] query
            context: [B, L2, C] key/value context, defaults to x for self-attention
            mask: [B, L2] or [B, L1, L2] attention mask
            pos_bias: [B, num_heads, L1, L2] position bias
        """
        context = x if context is None else context
        b = x.shape[0]
        n = self.num_heads
        c = self.head_dim

        q = self.q(x).reshape(b, -1, n, c)
        k = self.k(context).reshape(b, -1, n, c)
        v = self.v(context).reshape(b, -1, n, c)

        # Attention bias
        attn_bias = jnp.zeros((b, n, q.shape[1], k.shape[1]))
        if pos_bias is not None:
            attn_bias = attn_bias + pos_bias
        if mask is not None:
            # Expand mask to attention shape
            if mask.ndim == 2:
                mask = mask[:, None, None, :]  # [B, 1, 1, L2]
            else:
                mask = mask[:, None, :, :]  # [B, 1, L1, L2]
            attn_bias = jnp.where(mask == 0, jnp.finfo(x.dtype).min, attn_bias)

        attn = jnp.einsum("binc,bjnc->bnij", q, k) + attn_bias
        attn = jax.nn.softmax(attn, axis=-1)
        x = jnp.einsum("bnij,bjnc->binc", attn, v)

        x = x.reshape(b, -1, n * c)
        x = self.o(x)
        x = self.dropout(x, deterministic=deterministic)
        return x


class T5FeedForward(nnx.Module):
    """T5 Feed-forward network with gated activation."""

    def __init__(self, dim: int, dim_ffn: int, dropout: float = 0.1, *, rngs: nnx.Rngs):
        self.dim = dim
        self.dim_ffn = dim_ffn

        self.gate = nnx.Linear(dim, dim_ffn, use_bias=False, precision=Precision.HIGHEST, rngs=rngs)
        self.fc1 = nnx.Linear(dim, dim_ffn, use_bias=False, precision=Precision.HIGHEST, rngs=rngs)
        self.fc2 = nnx.Linear(dim_ffn, dim, use_bias=False, precision=Precision.HIGHEST, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x: Array, deterministic: bool = True) -> Array:
        x = self.fc1(x) * gelu(self.gate(x))
        x = self.dropout(x, deterministic=deterministic)
        x = self.fc2(x)
        x = self.dropout(x, deterministic=deterministic)
        return x


class T5RelativeEmbedding(nnx.Module):
    def __init__(self, num_buckets: int, num_heads: int, bidirectional: bool, max_dist: int = 128, *, rngs: nnx.Rngs):
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.max_dist = max_dist
        self.embedding = nnx.Embed(num_buckets, num_heads, rngs=rngs)

    def __call__(self, lq: int, lk: int) -> Array:
        """Compute relative position bias.

        Args:
            lq: Query sequence length
            lk: Key sequence length

        Returns:
            [1, num_heads, lq, lk] relative position bias
        """
        q_pos = jnp.arange(lq)[:, None]
        k_pos = jnp.arange(lk)[None, :]
        rel_pos = k_pos - q_pos

        rel_pos_buckets = self._relative_position_bucket(rel_pos)

        # Get embeddings
        rel_pos_embeds = self.embedding(rel_pos_buckets)  # [lq, lk, num_heads]
        rel_pos_embeds = rel_pos_embeds.transpose(2, 0, 1)[None, :, :, :]  # [1, num_heads, lq, lk]
        return rel_pos_embeds

    def _relative_position_bucket(self, rel_pos: Array) -> Array:
        """Convert relative positions to bucket indices."""
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = (rel_pos > 0).astype(jnp.int32) * num_buckets
            rel_pos = jnp.abs(rel_pos)
        else:
            num_buckets = self.num_buckets
            rel_buckets = 0
            rel_pos = -jnp.minimum(rel_pos, jnp.zeros_like(rel_pos))

        # Small vs large positions
        max_exact = num_buckets // 2
        is_small = rel_pos < max_exact

        # Logarithmic bucketing for large positions
        rel_pos_large = max_exact + (
            jnp.log(rel_pos.astype(jnp.float32) / max_exact)
            / math.log(self.max_dist / max_exact)
            * (num_buckets - max_exact)
        ).astype(jnp.int32)
        rel_pos_large = jnp.minimum(rel_pos_large, num_buckets - 1)

        rel_buckets = rel_buckets + jnp.where(is_small, rel_pos, rel_pos_large)
        return rel_buckets


class T5SelfAttention(nnx.Module):
    """T5 Self-attention block with feed-forward."""

    def __init__(
        self,
        dim: int,
        dim_attn: int,
        dim_ffn: int,
        num_heads: int,
        num_buckets: int,
        shared_pos: bool = True,
        dropout: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        self.shared_pos = shared_pos

        self.norm1 = RMSNorm(dim, rngs=rngs)
        self.attn = T5Attention(dim, dim_attn, num_heads, dropout, rngs=rngs)
        self.norm2 = RMSNorm(dim, rngs=rngs)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout, rngs=rngs)

        if not shared_pos:
            self.pos_embedding = T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True, rngs=rngs)
        else:
            self.pos_embedding = None

    def __call__(
        self, x: Array, mask: Optional[Array] = None, pos_bias: Optional[Array] = None, deterministic: bool = True
    ) -> Array:
        # Get position bias
        if self.shared_pos:
            e = pos_bias
        else:
            e = self.pos_embedding(x.shape[1], x.shape[1])

        x = x + self.attn(self.norm1(x), mask=mask, pos_bias=e, deterministic=deterministic)
        x = x + self.ffn(self.norm2(x), deterministic=deterministic)
        return x


class T5Encoder(nnx.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        dim_attn: int,
        dim_ffn: int,
        num_heads: int,
        num_layers: int,
        num_buckets: int,
        shared_pos: bool = True,
        dropout: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        self.dim = dim
        self.shared_pos = shared_pos

        self.token_embedding = nnx.Embed(vocab_size, dim, rngs=rngs)
        if shared_pos:
            self.pos_embedding = T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True, rngs=rngs)
        else:
            self.pos_embedding = None
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        self.blocks = nnx.List(
            [
                T5SelfAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets, shared_pos, dropout, rngs=rngs)
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(dim, rngs=rngs)

    def __call__(self, ids: Array, mask: Optional[Array] = None, deterministic: bool = True) -> Array:
        x = self.token_embedding(ids)
        x = self.dropout(x, deterministic=deterministic)

        # Compute shared position bias if needed
        e = self.pos_embedding(x.shape[1], x.shape[1]) if self.shared_pos else None

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask, pos_bias=e, deterministic=deterministic)

        x = self.norm(x)
        x = self.dropout(x, deterministic=deterministic)
        return x


@dataclasses.dataclass(frozen=True)
class T5Config:
    """Configuration for T5 Encoder."""

    vocab_size: int = 256384
    dim: int = 4096
    dim_attn: int = 4096
    dim_ffn: int = 10240
    num_heads: int = 64
    num_layers: int = 24
    num_buckets: int = 32
    shared_pos: bool = False  # UMT5 uses per-layer position embeddings
    dropout: float = 0.1

    @classmethod
    def umt5_xxl(cls) -> "T5Config":
        """UMT5-XXL configuration (~5B parameters)."""
        return cls(
            vocab_size=256384,
            dim=4096,
            dim_attn=4096,
            dim_ffn=10240,
            num_heads=64,
            num_layers=24,
            num_buckets=32,
            shared_pos=False,
            dropout=0.1,
        )

    @classmethod
    def umt5_base(cls) -> "T5Config":
        """UMT5-Base configuration (~580M parameters)."""
        return cls(
            vocab_size=256384,
            dim=768,
            dim_attn=768,
            dim_ffn=2048,
            num_heads=12,
            num_layers=12,
            num_buckets=32,
            shared_pos=False,
            dropout=0.1,
        )


class T5EncoderModel(nnx.Module):
    """T5 Encoder-only model for text encoding.

    Supports multiple T5 configurations (UMT5-XXL, UMT5-Base).
    """

    def __init__(self, config: T5Config, *, rngs: nnx.Rngs):
        """Initialize T5 encoder from config.

        Args:
            config: T5Config specifying model architecture
            rngs: Random number generators for initialization
        """
        self.config = config
        self.encoder = T5Encoder(
            vocab_size=config.vocab_size,
            dim=config.dim,
            dim_attn=config.dim_attn,
            dim_ffn=config.dim_ffn,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            num_buckets=config.num_buckets,
            shared_pos=config.shared_pos,
            dropout=config.dropout,
            rngs=rngs,
        )

    @classmethod
    def from_config(cls, config: T5Config, *, rngs: nnx.Rngs) -> "T5EncoderModel":
        """Create T5 encoder from configuration.

        Args:
            config: T5Config instance
            rngs: Random number generators

        Returns:
            T5EncoderModel instance
        """
        return cls(config, rngs=rngs)

    @classmethod
    def umt5_xxl(cls, *, rngs: nnx.Rngs) -> "T5EncoderModel":
        """Create UMT5-XXL encoder (~5B parameters).

        Args:
            rngs: Random number generators

        Returns:
            T5EncoderModel configured as UMT5-XXL
        """
        return cls(T5Config.umt5_xxl(), rngs=rngs)

    @classmethod
    def umt5_base(cls, *, rngs: nnx.Rngs) -> "T5EncoderModel":
        """Create UMT5-Base encoder (~580M parameters).

        Args:
            rngs: Random number generators

        Returns:
            T5EncoderModel configured as UMT5-Base
        """
        return cls(T5Config.umt5_base(), rngs=rngs)

    def __call__(self, input_ids: Array, attention_mask: Optional[Array] = None, deterministic: bool = True) -> Array:
        """Encode text.

        Args:
            input_ids: [B, L] token IDs
            attention_mask: [B, L] attention mask (1 for valid tokens, 0 for padding)
            deterministic: whether to disable dropout (True for inference)

        Returns:
            [B, L, dim] encoded text embeddings (dim depends on config)
        """
        return self.encoder(input_ids, mask=attention_mask, deterministic=deterministic)


__all__ = ["T5Config", "T5Encoder", "T5EncoderModel"]
