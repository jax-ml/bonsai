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

from typing import Type

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from bonsai.models.sam2.model_positional_encoding import apply_rotary_enc, compute_axial_cis
from bonsai.models.sam2.model_utils import MLP


class Attention(nnx.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        kv_in_dim: int | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        self.dropout_p = dropout

        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim"

        self.q_proj = nnx.Linear(embedding_dim, self.internal_dim, rngs=rngs)
        self.k_proj = nnx.Linear(self.kv_in_dim, self.internal_dim, rngs=rngs)
        self.v_proj = nnx.Linear(self.kv_in_dim, self.internal_dim, rngs=rngs)
        self.out_proj = nnx.Linear(self.internal_dim, embedding_dim, rngs=rngs)

    def _separate_heads(self, x: jax.Array) -> jax.Array:
        b, n, c = x.shape
        return x.reshape(b, n, self.num_heads, c // self.num_heads)

    def _recombine_heads(self, x: jax.Array) -> jax.Array:
        b, n, h, d = x.shape
        return x.reshape(b, n, h * d)

    def __call__(self, q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = self._separate_heads(q)  # (B, N, H, D)
        k = self._separate_heads(k)
        v = self._separate_heads(v)

        out = nnx.dot_product_attention(q, k, v, dropout_rate=self.dropout_p)
        out = self._recombine_heads(out)
        return self.out_proj(out)


class TwoWayAttentionBlock(nnx.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type = nnx.relu,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.self_attn = Attention(embedding_dim, num_heads, rngs=rngs)
        self.norm1 = nnx.LayerNorm(embedding_dim, rngs=rngs)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate, rngs=rngs
        )
        self.norm2 = nnx.LayerNorm(embedding_dim, rngs=rngs)

        self.mlp = MLP(embedding_dim, mlp_dim, embedding_dim, 2, activation, rngs=rngs)
        self.norm3 = nnx.LayerNorm(embedding_dim, rngs=rngs)

        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate, rngs=rngs
        )
        self.norm4 = nnx.LayerNorm(embedding_dim, rngs=rngs)

        self.skip_first_layer_pe = skip_first_layer_pe

    def __call__(self, queries, keys, query_pe, key_pe):
        # Self-attention
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            queries = queries + self.self_attn(q=q, k=q, v=queries)
        queries = self.norm1(queries)

        # Token-to-image attention
        q = queries + query_pe
        k = keys + key_pe
        queries = queries + self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = self.norm2(queries)

        # MLP
        queries = queries + self.mlp(queries)
        queries = self.norm3(queries)

        # Image-to-token attention
        q = queries + query_pe
        k = keys + key_pe
        keys = keys + self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = self.norm4(keys)

        return queries, keys


class TwoWayTransformer(nnx.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type = nnx.relu,
        attention_downsample_rate: int = 2,
        *,
        rngs: nnx.Rngs,
    ):
        self.blocks = [
            TwoWayAttentionBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                activation=activation,
                attention_downsample_rate=attention_downsample_rate,
                skip_first_layer_pe=(i == 0),
                rngs=rngs,
            )
            for i in range(depth)
        ]
        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate, rngs=rngs
        )
        self.norm_final_attn = nnx.LayerNorm(embedding_dim, rngs=rngs)

    def __call__(self, image_embedding, image_pe, point_embedding):
        B, C, H, W = image_embedding.shape
        image_embedding = image_embedding.transpose(0, 2, 3, 1).reshape(B, H * W, C)
        image_pe = image_pe.transpose(0, 2, 3, 1).reshape(B, H * W, C)

        queries = point_embedding
        keys = image_embedding

        for block in self.blocks:
            queries, keys = block(queries=queries, keys=keys, query_pe=point_embedding, key_pe=image_pe)

        q = queries + point_embedding
        k = keys + image_pe
        queries = queries + self.final_attn_token_to_image(q, k, keys)
        queries = self.norm_final_attn(queries)

        return queries, keys


class RoPEAttention(nnx.Module):
    """Multi-head attention with rotary positional encoding."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
        rope_k_repeat: bool = False,
        feat_sizes: tuple[int, ...] = (64, 64),
        kv_in_dim: int | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.q_proj = nnx.Linear(embedding_dim, embedding_dim, rngs=rngs)
        self.k_proj = nnx.Linear(self.kv_in_dim, self.internal_dim, rngs=rngs)
        self.v_proj = nnx.Linear(self.kv_in_dim, self.internal_dim, rngs=rngs)
        self.out_proj = nnx.Linear(self.internal_dim, embedding_dim, rngs=rngs)

        self.num_heads = num_heads
        self.head_dim = self.internal_dim // num_heads
        self.dropout_rate = dropout
        self.rope_k_repeat = rope_k_repeat

        H, W = feat_sizes
        self.rotary_freqs_cis = compute_axial_cis(self.head_dim, H, W, rope_theta)

    def _separate_heads(self, x: jnp.ndarray) -> jnp.ndarray:
        B, T, D = x.shape
        return x.reshape(B, T, self.num_heads, self.head_dim)

    def _combine_heads(self, x: jnp.ndarray) -> jnp.ndarray:
        B, T, H, C = x.shape
        return x.reshape(B, T, H * C)

    def _get_freqs_cis(self, seq_len: int) -> jnp.ndarray:
        """Handles dynamic recomputation if spatial extent changes."""
        if self.rotary_freqs_cis.shape[0] != seq_len:
            H = int(seq_len**0.5)
            W = -(-seq_len // H)
            return compute_axial_cis(self.head_dim, H, W)
        return self.rotary_freqs_cis

    def __call__(
        self,
        q: jnp.ndarray,  # [B, Tq, D]
        k: jnp.ndarray,  # [B, Tk, D]
        v: jnp.ndarray,  # [B, Tk, D]
        num_k_exclude_rope: int = 0,
    ) -> jnp.ndarray:
        q_proj = self._separate_heads(self.q_proj(q))  # [B, Tq, H, C]
        k_proj = self._separate_heads(self.k_proj(k))  # [B, Tk, H, C]
        v_proj = self._separate_heads(self.v_proj(v))  # [B, Tk, H, C]

        freqs = self._get_freqs_cis(q_proj.shape[2])

        # Apply RoPE to all but final num_k_exclude_rope keys
        if num_k_exclude_rope > 0:
            num_k_rope = k_proj.shape[2] - num_k_exclude_rope
            k_rope, k_pass = jnp.split(k_proj, [num_k_rope], axis=2)

            # Apply RoPE to q and k_rope only
            q_proj, k_rope = apply_rotary_enc(q_proj, k_rope, freqs, repeat_freqs_k=self.rope_k_repeat)

            # Concatenate back the untouched tail of k_proj
            k_proj = jnp.concatenate([k_rope, k_pass], axis=2)
        else:
            q_proj, k_proj = apply_rotary_enc(q_proj, k_proj, freqs, repeat_freqs_k=self.rope_k_repeat)

        attn_out = nnx.dot_product_attention(q_proj, k_proj, v_proj, dropout_rate=self.dropout_rate)
        out = self._combine_heads(attn_out)
        return self.out_proj(out)
