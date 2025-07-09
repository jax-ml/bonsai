# Copyright 2025 Google LLC
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

import flax.nnx as nnx
import jax.numpy as jnp

from bonsai.models.sam2.model_utils import get_activation_fn


class MemoryAttentionLayer(nnx.Module):
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        self_attention: nnx.Module,
        cross_attention: nnx.Module,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_queries: bool,
        pos_enc_at_cross_attn_keys: bool,
        *,
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        self.dropout_value = dropout
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention

        self.linear1 = nnx.Linear(d_model, dim_feedforward, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        self.linear2 = nnx.Linear(dim_feedforward, d_model, rngs=rngs)

        self.norm1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.norm2 = nnx.LayerNorm(d_model, rngs=rngs)
        self.norm3 = nnx.LayerNorm(d_model, rngs=rngs)
        self.dropout1 = nnx.Dropout(dropout, rngs=rngs)
        self.dropout2 = nnx.Dropout(dropout, rngs=rngs)
        self.dropout3 = nnx.Dropout(dropout, rngs=rngs)

        self.activation = get_activation_fn(activation)

        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def _forward_sa(self, tgt: jnp.ndarray, query_pos: jnp.ndarray | None) -> jnp.ndarray:
        x = self.norm1(tgt)
        q = k = x + query_pos if self.pos_enc_at_attn and query_pos is not None else x
        out = self.self_attn(q, k, x)
        return tgt + self.dropout1(out)

    def _forward_ca(
        self,
        tgt: jnp.ndarray,
        memory: jnp.ndarray,
        query_pos: jnp.ndarray | None,
        pos: jnp.ndarray | None,
        num_k_exclude_rope: int = 0,
    ) -> jnp.ndarray:
        x = self.norm2(tgt)

        q = x + query_pos if self.pos_enc_at_cross_attn_queries and query_pos is not None else x
        k = memory + pos if self.pos_enc_at_cross_attn_keys and pos is not None else memory

        kwds = {}
        if num_k_exclude_rope > 0 and hasattr(self.cross_attn_image, "num_k_exclude_rope"):
            kwds["num_k_exclude_rope"] = num_k_exclude_rope

        out = self.cross_attn_image(q=q, k=k, v=memory, **kwds)
        return tgt + self.dropout2(out)

    def __call__(
        self,
        tgt: jnp.ndarray,
        memory: jnp.ndarray,
        query_pos: jnp.ndarray | None = None,
        pos: jnp.ndarray | None = None,
        num_k_exclude_rope: int = 0,
    ) -> jnp.ndarray:
        tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
        x = self.norm3(tgt)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return tgt + self.dropout3(x)


class MemoryAttention(nnx.Module):
    def __init__(
        self,
        d_model: int,
        layers: list[nnx.Module],  # list of MemoryAttentionLayer
        pos_enc_at_input: bool,
        batch_first: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        self.layers = layers
        self.norm = nnx.LayerNorm(d_model, rngs=rngs)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first

    def __call__(
        self,
        curr: jnp.ndarray,  # [B, N, D] or [N, B, D]
        memory: jnp.ndarray,  # [B, M, D] or [M, B, D]
        curr_pos: jnp.ndarray | None = None,
        memory_pos: jnp.ndarray | None = None,
        num_obj_ptr_tokens: int = 0,
    ) -> jnp.ndarray:
        # Pre-input pos enc
        if self.pos_enc_at_input and curr_pos is not None:
            curr = curr + 0.1 * curr_pos

        # Permute to [T, B, D] if needed
        if self.batch_first:
            curr = jnp.transpose(curr, (1, 0, 2))
            memory = jnp.transpose(memory, (1, 0, 2))
            if curr_pos is not None:
                curr_pos = jnp.transpose(curr_pos, (1, 0, 2))
            if memory_pos is not None:
                memory_pos = jnp.transpose(memory_pos, (1, 0, 2))

        x = curr
        for layer in self.layers:
            x = layer(
                tgt=x,
                memory=memory,
                query_pos=curr_pos,
                pos=memory_pos,
                num_k_exclude_rope=num_obj_ptr_tokens,
            )

        x = self.norm(x)

        # Convert back to [B, T, D] if needed
        if self.batch_first:
            x = jnp.transpose(x, (1, 0, 2))

        return x
