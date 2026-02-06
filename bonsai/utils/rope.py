# Copyright 2026 The JAX Authors.
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

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array
from functools import partial


def segment_ids_to_positions(segment_ids: Array) -> Array:
    return jnp.cumsum(segment_ids, axis=-1)


def default_rope_params(
    positions: Array,
    head_dim: int,
    rope_theta: int = 1_000_000,
    factor: float = 1.0,
) -> tuple[Array, Array]:
    # Forked from: jax-llm-examples/qwen3/qwen3_jax/model.py;l=571
    fraction = jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim
    timescale = rope_theta**fraction
    rotational_frequency = 1.0 / timescale
    rotational_frequency /= factor
    attention_factor = 1.0
    return rotational_frequency, attention_factor


rope_functions = dict(
    default=default_rope_params,
)


def apply_rope(x: Array, sin: Array, cos: Array) -> Array:
    assert x.ndim == 4 and sin.ndim == 3 and cos.ndim == 3
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    # [B, T, head_dim] -> [B, h, T, head_dim]
    sin, cos = sin[:, :, None, :], cos[:, :, None, :]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(x.dtype)


class RoPE(nnx.Module):
    def __init__(self, *, rope_type: str, **rope_kwargs):
        self.rope_kwargs = rope_kwargs
        self.rope_fn = partial(rope_functions[rope_type], **rope_kwargs)

    def __call__(self, positions: Array) -> tuple[Array, Array]:
        rotational_frequency, attention_factor = self.rope_fn(positions)
        # Use high-precision einsum to prevent catastrophic bfloat16 rounding (ex: 257→256), as sin(257) differs from sin(256).
        sinusoid_inp = jnp.einsum("BT,k->BTk", positions, rotational_frequency, precision=jax.lax.Precision.HIGHEST)
        sin, cos = jnp.sin(sinusoid_inp) * attention_factor, jnp.cos(sinusoid_inp) * attention_factor
        return sin, cos
