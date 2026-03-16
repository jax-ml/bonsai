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

"""
This file implements some common KV cache strategies for transformer models.


Assumptions:
1. All inputs are left-padded.
2. Each cache should come with logic for computing a causal mask


Desired API:
# user initializes cache
cache = LayerCache(2, 4, 8, 128, jnp.bfloat16)

# user prefills cache
cache.prefill(k_new, v_new, segment_ids)

# user updates cache
cache.update(k_new, v_new)

# user computes causal mask
mask = cache.compute_causal_mask(input_len)


NOTE: This is still a work in progress and may have bugs.
NOTE: The masking strategy may depend on the cache strategy.
NOTE: Since these utilities will be shared across models, we need test cases in utils/tests/kv_cache_test.py
    The example tests provided there should be implemented and expanded upon.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, DTypeLike
from jax.sharding import PartitionSpec
from jax import P
from typing import TypeAlias
import logging

logging.basicConfig(level=logging.INFO)
logging.info("KV cache utilities are still in development.")
logging.info("Need further testing for these functions.")


def compute_left_pads(segment_ids: Array) -> Array:
    """Compute left pads from segment ids."""
    return jnp.sum(jnp.cumsum(segment_ids != 0, axis=-1) == 0, -1)


# NOTE: This is not truly a Protocol since it inherits from nnx.Module
# This gives a metaclass conflict if we try to inherit from both
class CacheProtocol(nnx.Module):
    """Protocol for KV cache."""

    def prefill(self, k_new: Array, v_new: Array): ...
    def update(self, k_new: Array, v_new: Array): ...
    def size(self): ...
    def compute_causal_mask(self, input_len: int): ...


Cache: TypeAlias = list[CacheProtocol]


class LayerCache(CacheProtocol):
    """Layer-wise KV cache.

    This cache stores the KV states for a single layer.
    This works by just pre-allocating the cache and updating it as we go.
    Note that a common error is to overflow the cache since jax out of bounds indexing does not raise an error.
    """

    def __init__(
        self,
        num_kv_heads: int,
        head_dim: int,
        batch_size: int,
        cache_size: int,
        dtype: DTypeLike,
        kv_shd: PartitionSpec | None = None,
        segment_ids: Array | None = None,
    ):
        # Create caches
        cache_shape = (batch_size, cache_size, num_kv_heads, head_dim)
        self.k_cache = nnx.Cache(jnp.zeros(cache_shape, dtype=dtype, out_sharding=kv_shd))
        self.v_cache = nnx.Cache(jnp.zeros(cache_shape, dtype=dtype, out_sharding=kv_shd))

        # Create start_ind and cur_ind
        # These are implementation details
        start_ind_shd = None if kv_shd is None else P(kv_shd[0])
        self.start_ind = nnx.Variable(-1 * jnp.ones((batch_size,), dtype=jnp.int32, out_sharding=start_ind_shd))
        self.cur_ind = nnx.Variable(jnp.zeros((), dtype=jnp.int32))

        # Initialize start_ind if segment_ids are provided
        # will raise error if not initialized before update or compute_causal_mask
        self.start_ind_initialized = False
        if segment_ids is not None:
            self._init_start_ind(segment_ids)

    @property
    def cache_size(self):
        return self.k_cache.shape[1]

    @property
    def batch_size(self):
        return self.k_cache.shape[0]

    def _init_start_ind(self, segment_ids: Array):
        left_pads = compute_left_pads(segment_ids)
        self.start_ind[...] = jnp.where(self.start_ind[...] < 0, left_pads, self.start_ind[...])
        self.start_ind_initialized = True

    def prefill(self, k_new: Array, v_new: Array, segment_ids: Array):
        return self.update(k_new, v_new)

    def update(self, k: Array, v: Array):
        assert self.start_ind_initialized, "Must initialize start_ind before updating LayerCache"
        slice_indices = (0, self.cur_ind[...], 0, 0)
        self.k_cache[...] = jax.lax.dynamic_update_slice(self.k_cache[...], k, slice_indices)
        self.v_cache[...] = jax.lax.dynamic_update_slice(self.v_cache[...], v, slice_indices)
        self.cur_ind[...] = self.cur_ind[...] + k.shape[1]

    def compute_causal_mask(self, input_len: int):
        assert self.start_ind_initialized, "Must initialize start_ind before computing causal mask"
        b, c = self.batch_size, self.cache_size
        seq_arange = jnp.arange(input_len)
        cache_arange = jnp.arange(c)
        causal_mask = (seq_arange[:, None] - cache_arange[None, :] >= -self.cur_ind) & (
            cache_arange[None, None, :] >= self.start_ind[:, None, None]
        )
        return causal_mask.astype(jnp.bool_)


class CyclicCache(CacheProtocol):
    """Cyclic KV cache.

    This cache stores the KV states for a single layer.
    This works by pre-allocating the cache up to size cache_size.
    Then, it updates the cache as we go, overwriting the oldest entries.

    #TODO: Assumptions:
    1. The number of input tokens does not exceed the cache size for pre-fill.
    2. The number of input tokens is always 1 after pre-fill.

    #TODO: Relax these assumptions after a first implementation
    """

    def __init__(
        self,
        num_kv_heads: int,
        head_dim: int,
        batch_size: int,
        cache_size: int,
        dtype: DTypeLike,
        kv_shd: PartitionSpec | None = None,
        segment_ids: Array | None = None,
    ):
        # Create caches
        cache_shape = (batch_size, cache_size, num_kv_heads, head_dim)
        self.k_cache = nnx.Cache(jnp.zeros(cache_shape, dtype=dtype, out_sharding=kv_shd))
        self.v_cache = nnx.Cache(jnp.zeros(cache_shape, dtype=dtype, out_sharding=kv_shd))

        # Create start_ind and cur_ind
        # These are implementation details
        start_ind_shd = None if kv_shd is None else P(kv_shd[0])
        self.start_ind = nnx.Variable(-1 * jnp.ones((batch_size,), dtype=jnp.int32, out_sharding=start_ind_shd))
        self.cur_ind = nnx.Variable(jnp.zeros((), dtype=jnp.int32))

        # Initialize start_ind if segment_ids are provided
        # will raise error if not initialized before update or compute_causal_mask
        self.start_ind_initialized = False
        if segment_ids is not None:
            self._init_start_ind(segment_ids)

    @property
    def cache_size(self):
        return self.k_cache.shape[1]

    @property
    def batch_size(self):
        return self.k_cache.shape[0]

    def _init_start_ind(self, segment_ids: Array):
        left_pads = compute_left_pads(segment_ids)
        self.start_ind[...] = jnp.where(self.start_ind[...] < 0, left_pads, self.start_ind[...])
        self.start_ind_initialized = True

    def prefill(self, k_new: Array, v_new: Array, segment_ids: Array):
        assert k_new.shape[1] <= self.cache_size, "Number of input tokens exceeds cache size"
        return self.update(k_new, v_new)

    def update(self, k: Array, v: Array):
        assert self.start_ind_initialized, "Must initialize start_ind before updating LayerCache"
        slice_indices = (0, self.cur_ind[...] % self.cache_size, 0, 0)
        self.k_cache[...] = jax.lax.dynamic_update_slice(self.k_cache[...], k, slice_indices)
        self.v_cache[...] = jax.lax.dynamic_update_slice(self.v_cache[...], v, slice_indices)
        self.cur_ind[...] = self.cur_ind[...] + k.shape[1]

    def compute_causal_mask(self, input_len: int):
        # TODO: Need to double check this logic and make sure it is correct with tests.
        assert self.start_ind_initialized, "Must initialize start_ind before computing causal mask"
        b, c = self.batch_size, self.cache_size
        seq_arange = jnp.arange(input_len)
        start_factor = self.cur_ind // c
        cache_arange = jnp.concatenate(
            [
                jnp.arange(start_factor * c, self.cur_ind),
                jnp.arange(self.cur_ind - start_factor * c, c),
            ]
        )
        causal_mask = (seq_arange[:, None] - cache_arange[None, :] >= -(self.cur_ind % c)) & (
            cache_arange[None, None, :] >= self.start_ind[:, None, None]
        )
        return causal_mask.astype(jnp.bool_)


if __name__ == "__main__":
    # Regular cache
    lc = LayerCache(num_kv_heads=1, head_dim=1, batch_size=2, cache_size=10, dtype=jnp.float32)
    segment_ids = jnp.array([[0, 0, 1, 1], [1, 1, 1, 1]])
    lc._init_start_ind(segment_ids)

    mask = lc.compute_causal_mask(4)

    print(mask)

    # Cyclic cache
    cc = CyclicCache(num_kv_heads=1, head_dim=1, batch_size=2, cache_size=4, dtype=jnp.float32)
    segment_ids = jnp.array([[0, 0, 1, 1], [1, 1, 1, 1]])
    cc._init_start_ind(segment_ids)

    cc.prefill(jnp.ones((2, 4, 1, 1)), jnp.ones((2, 4, 1, 1)), segment_ids)
    cc.update(2 * jnp.ones((2, 1, 1, 1)), 2 * jnp.ones((2, 1, 1, 1)))

    print(cc.k_cache[...].reshape((2, 4)))
    print(cc.v_cache[...].reshape((2, 4)))

    mask = cc.compute_causal_mask(4)
    print(mask)
