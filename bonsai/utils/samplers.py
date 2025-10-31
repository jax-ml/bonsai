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

from abc import ABC, abstractmethod
from enum import Enum

import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from jax.typing import ArrayLike


class SamplerType(Enum):
    GREEDY = "greedy"
    SAMPLER = "sampler"


class BaseSampler(ABC):
    """Base Sampler class for LLM sampling.

    Classes which inherit from this one should implement the __call__ method.
    The call_method expects logits from the model and keys for any random sampling.
    Inputs to the call method are of the shape (*B, V), where B is the batch shape
    and V is the vocabulary size. The output of each class is of shape B.
    """

    @abstractmethod
    def __call__(self):
        pass


class GreedySampler(BaseSampler):
    """Sampler class for doing greedy sampling from LLM outputs.

    Outputs are selected as the argmax over the last axis.
    """

    def __call__(self, x: ArrayLike, *, key: PRNGKey):
        return jnp.argmax(x, axis=-1, keepdims=True)


class Sampler(BaseSampler):
    """Sampler class for doing sampling with top k, top p, and temperature.

    This class expects logits, x, from an LLM.
    Probabilities are computed by softmax(x / temperature) in the last axis.
    The probabilities are then masked out to only include the top k
    probabilities and such that their cumulative sum is less than p.
    The outputs are then randomly sampled from this distribution.
    """

    def __init__(self, p: float, k: int, temperature: float):
        self.p = p
        self.k = k
        self.temperature = temperature

    def __call__(self, x: ArrayLike, *, key: PRNGKey):
        probs = jax.nn.softmax(x / self.temperature, axis=-1)

        # The folllowing is from tunix/generate/sampler.py
        probs_sorted, indices = jax.lax.top_k(probs, k=self.k)
        cumsum_probs = jnp.cumsum(probs_sorted, axis=-1)
        mask = cumsum_probs - probs_sorted > self.p
        probs_sorted = jnp.where(mask, 0.0, probs_sorted)
        probs_sorted /= jnp.sum(probs_sorted, axis=-1, keepdims=True)

        next_token = jax.random.categorical(key, logits=jnp.log(probs_sorted))

        next_token = jnp.take_along_axis(indices, next_token[..., None], axis=-1)
        return next_token


# TODO: Implement beam sampling
class BeamSampler(BaseSampler):
    pass


if __name__ == "__main__":
    batch_size = 4
    vocab_size = 64
    logits = jax.random.normal(jax.random.key(0), (batch_size, vocab_size))

    # GreedySampler Example
    greedy_sampler = GreedySampler()
    next_token = greedy_sampler(logits, key=jax.random.key(0))
    print(next_token)

    # Sampler Example
    kptemp_sampler = GreedySampler()
    next_token = kptemp_sampler(logits, key=jax.random.key(0))
    print(next_token)
