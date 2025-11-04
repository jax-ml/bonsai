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

import os

import jax
from absl.testing import absltest

from bonsai.utils.samplers import GreedySampler, Sampler, SamplerType


class TestSamplers(absltest.TestCase):
    def setUp(self):
        super().setUp()
        batch_size, vocab_size = 4, 64
        self.logits = jax.random.normal(jax.random.key(0), (batch_size, vocab_size))

    def test_greedy_sampler(self):
        greedy_sampler = GreedySampler()
        greedy_sampler(self.logits, key=jax.random.key(0))

    def test_sampler(self):
        kptemp_sampler = Sampler(top_p=0.1, top_k=10, temperature=1.0)
        kptemp_sampler(self.logits, key=jax.random.key(0))

    def test_sampler_invalid_args(self):
        self.assertRaises(ValueError, Sampler, temperature=1.0, top_p=0.0, top_k=10)
        self.assertRaises(ValueError, Sampler, temperature=1.0, top_p=1.1, top_k=10)
        self.assertRaises(ValueError, Sampler, temperature=1.0, top_p=0.5, top_k=0)
        self.assertRaises(ValueError, Sampler, temperature=-1.0, top_p=0.5, top_k=10)


if __name__ == "__main__":
    absltest.main()
