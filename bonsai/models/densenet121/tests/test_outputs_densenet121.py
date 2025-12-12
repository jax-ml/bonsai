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

import jax
import jax.numpy as jnp
import keras_hub
import numpy as np
import tensorflow as tf
from absl.testing import absltest
from flax import nnx
from huggingface_hub import snapshot_download

from bonsai.models.densenet121 import modeling, params


class TestModuleForwardPasses(absltest.TestCase):
    def setUp(self):
        super().setUp()
        jax.config.update("jax_default_matmul_precision", "float32")
        try:
            self.ref_model = keras_hub.models.ImageClassifier.from_preset("densenet_121_imagenet")
            model_ckpt_path = snapshot_download("keras/densenet_121_imagenet")
            graph_def, state = nnx.split(
                params.create_model_from_h5(model_ckpt_path, modeling.ModelConfig.densenet_121())
            )
            state = jax.tree.map(lambda x: x.astype(jnp.float32) if isinstance(x, jax.Array) else x, state)
            self.nnx_model = nnx.merge(graph_def, state)

        except Exception as e:
            self.skipTest(
                "Skipping test because tensorflow-text requires 3.12 or below: %s"
                "Manually install tensorflow-text and run if needed." % e
            )

    def test_full(self):
        jx = jax.random.uniform(jax.random.key(0), shape=(1, 224, 224, 3), dtype=jnp.float32)
        tx = tf.constant(jx)

        ty = self.ref_model(tx)
        ny = self.nnx_model(jx)

        np.testing.assert_allclose(ty.numpy(), ny, atol=1e-3)


if __name__ == "__main__":
    absltest.main()
