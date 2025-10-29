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

import jax.numpy as jnp
import keras_hub
import numpy as np
import tensorflow as tf
from absl.testing import absltest
from flax import nnx
from huggingface_hub import snapshot_download

from bonsai.models.vgg19 import modeling
from bonsai.models.vgg19 import params as params_lib


class TestModuleForwardPasses(absltest.TestCase):
    def setUp(self):
        super().setUp()
        try:
            self.ref_model = keras_hub.models.ImageClassifier.from_preset("vgg_19_imagenet")
            model_ckpt_path = snapshot_download("keras/vgg_19_imagenet")
            self.nnx_model = params_lib.create_model_from_h5(model_ckpt_path, modeling.ModelCfg.vgg_19())

            image = np.random.uniform(size=(1, 224, 224, 3)).astype(np.float32)
            self.jx = jnp.array(image)
            self.tx = tf.constant(image)
        except Exception as e:
            self.skipTest(
                "Skipping test because tensorflow-text requires 3.12 or below: %s"
                "Manually install tensorflow-text and run if needed." % e
            )

    def test_conv(self):
        ref_model = self.ref_model.backbone
        nnx_model = self.nnx_model.conv_block0

        ty = ref_model.layers[1](self.tx)
        ny = nnx.relu(nnx_model.conv_layers[0](self.jx))

        np.testing.assert_allclose(ty.numpy(), ny, atol=1e-3)

    def test_conv_block(self):
        ref_model = self.ref_model.backbone
        nnx_model = self.nnx_model.conv_block0

        tx = ref_model.layers[1](self.tx)
        tx = ref_model.layers[2](tx)
        ty = ref_model.layers[3](tx)
        ny = nnx_model(self.jx)

        np.testing.assert_allclose(ty.numpy(), ny, atol=1e-5)

    def test_full(self):
        ty = self.ref_model(self.tx)
        ny = self.nnx_model(self.jx)
        np.testing.assert_allclose(ty.numpy(), ny, atol=1e-3)


if __name__ == "__main__":
    absltest.main()
