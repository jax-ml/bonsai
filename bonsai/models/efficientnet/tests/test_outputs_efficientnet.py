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
import numpy as np
import timm
import torch
from absl.testing import absltest, parameterized
from flax import nnx

from bonsai.models.efficientnet import modeling, params


class TestModuleForwardPasses(parameterized.TestCase):
    def _get_models_and_input_size(version: int):
        nnx_name = f"efficientnet_b{version}"
        if version >= 5:
            timm_name = "tf_" + nnx_name + "_ap"
            block_configs = modeling.BlockConfigs.tf_block_config()
        else:
            timm_name = nnx_name
            block_configs = modeling.BlockConfigs.default_block_config()

        cfg = getattr(modeling.ModelCfg, f"b{version}")(1000)

        jax_model = params.create_model(cfg, block_configs, rngs=nnx.Rngs(0), mesh=None)

        # Download the pre-trained weights
        pretrained_weights = params.get_timm_pretrained_weights(nnx_name)

        nnx_model = params.load_pretrained_weights(jax_model, pretrained_weights)

        timm_model = timm.create_model(timm_name, pretrained=True)
        timm_model.eval()
        return nnx_model, timm_model, cfg.resolution

    @parameterized.parameters([0, 1, 2, 3, 4, 5, 6, 7])
    def test_full(self, version: int):
        nnx_model, timm_model, img_size = TestModuleForwardPasses._get_models_and_input_size(version)
        b = 32
        tx = torch.rand((b, 3, img_size, img_size), dtype=torch.float32)
        jx = jnp.permute_dims(tx.detach().cpu().numpy(), (0, 2, 3, 1))
        jy = nnx_model(jx, training=False)
        with torch.no_grad():
            ty = timm_model(tx)
        np.testing.assert_allclose(jy, ty.cpu().detach().numpy(), atol=1e-3)


if __name__ == "__main__":
    absltest.main()
