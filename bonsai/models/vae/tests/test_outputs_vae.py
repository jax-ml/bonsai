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
import torch
from absl.testing import absltest, parameterized
from diffusers.models import AutoencoderKL
from huggingface_hub import snapshot_download

from bonsai.models.vae import params


class TestModuleForwardPasses(parameterized.TestCase):
    def _get_models_and_input_size():
        weight = "stabilityai/sd-vae-ft-mse"
        model_ckpt_path = snapshot_download(weight)
        nnx_model = params.create_model_from_safe_tensors(file_dir=model_ckpt_path)
        dif_model = AutoencoderKL.from_pretrained(weight)

        return nnx_model, dif_model

    def test_full(self):
        nnx_model, dif_model = TestModuleForwardPasses._get_models_and_input_size()
        device = "cpu"
        dif_model.to(device).eval()

        batch = 32
        img_size = 256

        tx = torch.rand((batch, 3, img_size, img_size), dtype=torch.float32)
        jx = jnp.permute_dims(tx.detach().cpu().numpy(), (0, 2, 3, 1))
        jy = nnx_model(jx)
        with torch.no_grad():
            ty = dif_model(tx).sample
        np.testing.assert_allclose(jy, ty.permute(0, 2, 3, 1).cpu().detach().numpy(), atol=9e-1)


if __name__ == "__main__":
    absltest.main()
