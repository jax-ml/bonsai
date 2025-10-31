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
import numpy as np
import torch
from absl.testing import absltest
from huggingface_hub import snapshot_download
from transformers import ResNetForImageClassification

from bonsai.models.resnet50 import modeling as model_lib
from bonsai.models.resnet50 import params


class TestModuleForwardPasses(absltest.TestCase):
    def test_full_50(self):
        model_name = "microsoft/resnet-50"
        model_ckpt_path = snapshot_download("microsoft/resnet-50")
        bonsai_model = params.create_resnet50_from_pretrained(model_ckpt_path)
        baseline_model = ResNetForImageClassification.from_pretrained(model_name)

        batch_size, image_size = 8, 224
        random_inputs = jax.random.truncated_normal(
            jax.random.key(0), lower=-1, upper=1, shape=(batch_size, image_size, image_size, 3)
        )
        baseline_inputs = {"pixel_values": torch.tensor(random_inputs).to(torch.float32).permute(0, 3, 1, 2)}

        bonsai_outputs = model_lib.forward(bonsai_model, random_inputs)
        with torch.no_grad():
            baseline_outputs = baseline_model(**baseline_inputs).logits.cpu().detach().numpy()

        np.testing.assert_allclose(bonsai_outputs, baseline_outputs, rtol=1e-5)

    def test_full_152(self):
        model_name = "microsoft/resnet-152"
        model_ckpt_path = snapshot_download("microsoft/resnet-152")
        bonsai_model = params.create_resnet152_from_pretrained(model_ckpt_path)
        baseline_model = ResNetForImageClassification.from_pretrained(model_name)

        batch_size, image_size = 8, 224
        random_inputs = jax.random.truncated_normal(
            jax.random.key(0), lower=-1, upper=1, shape=(batch_size, image_size, image_size, 3)
        )
        baseline_inputs = {"pixel_values": torch.tensor(np.array(random_inputs)).to(torch.float32).permute(0, 3, 1, 2)}

        bonsai_outputs = model_lib.forward(bonsai_model, random_inputs)
        with torch.no_grad():
            baseline_outputs = baseline_model(**baseline_inputs).logits.cpu().detach().numpy()

        np.testing.assert_allclose(bonsai_outputs, baseline_outputs, rtol=1e-5)


if __name__ == "__main__":
    absltest.main()
