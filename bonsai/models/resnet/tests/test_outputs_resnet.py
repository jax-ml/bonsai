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
import numpy as np
import torch
from absl.testing import absltest, parameterized
from transformers import ResNetForImageClassification
from bonsai.models.resnet import modeling as model_lib


class TestModuleForwardPasses(parameterized.TestCase):
    @parameterized.named_parameters(
        ("resnet50", "microsoft/resnet-50"),
        ("resnet152", "microsoft/resnet-152"),
    )
    def test_full(self, model_name: str):
        bonsai_model = model_lib.ResNet.from_pretrained(model_name)
        baseline_model = ResNetForImageClassification.from_pretrained(model_name)

        batch_size, image_size = 8, 224
        random_inputs = jax.random.truncated_normal(
            jax.random.key(0), lower=-1, upper=1, shape=(batch_size, image_size, image_size, 3)
        )
        baseline_inputs = {
            "pixel_values": torch.tensor(np.asarray(random_inputs)).to(torch.float32).permute(0, 3, 1, 2)
        }

        bonsai_outputs = model_lib.forward(bonsai_model, random_inputs)
        with torch.no_grad():
            baseline_outputs = baseline_model(**baseline_inputs).logits.cpu().detach().numpy()

        np.testing.assert_allclose(bonsai_outputs, baseline_outputs, rtol=1e-5)


if __name__ == "__main__":
    absltest.main()
