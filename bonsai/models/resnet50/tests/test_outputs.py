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
import numpy as np  # 
import torch
from absl.testing import absltest
from huggingface_hub import snapshot_download
from transformers import ResNetForImageClassification

from bonsai.models.resnet50 import modeling as model_lib
from bonsai.models.resnet50 import params


class Resnet50Test(absltest.TestCase):
    def test_full(self):
        model_name = "microsoft/resnet-50"
        model_ckpt_path = snapshot_download("microsoft/resnet-50")
        bonsai_model = params.create_resnet50_from_pretrained(model_ckpt_path)
        baseline_model = ResNetForImageClassification.from_pretrained(model_name)

        batch_size, image_size = 8, 224
        random_inputs = jax.random.truncated_normal(
            jax.random.key(0), lower=-1, upper=1, shape=(batch_size, image_size, image_size, 3)
        )
        
        # Convert JAX array to NumPy array before converting to Torch
        baseline_inputs = {"pixel_values": torch.tensor(np.array(random_inputs)).to(torch.float32).permute(0, 3, 1, 2)}

        bonsai_outputs = model_lib.forward(bonsai_model, random_inputs)
        with torch.no_grad():
            baseline_outputs = baseline_model(**baseline_inputs).logits.cpu().detach().numpy()

        np.testing.assert_allclose(bonsai_outputs, baseline_outputs, rtol=1e-5)


#  new test class for ResNet-152

class Resnet152Test(absltest.TestCase):
    def test_full(self):
        model_name = "microsoft/resnet-152"
        model_ckpt_path = snapshot_download("microsoft/resnet-152")
        
        # --- Loading your JAX model ---
        bonsai_model = params.create_resnet152_from_pretrained(model_ckpt_path)
        # --- Loading the PyTorch baseline model ---
        baseline_model = ResNetForImageClassification.from_pretrained(model_name)

        batch_size, image_size = 8, 224
        # --- JAX input ---
        random_inputs = jax.random.truncated_normal(
            jax.random.key(0), lower=-1, upper=1, shape=(batch_size, image_size, image_size, 3)
        )
        # --- PyTorch input (this one is correct) ---
        baseline_inputs = {"pixel_values": torch.tensor(np.array(random_inputs)).to(torch.float32).permute(0, 3, 1, 2)}

        # --- Get JAX model output ---
        bonsai_outputs = model_lib.forward(bonsai_model, random_inputs)
        # --- Get PyTorch model output ---
        with torch.no_grad():
            baseline_outputs = baseline_model(**baseline_inputs).logits.cpu().detach().numpy()

        # --- Compare them! ---
        np.testing.assert_allclose(bonsai_outputs, baseline_outputs, rtol=1e-5)


if __name__ == "__main__":
    absltest.main()
    
    
# end of bonsai/models/resnet50/tests/test_outputs.py