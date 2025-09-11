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
import jax.numpy as jnp
from huggingface_hub import snapshot_download

from bonsai.models.resnet50 import modeling as model_lib
from bonsai.models.resnet50 import params


def full_test(MODEL_CP_PATH=None):
    # 1. Try additional imports
    try:
        import torch
        from transformers import ResNetForImageClassification
    except ModuleNotFoundError as e:
        print(f"Skipping model output verification: {e}")
        return

    # 2. Download safetensors file
    model_name = "microsoft/resnet-50"
    if MODEL_CP_PATH is None:
        MODEL_CP_PATH = "/tmp/models-bonsai/" + model_name.split("/")[1]

    if not os.path.isdir(MODEL_CP_PATH):
        snapshot_download(model_name, local_dir=MODEL_CP_PATH)

    safetensors_path = os.path.join(MODEL_CP_PATH, "model.safetensors")

    # 3. Load pretrained models
    bonsai_model = params.create_resnet50_from_pretrained(safetensors_path)
    baseline_model = ResNetForImageClassification.from_pretrained(model_name)

    # 4. Create inputs
    batch_size, image_size = 8, 224
    random_inputs = jax.random.truncated_normal(
        jax.random.key(0), lower=-1, upper=1, shape=(batch_size, image_size, image_size, 3)
    )
    baseline_inputs = {"pixel_values": torch.tensor(random_inputs).to(torch.float32).permute(0, 3, 1, 2)}

    # 5. Compute model outputs
    bonsai_outputs = model_lib.forward(bonsai_model, random_inputs)
    with torch.no_grad():
        baseline_outputs = baseline_model(**baseline_inputs).logits.cpu().detach().numpy()

    # 6. Compare model outputs
    atol = jnp.max(jnp.abs(bonsai_outputs - baseline_outputs))
    rtol = atol / jnp.max(jnp.abs(baseline_outputs))
    print(f"ATOL = {atol}\tRTOL = {rtol}")


if __name__ == "__main__":
    full_test()
