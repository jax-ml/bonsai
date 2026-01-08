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

import time

import jax
import jax.numpy as jnp
from huggingface_hub import snapshot_download

from bonsai.models.resnet import modeling as model_lib
from bonsai.models.resnet import params


def run_model(MODEL_CP_PATH=None):
    # 1. Download safetensors file
    model_ckpt_path = snapshot_download("microsoft/resnet-50")

    # 2. Load pretrained model
    config = model_lib.ModelConfig.resnet50()
    model = params.create_resnet_from_pretrained(model_ckpt_path, config)

    # 3. Prepare dummy input
    batch_size = 8
    image_size = 224
    dummy_input = jnp.ones((batch_size, image_size, image_size, 3), dtype=jnp.float32)

    # 4. Warmup + profiling
    # Warmup (triggers compilation)
    _ = model_lib.forward(model, dummy_input)
    jax.block_until_ready(_)

    # Profile a few steps
    jax.profiler.start_trace("/tmp/profile-resnet50")
    for _ in range(5):
        logits = model_lib.forward(model, dummy_input)
        jax.block_until_ready(logits)
    jax.profiler.stop_trace()

    # 5. Timed execution
    t0 = time.perf_counter()
    for _ in range(10):
        logits = model_lib.forward(model, dummy_input)
        jax.block_until_ready(logits)
    print(f"10 runs took {time.perf_counter() - t0:.4f} s")

    # 6. Show top-1 predicted class
    pred = jnp.argmax(logits, axis=-1)
    print("Predicted classes:", pred)


if __name__ == "__main__":
    run_model()

__all__ = ["run_model"]
