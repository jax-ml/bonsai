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

from bonsai.models.vae import modeling, params


def run_model():
    # 1. Download safetensors file
    model_ckpt_path = snapshot_download("stabilityai/sd-vae-ft-mse")
    config = modeling.ModelConfig.stable_diffusion_v1_5()
    model = params.create_model_from_safe_tensors(file_dir=model_ckpt_path, cfg=config)

    # 2. Prepare dummy input
    batch_size = 1
    image_size = 256
    dummy_input = jnp.ones((batch_size, image_size, image_size, 3), dtype=jnp.float32)

    # 3. Run a forward pass
    print("Running forward pass...")
    modeling.forward(model, dummy_input)
    print("Forward pass complete.")

    # 4. Warmup + profiling
    # Warmup (triggers compilation)
    _ = modeling.forward(model, dummy_input)
    jax.block_until_ready(_)

    # Profile a few steps
    jax.profiler.start_trace("/tmp/profile-vae")
    for _ in range(5):
        logits = modeling.forward(model, dummy_input)
        jax.block_until_ready(logits)
    jax.profiler.stop_trace()

    # 5. Timed execution
    t0 = time.perf_counter()
    for _ in range(2):
        logits = modeling.forward(model, dummy_input)
        jax.block_until_ready(logits)
    print(f"2 runs took {time.perf_counter() - t0:.4f} s")


if __name__ == "__main__":
    run_model()
