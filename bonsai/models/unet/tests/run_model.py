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
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx

from bonsai.models.unet import modeling, params


def run_model():
    # 1. Create model and PRNG key
    rngs = nnx.Rngs(params=0)
    config = modeling.ModelCfg(in_channels=3, num_classes=1)  # Example: RGB input, binary output
    model = params.create_model(cfg=config, rngs=rngs)

    # 2. Prepare dummy input
    batch_size = 4
    image_size = 128
    dummy_input = jnp.ones((batch_size, image_size, image_size, 3), dtype=jnp.float32)

    # 3. Warmup (triggers compilation)
    print("Starting JIT compilation (warmup)...")
    _ = modeling.UNet.forward(model, dummy_input)
    jax.block_until_ready(_)
    print("Warmup complete.")

    # 4. Timed execution
    num_runs = 10
    t0 = time.perf_counter()
    for _ in range(num_runs):
        logits = modeling.UNet.forward(model, dummy_input)
    jax.block_until_ready(logits)
    t1 = time.perf_counter()
    print(f"{num_runs} runs took {t1 - t0:.4f} s")
    print(f"Average inference time: {(t1 - t0) / num_runs * 1000:.2f} ms")

    # 5. Show output shape
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output logits shape: {logits.shape}")


if __name__ == "__main__":
    run_model()
