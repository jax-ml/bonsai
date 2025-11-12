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
import time

import jax
import jax.numpy as jnp
from flax import nnx
from huggingface_hub import snapshot_download

from bonsai.models.vit import modeling as model_lib
from bonsai.models.vit import params


def run_model():
    # Download safetensors file
    model_name = "google/vit-base-patch16-224"
    model_ckpt_path = snapshot_download(model_name)

    # Load pretrained model
    model = params.create_vit_from_pretrained(model_ckpt_path, model_lib.ModelConfig.vit_p16_224())
    graphdef, state = nnx.split(model)
    flat_state = jax.tree.leaves(state)

    # Prepare dummy input
    batch_size, channels, image_size = 8, 3, 224
    dummy_input = jnp.ones((batch_size, image_size, image_size, channels), dtype=jnp.float32)

    # Warmup (triggers compilation)
    _ = model_lib.forward(graphdef, flat_state, dummy_input, None).block_until_ready()

    # Profile a few steps
    jax.profiler.start_trace("/tmp/profile-vit")
    for _ in range(5):
        logits = model_lib.forward(graphdef, flat_state, dummy_input, None)
        jax.block_until_ready(logits)
    jax.profiler.stop_trace()

    # Timed execution
    t0 = time.perf_counter()
    for _ in range(10):
        logits = model_lib.forward(graphdef, flat_state, dummy_input, None).block_until_ready()
    print(f"Step time: {(time.perf_counter() - t0) / 10:.4f} s")

    # Show top-1 predicted class
    pred = jnp.argmax(logits, axis=-1)
    print("Predicted classes:", pred)


if __name__ == "__main__":
    run_model()

__all__ = ["run_model"]
