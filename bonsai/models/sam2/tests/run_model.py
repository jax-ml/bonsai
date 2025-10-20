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

import logging
import os
import time

import jax
import jax.numpy as jnp
from huggingface_hub import snapshot_download

from bonsai.models.sam2 import modeling, params


def convert_pt_to_safetensors(pt_path: str, output_path: str):
    """
    Converts a PyTorch .pt checkpoint to safetensors format.

    Args:
        pt_path: Path to the .pt file (must be loadable by torch).
        output_path: Path to save the safetensors file.
    """
    try:
        import safetensors.torch as st_torch
        import torch
    except ImportError:
        raise ImportError("This function requires PyTorch and safetensors[torch] installed.")

    logging.info(f"Loading PyTorch checkpoint from: {pt_path}")
    state_dict = torch.load(pt_path, map_location="cpu")["model"]

    # Strip "module." prefix if present (common in DDP models)
    state_dict = {(k.replace("module.", "") if k.startswith("module.") else k): v for k, v in state_dict.items()}

    logging.info(f"Saving safetensors to: {output_path}")
    st_torch.save_file(state_dict, output_path)
    logging.info("Conversion complete.")


def run_model(MODEL_CP_PATH=None):
    # 1. Download weights if needed
    model_name = "facebook/sam2-hiera-tiny"
    if MODEL_CP_PATH is None:
        MODEL_CP_PATH = "/tmp/models-bonsai/" + model_name.split("/")[1]
    if not os.path.isdir(MODEL_CP_PATH):
        snapshot_download(model_name, local_dir=MODEL_CP_PATH)

    pt_path = os.path.join(MODEL_CP_PATH, "sam2_hiera_tiny.pt")  # assuming this file name
    safetensors_path = "/tmp/sam2_model.safetensors"

    if not os.path.exists(safetensors_path):
        convert_pt_to_safetensors(pt_path, safetensors_path)

    # 2. Create SAM2 model and predictor
    config = modeling.SAM2Config.sam2_tiny()
    model_obj = params.create_sam2_from_pretrained(safetensors_path, config)
    model = modeling.SAM2ImagePredictor(model_obj)

    # 3. Prepare dummy input
    batch_size = 4
    image_size = 64
    dummy_images = [jnp.ones((image_size, image_size, 3), dtype=jnp.float32) for _ in range(batch_size)]
    dummy_points = [jnp.ones((1, 2), dtype=jnp.float32) for _ in range(batch_size)]
    dummy_labels = [jnp.ones((1,), dtype=jnp.float32) for _ in range(batch_size)]

    # 4. Setting images and warmup
    model.set_image_batch(dummy_images)

    # Predicting masks
    _ = modeling.forward(model, dummy_points, dummy_labels)
    jax.block_until_ready(_)

    # 5. Profiling
    jax.profiler.start_trace("/tmp/profile-sam2")
    for _ in range(5):
        masks_all, _, _ = modeling.forward(model, dummy_points, dummy_labels)
    jax.block_until_ready(masks_all)
    jax.profiler.stop_trace()

    # 6. Timing
    t0 = time.perf_counter()
    for _ in range(10):
        masks_all, _, _ = modeling.forward(model, dummy_points, dummy_labels)
    jax.block_until_ready(masks_all)
    print(f"10 forward passes took {time.perf_counter() - t0:.4f} s")


if __name__ == "__main__":
    run_model()

__all__ = ["run_model"]
