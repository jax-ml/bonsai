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

"""Run a small inference example for Gemma3."""

import time

import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from bonsai.models.gemma3 import modeling, params

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None


import os

import torch
from transformers import AutoProcessor


def make_input(processor, dtype=torch.float32, msg1=True):
    if msg1:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                    },
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
    else:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
                    },
                    {"type": "text", "text": "Describe this image in detail."},
                ],
            },
        ]

    out = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    out["pixel_values"] = out["pixel_values"].to(dtype=dtype)

    return out


def run_model():
    model_name: str = "google/gemma-3-4b-it"
    cfg = modeling.ModelConfig()
    access_token = os.environ["HF_TOKEN"]
    processor = AutoProcessor.from_pretrained(model_name, token=access_token, use_fast=False)
    model_ckpt_path = snapshot_download(model_name)
    bonsai_model = params.create_gemma3_from_pretrained(model_ckpt_path)

    # Make inputs
    t_inputs = make_input(processor)
    n_img = jnp.array(np.permute_dims(t_inputs["pixel_values"].detach().cpu().numpy(), (0, 2, 3, 1)))
    n_text = jnp.array(t_inputs["input_ids"].detach().cpu().numpy())
    n_tti = jnp.array(t_inputs["token_type_ids"].detach().cpu().numpy())

    gen_steps = 200
    batch_size, num_tokens = n_text.shape
    cache = modeling.init_cache(cfg, batch_size, num_tokens, gen_steps, jnp.float32)

    all_tokens = [n_text]
    for i in tqdm.trange(gen_steps):
        batch_size, num_tokens = n_text.shape
        segment_ids = jnp.ones((batch_size, num_tokens))
        out, cache = modeling.forward(bonsai_model, cache, n_text, n_img, segment_ids, n_tti)

        n_text = jnp.argmax(out, axis=-1)
        all_tokens.append(n_text)
        n_tti = n_tti[:, -1:None]  # this assumes that the input prompt ends with text
        n_img = None

    full_tokens = torch.tensor(jnp.concat(all_tokens, axis=1))
    out_tokens = processor.decode(full_tokens[0], skip_special_tokens=True)
    print(out_tokens)


if __name__ == "__main__":
    run_model()
