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

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
import tqdm
from huggingface_hub import snapshot_download
from transformers import Gemma3Processor

from bonsai.models.gemma3 import modeling, params
from bonsai.utils import Sampler


def make_input(processor, dtype=torch.float32, msg1=False):
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

    t_inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    t_inputs["pixel_values"] = t_inputs["pixel_values"].to(dtype=dtype)

    n_text = jnp.array(t_inputs["input_ids"].detach().cpu().numpy())
    n_img = jnp.array(np.permute_dims(t_inputs["pixel_values"].detach().cpu().numpy(), (0, 2, 3, 1)))
    n_tti = jnp.array(t_inputs["token_type_ids"].detach().cpu().numpy())

    return n_text, n_img, n_tti


def run_model():
    model_name: str = "google/gemma-3-4b-it"
    cfg = modeling.ModelConfig()
    access_token = os.environ["HF_TOKEN"]
    processor = Gemma3Processor.from_pretrained(model_name, token=access_token, use_fast=False)
    model_ckpt_path = snapshot_download(model_name)
    bonsai_config = modeling.ModelConfig.gemma3_4b()
    bonsai_model = params.create_gemma3_from_pretrained(model_ckpt_path, bonsai_config)
    eot_token_id = processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")

    # Make inputs
    n_text, n_img, n_tti = make_input(processor)

    gen_steps = 500
    batch_size, num_tokens = n_text.shape
    cache = modeling.init_cache(cfg, batch_size, num_tokens, gen_steps, jnp.float32)

    source_key = jax.random.key(0)
    sampler = jax.jit(Sampler(temperature=1.0, top_p=0.8, top_k=10))

    all_tokens = [n_text]
    for _ in tqdm.trange(gen_steps):
        batch_size, num_tokens = n_text.shape
        segment_ids = jnp.ones((batch_size, num_tokens))
        out, cache = modeling.forward(bonsai_model, cache, n_text, n_img, segment_ids, n_tti)

        source_key, key = jax.random.split(source_key)
        n_text = sampler(out, key=key)
        if jnp.all(n_text == eot_token_id):
            print("Hit end of token.")
            break
        all_tokens.append(n_text)
        n_tti = jnp.zeros((batch_size, 1), dtype=jnp.int32)
        n_img = None

    full_tokens = torch.tensor(jnp.concat(all_tokens, axis=1))
    out_tokens = processor.decode(full_tokens[0], skip_special_tokens=True)
    print(out_tokens)


if __name__ == "__main__":
    run_model()
