# Copyright 2026 The JAX Authors.
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

"""Run a small inference example for LLaDA-8B-Instruct."""

import jax
import jax.numpy as jnp
import torch
from huggingface_hub import snapshot_download
from jax.sharding import AxisType
from transformers import AutoTokenizer

from bonsai.models.llada import modeling, params


def run_model():
    # Create model and processor
    model_name = "GSAI-ML/LLaDA-8B-Instruct"
    processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, dtype=torch.float32)
    if processor.padding_side != "left":
        processor.padding_side = "left"
    assert processor.pad_token_id != 126336

    mesh = jax.make_mesh(((1, 1)), ("fsdp", "tp"), axis_types=(AxisType.Explicit, AxisType.Explicit))
    jax.set_mesh(mesh)
    model_ckpt_path = snapshot_download(model_name)
    config = modeling.ModelConfig.llada_8b_it(False, False, dtype=jnp.float32)
    model = params.create_llada_from_pretrained(model_ckpt_path, config, mesh=mesh)

    # Prepare inputs
    prompts = [
        "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
        "Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?",
    ]
    messages = [{"role": "user", "content": prompt} for prompt in prompts]
    prompts = [
        processor.apply_chat_template([message], add_generation_prompt=True, tokenize=False) for message in messages
    ]
    encoded_outputs = processor(prompts, add_special_tokens=False, padding=True, return_tensors="pt")
    input_ids = jnp.array(encoded_outputs["input_ids"].detach().cpu().numpy())
    attention_mask = encoded_outputs["attention_mask"].detach().cpu().numpy() > 0.5

    # Perform generation
    size = 32
    out = modeling.generate(
        model,
        input_ids,
        attention_mask,
        steps=size,
        gen_length=size,
        block_length=size,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=126336,
        logits_eos_inf=False,
        confidence_eos_eot_inf=False,
        key=jax.random.key(0),
    )
    out = torch.tensor(out)
    output = processor.batch_decode(out[:, input_ids.shape[1] :], skip_special_tokens=True)
    for i, o in enumerate(output):
        print(f"Output {i}:\t{o}")


if __name__ == "__main__":
    run_model()
