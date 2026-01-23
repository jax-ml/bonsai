# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from huggingface_hub import snapshot_download
from jax.sharding import AxisType
from transformers import AutoTokenizer

from bonsai.models.llama3_2 import modeling, params
from bonsai.utils import Sampler


def tokenize(tokenizer, prompts: list[str], shd=None):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    batch = tokenizer(prompts, padding=True, return_tensors="np")
    input_ids = jnp.array(batch["input_ids"], out_sharding=shd)
    attention_mask = jnp.array(batch["attention_mask"], out_sharding=shd)
    return input_ids, attention_mask


def run_model():
    # Choose a checkpoint and config; defaults to the 1B base variant.
    model_id = "meta-llama/Llama-3.2-1B"
    try:
        access_token = os.environ["HF_TOKEN"]
    except KeyError:
        print("\nError: HF_TOKEN is not set.", file=sys.stderr)
        print("Please set the HF_TOKEN environment variable and retry.", file=sys.stderr)
        sys.exit(1)

    model_ckpt_path = snapshot_download(model_id, token=access_token)

    # Default: no sharding (single-device friendly).
    config = modeling.ModelConfig.llama3_2_1b(use_fsdp=False, use_tp=False)
    fsdp, tp = modeling.ShardMode.FSDP.value, modeling.ShardMode.TP.value
    mesh = jax.make_mesh((1, 1), (fsdp, tp), axis_types=(AxisType.Explicit, AxisType.Explicit))
    jax.set_mesh(mesh)
    batch_shd = None

    prompts = [
        "The capital of France is",
        "The definition of a tokenizer in NLP is strictly defined as follows: A tokenizer is an algorithm that",
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_ckpt_path)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    tokens, attention_mask = tokenize(tokenizer, prompts, batch_shd)
    batch_size, token_len = tokens.shape

    generate_steps = 64
    model = params.create_model_from_safe_tensors(model_ckpt_path, config, mesh)
    cache = model.init_cache(config, batch_size, token_len, generate_steps)

    key = jax.random.key(0)
    sampler = Sampler(temperature=1.0, top_p=0.9, top_k=50)
    jit_sampler = jax.jit(sampler)

    # prefill
    logits, cache = modeling.forward(model, cache, tokens, pad_id, attention_mask=attention_mask)
    key, subkey = jax.random.split(key)
    next_tokens = jit_sampler(logits, key=subkey)

    # decode
    tokens_list = [next_tokens]
    finished = jnp.zeros((batch_size,), dtype=jnp.bool_)
    start = time.time()
    for _ in range(generate_steps):
        logits, cache = modeling.forward(model, cache, next_tokens, pad_id)
        key, subkey = jax.random.split(key)
        next_tokens = jit_sampler(logits, key=subkey)
        finished = finished | (next_tokens.squeeze(-1) == tokenizer.eos_token_id)
        tokens_list.append(next_tokens)
        if finished.all():
            break

    elapsed = time.time() - start
    all_output_tokens = jax.device_get(jnp.concatenate(tokens_list, axis=-1))
    print(f"Generated {all_output_tokens.shape[1]} tokens in {elapsed:.3f}s")
    for i, prompt in enumerate(prompts):
        print(f"Prompt:\n {prompt}")
        seq_tokens = all_output_tokens[i]
        eos_idx = np.where(seq_tokens == tokenizer.eos_token_id)[0]
        if eos_idx.size > 0:
            seq_tokens = seq_tokens[: eos_idx[0]]
        decoded = tokenizer.decode(seq_tokens, skip_special_tokens=True)
        print(f"Completion:\n {decoded}\n\n")


if __name__ == "__main__":
    run_model()


__all__ = ["run_model"]
