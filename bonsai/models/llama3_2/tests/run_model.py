# Copyright 2025 The JAX Authors.
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

import argparse
import dataclasses
import os
import sys
import time
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from huggingface_hub import snapshot_download

from transformers import AutoTokenizer

from bonsai.models.llama3_2 import modeling, params
from bonsai.utils import Sampler


@dataclasses.dataclass(frozen=True)
class Args:
    model_size: Literal["1B", "3B"]
    use_base_model: bool = False


def _parse_args(argv: list[str]) -> Args:
    def _size(value: str) -> str:
        value = value.strip().upper()
        if value not in ("1B", "3B"):
            raise argparse.ArgumentTypeError("size must be 1B or 3B")
        return value

    parser = argparse.ArgumentParser(description="Run a small Llama 3.2 inference example.")
    parser.add_argument(
        "--size",
        type=_size,
        default="1B",
        help="Model size to load. Choices: 1B or 3B. Default: 1B.",
    )
    parser.add_argument(
        "--base",
        action="store_true",
        help="Use the base (non-instruct) checkpoint. Default uses Instruct.",
    )
    parsed = parser.parse_args(argv)
    return Args(model_size=parsed.size, use_base_model=parsed.base)


def tokenize(tokenizer, prompts: list[str], use_chat_template: bool = True):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    use_template = use_chat_template and getattr(tokenizer, "chat_template", None) is not None

    if use_template:
        lines = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
            for prompt in prompts
        ]
    else:
        lines = prompts

    batch = tokenizer(lines, padding=True, return_tensors="np", add_special_tokens=not use_template)
    input_ids = jnp.array(batch["input_ids"])
    attention_mask = jnp.array(batch["attention_mask"])
    return input_ids, attention_mask


def run_model():
    args = _parse_args(sys.argv[1:])

    # Choose a checkpoint and config; defaults to the 1B Instruct variant.
    model_size = args.model_size
    use_base_model = args.use_base_model

    model_id = f"meta-llama/Llama-3.2-{model_size}" if use_base_model else f"meta-llama/Llama-3.2-{model_size}-Instruct"
    try:
        access_token = os.environ["HF_TOKEN"]
    except KeyError:
        print("\nError: HF_TOKEN is not set.", file=sys.stderr)
        print("Please set the HF_TOKEN environment variable and retry.", file=sys.stderr)
        sys.exit(1)

    model_ckpt_path = snapshot_download(model_id, token=access_token)

    # Default: no sharding (single-device friendly).
    config_fn = modeling.ModelConfig.llama3_2_1b if model_size == "1B" else modeling.ModelConfig.llama3_2_3b
    config = config_fn(use_fsdp=False, use_tp=False)

    instruct_prompts = [
        "Summarize what a tokenizer does in one paragraph.",
        "Write a short, friendly explanation of gradient descent for beginners.",
    ]
    base_prompts = [
        "The capital of Japan is",
        "The definition of a tokenizer in NLP is strictly defined as follows: A tokenizer is an algorithm that",
    ]
    prompts = base_prompts if use_base_model else instruct_prompts

    tokenizer = AutoTokenizer.from_pretrained(model_ckpt_path)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    tokens, attention_mask = tokenize(tokenizer, prompts, use_chat_template=not use_base_model)
    batch_size, token_len = tokens.shape

    generate_steps = 64
    model = params.create_model_from_safe_tensors(model_ckpt_path, config)
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
        print(f"User:\n {prompt}")
        seq_tokens = all_output_tokens[i]
        eos_idx = np.where(seq_tokens == tokenizer.eos_token_id)[0]
        if eos_idx.size > 0:
            seq_tokens = seq_tokens[: eos_idx[0]]
        decoded = tokenizer.decode(seq_tokens, skip_special_tokens=True)
        print(f"Answer:\n {decoded}\n\n")


if __name__ == "__main__":
    run_model()


__all__ = ["run_model"]
