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

import math
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from huggingface_hub import snapshot_download
from jax.sharding import AxisType
from jax.sharding import PartitionSpec as P
from transformers import AutoTokenizer

from bonsai.models.qwen3 import modeling, params
from bonsai.utils import GreedySampler, Sampler


def tokenize(tokenizer, input: list[str]):
    pad_idx = tokenizer.pad_token_id
    lines = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": l}], tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        for l in input
    ]
    lines = [tokenizer.encode(line) for line in lines]
    max_len = max(len(line) for line in lines)  # Right-align, left-padding to the max token length.
    return jnp.array([np.pad(l, (max_len - len(l), 0), constant_values=pad_idx) for l in lines])


def run_model():
    model_ckpt_path = snapshot_download("Qwen/Qwen3-4B")

    # Sharding is set False by default. Configure to meet device needs.
    config = modeling.ModelConfig.qwen3_4b(use_sharding=False)
    mesh = jax.make_mesh((1, 1), ("fsdp", "tp"), axis_types=(AxisType.Explicit, AxisType.Explicit))
    jax.set_mesh(mesh)

    query = [
        "Why is the sky blue instead of any other color like purple?",
        "Who am I?",
        "Tell me 10 flavors of ice creams.",
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_ckpt_path)
    tokens = tokenize(tokenizer, query)
    batch_size, token_len = tokens.shape

    generate_steps = 256
    cache = modeling.init_cache(config, batch_size, token_len, generate_steps)

    model = params.create_model_from_safe_tensors(model_ckpt_path, config, mesh)
    graphdef, state = nnx.split((model, cache))
    state = jax.tree.leaves(state)  # Better perf from flattened jax state due to no pytree trasversals.

    key = jax.random.key(0)
    sampler = Sampler(temperature=1.0, top_p=0.8, top_k=10)

    # prefill
    logits, state = modeling.forward(graphdef, state, tokens, tokenizer.pad_token_id)
    next_tokens = sampler(logits, key=key)

    # decode
    tokens_list = [next_tokens]
    finished = jnp.zeros((batch_size,), dtype=jnp.bool_)
    for i in range(generate_steps):
        logits, state = modeling.forward(graphdef, state, next_tokens, tokenizer.pad_token_id)
        next_tokens = sampler(logits, key=key)
        finished = finished | (next_tokens.squeeze(-1) == tokenizer.eos_token_id)
        tokens_list.append(next_tokens)
        if finished.all():
            break

    all_output_tokens = jax.device_get(jnp.concatenate(tokens_list, axis=-1))
    for i, q in enumerate(query):
        print(f"User:\n {q}")
        seq_tokens = all_output_tokens[i]
        eos_idx = np.where(seq_tokens == tokenizer.eos_token_id)[0]
        if eos_idx.size > 0:
            seq_tokens = seq_tokens[: eos_idx[0]]
        decoded = tokenizer.decode(seq_tokens, skip_special_tokens=True)
        print(f"Answer:\n {decoded}\n\n")


if __name__ == "__main__":
    run_model()


__all__ = ["run_model"]
