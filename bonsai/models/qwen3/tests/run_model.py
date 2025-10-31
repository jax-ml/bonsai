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
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
from flax import nnx
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from bonsai.models.qwen3 import modeling, params
from bonsai.utils import GreedySampler, Sampler, SamplerType

FLAGS = flags.FLAGS
flags.DEFINE_enum(
    "sampler_type",
    SamplerType.GREEDY.value,
    enum_values=[SamplerType.GREEDY.value, SamplerType.SAMPLER.value],
    help="The type of sampler for picking next tokens from LLM outputs.",
)


def tokenize(tokenizer, input: list[str]):
    pad_idx = tokenizer.pad_token_id
    lines = [
        tokenizer.apply_chat_template([{"role": "user", "content": l}], tokenize=False, add_generation_prompt=True)
        for l in input
    ]
    lines = [tokenizer.encode(line) for line in lines]
    max_l = max(len(line) for line in lines)  # left-pad to max line length.
    buffer_len = 2 ** math.ceil(math.log2(max(max_l, 1)))  # right-pad to buffer length.
    return jnp.array([np.pad(l, (max_l - len(l), buffer_len - max_l), constant_values=pad_idx) for l in lines]), max_l


def run_model(sampler_type: str = "greedy"):
    model_ckpt_path = snapshot_download("Qwen/Qwen3-0.6B")

    query = ["Why is the sky blue instead of any other color like purple?", "Who am I?"]

    tokenizer = AutoTokenizer.from_pretrained(model_ckpt_path)
    tokens, max_len = tokenize(tokenizer, query)
    batch_size, _ = tokens.shape

    cache_size, gen_steps = 128, 11
    assert cache_size >= max_len + gen_steps, f"Cache size ({cache_size}) must be >= {max_len} + {gen_steps}"

    config = modeling.ModelCfg.qwen3_0_6b()
    fsdp, tp = 1, jax.device_count()  # change this to meet your sharding setup.
    mesh = jax.make_mesh((fsdp, tp), ("fsdp", "tp"))
    with mesh:
        model = params.create_model_from_safe_tensors(model_ckpt_path, config, mesh)
        cache = modeling.init_cache(
            num_layers=config.num_layers,
            batch_size=batch_size,
            cache_size=cache_size,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
        )
    graphdef, state = nnx.split((model, cache))
    state = jax.tree.leaves(state)  # Better perf from flattened jax state due to no pytree trasversals.

    # prefill
    if sampler_type == "greedy":
        sampler = GreedySampler()
    elif sampler_type == "sampler":
        sampler = Sampler(0.8, 10, 1.0)
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")

    key = jax.random.key(0)
    next_tokens, state = modeling.forward(graphdef, state, tokens, tokenizer.pad_token_id, sampler, key)

    # decode - warmup
    tokens_list = [next_tokens]
    next_tokens, state = modeling.forward(graphdef, state, next_tokens, tokenizer.pad_token_id, sampler, key)
    tokens_list.append(next_tokens)

    # profile
    jax.profiler.start_trace("/tmp/profile-data")
    for i in range(5):
        next_tokens, state = modeling.forward(graphdef, state, next_tokens, tokenizer.pad_token_id, sampler, key)
        tokens_list.append(next_tokens)
    jax.block_until_ready(tokens_list)
    jax.profiler.stop_trace()

    # measure time:
    t = time.perf_counter()
    decode_steps = 128
    for i in range(decode_steps):
        next_tokens, state = modeling.forward(graphdef, state, next_tokens, tokenizer.pad_token_id, sampler, key)
        tokens_list.append(next_tokens)
    jax.block_until_ready(tokens_list)
    print(f"Time: {(time.perf_counter() - t) / decode_steps:.4f} s per step")

    tokens_list = jnp.concatenate(tokens_list, axis=-1)
    for i, q in enumerate(query):
        print(f"User:\n {q}")
        print(f"Answer:\n {tokenizer.decode(tokens_list[i], skip_special_tokens=True)}\n\n")


def main(argv):
    run_model(FLAGS.sampler_type)


if __name__ == "__main__":
    app.run(main)


__all__ = ["run_model"]
