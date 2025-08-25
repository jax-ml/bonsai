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
from flax import nnx
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from bonsai.models.llada_8b import modeling as modeling
from bonsai.models.llada_8b import params as params

# Config / paths
HF_REPO = "GSAI-ML/LLaDA-8B-Instruct"
MODEL_CP_PATH = "/tmp/models-bonsai/" + HF_REPO.split("/")[1]

if not os.path.isdir(MODEL_CP_PATH):
    snapshot_download(HF_REPO, local_dir=MODEL_CP_PATH)


# Tokenization helper
def tokenize(tokenizer, inputs: list[str]):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        # fall back to eos as pad if the tokenizer doesn't define pad
        pad_id = tokenizer.eos_token_id
    lines = [
        tokenizer.apply_chat_template([{"role": "user", "content": l}], tokenize=False, add_generation_prompt=True)
        for l in inputs
    ]
    encoded = [tokenizer.encode(s) for s in lines]
    max_l = max(len(e) for e in encoded) if encoded else 1
    buffer_len = 2 ** math.ceil(math.log2(max(max_l, 1)))
    batch = np.stack([np.pad(e, (0, buffer_len - len(e)), constant_values=pad_id) for e in encoded], axis=0)
    return jnp.array(batch), pad_id, max_l, buffer_len


# Demo queries
queries = [
    "Why is the sky blue instead of a different color like purple?",
    "Who invented the World Wide Web?",
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_CP_PATH, use_fast=True)
tokens, pad_id, max_len, token_len = tokenize(tokenizer, queries)

# Model + params
cfg = modeling.ModelConfig.llada_8b_instruct()
# Load weights into a constructed model
model = params.create_llada_from_pretrained(MODEL_CP_PATH, cfg)
graphdef, state = nnx.split((model, None))
state = jax.tree.leaves(state)

# Generation loop
batch_size = tokens.shape[0]
gen_steps = 32  # how many new tokens to generate per sequence

# Prefill with the prompt
next_tokens, state = modeling.forward(graphdef, state, tokens, pad_id)

# Decode loop (one token at a time)
generated = [next_tokens]
t_start = None
for step in range(gen_steps):
    if step == 1:
        # avoid XLA warmup in timing
        jax.profiler.start_trace("/tmp/profile-llada")  # optional profiling
    next_tokens, state = modeling.forward(graphdef, state, next_tokens, pad_id)
    generated.append(next_tokens)
    if step == 5:
        jax.block_until_ready(generated)
        jax.profiler.stop_trace()
        t_start = time.perf_counter()

jax.block_until_ready(generated)
if t_start is not None:
    print(f"Time for steps 6..{gen_steps}: {time.perf_counter() - t_start:.4f} s")

# Pretty-print outputs
generated = jnp.concatenate(generated, axis=-1)  # (B, T_gen)
for i, q in enumerate(queries):
    text = tokenizer.decode(np.array(generated[i]).tolist(), skip_special_tokens=True)
    print(f"\nUser:\n{q}\n\nAnswer:\n{text}\n")
