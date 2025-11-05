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
from jax import lax
from transformers import AutoTokenizer

from bonsai.models.llada_8b import modeling as modeling
from bonsai.models.llada_8b import params as params

# Config / paths
HF_REPO = "GSAI-ML/LLaDA-8B-Instruct"
MODEL_CP_PATH = snapshot_download(HF_REPO, local_dir="/tmp/models-bonsai")


# Tokenization helper
def tokenize(tokenizer, inputs: list[str]):
    pad_id = 126336  # Mask Token
    lines = [
        tokenizer.apply_chat_template([{"role": "user", "content": l}], tokenize=False, add_generation_prompt=True)
        for l in inputs
    ]
    encoded = [tokenizer(s)["input_ids"] for s in lines]
    max_l = max(len(e) for e in encoded) if encoded else 1
    buffer_len = 2 ** math.ceil(math.log2(max(max_l, 1)))
    batch = np.stack([np.pad(e, (0, buffer_len - len(e)), constant_values=pad_id) for e in encoded], axis=0)
    return jnp.array(batch), pad_id, max_l, buffer_len


# Benchmarking helper
def generate_for_benchmark(
    graphdef: nnx.GraphDef[nnx.Module],
    state: nnx.State,
    prompt: jax.Array,
    *,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    rng: jax.Array = jax.random.PRNGKey(0),
    profile_dir: str | None = None,
    profile_start_step: int | None = None,  # global step index to start profiling
    profile_stop_step: int | None = None,  # global step index to stop profiling
):
    """
    Returns:
      x: final tokens (B, prompt_len + gen_length)
      step_times: list[float] of per-step latencies (seconds), length == `steps`
    """
    B, prompt_len = prompt.shape
    x = jnp.full((B, prompt_len + gen_length), mask_id, dtype=jnp.int32).at[:, :prompt_len].set(prompt)
    prompt_index = x != mask_id

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    step_times: list[float] = []
    global_step = 0

    for block_idx in range(num_blocks):
        start = prompt_len + block_idx * block_length
        stop = prompt_len + (block_idx + 1) * block_length

        # Per-block transfer schedule matches the reference
        block_mask = lax.dynamic_slice_in_dim(x, start_index=start, slice_size=block_length, axis=1)
        block_mask_index = block_mask == mask_id
        num_transfer_tokens = modeling.get_num_transfer_tokens(
            block_mask_index, steps_per_block
        )  # (B, steps_per_block)

        for i in range(steps_per_block):
            if profile_dir and profile_start_step is not None and global_step == profile_start_step:
                jax.profiler.start_trace(profile_dir)

            t0 = time.perf_counter()
            x, rng = modeling.generate_step(
                graphdef=graphdef,
                state=state,
                x=x,
                prompt_index=prompt_index,
                rng=rng,
                step_idx=i,
                start=start,
                stop=stop,
                num_transfer_tokens=num_transfer_tokens,
                temperature=temperature,
                cfg_scale=cfg_scale,
                remasking=remasking,
                mask_id=mask_id,
            )
            _ = x.block_until_ready()  # ensure device sync for accurate timing
            step_times.append(time.perf_counter() - t0)

            if profile_dir and profile_stop_step is not None and global_step == profile_stop_step:
                jax.profiler.stop_trace()

            global_step += 1

    return x, step_times


def decode_batch(tokenizer, seqs):
    if hasattr(tokenizer, "batch_decode"):
        return tokenizer.batch_decode(seqs, skip_special_tokens=True)
    return [tokenizer.decode(s) for s in seqs]


def run_model():
    # Demo prompts
    prompt = [
        "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
        "Johnny picked 8 apples this morning and put them on his desk. Bonnie eats 3 of them. How many apples does Johnny have left?",
    ]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CP_PATH, use_fast=True)
    tokens, _, _, _ = tokenize(tokenizer, prompt)
    print("Tokenized batch")

    cfg = modeling.ModelConfig.llada_8b_instruct()
    model = params.create_llada_from_pretrained(MODEL_CP_PATH, cfg)
    graphdef, state = nnx.split(model)
    print("Loaded model")

    x_final, step_times = generate_for_benchmark(
        graphdef,
        state,
        tokens,
        steps=128,
        gen_length=128,
        block_length=32,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        rng=jax.random.PRNGKey(0),
        profile_dir="/tmp/profile-data",
        profile_start_step=1,
        profile_stop_step=5,
    )

    print("Per-step (s):", [f"{t:.4f}" for t in step_times])
    print(f"Avg: {sum(step_times) / len(step_times):.4f} s")

    # Decode
    decoded_full = decode_batch(tokenizer, np.asarray(x_final))

    print("Full text (prompt + generated)")
    for i, text in enumerate(decoded_full):
        print(f"[Sample {i}] {text}\n")


if __name__ == "__main__":
    run_model()
