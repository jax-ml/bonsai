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
from functools import partial

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
MODEL_CP_PATH = "/tmp/models-bonsai/" + HF_REPO.split("/")[1]

if not os.path.isdir(MODEL_CP_PATH):
    snapshot_download(HF_REPO, local_dir=MODEL_CP_PATH)


# Tokenization helper
def tokenize(tokenizer, inputs: list[str]):
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id  # fall back to eos
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
prompt = ["Why is the sky blue instead of a different color like purple?", "Who am I?"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_CP_PATH, use_fast=True)
tokens, pad_id, max_len, token_len = tokenize(tokenizer, prompt)
batch_size = tokens.shape[0]

cfg = modeling.ModelConfig.llada_8b_instruct()
model = params.create_llada_from_pretrained(MODEL_CP_PATH, cfg)
graphdef, state = nnx.split(model)


def generate_for_benchmark(
    graphdef: nnx.GraphDef[nnx.Module],
    init_state: nnx.State,
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
    profile_start_step: int | None = None,  # global step index at which to start profiling
    profile_stop_step: int | None = None,  # global step index at which to stop profiling
):
    """
    Returns:
      x: final tokens (B, Lp+gen_length)
      state: final nnx.State
      step_times: list[float] of per-step latencies (seconds), length == steps
    """
    B, Lp = prompt.shape
    x = jnp.full((B, Lp + gen_length), mask_id, dtype=jnp.int32).at[:, :Lp].set(prompt)
    prompt_index = x != mask_id

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    state = init_state

    # --- define do_step exactly like the scan body, but as a pure function
    def _do_step(x, state, rng, i, stop, num_transfer_tokens, *, remasking: str):
        mask_index = x == mask_id

        # CFG path: run conditional + unconditional; keep state from the conditional path
        if cfg_scale > 0.0:
            un_x = jnp.where(prompt_index, mask_id, x)
            x_stack = jnp.concatenate([x, un_x], axis=0)  # (2B, L)
            logits_both, state = modeling.forward(graphdef, state, x_stack)  # (2B,L,V)
            logits, logits_un = jnp.split(logits_both, 2, axis=0)
            logits = logits_un + (cfg_scale + 1.0) * (logits - logits_un)
            state_next = state
        else:
            logits, state_next = modeling.forward(graphdef, state, x)

        # Noise + argmax
        rng, sub = jax.random.split(rng)
        logits_noisy = modeling.add_gumbel_noise(logits, temperature, sub)
        x0 = jnp.argmax(logits_noisy, axis=-1).astype(jnp.int32)

        # Confidence
        if remasking == "low_confidence":
            p = jax.nn.softmax(logits, axis=-1)
            x0_p = jnp.squeeze(jnp.take_along_axis(p, x0[..., None], axis=-1), axis=-1)
        elif remasking == "random":
            rng, sub = jax.random.split(rng)
            x0_p = jax.random.uniform(sub, shape=x0.shape, dtype=logits.dtype)
        else:
            raise NotImplementedError(remasking)

        # Forbid beyond current block
        neg_inf = jnp.array(-jnp.inf, dtype=x0_p.dtype)
        pos = jnp.arange(x0.shape[1])[None, :]
        x0_p = jnp.where(pos >= stop, neg_inf, x0_p)

        x0_sel = jnp.where(mask_index, x0, x)
        conf = jnp.where(mask_index, x0_p, neg_inf)

        # variable-k per row
        k_vec = num_transfer_tokens[:, i]
        transfer_index = modeling.row_topk_mask_vmapped(conf, k_vec)

        x = jnp.where(transfer_index, x0_sel, x)
        return x, state_next, rng

    # JIT the step function for realistic per-step timing.
    do_step = partial(_do_step, remasking=remasking)
    do_step_jit = jax.jit(do_step, donate_argnames=["state"])  # donate state

    # Optional warmup compile outside the timing window
    # (uses the very first block/step's shapes & branches)
    start0 = Lp + 0 * block_length
    stop0 = Lp + (0 + 1) * block_length
    block_mask0 = lax.dynamic_slice_in_dim(x, start_index=start0, slice_size=block_length, axis=1)
    block_mask_index0 = block_mask0 == mask_id
    num_transfer_tokens0 = modeling.get_num_transfer_tokens(block_mask_index0, steps_per_block)
    x_w, state_w, rng = do_step_jit(x, state, rng, jnp.int32(0), jnp.int32(stop0), num_transfer_tokens0)
    _ = x_w.block_until_ready()  # ensure compile happens before we start measuring

    step_times: list[float] = []
    global_step = 0

    # Main loop with per-step timing
    for b_idx in range(num_blocks):
        start = Lp + b_idx * block_length
        stop = Lp + (b_idx + 1) * block_length

        block_mask = lax.dynamic_slice_in_dim(x, start_index=start, slice_size=block_length, axis=1)
        block_mask_index = block_mask == mask_id
        num_transfer_tokens = modeling.get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            # Start trace exactly at profile_start_step if requested
            if profile_dir and profile_start_step is not None and global_step == profile_start_step:
                jax.profiler.start_trace(profile_dir)

            t0 = time.perf_counter()
            x, state, rng = do_step_jit(x, state, rng, jnp.int32(i), jnp.int32(stop), num_transfer_tokens)
            _ = x.block_until_ready()  # sync to measure device time
            dt = time.perf_counter() - t0
            step_times.append(dt)

            # Stop trace after profile_stop_step if requested
            if profile_dir and profile_stop_step is not None and global_step == profile_stop_step:
                jax.profiler.stop_trace()

            global_step += 1

    return x, state, step_times


x_final, state_final, step_times = generate_for_benchmark(
    graphdef,
    state,
    prompt,
    steps=64,
    gen_length=32,
    block_length=32,
    temperature=0.2,
    cfg_scale=1.0,
    remasking="low_confidence",
    mask_id=126336,
    rng=jax.random.PRNGKey(0),
    profile_dir="/tmp/profile-data",
    profile_start_step=1,
    profile_stop_step=5,
)

print("Per-step (s):", [f"{t:.4f}" for t in step_times])
print(f"Avg: {sum(step_times)/len(step_times):.4f} s")

# Decode
B, L_total = x_final.shape


def decode_batch(seqs):
    if hasattr(tokenizer, "batch_decode"):
        return tokenizer.batch_decode(seqs, skip_special_tokens=False)
    return [tokenizer.decode(s) for s in seqs]


decoded_full = decode_batch(np.asarray(x_final))


print("Full text (prompt + generated)")
for i, text in enumerate(decoded_full):
    print(f"[Sample {i}] {text}\n")
