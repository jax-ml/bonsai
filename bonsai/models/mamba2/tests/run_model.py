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

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from transformers import AutoTokenizer

from bonsai.models.mamba2 import modeling, params


@jax.jit
def _decode_step(
    model: modeling.Mamba2ForCausalLM,
    tokens: jnp.ndarray,
    cur: jnp.ndarray,
) -> jnp.ndarray:
    out = model(tokens, labels=None)
    prev_logits = jax.lax.dynamic_index_in_dim(out["logits"], cur - jnp.int32(1), axis=1, keepdims=False)
    next_tok = jnp.argmax(prev_logits, axis=-1)
    tokens = tokens.at[:, cur].set(next_tok)
    return tokens


def _greedy_generate(
    model: modeling.Mamba2ForCausalLM,
    prompt_ids_1d: jnp.ndarray,
    *,
    max_new_tokens: int,
    pad_id: int,
    buffer_len: int,
) -> jnp.ndarray:
    prompt_len = int(prompt_ids_1d.shape[0])
    total_len = prompt_len + max_new_tokens
    if buffer_len < total_len:
        raise ValueError(f"buffer_len ({buffer_len}) must be >= prompt_len+max_new_tokens ({total_len})")

    # Fixed shape buffer: (batch=1, buffer_len)
    tokens = jnp.full((1, buffer_len), int(pad_id), dtype=jnp.int32)
    tokens = tokens.at[0, :prompt_len].set(jnp.asarray(prompt_ids_1d, dtype=jnp.int32))

    # Warmup compile once for this (1, buffer_len) shape.
    tokens = _decode_step(model, tokens, jnp.asarray(prompt_len, dtype=jnp.int32))
    jax.block_until_ready(tokens)

    # Now generate the remaining tokens.
    for t in range(1, max_new_tokens + 1):
        cur = jnp.asarray(prompt_len + t, dtype=jnp.int32)  # dynamic scalar => no per-token recompiles
        tokens = _decode_step(model, tokens, cur)

    out = tokens[:, :total_len]
    jax.block_until_ready(out)
    return out


def _to_host_1d(x: jnp.ndarray) -> jnp.ndarray:
    try:
        return x.get(out_sharding=P(None))
    except Exception:
        return jax.device_get(x)


@jax.jit
def _cached_decode_step(
    model: modeling.Mamba2ForCausalLM,
    token: jnp.ndarray,
    cache: modeling.Mamba2Cache,
) -> tuple[jnp.ndarray, modeling.Mamba2Cache]:
    """Single decode step with cache.

    Args:
        model: The Mamba2 model
        token: Single token (batch, 1)
        cache: Current cache state

    Returns:
        next_token: Next predicted token (batch,)
        new_cache: Updated cache
    """
    out = model(token, labels=None, cache=cache)
    logits = out["logits"][:, -1, :]
    next_token = jnp.argmax(logits, axis=-1)
    return next_token, out["cache"]


def _greedy_generate_cached(
    model: modeling.Mamba2ForCausalLM,
    prompt_ids_1d: jnp.ndarray,
    *,
    max_new_tokens: int,
    cfg: modeling.Mamba2Config,
) -> jnp.ndarray:
    """Greedy generation with SSM state caching (O(n) complexity).

    Args:
        model: Mamba2ForCausalLM model
        prompt_ids_1d: Prompt token IDs (seq_len,)
        max_new_tokens: Number of tokens to generate
        cfg: Model config for cache initialization

    Returns:
        Generated tokens including prompt (1, prompt_len + max_new_tokens)
    """
    prompt_len = int(prompt_ids_1d.shape[0])

    # Prefill: Process prompt in one pass to initialize cache
    prompt_2d = jnp.expand_dims(prompt_ids_1d, axis=0)  # (1, prompt_len)
    out = model(prompt_2d, labels=None, cache=None)
    cache = out["cache"]

    # Get first generated token from prefill
    logits = out["logits"][:, -1, :]
    next_token = jnp.argmax(logits, axis=-1, keepdims=True)  # (1, 1)

    generated = [prompt_2d, next_token]

    # Decode: Generate remaining tokens one at a time using cache
    for _ in range(1, max_new_tokens):
        next_token, cache = _cached_decode_step(model, next_token, cache)
        next_token = jnp.expand_dims(next_token, axis=-1)  # (1, 1)
        generated.append(next_token)

    result = jnp.concatenate(generated, axis=1)
    jax.block_until_ready(result)
    return result


def run_model(*, max_new_tokens: int = 32, use_cache: bool = True) -> None:
    """Run the Mamba2 generation smoke test.

    Args:
        max_new_tokens: Number of tokens to generate
        use_cache: If True, use cached generation (O(n)). If False, use non-cached (O(n²))
    """
    query = [
        "Why is the sky blue instead of any other color like purple?",
        "What is the capital city of England?",
    ]

    cfg = modeling.Mamba2Config(
        vocab_size=50288,
        hidden_size=768,
        state_size=128,
        num_hidden_layers=24,
        head_dim=64,
        expand=2,
        conv_kernel=4,
    )

    model = modeling.Mamba2ForCausalLM.from_pretrained("state-spaces/mamba2-130m", cfg=cfg)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    prompt_ids_list = [jnp.asarray(tokenizer.encode(q), dtype=jnp.int32) for q in query]

    if use_cache:
        print("Using cached generation (O(n) complexity)\n")
        for q, prompt_ids in zip(query, prompt_ids_list):
            tokens_2d = _greedy_generate_cached(
                model,
                prompt_ids,
                max_new_tokens=max_new_tokens,
                cfg=cfg,
            )

            host_ids = _to_host_1d(tokens_2d.at[0].get())
            generated_ids_only = host_ids[len(prompt_ids) :]
            text = tokenizer.decode(generated_ids_only.tolist(), skip_special_tokens=True)

            print(f"User:\n {q}")
            print(f"Answer:\n {text.strip()}\n")
    else:
        print("Using non-cached generation (O(n^2) complexity)\n")
        # Determine padding id robustly.
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            pad_id = cfg.pad_token_id

        # Share a single static buffer length across all prompts to reuse compilation.
        max_prompt_len = max(int(x.shape[0]) for x in prompt_ids_list)
        buffer_len = max_prompt_len + max_new_tokens

        for q, prompt_ids in zip(query, prompt_ids_list):
            tokens_2d = _greedy_generate(
                model,
                prompt_ids,
                max_new_tokens=max_new_tokens,
                pad_id=int(pad_id),
                buffer_len=buffer_len,
            )

            host_ids = _to_host_1d(tokens_2d.at[0].get())
            generated_ids_only = host_ids[len(prompt_ids) :]
            text = tokenizer.decode(generated_ids_only.tolist(), skip_special_tokens=True)

            print(f"User:\n {q}")
            print(f"Answer:\n {text.strip()}\n")


def run_forecaster() -> None:
    """Run a tiny Mamba2Forecaster smoke test (shape-only)."""

    model = params.create_random_forecaster(
        input_dim=10,
        d_model=64,
        n_layers=2,
        output_dim=1,
        forecast_horizon=24,
        seed=42,
    )

    x = jax.random.normal(jax.random.PRNGKey(0), (4, 100, 10))
    y = model(x)
    jax.block_until_ready(y)

    print(f"Forecaster input shape:  {tuple(x.shape)}")
    print(f"Forecaster output shape: {tuple(y.shape)}")


if __name__ == "__main__":
    run_model()
    run_forecaster()


__all__ = ["run_forecaster", "run_model"]
