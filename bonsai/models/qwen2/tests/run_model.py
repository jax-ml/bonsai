import jax
import os
import jax.numpy as jnp
import numpy as np
from jax import P
from jax._src.mesh import AxisType
from transformers import AutoTokenizer

from bonsai.models.qwen2 import modeling, params
from bonsai.utils import GreedySampler, Sampler


def tokenize(tokenizer, input: list[str], shd: P | None = None):
    pad_idx = tokenizer.pad_token_id
    lines = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": l}], tokenize=False, add_generation_prompt=True
        )
        for l in input
    ]
    lines = [tokenizer.encode(line) for line in lines]
    max_len = max(len(line) for line in lines)  # Right-align, left-padding to the max token length.
    return jnp.array([np.pad(l, (max_len - len(l), 0), constant_values=pad_idx) for l in lines], out_sharding=shd)


def run_model():

    model_ckpt_path = os.path.expanduser("~/.cache/modelscope/hub/models/Qwen/Qwen2-7B")

    # Disable sharding - run on single GPU
    config = modeling.ModelConfig.qwen2_7b(use_sharding=False)
    # mesh, batch_shd = None, None

    mesh = jax.make_mesh((1, 4), ("fsdp", "tp"), axis_types=(AxisType.Explicit, AxisType.Explicit))
    batch_shd = P("fsdp", None)
    jax.set_mesh(mesh)

    query = [
        "why sky is blue?",
    ]

    tokenizer = AutoTokenizer.from_pretrained(
        model_ckpt_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    print(f"Tokenizer special tokens:")
    print(f"  eos_token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"  pad_token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    print()
    tokens = tokenize(tokenizer, query, batch_shd)
    batch_size, token_len = tokens.shape

    generate_steps = 1024
    print(f"\nLoading model...")
    model = params.create_model_from_safe_tensors(model_ckpt_path, config, mesh)
    print(f"Model loaded successfully\n")

    cache = model.init_cache(config, batch_size, token_len, generate_steps)

    key = jax.random.key(0)
    sampler = Sampler(temperature=0.7, top_p=0.9, top_k=20)
    jit_sampler = jax.jit(sampler)

    logits, cache = modeling.forward(model, cache, tokens, tokenizer.pad_token_id)

    tokens_list = []
    finished = jnp.zeros((batch_size,), dtype=jnp.bool_)

    im_end_token_id = tokenizer.encode("<|im_end|>")[0]
    for i in range(generate_steps):
        # CRITICAL: Split key for each step to avoid deterministic sampling
        key, subkey = jax.random.split(key)
        next_tokens = jit_sampler(logits, key=subkey)

        current_token_id = int(next_tokens.squeeze(-1)[0])

        # Only check for actual EOS token
        is_eos = (next_tokens.squeeze(-1) == tokenizer.eos_token_id)
        is_im_end = (next_tokens.squeeze(-1) == im_end_token_id)


        finished = finished | is_eos | is_im_end

        tokens_list.append(next_tokens)

        if finished.all():
            print(f"✓ Generation stopped at step {i+1}/{generate_steps} (EOS token reached)")
            break

        # Continue generation
        logits, cache = modeling.forward(model, cache, next_tokens, tokenizer.pad_token_id)

    all_output_tokens = jax.device_get(jnp.concatenate(tokens_list, axis=-1))
    for i, q in enumerate(query):
        print(f"User:\n {q}")
        seq_tokens = all_output_tokens[i]
        eos_idx = np.where(seq_tokens == tokenizer.eos_token_id)[0]
        if eos_idx.size > 0:
            seq_tokens = seq_tokens[: eos_idx[0]]
        decoded = tokenizer.decode(seq_tokens, skip_special_tokens=True)
        print(f"Answer ({len(seq_tokens)} tokens):\n {decoded}\n\n")


if __name__ == "__main__":
    run_model()


__all__ = ["run_model"]