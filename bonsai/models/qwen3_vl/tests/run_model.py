import time

import jax
import jax.numpy as jnp
import numpy as np
from huggingface_hub import snapshot_download
from jax._src.mesh import AxisType
from transformers import AutoProcessor

from bonsai.models.qwen3_vl import modeling
from bonsai.models.qwen3_vl import params

# ============================================================================
# MESH CONFIGURATION - Modify these to enable sharding

# Set USE_SHARDING=True and MESH_SHAPE to enable distributed inference
# Examples:
#   MESH_SHAPE = (2, 4)  -> 2 FSDP devices x 4 TP devices (8 total)
#   MESH_SHAPE = (1, 4)  -> Pure tensor parallelism on 4 devices
#   MESH_SHAPE = (4, 1)  -> Pure FSDP on 4 devices
USE_SHARDING = False
MESH_SHAPE = (1, 1)  # (fsdp, tp) - only used when USE_SHARDING=True
# ============================================================================

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
EOS_TOKEN_ID = 151643


def generate(model, cache, input_ids, max_new_tokens: int = 50):
    """Fast greedy generation with JIT-compiled forward and KV-cache."""
    batch_size, seq_len = input_ids.shape
    generated_tokens = []

    # Prefill
    start = time.time()
    logits, cache = modeling.forward(model, cache, input_ids)
    logits.block_until_ready()
    prefill_time = time.time() - start
    print(f"   Prefill: {prefill_time:.2f}s ({seq_len} tokens)")

    next_token = jnp.argmax(logits, axis=-1, keepdims=True)
    generated_tokens.append(next_token)

    # Decode loop
    start = time.time()
    for step in range(max_new_tokens - 1):
        logits, cache = modeling.forward(model, cache, next_token)
        next_token = jnp.argmax(logits, axis=-1, keepdims=True)
        generated_tokens.append(next_token)

        # Stop on EOS
        if int(next_token[0, 0]) == EOS_TOKEN_ID:
            break

    decode_time = time.time() - start
    tokens_generated = len(generated_tokens)
    if decode_time > 0:
        print(f"   Decode: {decode_time:.2f}s ({tokens_generated} tokens, {tokens_generated / decode_time:.1f} tok/s)")
    else:
        print(f"   Decode: {decode_time:.2f}s ({tokens_generated} tokens)")

    return jnp.concatenate([input_ids] + generated_tokens, axis=1)


def generate_with_vision(
    model, cache, input_ids, pixel_values, image_grid_thw, token_type_ids, max_new_tokens: int = 50
):
    """Generation with vision inputs."""
    batch_size, seq_len = input_ids.shape
    generated_tokens = []

    # Prefill with vision
    start = time.time()
    logits, cache = modeling.forward_vision(model, cache, input_ids, pixel_values, image_grid_thw, token_type_ids)
    logits.block_until_ready()
    prefill_time = time.time() - start
    print(f"   Prefill (with vision): {prefill_time:.2f}s ({seq_len} tokens)")

    next_token = jnp.argmax(logits, axis=-1, keepdims=True)
    generated_tokens.append(next_token)

    # Decode loop (text-only after prefill)
    start = time.time()
    for step in range(max_new_tokens - 1):
        logits, cache = modeling.forward(model, cache, next_token)
        next_token = jnp.argmax(logits, axis=-1, keepdims=True)
        generated_tokens.append(next_token)

        if int(next_token[0, 0]) == EOS_TOKEN_ID:
            break

    decode_time = time.time() - start
    tokens_generated = len(generated_tokens)
    if decode_time > 0:
        print(f"   Decode: {decode_time:.2f}s ({tokens_generated} tokens, {tokens_generated / decode_time:.1f} tok/s)")

    return jnp.concatenate([input_ids] + generated_tokens, axis=1)


def main():
    print("=" * 60)
    print("Qwen3-VL JAX Fast Generation Demo")
    print("=" * 60)

    # Setup mesh context if sharding enabled
    use_fsdp = USE_SHARDING and MESH_SHAPE[0] > 1
    use_tp = USE_SHARDING and MESH_SHAPE[1] > 1

    if USE_SHARDING:
        print(f"\nSharding enabled: MESH_SHAPE={MESH_SHAPE}, use_fsdp={use_fsdp}, use_tp={use_tp}")
        mesh = jax.make_mesh(MESH_SHAPE, ("fsdp", "tp"), axis_types=(AxisType.Explicit, AxisType.Explicit))
        jax.set_mesh(mesh)
    else:
        print("\nSharding disabled (running on single device)")

    # Download model
    print("\n1. Downloading model...")
    model_path = snapshot_download(MODEL_ID)

    # Load Flax model
    print("\n2. Loading Flax model...")
    start = time.time()
    flax_config = modeling.Qwen3VLConfig.qwen3vl_2b(use_fsdp=use_fsdp, use_tp=use_tp)
    flax_model = params.create_model_from_safe_tensors(model_path, flax_config)
    print(f"   Model loaded in {time.time() - start:.2f}s")

    # Load processor
    processor = AutoProcessor.from_pretrained(model_path)

    # =========================================================================
    # Example 1: Text-only generation
    # =========================================================================
    print("\n" + "=" * 60)
    print("Example 1: Text-only Generation")
    print("=" * 60)

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "What is the capital of France? Answer in one word."}],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    input_ids = jnp.array(inputs["input_ids"].numpy())
    batch_size, seq_len = input_ids.shape

    # When sharding, batch size must be >= fsdp for divisibility
    if USE_SHARDING and batch_size < MESH_SHAPE[0]:
        print(f"WARNING: batch_size={batch_size} < fsdp={MESH_SHAPE[0]}, padding batch")
        pad_size = MESH_SHAPE[0] - batch_size
        input_ids = jnp.concatenate([input_ids, jnp.zeros((pad_size, seq_len), dtype=jnp.int32)], axis=0)
        batch_size = MESH_SHAPE[0]

    print(f"\n3. Input: {seq_len} tokens (batch_size={batch_size})")

    # Warm up JIT - use minimum batch size that works with sharding
    print("\n4. Warming up JIT (first run is slow)...")
    warm_batch = MESH_SHAPE[0] if USE_SHARDING else 1
    warm_input = jnp.zeros((warm_batch, 1), dtype=jnp.int32)
    warm_cache = modeling.init_cache(flax_config, warm_batch, 1, 10)
    _ = modeling.forward(flax_model, warm_cache, warm_input)
    print("   JIT warm-up complete")

    # Generate
    print("\n5. Generating...")
    cache = modeling.init_cache(flax_config, batch_size, seq_len, generate_steps=100)
    generated_ids = generate(flax_model, cache, input_ids, max_new_tokens=30)

    # Decode
    generated_ids_trimmed = generated_ids[:, seq_len:]
    output_text = processor.batch_decode(
        np.asarray(generated_ids_trimmed),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    print("\n" + "-" * 40)
    print("Q: What is the capital of France?")
    print(f"A: {output_text[0]}")
    print("-" * 40)

    # =========================================================================
    # Example 2: Second text generation (JIT is warm)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Example 2: Second Generation (JIT warm)")
    print("=" * 60)

    messages2 = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "How to cook lasagna? Answer in 20 words."}],
        }
    ]

    inputs2 = processor.apply_chat_template(
        messages2,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    input_ids2 = jnp.array(inputs2["input_ids"].numpy())
    batch2 = MESH_SHAPE[0] if USE_SHARDING else 1
    if USE_SHARDING and input_ids2.shape[0] < batch2:
        input_ids2 = jnp.concatenate(
            [input_ids2, jnp.zeros((batch2 - 1, input_ids2.shape[1]), dtype=jnp.int32)], axis=0
        )
    seq_len2 = input_ids2.shape[1]
    print(f"\nInput: {seq_len2} tokens")

    cache2 = modeling.init_cache(flax_config, batch2, seq_len2, generate_steps=100)
    generated_ids2 = generate(flax_model, cache2, input_ids2, max_new_tokens=60)

    generated_ids_trimmed2 = generated_ids2[:, seq_len2:]
    output_text2 = processor.batch_decode(
        np.asarray(generated_ids_trimmed2),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    print("\n" + "-" * 40)
    print("Q: How to cook lasagna?")
    print(f"A: {output_text2[0]}")
    print("-" * 40)

    # =========================================================================
    # Example 3: Image + Text generation
    # =========================================================================
    print("\n" + "=" * 60)
    print("Example 3: Image + Text Generation")
    print("=" * 60)

    messages_vision = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.jpeg",
                },
                {"type": "text", "text": "What is in this image? Answer in detail"},
            ],
        }
    ]

    print("\n3. Processing image input...")
    inputs_vision = processor.apply_chat_template(
        messages_vision,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    input_ids_vision = jnp.array(inputs_vision["input_ids"].numpy())
    seq_len_vision = input_ids_vision.shape[1]
    print(f"   Input IDs: {seq_len_vision} tokens")

    # Check for vision inputs
    if "pixel_values" in inputs_vision:
        pixel_values = jnp.array(inputs_vision["pixel_values"].numpy())
        image_grid_thw = jnp.array(inputs_vision["image_grid_thw"].numpy())

        # Create token_type_ids (1 for image tokens, 0 for text)
        # Image token ID is 151655
        token_type_ids = (input_ids_vision == flax_config.image_token_id).astype(jnp.int32)

        print(f"   Pixel values: {pixel_values.shape}")
        print(f"   Image grid THW: {image_grid_thw}")
        print(f"   Image tokens in sequence: {int(token_type_ids.sum())}")

        # Initialize cache
        batch_vision = MESH_SHAPE[0] if USE_SHARDING else 1
        if USE_SHARDING and input_ids_vision.shape[0] < batch_vision:
            input_ids_vision = jnp.concatenate(
                [input_ids_vision, jnp.zeros((batch_vision - 1, input_ids_vision.shape[1]), dtype=jnp.int32)], axis=0
            )
            pixel_values = jnp.concatenate(
                [pixel_values, jnp.zeros((batch_vision - 1, *pixel_values.shape[1:]), dtype=pixel_values.dtype)], axis=0
            )
            image_grid_thw = jnp.concatenate([image_grid_thw, image_grid_thw], axis=0)[:batch_vision]
            token_type_ids = jnp.concatenate(
                [token_type_ids, jnp.zeros((batch_vision - 1, token_type_ids.shape[1]), dtype=jnp.int32)], axis=0
            )
        cache_vision = modeling.init_cache(flax_config, batch_vision, seq_len_vision, generate_steps=200)

        print("\n4. Generating with vision...")
        generated_ids_vision = generate_with_vision(
            flax_model,
            cache_vision,
            input_ids_vision,
            pixel_values,
            image_grid_thw,
            token_type_ids,
            max_new_tokens=200,
        )

        # Decode
        generated_ids_trimmed_vision = generated_ids_vision[:, seq_len_vision:]
        output_text_vision = processor.batch_decode(
            np.asarray(generated_ids_trimmed_vision),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        print("\n" + "-" * 40)
        print("Q: What is in this image?")
        print(f"A: {output_text_vision[0]}")
        print("-" * 40)
    else:
        print("   No pixel_values in inputs - vision processing may have failed.")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
