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


def generate_with_vision(
    model, cache, input_ids, pixel_values, image_grid_thw, token_type_ids, max_new_tokens: int = 50
):
    """Generation with vision inputs."""
    batch_size, seq_len = input_ids.shape
    generated_tokens = []

    # Prefill with vision
    logits, cache = modeling.forward_vision(model, cache, input_ids, pixel_values, image_grid_thw, token_type_ids)

    next_token = jnp.argmax(logits, axis=-1, keepdims=True)
    generated_tokens.append(next_token)

    # Decode loop (text-only after prefill)
    for step in range(max_new_tokens - 1):
        logits, cache = modeling.forward(model, cache, next_token)
        next_token = jnp.argmax(logits, axis=-1, keepdims=True)
        generated_tokens.append(next_token)

        # Stop on EOS - use numpy to avoid sharding issues with indexing
        if int(np.array(next_token)[0, 0]) == EOS_TOKEN_ID:
            break

    # Use numpy for concatenation to avoid sharding mismatches
    all_tokens = [np.array(input_ids)] + [np.array(t) for t in generated_tokens]
    return jnp.array(np.concatenate(all_tokens, axis=1))


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

    print("\n" + "=" * 60)
    print("3. Example : Image + Text Generation")
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
