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

"""Example script for running Wan2.1-T2V-1.3B text-to-video generation."""

import time

import jax
import jax.numpy as jnp
from flax import nnx
from huggingface_hub import snapshot_download

from bonsai.models.wan2 import modeling, params


def get_t5_text_embeddings(prompt: str, max_length: int = 512):
    """
    Get text embeddings from T5 encoder.

    1. Load a pretrained T5 model (e.g., google-t5/t5-base or UMT5)
    2. Tokenize the prompt
    3. Encode to get embeddings
    """
    try:
        from transformers import AutoTokenizer, FlaxAutoModel

        tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
        model = FlaxAutoModel.from_pretrained("google/umt5-xxl")

        # Tokenize with padding
        inputs = tokenizer(
            prompt,
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

        # Get embeddings
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state  # [1, seq_len, 4096]

        return embeddings

    except ImportError:
        print("transformers not installed, using dummy embeddings")
        print("Install with: pip install transformers")
        # Return dummy embeddings
        return jnp.zeros((1, max_length, 4096))


def decode_video_latents(latents: jax.Array):
    """
    Decode video latents to RGB frames using Wan-VAE.

    NOTE: This is a placeholder. In a real implementation, you would:
    1. Load the Wan-VAE decoder from the checkpoint
    2. Decode latents to RGB video

    For now, we return a placeholder video.
    """
    print("⚠ VAE decoder not implemented, returning dummy video")
    b, t, _h, _w, _c = latents.shape
    # Upsample from latent size (60x60) to 480p
    video_h, video_w = 480, 480
    video = jnp.zeros((b, t, video_h, video_w, 3))
    return video


def run_model():
    """Run Wan2.1-T2V-1.3B text-to-video generation."""

    print("=" * 60)
    print("Wan2.1-T2V-1.3B Text-to-Video Generation Demo")
    print("=" * 60)

    # Configuration
    model_ckpt_path = snapshot_download("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    config = modeling.ModelConfig.wan2_1_1_3b(use_sharding=False)

    # For sharding (multi-GPU), uncomment:
    # from jax.sharding import AxisType
    # mesh = jax.make_mesh((2, 2), ("fsdp", "tp"), axis_types=(AxisType.Explicit, AxisType.Explicit))
    # jax.set_mesh(mesh)

    # Text prompts
    prompts = [
        "A beautiful sunset over the ocean with waves crashing on the shore",
    ]

    print(f"\nPrompt: {prompts[0]}")
    print(f"Model: Wan2.1-T2V-1.3B ({config.num_layers} layers, {config.hidden_dim} dim)")
    print(f"Video: {config.num_frames} frames @ 480p")

    # Step 1: Get text embeddings
    print("\n[1/4] Encoding text with T5...")
    text_embeds = get_t5_text_embeddings(prompts[0], max_length=config.max_text_len)
    print(f"      Text embeddings shape: {text_embeds.shape}")

    # Step 2: Load DiT model
    print("\n[2/4] Loading Diffusion Transformer weights...")
    try:
        model = params.create_model_from_safe_tensors(model_ckpt_path, config, mesh=None, load_transformer_only=True)
        print("      Model loaded successfully")
    except Exception as e:
        print(f"      Could not load weights: {e}")
        print("      Creating model from scratch (random weights)...")
        model = modeling.Wan2DiT(config, rngs=nnx.Rngs(params=0))

    # Step 3: Generate video latents
    print("\n[3/4] Generating video latents...")
    print(f"      Using {config.num_inference_steps} diffusion steps")
    print(f"      Guidance scale: {config.guidance_scale}")

    key = jax.random.PRNGKey(42)
    start_time = time.time()

    latents = modeling.generate_video(
        model=model,
        text_embeds=text_embeds,
        num_frames=config.num_frames,
        latent_size=config.latent_size,
        num_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        key=key,
    )

    generation_time = time.time() - start_time
    print(f"      ✓ Generated latents in {generation_time:.2f}s")
    print(f"      Latents shape: {latents.shape}")

    # Step 4: Decode to video
    print("\n[4/4] Decoding latents to video...")
    video = decode_video_latents(latents)
    print(f"      Video shape: {video.shape}")

    # Summary
    print("\n" + "=" * 60)
    print("✓ Generation Complete!")
    print("=" * 60)
    print(f"Total time: {generation_time:.2f}s")
    print(f"FPS: {config.num_frames / generation_time:.2f}")
    print("\nTo save the video, implement VAE decoder and use:")
    print("  import imageio")
    print("  imageio.mimsave('output.mp4', video[0], fps=30)")
    print()

    return video


def run_simple_forward_pass():
    """
    Simple forward pass test without full generation pipeline.
    Useful for testing the model architecture.
    """
    print("=" * 60)
    print("Wan2.1-T2V-1.3B Simple Forward Pass Test")
    print("=" * 60)

    config = modeling.ModelConfig.wan2_1_1_3b()
    print(f"\nConfig: {config.num_layers} layers, {config.hidden_dim} dim")

    # Create model
    print("\n[1/2] Creating model...")
    model = modeling.Wan2DiT(config, rngs=nnx.Rngs(params=0, dropout=0))
    print("      Model created")

    # Create dummy inputs
    batch_size = 1
    latents = jnp.zeros(
        (batch_size, config.num_frames, config.latent_size[0], config.latent_size[1], config.input_dim)
    )  # [B, T, H, W, C]
    text_embeds = jnp.zeros((batch_size, config.max_text_len, config.text_embed_dim))
    timestep = jnp.array([25])  # Middle of diffusion process

    print("\n[2/2] Running forward pass...")
    print(f"      Latents: {latents.shape}")
    print(f"      Text embeds: {text_embeds.shape}")
    print(f"      Timestep: {timestep}")

    # Forward pass
    start_time = time.time()
    predicted_noise = model(latents, text_embeds, timestep, deterministic=True)
    forward_time = time.time() - start_time

    print("\n✓ Forward pass complete!")
    print(f"  Output shape: {predicted_noise.shape}")
    print(f"  Time: {forward_time:.3f}s")
    print(f"  Output range: [{predicted_noise.min():.3f}, {predicted_noise.max():.3f}]")
    print()

    return predicted_noise


if __name__ == "__main__":
    # Choose which demo to run:

    # Option 1: Full generation pipeline (requires checkpoint download)
    # run_model()

    # Option 2: Simple forward pass test (no checkpoint required)
    run_simple_forward_pass()


__all__ = ["run_model", "run_simple_forward_pass"]
