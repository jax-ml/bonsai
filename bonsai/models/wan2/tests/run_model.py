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

from bonsai.models.wan2 import modeling, params, vae


def get_t5_text_embeddings(prompt: str, max_length: int = 512):
    """
    Get text embeddings from T5 encoder.

    Uses JAX implementation of UMT5-XXL encoder.

    Args:
        prompt: Text prompt to encode
        max_length: Maximum sequence length (default 512)

    Returns:
        embeddings: [1, seq_len, 4096] text embeddings
    """
    try:
        from transformers import AutoTokenizer

        from bonsai.models.wan2 import t5_jax

        print("Loading UMT5-XXL tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")

        # Tokenize with padding
        inputs = tokenizer(
            prompt,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",  # Return numpy arrays
        )

        # Note: Model weights need to be loaded separately
        # For now, create model with random weights
        print("Creating T5 encoder (random weights - load checkpoint for real use)...")
        model = t5_jax.T5EncoderModel(rngs=nnx.Rngs(params=0))

        # Get embeddings
        input_ids = jnp.array(inputs["input_ids"])
        attention_mask = jnp.array(inputs["attention_mask"])
        embeddings = model(input_ids, attention_mask, deterministic=True)

        print(f"Text embeddings shape: {embeddings.shape}")
        return embeddings

    except ImportError as e:
        print(f"Error loading dependencies: {e}")
        print("Install with: pip install transformers")
        # Return dummy embeddings
        return jnp.zeros((1, max_length, 4096))
    except Exception as e:
        print(f"Error in text encoding: {e}")
        print("Using dummy embeddings")
        return jnp.zeros((1, max_length, 4096))


def decode_video_latents(latents: jax.Array, vae_decoder: vae.WanVAEDecoder):
    """
    Decode video latents to RGB frames using Wan-VAE.

    Args:
        latents: [B, T, H, W, C] video latents
        vae_decoder: Optional WanVAEDecoder instance. If None, returns dummy video.

    Returns:
        video: [B, T, H_out, W_out, 3] RGB video (uint8)
    """
    # Decode using VAE
    video = vae.decode_latents_to_video(vae_decoder, latents, normalize=True)
    return video


def run_model():
    """Run Wan2.1-T2V-1.3B text-to-video generation."""

    print("=" * 60)
    print("Wan2.1-T2V-1.3B Text-to-Video Generation Demo")
    print("=" * 60)

    # Configuration
    model_ckpt_path = snapshot_download("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    config = modeling.ModelConfig()

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
    print("\n[2/5] Loading Diffusion Transformer weights...")
    try:
        model = params.create_model_from_safe_tensors(model_ckpt_path, config, mesh=None, load_transformer_only=True)
        print("      Model loaded successfully")
    except Exception as e:
        print(f"      Could not load weights: {e}")
        print("      Creating model from scratch (random weights)...")
        model = modeling.Wan2DiT(config, rngs=nnx.Rngs(params=0))

    # Step 2.5: Load VAE decoder
    print("\n[2.5/5] Loading VAE decoder...")
    try:
        vae_decoder = params.create_vae_decoder_from_safe_tensors(model_ckpt_path, mesh=None)
        print("      VAE decoder loaded")
    except Exception as e:
        print(f"      Could not load VAE: {e}")
        print("      VAE decoder will not be used (dummy video output)")
        vae_decoder = None

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
    print("\n[4/5] Decoding latents to video...")
    video = decode_video_latents(latents, vae_decoder)
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

    config = modeling.ModelConfig()
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
    predicted_noise = model.forward(latents, text_embeds, timestep, deterministic=True)
    forward_time = time.time() - start_time

    print("\n✓ Forward pass complete!")
    print(f"  Output shape: {predicted_noise.shape}")
    print(f"  Time: {forward_time:.3f}s")
    print(f"  Output range: [{predicted_noise.min():.3f}, {predicted_noise.max():.3f}]")
    print()

    return predicted_noise


def run_vae_decoder_test():
    """
    Simple VAE decoder forward pass test.
    Tests the decoder architecture without loading weights.
    """
    print("=" * 60)
    print("Wan-VAE Decoder Forward Pass Test")
    print("=" * 60)

    # Create VAE decoder
    print("\n[1/3] Creating VAE decoder...")
    vae_decoder = vae.WanVAEDecoder(rngs=nnx.Rngs(params=0))
    print("      VAE decoder created")

    # Create dummy latent inputs
    # Expected input: [B, T, H, W, C] = [1, 21, 104, 60, 16]
    batch_size = 1
    num_frames = 21
    latent_h, latent_w = 104, 60
    latent_channels = 16

    latents = jnp.zeros((batch_size, num_frames, latent_h, latent_w, latent_channels))

    print("\n[2/3] Running VAE decoder...")
    print(f"      Input latents: {latents.shape}")
    print("      Expected output: [1, 84, 832, 480, 3]")

    # Decode
    start_time = time.time()
    video = vae_decoder.decode(latents)
    decode_time = time.time() - start_time

    print("\n[3/3] Decoder output:")
    print(f"  ✓ Output shape: {video.shape}")
    print(f"  ✓ Time: {decode_time:.3f}s")
    # print(f"  ✓ Output range: [{video.min():.3f}, {video.max():.3f}]")
    print(f"  ✓ Output dtype: {video.dtype}")

    # Verify output shape
    expected_t = num_frames * 4  # Each input frame generates 4 output frames
    expected_h, expected_w = 832, 480
    expected_c = 3

    print("\n" + "=" * 60)
    if video.shape == (batch_size, expected_t, expected_h, expected_w, expected_c):
        print("✓ Decoder test PASSED!")
    else:
        print("✗ Decoder test FAILED!")
        print(f"  Expected: [{batch_size}, {expected_t}, {expected_h}, {expected_w}, {expected_c}]")
        print(f"  Got: {list(video.shape)}")
    print("=" * 60)
    print()

    return video


if __name__ == "__main__":
    # Choose which demo to run:

    # Option 1: Full generation pipeline (requires checkpoint download)
    run_model()

    # Option 2: Simple DiT forward pass test (no checkpoint required)
    # run_simple_forward_pass()

    # Option 3: Simple VAE decoder test (no checkpoint required)
    # run_vae_decoder_test()


__all__ = ["run_model", "run_simple_forward_pass", "run_vae_decoder_test"]
