"""Example script for running Wan2.1-T2V-1.3B text-to-video generation."""

import time

import jax
import jax.numpy as jnp
from flax import nnx
from huggingface_hub import snapshot_download
import traceback
from transformers import AutoTokenizer
from ..scheduler import FlaxUniPCMultistepScheduler

from bonsai.models.wan2 import modeling, params, vae


def get_t5_text_embeddings(
    prompt: str,
    model_ckpt_path: str,
    max_length: int = 512,
):
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
        inputs = tokenizer(
            prompt,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",  # Return numpy arrays
        )
        model = params.create_t5_encoder_from_safe_tensors(model_ckpt_path, mesh=None)

        input_ids = jnp.array(inputs["input_ids"])
        attention_mask = jnp.array(inputs["attention_mask"])
        embeddings = model(input_ids, attention_mask, deterministic=True)

        return embeddings
    except Exception as e:
        print(f"Error in text encoding: {e}")
        traceback.print_exc()
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
    print("=" * 60)
    print("Wan2.1-T2V-1.3B Text-to-Video Generation Demo")
    print("=" * 60)

    model_ckpt_path = snapshot_download("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    config = modeling.ModelConfig()

    # For sharding (multi-GPU), uncomment:
    # from jax.sharding import AxisType
    # mesh = jax.make_mesh((2, 2), ("fsdp", "tp"), axis_types=(AxisType.Explicit, AxisType.Explicit))
    # jax.set_mesh(mesh)
    scheduler = FlaxUniPCMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        solver_order=2,  # Order 2 for guided sampling
        prediction_type="flow_prediction",
        use_flow_sigmas=True,  # Enable flow-based sigma schedule
        flow_shift=3.0,  # 5.0 for 720P, 3.0 for 480P
        timestep_spacing="linspace",
        predict_x0=True,
        solver_type="bh2",
        lower_order_final=True,
        dtype=jnp.float32,
    )

    # Create initial state
    scheduler_state = scheduler.create_state()

    scheduler_state = scheduler.set_timesteps(
        scheduler_state, num_inference_steps=1000, shape=latents.shape
    )

    prompts = [
        "A beautiful sunset over the ocean with waves crashing on the shore",
    ]

    print(f"\nPrompt: {prompts[0]}")
    print(f"Model: Wan2.1-T2V-1.3B ({config.num_layers} layers, {config.hidden_dim} dim)")
    print(f"Video: {config.num_frames} frames @ 480p")

    print("\n[1/4] Encoding text with T5...")
    text_embeds = get_t5_text_embeddings(prompts[0], max_length=config.max_text_len, model_ckpt_path=model_ckpt_path)

    print("\n[2/5] Loading Diffusion Transformer weights...")
    model = params.create_model_from_safe_tensors(model_ckpt_path, config, mesh=None)
    print("\n[2.5/5] Loading VAE decoder...")
    vae_decoder = params.create_vae_decoder_from_safe_tensors(model_ckpt_path, mesh=None)
    print("Model loaded successfully")

    print("\n[3/4] Generating video latents...")
    print(f"Using {config.num_inference_steps} diffusion steps")
    print(f"Guidance scale: {config.guidance_scale}")

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
        scheduler=scheduler,
        scheduler_state=scheduler_state,
    )

    generation_time = time.time() - start_time
    print(f"✓ Generated latents in {generation_time:.2f}s")
    print(f"Latents shape: {latents.shape}")

    # Step 4: Decode to video
    print("\n[4/5] Decoding latents to video...")
    video = decode_video_latents(latents, vae_decoder)
    generation_time = time.time() - start_time
    print(f"Video shape: {video.shape}")

    # Summary
    print("\n" + "=" * 60)
    print("✓ Generation Complete!")
    print("=" * 60)
    print(f"Total time: {generation_time:.2f}s")
    print(f"FPS: {config.num_frames / generation_time:.2f}")
    vae.save_video(video, "generated_video.mp4")
    print("Video saved to generated_video.mp4")

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
    run_model()



__all__ = ["run_model", "run_simple_forward_pass", "run_vae_decoder_test"]
