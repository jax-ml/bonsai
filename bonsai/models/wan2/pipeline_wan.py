"""Example script for running Wan2.1-T2V-1.3B text-to-video generation."""

import argparse
import time
import traceback
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from huggingface_hub import snapshot_download
from jaxtyping import Array
from transformers import AutoTokenizer

from bonsai.models.wan2 import params, transformer_wan, umt5, vae_wan
from bonsai.models.wan2.transformer_wan import TransformerWanModelConfig, Wan2DiT
from bonsai.models.wan2.unipc_multistep_scheduler import FlaxUniPCMultistepScheduler, UniPCMultistepSchedulerState

jax.config.update("jax_debug_nans", True)


def get_t5_text_embeddings(
    prompt: str,
    tokenizer: AutoTokenizer = None,
    text_encoder: umt5.T5EncoderModel = None,
    max_length: int = 512,
    dtype: jnp.dtype = jnp.float32,
):
    try:
        inputs = tokenizer(
            prompt,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",  # Return numpy arrays
        )
        input_ids = jnp.array(inputs["input_ids"])
        attention_mask = jnp.array(inputs["attention_mask"])
        seq_lens = jnp.sum(attention_mask, axis=1).astype(jnp.int32)
        print(f"seq_lens: {seq_lens}")
        embeddings = text_encoder(input_ids, attention_mask, deterministic=True)
        prompt_embeds = jnp.asarray(embeddings, dtype=dtype)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = jnp.stack(
            [
                jnp.concatenate(
                    [u, jnp.zeros((max_length - u.shape[0], u.shape[1]), dtype=u.dtype)],
                    axis=0,
                )
                for u in prompt_embeds
            ],
            axis=0,
        )
        return prompt_embeds
    except Exception as e:
        print(f"Error in text encoding: {e}")
        traceback.print_exc()


def decode_video_latents(latents: jax.Array, vae_decoder: vae_wan.WanVAEDecoder):
    """
    Decode video latents to RGB frames using Wan-VAE.

    Args:
        latents: [B, T, H, W, C] video latents
        vae_decoder: Optional WanVAEDecoder instance. If None, returns dummy video.

    Returns:
        video: [B, T, H_out, W_out, 3] RGB video (uint8)
    """
    # Decode using VAE
    video = vae_wan.decode_latents_to_video(vae_decoder, latents, normalize=True)
    return video


def generate_video(
    model: Wan2DiT,
    latents: Array,
    text_embeds: Array,
    negative_embeds: Array,
    num_steps: int = 50,
    guidance_scale: float = 5.5,
    scheduler: Optional[FlaxUniPCMultistepScheduler] = None,
    scheduler_state: Optional[UniPCMultistepSchedulerState] = None,
) -> Array:
    """
    Generate video from text embeddings using the diffusion model.

    Args:
        model: Wan2DiT model
        text_embeds: [B, seq_len, text_dim] text embeddings from UMT5
        num_frames: Number of frames to generate
        latent_size: Spatial size of latents
        num_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale (5-6 recommended)

    Returns:
        latents: [B, T, H, W, C] generated video latents
    """
    b = text_embeds.shape[0]

    # Initialize random noise
    scheduler_state = scheduler.set_timesteps(
        scheduler_state, num_inference_steps=num_steps, shape=latents.transpose(0, 4, 1, 2, 3).shape
    )

    for t_idx in range(num_steps):
        # Scheduler needs scalar timestep, model needs batched timestep
        t_scalar = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[t_idx]
        t_batch = jnp.full((b,), t_scalar, dtype=jnp.int32)

        # Classifier-free guidance
        if guidance_scale != 1.0:
            noise_pred_cond = model.forward(latents, text_embeds, t_batch, deterministic=True)
            noise_pred_uncond = model.forward(latents, negative_embeds, t_batch, deterministic=True)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = model.forward(latents, text_embeds, t_batch, deterministic=True)

        latents, scheduler_state = scheduler.step(
            scheduler_state, noise_pred.transpose(0, 4, 1, 2, 3), t_scalar, latents.transpose(0, 4, 1, 2, 3)
        )
        latents = latents.transpose(0, 2, 3, 4, 1)  # back to channel-last

    return latents


def run_model(prompt: Optional[str] = None, neg_prompt: Optional[str] = None) -> jax.Array:
    print("=" * 60)
    print("Wan2.1-T2V-1.3B Text-to-Video Generation Demo")
    print("=" * 60)

    model_ckpt_path = snapshot_download("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    config = TransformerWanModelConfig()

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
    scheduler_state = scheduler.create_state()

    if prompt is not None:
        prompts = [prompt]
    else:
        prompts = [
            "A curious racoon",
        ]

    print(f"\nPrompt: {prompts[0]}")
    print(f"Model: Wan2.1-T2V-1.3B ({config.num_layers} layers, {config.hidden_dim} dim)")
    print(f"Video: {config.num_frames} frames @ 480p")

    tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
    umt5_encoder = params.create_t5_encoder_from_safe_tensors(model_ckpt_path, mesh=None)

    print("\n[1/4] Encoding text with UMT5...")
    text_embeds = get_t5_text_embeddings(prompts[0], tokenizer, umt5_encoder, max_length=config.max_text_len)
    if neg_prompt is not None:
        negative_prompts = [neg_prompt]
    else:
        negative_prompts = ["blurry"]
    negative_embeds = get_t5_text_embeddings(
        negative_prompts[0], tokenizer, umt5_encoder, max_length=config.max_text_len
    )

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

    latents = jax.random.normal(
        key, (1, config.num_frames, config.latent_size[0], config.latent_size[1], config.latent_input_dim)
    )

    latents = generate_video(
        model=model,
        latents=latents,
        text_embeds=text_embeds,
        negative_embeds=negative_embeds,
        num_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        scheduler=scheduler,
        scheduler_state=scheduler_state,
    )

    generation_time = time.time() - start_time
    print(f"✓ Generated latents in {generation_time:.2f}s")
    print(f"Latents shape: {latents.shape}")
    print(latents[0, 1:, :, 25:, :].mean())
    print(f"Has NaN: {jnp.isnan(latents).any()}")
    print(f"Has Inf: {jnp.isinf(latents).any()}")

    print("\n[4/5] Decoding latents to video...")
    video = decode_video_latents(latents, vae_decoder)
    generation_time = time.time() - start_time
    print(f"Video shape: {video.shape}")
    print(video[0, 1:, :, 235:, :].mean())

    print("\n" + "=" * 60)
    print("✓ Generation Complete!")
    print("=" * 60)
    print(f"Total time: {generation_time:.2f}s")
    print(f"FPS: {config.num_frames / generation_time:.2f}")
    vae_wan.save_video(video, "generated_video.mp4")
    print("Video saved to generated_video.mp4")

    return video


def main():
    parser = argparse.ArgumentParser(description="Wan2.1-T2V-1.3B Text-to-Video Generation Demo")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for video generation")
    parser.add_argument("--neg_prompt", type=str, default=None, help="Negative text prompt for video generation")
    args = parser.parse_args()
    run_model(args.prompt, args.neg_prompt)


if __name__ == "__main__":
    main()

__all__ = ["run_model"]
