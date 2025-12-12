"""Example script for running Wan2.1-T2V-1.3B text-to-video generation."""

import time

import jax
import jax.numpy as jnp
from flax import nnx
from huggingface_hub import snapshot_download
import traceback
from transformers import AutoTokenizer
from bonsai.models.wan2 import params, transformer_wan, umt5, vae_wan
from bonsai.models.wan2 import unipc_multistep_scheduler as scheduler_module
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


def run_model():
    print("=" * 60)
    print("Wan2.1-T2V-1.3B Text-to-Video Generation Demo")
    print("=" * 60)

    model_ckpt_path = snapshot_download("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    config = transformer_wan.TransformerWanModelConfig()

    # For sharding (multi-GPU), uncomment:
    # from jax.sharding import AxisType
    # mesh = jax.make_mesh((2, 2), ("fsdp", "tp"), axis_types=(AxisType.Explicit, AxisType.Explicit))
    # jax.set_mesh(mesh)
    scheduler = scheduler_module.FlaxUniPCMultistepScheduler(
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
    negative_prompts = [""]  # Empty negative prompt
    negative_embeds = get_t5_text_embeddings(negative_prompts[0], tokenizer, umt5_encoder, max_length=config.max_text_len)

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

    latents = jax.random.normal(key, (1, config.num_frames, config.latent_size[0], config.latent_size[1], config.latent_input_dim))

    latents = transformer_wan.generate_video(
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
    print(latents[0,1:,:,25:,:].mean())
    print(f"Has NaN: {jnp.isnan(latents).any()}")
    print(f"Has Inf: {jnp.isinf(latents).any()}")

    # Step 4: Decode to video
    print("\n[4/5] Decoding latents to video...")
    video = decode_video_latents(latents, vae_decoder)
    generation_time = time.time() - start_time
    print(f"Video shape: {video.shape}")
    print(video[0,1:,:,235:,:].mean())

    # Summary
    print("\n" + "=" * 60)
    print("✓ Generation Complete!")
    print("=" * 60)
    print(f"Total time: {generation_time:.2f}s")
    print(f"FPS: {config.num_frames / generation_time:.2f}")
    vae_wan.save_video(video, "generated_video.mp4")
    print("Video saved to generated_video.mp4")

    return video

if __name__ == "__main__":
    run_model()



__all__ = ["run_model", "run_simple_forward_pass", "run_vae_decoder_test"]
