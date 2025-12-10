"""Example script for running Wan2.1-T2V-1.3B text-to-video generation."""

import time

import jax
import jax.numpy as jnp
from flax import nnx
from huggingface_hub import snapshot_download
import traceback
from transformers import AutoTokenizer
from bonsai.models.wan2 import modeling, params, vae, t5
from bonsai.models.wan2 import scheduler as scheduler_module
from diffusers import AutoModel, AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video
import torch
import numpy as np


def compare_outputs(jax_output: jax.Array, torch_output, name: str, rtol: float = 1e-2, atol: float = 1e-4):
    print(f"torch dtype: {torch_output.dtype}")
    if torch_output.dtype == torch.bfloat16:
        torch_output = torch_output.float()

    if isinstance(torch_output, torch.Tensor):
        torch_np = torch_output.detach().cpu().numpy()
    else:
        torch_np = np.array(torch_output)

    jax_np = np.array(jax_output)

    print(f"\n{'=' * 80}")
    print(f"Comparing: {name}")
    print(f"{'=' * 80}")
    print(f"JAX shape:   {jax_np.shape}")
    print(f"Torch shape: {torch_np.shape}")
    print(f"JAX dtype:   {jax_np.dtype}")
    print(f"Torch dtype: {torch_np.dtype}")

    if jax_np.shape != torch_np.shape:
        print("Shape mismatch!")
        return False

    abs_diff = np.abs(jax_np - torch_np)
    rel_diff = abs_diff / (np.abs(torch_np) + 1e-10)

    max_abs_diff = np.max(abs_diff)
    max_rel_diff = np.max(rel_diff)
    mean_abs_diff = np.mean(abs_diff)
    mean_rel_diff = np.mean(rel_diff)

    print("\nStatistics:")
    print(f"  Max absolute difference: {max_abs_diff:.2e}")
    print(f"  Max relative difference: {max_rel_diff:.2e}")
    print(f"  Mean absolute difference: {mean_abs_diff:.2e}")
    print(f"  Mean relative difference: {mean_rel_diff:.2e}")

    print(f"\nJAX output range:   [{np.min(jax_np):.4f}, {np.max(jax_np):.4f}]")
    print(f"Torch output range: [{np.min(torch_np):.4f}, {np.max(torch_np):.4f}]")

    close = np.allclose(jax_np, torch_np, rtol=rtol, atol=atol)

    if close:
        print(f"\n✅ Outputs match within tolerance (rtol={rtol}, atol={atol})")
    else:
        print(f"\n❌ Outputs do NOT match (rtol={rtol}, atol={atol})")
        # Show some mismatched locations
        mismatch_mask = ~np.isclose(jax_np, torch_np, rtol=rtol, atol=atol)
        n_mismatches = np.sum(mismatch_mask)
        print(f"  Number of mismatches: {n_mismatches} / {jax_np.size} ({100 * n_mismatches / jax_np.size:.2f}%)")

    return close


def get_t5_text_embeddings(
    prompt: str,
    tokenizer: AutoTokenizer = None,
    text_encoder: t5.T5EncoderModel = None,
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

    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    model_ckpt_path = snapshot_download(model_id)
    config = modeling.ModelConfig()

    vae = AutoencoderKLWan.from_pretrained(model_ckpt_path, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    flow_shift = 3.0
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
    pipe.to("cpu")
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

    # Test parameters
    prompt = "A cat walking in a garden"
    negative_prompt = "blurry, low quality"
    height, width = 480, 832
    num_frames = 81
    guidance_scale = 5.0
    num_inference_steps = 50

    prompts = [
        "A curious racoon",
    ]

    print(f"\nPrompt: {prompts[0]}")
    print(f"Model: Wan2.1-T2V-1.3B ({config.num_layers} layers, {config.hidden_dim} dim)")
    print(f"Video: {config.num_frames} frames @ 480p")

    tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
    umt5_encoder = params.create_t5_encoder_from_safe_tensors(model_ckpt_path, mesh=None)
    
    print("\n[1/4] Encoding text with T5...")
    text_embeds = get_t5_text_embeddings(prompts[0], tokenizer, umt5_encoder, max_length=config.max_text_len, model_ckpt_path=model_ckpt_path)
    negative_prompts = [""]  # Empty negative prompt
    negative_embeds = get_t5_text_embeddings(negative_prompts[0], tokenizer, umt5_encoder, max_length=config.max_text_len, model_ckpt_path=model_ckpt_path)

    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompts,
            negative_prompt=negative_prompts,
            do_classifier_free_guidance=True,
            num_videos_per_prompt=1,
            device="cuda"
        )
    
    compare_outputs(text_embeds, prompt_embeds, "T5 Text Embeddings")
    compare_outputs(negative_embeds, negative_prompt_embeds, "T5 Negative Text Embeddings")

    latent_height = height // pipe.vae_scale_factor_spatial
    latent_width = width // pipe.vae_scale_factor_spatial
    latent_frames = (num_frames - 1) // pipe.vae_scale_factor_temporal + 1
    print(f"Latent video size: {latent_frames} x {latent_height} x {latent_width}")

    # Generate same random noise (use same seed!)
    key = jax.random.PRNGKey(42)
    generator = torch.Generator(device="cpu").manual_seed(42)
    latents = torch.randn(
        (1, vae.config.latent_channels, latent_frames, latent_height, latent_width),
        generator=generator,
        device="cpu",
        dtype=torch.float32
    )
    latents_jax = jnp.array(latents.numpy()).transpose(0, 2, 3, 4, 1)  # 调整维度顺序

    print("Initial latents shape:", latents.shape)
    print("Initial latents range:", latents.min().item(), latents.max().item())

    print("\n[2/5] Loading Diffusion Transformer weights...")
    model = params.create_model_from_safe_tensors(model_ckpt_path, config, mesh=None)
    print("\n[2.5/5] Loading VAE decoder...")
    vae_decoder = params.create_vae_decoder_from_safe_tensors(model_ckpt_path, mesh=None)
    print("Model loaded successfully")

    print("\n[3/4] Generating video latents...")
    print(f"Using {num_inference_steps} diffusion steps")
    print(f"Guidance scale: {config.guidance_scale}")

    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        latents=latents,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        output_type="latent",  # Important! Get latents instead of decoded video
    )
    denoised_latents = output.frames.transpose(0,2,3,4,1)  # Shape: (B, C, T, H, W)

    start_time = time.time()

    latents_jax = modeling.generate_video(
        model=model,
        latents=latents_jax,
        text_embeds=text_embeds,
        negative_embeds=negative_embeds,
        num_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        scheduler=scheduler,
        scheduler_state=scheduler_state,
    )
    compare_outputs(latents_jax, denoised_latents, "Final Generated Latents", rtol=1e-2, atol=1e-3)

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

if __name__ == "__main__":
    run_model()

__all__ = ["run_model"]
