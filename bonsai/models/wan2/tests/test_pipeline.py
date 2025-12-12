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
from diffusers import AutoModel, AutoencoderKLWan, WanPipeline, WanTransformer3DModel
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

    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    model_ckpt_path = snapshot_download(model_id)
    config = transformer_wan.ModelConfig()

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
    height, width = 240, 240
    num_frames = 41
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
    
    print("\n[1/4] Encoding text with UMT5...")
    text_embeds = get_t5_text_embeddings(prompts[0], tokenizer, umt5_encoder, max_length=config.max_text_len)
    negative_prompts = [""]  # Empty negative prompt
    negative_embeds = get_t5_text_embeddings(negative_prompts[0], tokenizer, umt5_encoder, max_length=config.max_text_len)

    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompts[0],
            max_sequence_length=config.max_text_len,
            negative_prompt=negative_prompts[0],
            do_classifier_free_guidance=True,
            num_videos_per_prompt=1,
            device="cpu"
        )
    
    compare_outputs(text_embeds, prompt_embeds, "UMT5 Text Embeddings")
    compare_outputs(negative_embeds, negative_prompt_embeds, "UMT5 Negative Text Embeddings")

    latent_height = height // pipe.vae_scale_factor_spatial
    latent_width = width // pipe.vae_scale_factor_spatial
    latent_frames = (num_frames - 1) // pipe.vae_scale_factor_temporal + 1
    print(f"Latent video size: {latent_frames} x {latent_height} x {latent_width}")

    print("\n[2/5] Loading Diffusion Transformer weights...")
    model = params.create_model_from_safe_tensors(model_ckpt_path, config, mesh=None)
    print("\n[2.5/5] Loading VAE decoder...")
    vae_decoder = params.create_vae_decoder_from_safe_tensors(model_ckpt_path, mesh=None)
    print("Model loaded successfully")

    print("\n[3/4] Generating video latents...")
    print(f"Using {num_inference_steps} diffusion steps")
    print(f"Guidance scale: {config.guidance_scale}")

    # Generate same random noise (use same seed!)
    key = jax.random.PRNGKey(42)
    generator = torch.Generator(device="cpu").manual_seed(42)
    latents = torch.randn(
        (1, config.latent_input_dim, latent_frames, latent_height, latent_width),
        generator=generator,
        device="cpu",
        dtype=torch.float32
    )
    latents_jax = jnp.array(latents.numpy()).transpose(0, 2, 3, 4, 1)  # 调整维度顺序

    print("Initial latents shape:", latents.shape)
    print("Initial latents range:", latents.min().item(), latents.max().item())

    pipe.scheduler.set_timesteps(num_inference_steps, device="cpu")
    timesteps = pipe.scheduler.timesteps
    scheduler_state = scheduler.set_timesteps(scheduler_state, num_inference_steps=num_inference_steps, shape=latents.shape)
    # print(f"schecduler_state: {scheduler_state}")
    b=1

    for i, t in enumerate(timesteps):
        print(f"Step {i}: Timestep {t}")
        pipe._current_timestep = t
        current_model = pipe.transformer
        timestep = t.expand(latents.shape[0])

        t_scalar = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[i]
        t_batch = jnp.full((b,), t_scalar, dtype=jnp.int32)

        compare_outputs(t_batch, timestep, f"Timestep Batch at step {i}")
        latent_model_input = latents.to(torch.bfloat16)
        with current_model.cache_context("cond"):
            noise_pred = current_model(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                attention_kwargs=None,
                return_dict=False,
            )[0]
        print(f"noise_pred shape (torch): {noise_pred.shape}")
        noise_pred_cond = model.forward(latents_jax, text_embeds, t_batch, deterministic=True)
        compare_outputs(noise_pred_cond, noise_pred.permute(0,2,3,4,1), f"Noise Prediction Cond at step {i}", rtol=1e-2, atol=1e-3)
        with current_model.cache_context("uncond"):
            noise_uncond = current_model(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=negative_prompt_embeds,
                attention_kwargs=None,
                return_dict=False,
            )
        noise_uncond = noise_uncond[0]
        print(f"noise_uncond shape (torch): {noise_uncond.shape}")
        noise_pred_uncond = model.forward(latents_jax, negative_embeds, t_batch, deterministic=True)
        compare_outputs(noise_pred_uncond, noise_uncond.permute(0,2,3,4,1), f"Noise Prediction Uncond at step {i}", rtol=1e-2, atol=1e-3)

        noise_pred_jax = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        noise_pred_torch = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

        latents = pipe.scheduler.step(noise_pred_torch, t, latents, return_dict=False)[0]
        latents_jax, scheduler_state = scheduler.step(scheduler_state, noise_pred_jax, t_scalar, latents)

        compare_outputs(latents_jax, latents, f"Latents at step {i}", rtol=1e-2, atol=1e-3)


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

    latents_jax = transformer_wan.generate_video(
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

def run_simple_forward_pass():
    """
    Simple forward pass test without full generation pipeline.
    Useful for testing the model architecture.
    """
    print("=" * 60)
    print("Wan2.1-T2V-1.3B Simple Forward Pass Test")
    print("=" * 60)

    config = transformer_wan.ModelConfig()
    print(f"\nConfig: {config.num_layers} layers, {config.hidden_dim} dim")

    # Create model
    print("\n[1/2] Creating model...")
    model = transformer_wan.Wan2DiT(config, rngs=nnx.Rngs(params=0, dropout=0))
    print("      Model created")

    # Create dummy inputs
    batch_size = 1
    latents = jnp.zeros(
        (batch_size, config.num_frames, config.latent_size[0], config.latent_size[1], config.latent_input_dim)
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
    vae_decoder = vae_wan.WanVAEDecoder(rngs=nnx.Rngs(params=0))
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

__all__ = ["run_model"]
