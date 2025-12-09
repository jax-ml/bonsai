"""
Example: Using FlaxUniPCMultistepScheduler with JAX Wan implementation

This shows how to use the JAX/Flax port of UniPCMultistepScheduler with flow_shift
for the Wan 2.1 T2V model.
"""

import jax
import jax.numpy as jnp
from diffusers.schedulers.scheduling_unipc_multistep_flax import (
    FlaxUniPCMultistepScheduler,
)

# ============================================================================
# 1. Initialize the scheduler with flow_shift
# ============================================================================

# For Wan2.1-T2V-1.3B:
# - Use flow_shift=5.0 for 720P generation
# - Use flow_shift=3.0 for 480P generation
# - solver_order=2 is recommended for guided sampling (with CFG)

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

# ============================================================================
# 2. Set timesteps for inference
# ============================================================================

num_inference_steps = 50
batch_size = 1
num_channels = 16
num_frames = 9
height = 60  # Latent height (720 / 12)
width = 104  # Latent width (1280 / 12)

latent_shape = (batch_size, num_channels, num_frames, height, width)

scheduler_state = scheduler.set_timesteps(scheduler_state, num_inference_steps=num_inference_steps, shape=latent_shape)

print(f"Timesteps: {scheduler_state.timesteps[:10]}...")  # Show first 10
print(f"Sigmas: {scheduler_state.sigmas[:10]}...")

# ============================================================================
# 3. Prepare initial latents
# ============================================================================

rng = jax.random.PRNGKey(42)
latents = jax.random.normal(rng, latent_shape, dtype=jnp.bfloat16)

# Scale by initial noise sigma
latents = latents * scheduler_state.init_noise_sigma

# ============================================================================
# 4. Denoising loop with CFG
# ============================================================================

# Assume you have:
# - your_wan_transformer: Your JAX Wan transformer implementation
# - prompt_embeds: Text embeddings from UMT5 (shape: [B, 512, 4096])
# - negative_prompt_embeds: Negative text embeddings

guidance_scale = 7.5
timesteps = scheduler_state.timesteps

print(f"\nStarting denoising loop with {len(timesteps)} steps...")

for i, t in enumerate(timesteps):
    print(f"Step {i + 1}/{len(timesteps)}, timestep: {t}")

    # Prepare timestep (broadcast to batch)
    timestep_tensor = jnp.array([t] * batch_size, dtype=jnp.int32)

    # ========================================================================
    # Conditional forward pass
    # ========================================================================
    # noise_pred_cond = your_wan_transformer(
    #     hidden_states=latents,
    #     timestep=timestep_tensor,
    #     encoder_hidden_states=prompt_embeds
    # )
    # For this example, simulate with random noise
    noise_pred_cond = jax.random.normal(rng, latents.shape, dtype=jnp.bfloat16)

    # ========================================================================
    # Unconditional forward pass (for CFG)
    # ========================================================================
    # noise_pred_uncond = your_wan_transformer(
    #     hidden_states=latents,
    #     timestep=timestep_tensor,
    #     encoder_hidden_states=negative_prompt_embeds
    # )
    # For this example, simulate with random noise
    noise_pred_uncond = jax.random.normal(rng, latents.shape, dtype=jnp.bfloat16)

    # ========================================================================
    # Apply classifier-free guidance
    # ========================================================================
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

    # ========================================================================
    # Scheduler step: x_t -> x_{t-1}
    # ========================================================================
    scheduler_output = scheduler.step(
        state=scheduler_state, model_output=noise_pred, timestep=t, sample=latents, return_dict=True
    )

    latents = scheduler_output.prev_sample
    scheduler_state = scheduler_output.state

    if i % 10 == 0:
        print(f"  Latent stats - mean: {latents.mean():.4f}, std: {latents.std():.4f}")

print("\nDenoising complete!")
print(f"Final latent shape: {latents.shape}")

# ============================================================================
# 5. Decode latents to video
# ============================================================================

# video_frames = your_wan_vae_decoder(latents)
#
# # Post-process (denormalize from [-1, 1] to [0, 255])
# video_frames = (video_frames / 2 + 0.5).clip(0, 1)
# video_frames = (video_frames * 255).astype(jnp.uint8)
#
# # video_frames shape: (B, 3, T*8, H*12, W*12) for 720P
# # e.g., (1, 3, 72, 720, 1280) for 9 latent frames

print("\n" + "=" * 80)
print("Example complete!")
print("=" * 80)
print("\nKey points:")
print("1. Use flow_shift=5.0 for 720P, 3.0 for 480P")
print("2. solver_order=2 is recommended for CFG-guided sampling")
print("3. The scheduler handles the flow-based sigma transformation automatically")
print("4. Each step updates both latents and scheduler_state")
print("5. The state is immutable (Flax dataclass) - always use the returned state")
