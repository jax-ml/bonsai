import unittest

import jax
import jax.numpy as jnp
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

from .scheduler import FlaxUniPCMultistepScheduler


class FlaxUniPCMultistepSchedulerTest(unittest.TestCase):
    def test_set_timesteps(self):
        """Test timestep generation"""
        scheduler = FlaxUniPCMultistepScheduler(
            use_flow_sigmas=True,
            flow_shift=5.0,
        )
        state = scheduler.create_state()

        num_inference_steps = 50
        shape = (1, 16, 9, 60, 104)

        state = scheduler.set_timesteps(state, num_inference_steps, shape)

        self.assertEqual(state.num_inference_steps, num_inference_steps)
        self.assertEqual(len(state.timesteps), num_inference_steps)
        self.assertEqual(len(state.sigmas), num_inference_steps)

    def test_step_shape(self):
        """Test that step produces correct output shape"""
        scheduler = FlaxUniPCMultistepScheduler(
            use_flow_sigmas=True,
            flow_shift=5.0,
            dtype=jnp.float32,
        )
        state = scheduler.create_state()

        shape = (1, 4, 2, 8, 8)
        num_inference_steps = 10

        state = scheduler.set_timesteps(state, num_inference_steps, shape)

        # Create dummy inputs
        rng = jax.random.PRNGKey(0)
        sample = jax.random.normal(rng, shape)
        model_output = jax.random.normal(rng, shape)
        timestep = state.timesteps[0]

        # Run step
        output = scheduler.step(state, model_output, timestep, sample, return_dict=True)

        self.assertEqual(output.prev_sample.shape, shape)
        self.assertIsNotNone(output.state)

    def test_full_loop(self):
        """Test a full denoising loop"""
        scheduler = FlaxUniPCMultistepScheduler(
            use_flow_sigmas=True,
            flow_shift=5.0,
            solver_order=2,
            dtype=jnp.float32,
        )
        state = scheduler.create_state()

        shape = (1, 4, 2, 8, 8)
        num_inference_steps = 10

        state = scheduler.set_timesteps(state, num_inference_steps, shape)

        # Initialize latents
        rng = jax.random.PRNGKey(42)
        latents = jax.random.normal(rng, shape)
        latents = latents * state.init_noise_sigma

        # Denoising loop
        for i, t in enumerate(state.timesteps):
            # Simulate model output
            model_output = jax.random.normal(rng, shape)

            # Scheduler step
            output = scheduler.step(state, model_output, t, latents, return_dict=True)

            latents = output.prev_sample
            state = output.state

        # Check final output
        self.assertEqual(latents.shape, shape)
        self.assertTrue(jnp.isfinite(latents).all())

    def test_flow_shift_values(self):
        """Test that different flow_shift values produce different timesteps"""
        shape = (1, 4, 2, 8, 8)
        num_inference_steps = 20

        # Test with flow_shift=3.0 (480P)
        scheduler_480p = FlaxUniPCMultistepScheduler(
            use_flow_sigmas=True,
            flow_shift=3.0,
        )
        state_480p = scheduler_480p.create_state()
        state_480p = scheduler_480p.set_timesteps(state_480p, num_inference_steps, shape)

        # Test with flow_shift=5.0 (720P)
        scheduler_720p = FlaxUniPCMultistepScheduler(
            use_flow_sigmas=True,
            flow_shift=5.0,
        )
        state_720p = scheduler_720p.create_state()
        state_720p = scheduler_720p.set_timesteps(state_720p, num_inference_steps, shape)

        # Timesteps should be different
        self.assertFalse(jnp.allclose(state_480p.sigmas, state_720p.sigmas))

    def test_add_noise(self):
        """Test add_noise function"""
        scheduler = FlaxUniPCMultistepScheduler()
        state = scheduler.create_state()

        shape = (2, 4, 8, 8)
        rng = jax.random.PRNGKey(0)

        original_samples = jax.random.normal(rng, shape)
        noise = jax.random.normal(rng, shape)
        timesteps = jnp.array([100, 200])

        noisy_samples = scheduler.add_noise(state, original_samples, noise, timesteps)

        self.assertEqual(noisy_samples.shape, shape)
        self.assertTrue(jnp.isfinite(noisy_samples).all())


def test_scheduler_with_model():
    """
    Example: Using FlaxUniPCMultistepScheduler with JAX Wan implementation

    This shows how to use the JAX/Flax port of UniPCMultistepScheduler with flow_shift
    for the Wan 2.1 T2V model.
    """
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

    scheduler_state = scheduler.set_timesteps(
        scheduler_state, num_inference_steps=num_inference_steps, shape=latent_shape
    )

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
        _timestep_tensor = jnp.array([t] * batch_size, dtype=jnp.int32)

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


if __name__ == "__main__":
    unittest.main()
