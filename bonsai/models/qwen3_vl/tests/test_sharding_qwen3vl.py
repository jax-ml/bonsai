import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
import unittest

from flax import nnx
from jax._src.mesh import AxisType
from jax.sharding import PartitionSpec as P

from bonsai.models.qwen3_vl import modeling


def get_test_config(use_fsdp=False, use_tp=False):
    """Get a small test config with optional sharding."""
    return modeling.Qwen3VLConfig(
        vision_config=modeling.Qwen3VLVisionConfig(
            depth=2,
            hidden_size=64,
            intermediate_size=128,
            num_heads=4,
            in_channels=3,
            patch_size=14,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=128,
            num_position_embeddings=256,
            deepstack_visual_indexes=(0,),
            shd_cfg=modeling.VisionShardingConfig.default(use_fsdp, use_tp)
            if (use_fsdp or use_tp)
            else modeling.VisionShardingConfig.no_sharding(),
        ),
        text_config=modeling.Qwen3VLTextConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=32,
            shd_cfg=modeling.TextShardingConfig.default(use_fsdp, use_tp)
            if (use_fsdp or use_tp)
            else modeling.TextShardingConfig.no_sharding(),
        ),
    )


@unittest.skipIf(jax.device_count() < 4, "Atleast 4 devices required")
class TestSharding(absltest.TestCase):
    """Test sharding with simulated 8-device mesh."""

    @classmethod
    def setUpClass(cls):
        print(f"JAX devices: {jax.devices()}")
        print(f"Device count: {len(jax.devices())}")
        assert len(jax.devices()) == 8, f"Expected 8 simulated devices, got {len(jax.devices())}"

    def test_sharding_config_creation(self):
        """Test that sharding configs are created correctly."""
        # No sharding
        cfg_unsharded = modeling.Qwen3VLConfig.qwen3vl_2b()
        self.assertEqual(cfg_unsharded.text_config.shd_cfg.q_weight, P(None, None))

        # With sharding
        cfg_sharded = modeling.Qwen3VLConfig.qwen3vl_2b(use_fsdp=True, use_tp=True)
        self.assertEqual(cfg_sharded.text_config.shd_cfg.q_weight, P("fsdp", "tp"))
        self.assertEqual(cfg_sharded.vision_config.shd_cfg.attn_qkv_kernel, P("fsdp", "tp"))

    def test_text_mlp_sharded_vs_unsharded(self):
        """Test text MLP output is numerically equivalent with/without sharding."""
        # Setup mesh for sharded test
        mesh = jax.make_mesh((2, 2), ("fsdp", "tp"), axis_types=(AxisType.Explicit, AxisType.Explicit))

        # Create unsharded model and input
        cfg_unsharded = get_test_config(use_fsdp=False, use_tp=False)
        rngs = nnx.Rngs(42)
        mlp_unsharded = modeling.Qwen3VLMLP(cfg_unsharded.text_config, rngs=rngs)

        # Capture weights from unsharded model using [...] indexing
        gate_kernel = np.array(mlp_unsharded.gate_proj.kernel[...])
        up_kernel = np.array(mlp_unsharded.up_proj.kernel[...])
        down_kernel = np.array(mlp_unsharded.down_proj.kernel[...])

        # Use batch=2 so it's divisible by fsdp=2
        x = jnp.ones((2, 8, 128), dtype=jnp.float32)  # (batch, seq, hidden)
        out_unsharded = mlp_unsharded(x)

        # Create sharded model with same weights
        jax.set_mesh(mesh)
        cfg_sharded = get_test_config(use_fsdp=True, use_tp=True)
        rngs = nnx.Rngs(42)
        mlp_sharded = modeling.Qwen3VLMLP(cfg_sharded.text_config, rngs=rngs)

        # Copy weights to sharded model using [...] indexing
        mlp_sharded.gate_proj.kernel[...] = jnp.array(gate_kernel)
        mlp_sharded.up_proj.kernel[...] = jnp.array(up_kernel)
        mlp_sharded.down_proj.kernel[...] = jnp.array(down_kernel)

        # Recreate input inside mesh context
        x_sharded = jnp.ones((2, 8, 128), dtype=jnp.float32)
        out_sharded = mlp_sharded(x_sharded)

        # Numerical comparison
        np.testing.assert_allclose(
            np.array(out_unsharded),
            np.array(out_sharded),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Sharded MLP output differs from unsharded",
        )
        print("Text MLP: sharded vs unsharded numerical match ✓")

        # Reset mesh
        jax.set_mesh(jax.make_mesh((1,), ("dummy",), axis_types=(AxisType.Explicit,)))

    def test_vision_mlp_sharded_vs_unsharded(self):
        """Test vision MLP output is numerically equivalent with/without sharding."""
        mesh = jax.make_mesh((2, 2), ("fsdp", "tp"), axis_types=(AxisType.Explicit, AxisType.Explicit))

        # Create unsharded model
        cfg_unsharded = get_test_config(use_fsdp=False, use_tp=False)
        rngs = nnx.Rngs(42)
        mlp_unsharded = modeling.Qwen3VLVisionMLP(cfg_unsharded.vision_config, rngs=rngs)

        fc1_kernel = np.array(mlp_unsharded.linear_fc1.kernel[...])
        fc2_kernel = np.array(mlp_unsharded.linear_fc2.kernel[...])
        fc1_bias = np.array(mlp_unsharded.linear_fc1.bias[...])
        fc2_bias = np.array(mlp_unsharded.linear_fc2.bias[...])

        x = jnp.ones((16, 64), dtype=jnp.float32)  # (seq, hidden)
        out_unsharded = mlp_unsharded(x)

        # Create sharded model
        jax.set_mesh(mesh)
        cfg_sharded = get_test_config(use_fsdp=True, use_tp=True)
        rngs = nnx.Rngs(42)
        mlp_sharded = modeling.Qwen3VLVisionMLP(cfg_sharded.vision_config, rngs=rngs)

        mlp_sharded.linear_fc1.kernel[...] = jnp.array(fc1_kernel)
        mlp_sharded.linear_fc2.kernel[...] = jnp.array(fc2_kernel)
        mlp_sharded.linear_fc1.bias[...] = jnp.array(fc1_bias)
        mlp_sharded.linear_fc2.bias[...] = jnp.array(fc2_bias)

        # Recreate input inside mesh context
        x_sharded = jnp.ones((16, 64), dtype=jnp.float32)
        out_sharded = mlp_sharded(x_sharded)

        np.testing.assert_allclose(
            np.array(out_unsharded),
            np.array(out_sharded),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Sharded Vision MLP output differs from unsharded",
        )
        print("Vision MLP: sharded vs unsharded numerical match ✓")

        jax.set_mesh(jax.make_mesh((1,), ("dummy",), axis_types=(AxisType.Explicit,)))

    def test_full_model_creation_with_sharding(self):
        """Test full model can be created with sharding enabled."""
        mesh = jax.make_mesh((2, 2), ("fsdp", "tp"), axis_types=(AxisType.Explicit, AxisType.Explicit))
        jax.set_mesh(mesh)

        cfg = get_test_config(use_fsdp=True, use_tp=True)
        rngs = nnx.Rngs(0)

        model = modeling.Qwen3VLForConditionalGeneration(cfg, rngs=rngs)

        # Just verify model was created successfully
        self.assertIsNotNone(model)
        print("Full model created with sharding ✓")

        jax.set_mesh(jax.make_mesh((1,), ("dummy",), axis_types=(AxisType.Explicit,)))

    def test_text_model_forward_with_sharding(self):
        """Test text model can run forward pass with sharding enabled."""
        mesh = jax.make_mesh((2, 2), ("fsdp", "tp"), axis_types=(AxisType.Explicit, AxisType.Explicit))
        jax.set_mesh(mesh)

        # Create sharded model
        cfg = get_test_config(use_fsdp=True, use_tp=True)
        rngs = nnx.Rngs(42)
        text_model = modeling.Qwen3VLTextModel(cfg.text_config, rngs=rngs)

        # Create input - batch=2 divisible by fsdp=2
        batch, seq_len = 2, 8
        inputs_embeds = jnp.ones((batch, seq_len, 128), dtype=jnp.float32)
        cache = modeling.init_cache(cfg, batch_size=batch, token_len=seq_len, generate_steps=4)
        positions = jnp.arange(seq_len)[None, :].repeat(batch, axis=0)
        sin, cos = modeling._generate_rope(positions, cfg.text_config.head_dim, cfg.text_config.rope_theta)

        # Forward pass should complete without error
        output = text_model(inputs_embeds, cache, sin, cos, mask=None)

        self.assertEqual(output.shape, (batch, seq_len, 128))
        print("Text Model: forward pass with sharding ✓")

        jax.set_mesh(jax.make_mesh((1,), ("dummy",), axis_types=(AxisType.Explicit,)))

    def test_vision_attention_sharded_vs_unsharded(self):
        """Test vision attention output is numerically equivalent with/without sharding."""
        mesh = jax.make_mesh((2, 2), ("fsdp", "tp"), axis_types=(AxisType.Explicit, AxisType.Explicit))

        # Create unsharded model
        cfg_unsharded = get_test_config(use_fsdp=False, use_tp=False)
        rngs = nnx.Rngs(42)
        attn_unsharded = modeling.Qwen3VLVisionAttention(cfg_unsharded.vision_config, rngs=rngs)

        # Get weights
        graphdef, state = nnx.split(attn_unsharded)
        flat_state = nnx.to_flat_state(state)
        state_arrays = {k: np.array(v[...]) for k, v in zip(flat_state.paths, flat_state.leaves)}

        # Create input - (seq, hidden)
        seq_len = 16
        hidden_size = cfg_unsharded.vision_config.hidden_size
        x = jnp.ones((seq_len, hidden_size), dtype=jnp.float32)

        # Create RoPE cos/sin for vision
        # VisionAttention expects (seq_len, head_dim) for cos/sin
        head_dim = cfg_unsharded.vision_config.head_dim
        # Simple RoPE: just use arange positions with full head_dim
        positions = jnp.arange(seq_len, dtype=jnp.float32)
        theta = 10000.0
        freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
        angles = jnp.outer(positions, freqs)  # (seq_len, head_dim/2)
        # Repeat to match full head_dim
        angles = jnp.concatenate([angles, angles], axis=-1)  # (seq_len, head_dim)
        cos_vals = jnp.cos(angles)
        sin_vals = jnp.sin(angles)

        out_unsharded = attn_unsharded(x, (cos_vals, sin_vals))

        # Create sharded model
        jax.set_mesh(mesh)
        cfg_sharded = get_test_config(use_fsdp=True, use_tp=True)
        rngs = nnx.Rngs(42)
        attn_sharded = modeling.Qwen3VLVisionAttention(cfg_sharded.vision_config, rngs=rngs)

        # Copy weights
        sharded_graphdef, sharded_state = nnx.split(attn_sharded)
        sharded_flat_state = nnx.to_flat_state(sharded_state)
        for k, v in zip(sharded_flat_state.paths, sharded_flat_state.leaves):
            if k in state_arrays:
                v[...] = jnp.array(state_arrays[k])

        # Recreate ALL inputs inside mesh context
        x_sharded = jnp.ones((seq_len, hidden_size), dtype=jnp.float32)
        positions_sharded = jnp.arange(seq_len, dtype=jnp.float32)
        freqs_sharded = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
        angles_sharded = jnp.outer(positions_sharded, freqs_sharded)
        angles_sharded = jnp.concatenate([angles_sharded, angles_sharded], axis=-1)
        cos_sharded = jnp.cos(angles_sharded)
        sin_sharded = jnp.sin(angles_sharded)

        out_sharded = attn_sharded(x_sharded, (cos_sharded, sin_sharded))

        np.testing.assert_allclose(
            np.array(out_unsharded),
            np.array(out_sharded),
            rtol=1e-4,
            atol=1e-4,
            err_msg="Sharded VisionAttention output differs from unsharded",
        )
        print("Vision Attention: sharded vs unsharded numerical match ✓")

        jax.set_mesh(jax.make_mesh((1,), ("dummy",), axis_types=(AxisType.Explicit,)))

    def test_full_vision_text_forward(self):
        """Test full vision+text forward pass with sharding (FSDP=4, TP=2)."""
        mesh = jax.make_mesh((4, 2), ("fsdp", "tp"), axis_types=(AxisType.Explicit, AxisType.Explicit))
        jax.set_mesh(mesh)

        cfg = get_test_config(use_fsdp=True, use_tp=True)
        rngs = nnx.Rngs(0)
        fsdp = modeling.ShardMode.FSDP.value
        model = modeling.Qwen3VLForConditionalGeneration(cfg, rngs=rngs)

        batch_size = 4  # divisible by fsdp=4
        num_tokens = 128
        patch_size = cfg.vision_config.patch_size  # 14
        temporal_patch_size = cfg.vision_config.temporal_patch_size  # 2
        in_channels = cfg.vision_config.in_channels  # 3
        spatial_merge_size = cfg.vision_config.spatial_merge_size  # 2
        per_patch_features = in_channels * temporal_patch_size * patch_size * patch_size  # 3 * 2 * 14 * 14 = 1176

        # Each image: grid (T=1, H=patch_size, W=patch_size) -> patch_size*patch_size patches before merge
        grid_h, grid_w = patch_size, patch_size  # 14x14 = 196 patches per image
        num_patches_per_image = grid_h * grid_w  # 196
        merged_patches_per_image = num_patches_per_image // (spatial_merge_size**2)  # 196 // 4 = 49
        total_patches = num_patches_per_image * batch_size  # 196 * 4 = 784

        # Create pixel_values: (total_patches, per_patch_features)
        key = jax.random.key(0)
        pixel_values = jax.random.uniform(
            key,
            (total_patches, per_patch_features),
            dtype=jnp.float32,
            minval=-1,
            maxval=1,
        )

        # Create text input_ids: (batch, num_tokens)
        n_text = jax.device_put(
            np.arange(batch_size * num_tokens).reshape(batch_size, -1) % cfg.text_config.vocab_size,
            device=P(fsdp),
        )

        # Create token_type_ids: 1 for image token positions, 0 for text
        # Mark merged_patches_per_image consecutive positions as image tokens per batch
        token_type_ids = np.zeros((batch_size, num_tokens), dtype=np.int32)
        token_type_ids[:, 12 : 12 + merged_patches_per_image] = 1
        n_tti = jax.device_put(token_type_ids, device=P(fsdp))

        # One image per batch item -> batch_size entries in grid_thw
        image_grid_thw = ((1, grid_h, grid_w),) * batch_size

        # Init cache
        cache = modeling.init_cache(cfg, batch_size, num_tokens, 1, jnp.float32)

        # Forward pass - should complete without error
        out = model(n_text, cache, pixel_values, image_grid_thw, n_tti)

        # Verify output shape: (batch, seq_len, vocab_size)
        self.assertEqual(out.shape, (batch_size, num_tokens, cfg.text_config.vocab_size))

        # Verify output is sharded correctly
        from jax.sharding import NamedSharding

        self.assertIsInstance(out.sharding, NamedSharding)
        expected_logit_shd = P(cfg.text_config.shd_cfg.act_btd[0], None, None)
        self.assertEqual(out.sharding.spec, expected_logit_shd)
        print(f"Full vision+text forward: shape={out.shape}, sharding={out.sharding.spec} ✓")

        jax.set_mesh(jax.make_mesh((1,), ("dummy",), axis_types=(AxisType.Explicit,)))


if __name__ == "__main__":
    absltest.main()
