import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
import unittest

from flax import nnx
from jax._src.mesh import AxisType
from jax.sharding import NamedSharding, PartitionSpec as P

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


@unittest.skipIf(jax.device_count() < 8, "At least 8 devices required")
class TestSharding(absltest.TestCase):
    """Test sharding with simulated 8-device mesh (FSDP=4, TP=2)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.mesh = jax.make_mesh((4, 2), ("fsdp", "tp"), axis_types=(AxisType.Explicit, AxisType.Explicit))
        jax.set_mesh(cls.mesh)
        cls.cfg_unsharded = get_test_config(use_fsdp=False, use_tp=False)
        cls.cfg_sharded = get_test_config(use_fsdp=True, use_tp=True)

    def test_sharding_config_creation(self):
        """Test that sharding configs are created correctly."""
        cfg_unsharded = modeling.Qwen3VLConfig.qwen3vl_2b()
        self.assertEqual(cfg_unsharded.text_config.shd_cfg.q_weight, P(None, None))

        cfg_sharded = modeling.Qwen3VLConfig.qwen3vl_2b(use_fsdp=True, use_tp=True)
        self.assertEqual(cfg_sharded.text_config.shd_cfg.q_weight, P("fsdp", "tp"))
        self.assertEqual(cfg_sharded.vision_config.shd_cfg.attn_qkv_kernel, P("fsdp", "tp"))

    def test_text_mlp_sharded_vs_unsharded(self):
        """Test text MLP output is numerically equivalent with/without sharding."""
        rngs = nnx.Rngs(42)
        mlp_unsharded = modeling.Qwen3VLMLP(self.cfg_unsharded.text_config, rngs=rngs)

        gate_kernel = np.array(mlp_unsharded.gate_proj.kernel[...])
        up_kernel = np.array(mlp_unsharded.up_proj.kernel[...])
        down_kernel = np.array(mlp_unsharded.down_proj.kernel[...])

        x = jnp.ones((4, 8, 128), dtype=jnp.float32)
        out_unsharded = mlp_unsharded(x)

        rngs = nnx.Rngs(42)
        mlp_sharded = modeling.Qwen3VLMLP(self.cfg_sharded.text_config, rngs=rngs)

        mlp_sharded.gate_proj.kernel[...] = jnp.array(gate_kernel)
        mlp_sharded.up_proj.kernel[...] = jnp.array(up_kernel)
        mlp_sharded.down_proj.kernel[...] = jnp.array(down_kernel)

        x_sharded = jnp.ones((4, 8, 128), dtype=jnp.float32)
        out_sharded = mlp_sharded(x_sharded)

        np.testing.assert_allclose(
            np.array(out_unsharded),
            np.array(out_sharded),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Sharded MLP output differs from unsharded",
        )

    def test_vision_mlp_sharded_vs_unsharded(self):
        """Test vision MLP output is numerically equivalent with/without sharding."""
        rngs = nnx.Rngs(42)
        mlp_unsharded = modeling.Qwen3VLVisionMLP(self.cfg_unsharded.vision_config, rngs=rngs)

        fc1_kernel = np.array(mlp_unsharded.linear_fc1.kernel[...])
        fc2_kernel = np.array(mlp_unsharded.linear_fc2.kernel[...])
        fc1_bias = np.array(mlp_unsharded.linear_fc1.bias[...])
        fc2_bias = np.array(mlp_unsharded.linear_fc2.bias[...])

        x = jnp.ones((16, 64), dtype=jnp.float32)
        out_unsharded = mlp_unsharded(x)

        rngs = nnx.Rngs(42)
        mlp_sharded = modeling.Qwen3VLVisionMLP(self.cfg_sharded.vision_config, rngs=rngs)

        mlp_sharded.linear_fc1.kernel[...] = jnp.array(fc1_kernel)
        mlp_sharded.linear_fc2.kernel[...] = jnp.array(fc2_kernel)
        mlp_sharded.linear_fc1.bias[...] = jnp.array(fc1_bias)
        mlp_sharded.linear_fc2.bias[...] = jnp.array(fc2_bias)

        x_sharded = jnp.ones((16, 64), dtype=jnp.float32)
        out_sharded = mlp_sharded(x_sharded)

        np.testing.assert_allclose(
            np.array(out_unsharded),
            np.array(out_sharded),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Sharded Vision MLP output differs from unsharded",
        )

    def test_full_model_creation_with_sharding(self):
        """Test full model can be created with sharding enabled."""
        rngs = nnx.Rngs(0)
        model = modeling.Qwen3VLForConditionalGeneration(self.cfg_sharded, rngs=rngs)
        self.assertIsNotNone(model)

    def test_text_model_forward_with_sharding(self):
        """Test text model can run forward pass with sharding enabled."""
        rngs = nnx.Rngs(42)
        text_model = modeling.Qwen3VLTextModel(self.cfg_sharded.text_config, rngs=rngs)

        batch, seq_len = 4, 8
        inputs_embeds = jnp.ones((batch, seq_len, 128), dtype=jnp.float32)
        cache = modeling.init_cache(self.cfg_sharded, batch_size=batch, token_len=seq_len, generate_steps=4)
        positions = jnp.arange(seq_len)[None, :].repeat(batch, axis=0)
        sin, cos = modeling._generate_rope(
            positions, self.cfg_sharded.text_config.head_dim, self.cfg_sharded.text_config.rope_theta
        )

        output = text_model(inputs_embeds, cache, sin, cos, mask=None)
        self.assertEqual(output.shape, (batch, seq_len, 128))

    def test_vision_attention_sharded_vs_unsharded(self):
        """Test vision attention output is numerically equivalent with/without sharding."""
        rngs = nnx.Rngs(42)
        attn_unsharded = modeling.Qwen3VLVisionAttention(self.cfg_unsharded.vision_config, rngs=rngs)

        graphdef, state = nnx.split(attn_unsharded)
        flat_state = nnx.to_flat_state(state)
        state_arrays = {k: np.array(v[...]) for k, v in zip(flat_state.paths, flat_state.leaves)}

        seq_len = 16
        hidden_size = self.cfg_unsharded.vision_config.hidden_size
        head_dim = self.cfg_unsharded.vision_config.head_dim
        x = jnp.ones((seq_len, hidden_size), dtype=jnp.float32)

        theta = 10000.0
        positions = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
        angles = jnp.concatenate([jnp.outer(positions, freqs)] * 2, axis=-1)
        cos_vals = jnp.cos(angles)
        sin_vals = jnp.sin(angles)

        out_unsharded = attn_unsharded(x, (cos_vals, sin_vals))

        rngs = nnx.Rngs(42)
        attn_sharded = modeling.Qwen3VLVisionAttention(self.cfg_sharded.vision_config, rngs=rngs)

        sharded_graphdef, sharded_state = nnx.split(attn_sharded)
        sharded_flat_state = nnx.to_flat_state(sharded_state)
        for k, v in zip(sharded_flat_state.paths, sharded_flat_state.leaves):
            if k in state_arrays:
                v[...] = jnp.array(state_arrays[k])

        x_sharded = jnp.ones((seq_len, hidden_size), dtype=jnp.float32)
        positions_sharded = jnp.arange(seq_len, dtype=jnp.float32)
        freqs_sharded = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
        angles_sharded = jnp.concatenate([jnp.outer(positions_sharded, freqs_sharded)] * 2, axis=-1)
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

    def test_full_vision_text_forward(self):
        """Test full vision+text forward pass with sharding."""
        rngs = nnx.Rngs(0)
        fsdp = modeling.ShardMode.FSDP.value
        model = modeling.Qwen3VLForConditionalGeneration(self.cfg_sharded, rngs=rngs)

        batch_size = 4
        num_tokens = 128
        patch_size = self.cfg_sharded.vision_config.patch_size
        temporal_patch_size = self.cfg_sharded.vision_config.temporal_patch_size
        in_channels = self.cfg_sharded.vision_config.in_channels
        spatial_merge_size = self.cfg_sharded.vision_config.spatial_merge_size
        per_patch_features = in_channels * temporal_patch_size * patch_size * patch_size

        grid_h, grid_w = patch_size, patch_size
        num_patches_per_image = grid_h * grid_w
        merged_patches_per_image = num_patches_per_image // (spatial_merge_size**2)
        total_patches = num_patches_per_image * batch_size

        key = jax.random.key(0)
        pixel_values = jax.random.uniform(
            key,
            (total_patches, per_patch_features),
            dtype=jnp.float32,
            minval=-1,
            maxval=1,
        )

        n_text = jax.device_put(
            np.arange(batch_size * num_tokens).reshape(batch_size, -1) % self.cfg_sharded.text_config.vocab_size,
            device=P(fsdp),
        )

        token_type_ids = np.zeros((batch_size, num_tokens), dtype=np.int32)
        token_type_ids[:, 12 : 12 + merged_patches_per_image] = 1
        n_tti = jax.device_put(token_type_ids, device=P(fsdp))

        image_grid_thw = ((1, grid_h, grid_w),) * batch_size
        cache = modeling.init_cache(self.cfg_sharded, batch_size, num_tokens, 1, jnp.float32)

        out = model(n_text, cache, pixel_values, image_grid_thw, n_tti)

        self.assertEqual(out.shape, (batch_size, num_tokens, self.cfg_sharded.text_config.vocab_size))
        self.assertIsInstance(out.sharding, NamedSharding)
        expected_logit_shd = P(self.cfg_sharded.text_config.shd_cfg.act_btd[0], None, None)
        self.assertEqual(out.sharding.spec, expected_logit_shd)


if __name__ == "__main__":
    absltest.main()
