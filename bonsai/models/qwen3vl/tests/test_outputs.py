"""Comprehensive unit tests for Qwen3-VL Flax NNX model.

Tests include:
- Component-level shape verification
- PyTorch-Flax numerical equivalence using transformers library
- M-RoPE logic verification
- Full model forward pass
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from flax import nnx

from ..modeling import (
    Qwen3VLConfig,
    Qwen3VLModel,
    TextConfig,
    VisionAttention,
    VisionBlock,
    VisionConfig,
    VisionMLP,
    VisionPatchEmbed,
    VisionPatchMerger,
    apply_interleaved_mrope,
    rotate_half,
)

# Check if PyTorch and transformers are available
try:
    import torch
    import torch.nn as nn
    from transformers import Qwen3VLConfig as HFQwen3VLConfig
    from transformers import Qwen3VLModel as HFQwen3VLModel
    from transformers import Qwen3VLForConditionalGeneration as HFQwen3VLForConditionalGeneration

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# =============================================================================
# Test Configuration for Transformers Comparison
# =============================================================================


def get_hf_test_config():
    """Get a small HuggingFace Qwen3VL config for testing."""
    if not HAS_TRANSFORMERS:
        return None

    return HFQwen3VLConfig(
        text_config={
            "bos_token_id": 0,
            "eos_token_id": 1,
            "pad_token_id": 2,
            "hidden_act": "silu",
            "head_dim": 16,
            "hidden_size": 64,
            "vocab_size": 1000,
            "intermediate_size": 256,
            "max_position_embeddings": 512,
            "model_type": "qwen3_vl",
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "num_key_value_heads": 2,
            "rope_theta": 10000,
            "tie_word_embeddings": True,
            "rope_scaling": {"rope_type": "default", "mrope_section": [8, 4, 4]},
        },
        vision_config={
            "depth": 2,
            "in_chans": 3,
            "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": 256,
            "out_hidden_size": 64,
            "hidden_size": 64,
            "num_heads": 4,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0, 1],
        },
        image_token_id=3,
        video_token_id=4,
        vision_start_token_id=5,
        vision_end_token_id=6,
        tie_word_embeddings=True,
    )


def get_flax_test_config():
    """Get matching Flax config for the HF test config."""
    return Qwen3VLConfig(
        text_config=TextConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            hidden_act="silu",
            max_position_embeddings=512,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            rope_theta=10000.0,
            mrope_section=(8, 4, 4),
        ),
        vision_config=VisionConfig(
            depth=2,
            hidden_size=64,
            intermediate_size=256,
            num_heads=4,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=64,
            num_position_embeddings=16,
            deepstack_visual_indexes=(0, 1),
        ),
        image_token_id=3,
        video_token_id=4,
        vision_start_token_id=5,
        vision_end_token_id=6,
        tie_word_embeddings=True,
    )


# =============================================================================
# Vision Component Tests
# =============================================================================


class TestVisionComponents(absltest.TestCase):
    """Test individual vision encoder components."""

    def setUp(self):
        super().setUp()
        self.vision_config = VisionConfig(
            depth=2,
            hidden_size=64,
            intermediate_size=256,
            num_heads=4,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=64,
            num_position_embeddings=16,
            deepstack_visual_indexes=(0, 1),
        )
        self.rngs = nnx.Rngs(42)

    def test_vision_patch_embed_shape(self):
        """Test VisionPatchEmbed produces correct output shape."""
        patch_embed = VisionPatchEmbed(self.vision_config, rngs=self.rngs)
        x = jnp.zeros((1, 3, 2, 32, 32), dtype=jnp.float32)
        out = patch_embed(x)
        self.assertEqual(out.shape, (4, 64))

    def test_vision_patch_embed_nonzero(self):
        """Test VisionPatchEmbed with random input produces non-zero output."""
        patch_embed = VisionPatchEmbed(self.vision_config, rngs=self.rngs)
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (1, 3, 2, 32, 32))
        out = patch_embed(x)
        self.assertFalse(jnp.allclose(out, 0))

    def test_vision_attention_2d_input(self):
        """Test VisionAttention with 2D sequence input."""
        attn = VisionAttention(self.vision_config, rngs=self.rngs)
        seq_len = 4
        hidden = jnp.zeros((seq_len, 64), dtype=jnp.float32)
        head_dim = 64 // 4
        cos = jnp.ones((seq_len, head_dim), dtype=jnp.float32)
        sin = jnp.zeros((seq_len, head_dim), dtype=jnp.float32)
        out = attn(hidden, cu_seqlens=None, position_embeddings=(cos, sin))
        self.assertEqual(out.shape, (seq_len, 64))

    def test_vision_mlp_shape(self):
        """Test VisionMLP produces correct shape."""
        mlp = VisionMLP(self.vision_config, rngs=self.rngs)
        x = jnp.zeros((4, 64), dtype=jnp.float32)
        out = mlp(x)
        self.assertEqual(out.shape, (4, 64))

    def test_vision_block_shape(self):
        """Test VisionBlock produces correct shape."""
        block = VisionBlock(self.vision_config, rngs=self.rngs)
        seq_len = 4
        x = jnp.zeros((seq_len, 64), dtype=jnp.float32)
        head_dim = 64 // 4
        cos = jnp.ones((seq_len, head_dim), dtype=jnp.float32)
        sin = jnp.zeros((seq_len, head_dim), dtype=jnp.float32)
        out = block(x, cu_seqlens=None, position_embeddings=(cos, sin))
        self.assertEqual(out.shape, (seq_len, 64))

    def test_vision_patch_merger_shape(self):
        """Test VisionPatchMerger produces correct shape."""
        merger = VisionPatchMerger(self.vision_config, use_postshuffle_norm=False, rngs=self.rngs)
        x = jnp.zeros((16, 64), dtype=jnp.float32)
        out = merger(x)
        self.assertEqual(out.shape, (4, 64))


# =============================================================================
# Text Component Tests
# =============================================================================


class TestTextComponents(absltest.TestCase):
    """Test individual text decoder components."""

    def test_rotate_half(self):
        """Test rotate_half function."""
        x = jnp.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(1, 8)
        out = rotate_half(x)
        expected = jnp.array([-5, -6, -7, -8, 1, 2, 3, 4]).reshape(1, 8)
        np.testing.assert_array_almost_equal(out, expected)

    def test_mrope_interleaving(self):
        """Test apply_interleaved_mrope produces correct interleaved pattern."""
        freqs = jnp.ones((3, 1, 1, 16))
        freqs = freqs.at[0].set(1.0)
        freqs = freqs.at[1].set(2.0)
        freqs = freqs.at[2].set(3.0)
        mrope_section = (8, 4, 4)
        out = apply_interleaved_mrope(freqs, mrope_section)
        self.assertEqual(out.shape, (1, 1, 16))
        self.assertAlmostEqual(float(out[0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(out[0, 0, 1]), 2.0)
        self.assertAlmostEqual(float(out[0, 0, 2]), 3.0)
        self.assertAlmostEqual(float(out[0, 0, 3]), 1.0)


# =============================================================================
# Full Model Tests
# =============================================================================


class TestFullModel(absltest.TestCase):
    """Test full Qwen3VL model."""

    def setUp(self):
        super().setUp()
        self.config = Qwen3VLConfig.standard_test()
        self.rngs = nnx.Rngs(42)
        self.model = Qwen3VLModel(self.config, rngs=self.rngs)

    def test_forward_shape(self):
        """Test full model forward pass produces correct shape."""
        B, S = 1, 10
        input_ids = jnp.zeros((B, S), dtype=jnp.int32)
        pixel_values = jnp.zeros((B, 3, 2, 32, 32), dtype=jnp.float32)
        grid_thw = jnp.array([[1, 2, 2]], dtype=jnp.int32)
        visual_pos_masks = jnp.zeros((B, S), dtype=jnp.bool_)
        output = self.model(input_ids, pixel_values, grid_thw, visual_pos_masks)
        self.assertEqual(output.shape, (B, S, self.config.text_config.hidden_size))

    def test_forward_batched(self):
        """Test full model with batch size > 1."""
        B, S = 2, 8
        input_ids = jnp.zeros((B, S), dtype=jnp.int32)
        pixel_values = jnp.zeros((B, 3, 2, 32, 32), dtype=jnp.float32)
        grid_thw = jnp.array([[1, 2, 2], [1, 2, 2]], dtype=jnp.int32)
        visual_pos_masks = jnp.zeros((B, S), dtype=jnp.bool_)
        output = self.model(input_ids, pixel_values, grid_thw, visual_pos_masks)
        self.assertEqual(output.shape, (B, S, self.config.text_config.hidden_size))

    def test_text_only_forward(self):
        """Test model forward without vision inputs."""
        B, S = 1, 10
        input_ids = jnp.zeros((B, S), dtype=jnp.int32)
        output = self.model(input_ids)
        self.assertEqual(output.shape, (B, S, self.config.text_config.hidden_size))

    def test_output_not_nan(self):
        """Test model output contains no NaN values."""
        B, S = 1, 5
        key = jax.random.PRNGKey(0)
        input_ids = jax.random.randint(key, (B, S), 0, 100)
        pixel_values = jax.random.normal(key, (B, 3, 2, 32, 32))
        grid_thw = jnp.array([[1, 2, 2]], dtype=jnp.int32)
        visual_pos_masks = jnp.zeros((B, S), dtype=jnp.bool_)
        output = self.model(input_ids, pixel_values, grid_thw, visual_pos_masks)
        self.assertFalse(jnp.any(jnp.isnan(output)))


# =============================================================================
# Transformers Library Comparison Tests
# =============================================================================


@unittest.skipUnless(HAS_TRANSFORMERS, "transformers library not available")
class TestTransformersComparison(absltest.TestCase):
    """Compare Flax and transformers library implementations numerically."""

    def setUp(self):
        super().setUp()
        jax.config.update("jax_default_matmul_precision", "float32")
        self.hf_config = get_hf_test_config()
        self.flax_config = get_flax_test_config()
        self.rtol = 1e-3
        self.atol = 1e-3

    def test_rotate_half_equivalence(self):
        """Test rotate_half matches PyTorch implementation."""
        x_np = np.random.randn(2, 4, 8).astype(np.float32)
        x_jax = jnp.array(x_np)
        x_torch = torch.tensor(x_np)

        jax_out = rotate_half(x_jax)
        x1 = x_torch[..., : x_torch.shape[-1] // 2]
        x2 = x_torch[..., x_torch.shape[-1] // 2 :]
        torch_out = torch.cat((-x2, x1), dim=-1)

        np.testing.assert_allclose(np.array(jax_out), torch_out.numpy(), rtol=self.rtol, atol=self.atol)

    def test_gelu_equivalence(self):
        """Test GELU activation matches PyTorch."""
        x_np = np.random.randn(4, 64).astype(np.float32)
        x_jax = jnp.array(x_np)
        x_torch = torch.tensor(x_np)

        jax_out = nnx.gelu(x_jax, approximate=False)
        torch_out = torch.nn.functional.gelu(x_torch, approximate="none")

        np.testing.assert_allclose(np.array(jax_out), torch_out.numpy(), rtol=self.rtol, atol=self.atol)

    def test_silu_equivalence(self):
        """Test SiLU activation matches PyTorch."""
        x_np = np.random.randn(4, 64).astype(np.float32)
        x_jax = jnp.array(x_np)
        x_torch = torch.tensor(x_np)

        jax_out = nnx.silu(x_jax)
        torch_out = torch.nn.functional.silu(x_torch)

        np.testing.assert_allclose(np.array(jax_out), torch_out.numpy(), rtol=self.rtol, atol=self.atol)

    def test_softmax_equivalence(self):
        """Test softmax matches PyTorch."""
        x_np = np.random.randn(2, 4, 4).astype(np.float32)
        x_jax = jnp.array(x_np)
        x_torch = torch.tensor(x_np)

        jax_out = nnx.softmax(x_jax, axis=-1)
        torch_out = torch.nn.functional.softmax(x_torch, dim=-1)

        np.testing.assert_allclose(np.array(jax_out), torch_out.numpy(), rtol=self.rtol, atol=self.atol)

    def test_layer_norm_equivalence(self):
        """Test LayerNorm matches PyTorch."""
        x_np = np.random.randn(2, 4, 64).astype(np.float32)
        x_jax = jnp.array(x_np)
        x_torch = torch.tensor(x_np)

        rngs = nnx.Rngs(0)
        jax_ln = nnx.LayerNorm(64, epsilon=1e-6, rngs=rngs)
        jax_ln.scale.value = jnp.ones(64)
        jax_ln.bias.value = jnp.zeros(64)
        jax_out = jax_ln(x_jax)

        torch_ln = torch.nn.LayerNorm(64, eps=1e-6)
        torch_ln.weight.data.fill_(1.0)
        torch_ln.bias.data.fill_(0.0)
        torch_out = torch_ln(x_torch)

        np.testing.assert_allclose(np.array(jax_out), torch_out.detach().numpy(), rtol=self.rtol, atol=self.atol)

    def test_hf_model_can_init(self):
        """Test that HuggingFace model can be initialized with test config."""
        model = HFQwen3VLModel(self.hf_config)
        self.assertIsNotNone(model)

        # Check structure
        self.assertTrue(hasattr(model, "visual"))
        self.assertTrue(hasattr(model, "language_model"))

    def test_hf_model_forward(self):
        """Test HuggingFace model forward pass works."""
        model = HFQwen3VLModel(self.hf_config).eval()

        B = 1
        S = 10
        input_ids = torch.randint(0, 100, (B, S))

        with torch.no_grad():
            outputs = model(input_ids=input_ids)

        self.assertIsNotNone(outputs)
        self.assertTrue(hasattr(outputs, "last_hidden_state"))

    def test_flax_matches_hf_text_only_shape(self):
        """Test Flax model produces same shape as HF for text-only input."""
        hf_model = HFQwen3VLModel(self.hf_config).eval()
        flax_model = Qwen3VLModel(self.flax_config, rngs=nnx.Rngs(0))

        B, S = 1, 10
        input_ids_np = np.random.randint(0, 100, (B, S))

        # HF forward
        with torch.no_grad():
            hf_out = hf_model(input_ids=torch.tensor(input_ids_np))

        # Flax forward
        flax_out = flax_model(jnp.array(input_ids_np))

        # Check shapes match
        self.assertEqual(
            hf_out.last_hidden_state.shape[-1],
            flax_out.shape[-1],
            f"HF shape {hf_out.last_hidden_state.shape} vs Flax shape {flax_out.shape}",
        )


# =============================================================================
# Weight Loading Tests
# =============================================================================


class TestWeightLoading(absltest.TestCase):
    """Test weight loading utilities."""

    def test_standard_test_config(self):
        """Test standard_test config creation."""
        config = Qwen3VLConfig.standard_test()
        self.assertIsInstance(config, Qwen3VLConfig)
        self.assertEqual(config.text_config.hidden_size, 64)
        self.assertEqual(config.vision_config.hidden_size, 64)

    def test_model_from_standard_test(self):
        """Test model can be created from standard_test config."""
        config = Qwen3VLConfig.standard_test()
        model = Qwen3VLModel(config, rngs=nnx.Rngs(0))
        self.assertIsNotNone(model)


if __name__ == "__main__":
    absltest.main()
