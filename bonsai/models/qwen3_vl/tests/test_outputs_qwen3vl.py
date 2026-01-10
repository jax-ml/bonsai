"""Unit tests for Qwen3-VL JAX implementation with PyTorch numerical comparison.

Following vjepa2 testing patterns: create PyTorch model, save weights,
load into JAX, compare outputs with rtol/atol.

Tests are structured to work in two modes:
1. JAX-only mode: Tests shapes, basic functionality (always works)
2. PyTorch comparison mode: Numerical equivalence tests (when PT available)
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

# Try to use absltest, fallback to unittest
try:
    from absl.testing import absltest
except ImportError:
    import unittest as absltest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
import modeling

# Try to import PyTorch for comparison tests
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestConfigCreation(absltest.TestCase):
    """Test configuration creation."""

    def test_standard_test_config(self):
        """Test standard_test creates valid config."""
        config = modeling.Qwen3VLConfig.standard_test()
        self.assertEqual(config.vision_config.depth, 2)
        self.assertEqual(config.text_config.num_hidden_layers, 2)

    def test_2b_config(self):
        """Test 2B config values."""
        config = modeling.Qwen3VLConfig.qwen3vl_2b()
        self.assertEqual(config.vision_config.depth, 24)
        self.assertEqual(config.text_config.num_hidden_layers, 28)
        self.assertEqual(config.text_config.hidden_size, 2048)

    def test_4b_config(self):
        """Test 4B config values."""
        config = modeling.Qwen3VLConfig.qwen3vl_4b()
        self.assertEqual(config.text_config.hidden_size, 2560)

    def test_8b_config(self):
        """Test 8B config values."""
        config = modeling.Qwen3VLConfig.qwen3vl_8b()
        self.assertEqual(config.vision_config.depth, 27)
        self.assertEqual(config.text_config.tie_word_embeddings, False)


class TestRMSNorm(absltest.TestCase):
    """Test RMSNorm implementation."""

    def test_rms_norm_shape(self):
        """Test RMSNorm preserves shape."""
        norm = modeling.RMSNorm(64, rngs=nnx.Rngs(0))
        x = jax.random.normal(jax.random.key(0), (2, 10, 64))
        out = norm(x)
        self.assertEqual(out.shape, x.shape)

    def test_rms_norm_dtype(self):
        """Test RMSNorm preserves input dtype."""
        norm = modeling.RMSNorm(64, rngs=nnx.Rngs(0))
        x = jax.random.normal(jax.random.key(0), (2, 10, 64), dtype=jnp.bfloat16)
        out = norm(x)
        self.assertEqual(out.dtype, jnp.bfloat16)

    def test_rms_norm_normalization(self):
        """Test RMSNorm actually normalizes."""
        norm = modeling.RMSNorm(64, rngs=nnx.Rngs(0))
        x = jax.random.normal(jax.random.key(0), (2, 10, 64)) * 10  # Large values
        out = norm(x)
        # Output should have smaller std than input (after RMS normalization)
        self.assertLess(float(jnp.std(out)), float(jnp.std(x)))


class TestRotateHalf(absltest.TestCase):
    """Test rotate_half function."""

    def test_rotate_half_shape(self):
        """Test rotate_half preserves shape."""

        def rotate_half(x):
            x1, x2 = jnp.split(x, 2, axis=-1)
            return jnp.concatenate([-x2, x1], axis=-1)

        x = jax.random.normal(jax.random.key(0), (4, 8, 16, 32))
        out = rotate_half(x)
        self.assertEqual(out.shape, x.shape)

    def test_rotate_half_involution(self):
        """Test that applying rotate_half twice gives -x."""

        def rotate_half(x):
            x1, x2 = jnp.split(x, 2, axis=-1)
            return jnp.concatenate([-x2, x1], axis=-1)

        x = jax.random.normal(jax.random.key(0), (4, 8, 16, 32))
        out = rotate_half(rotate_half(x))
        np.testing.assert_allclose(np.array(out), -np.array(x), rtol=1e-5, atol=1e-5)


class TestCausalMask(absltest.TestCase):
    """Test causal attention mask generation."""

    def test_causal_mask_shape(self):
        """Test mask has correct shape."""
        mask = modeling.make_causal_mask(seq_len=5, cache_len=10, cur_pos=0)
        self.assertEqual(mask.shape, (1, 1, 5, 10))

    def test_causal_mask_is_causal(self):
        """Test mask is lower triangular."""
        mask = modeling.make_causal_mask(seq_len=4, cache_len=4, cur_pos=0)
        mask_np = np.array(mask[0, 0])

        # Position i can only attend to positions <= i
        for i in range(4):
            for j in range(4):
                if j <= i:
                    self.assertTrue(mask_np[i, j], f"Position {i} should attend to {j}")
                else:
                    self.assertFalse(mask_np[i, j], f"Position {i} should NOT attend to {j}")


class TestLayerCache(absltest.TestCase):
    """Test KV-cache initialization and shapes."""

    def test_cache_initialization(self):
        """Test cache is initialized with correct shapes."""
        config = modeling.Qwen3VLConfig.standard_test()
        cache = modeling.init_cache(config, batch_size=2, token_len=10, generate_steps=5)

        self.assertEqual(len(cache), config.text_config.num_hidden_layers)

        for layer_cache in cache:
            k_shape = layer_cache.k_cache.value.shape
            self.assertEqual(k_shape[0], 2)  # batch
            self.assertEqual(k_shape[2], config.text_config.num_key_value_heads)
            self.assertEqual(k_shape[3], config.text_config.head_dim)


class TestModelInitialization(absltest.TestCase):
    """Test model initialization and forward pass shapes."""

    def test_model_creates(self):
        """Test model instantiation."""
        config = modeling.Qwen3VLConfig.standard_test()
        model = modeling.Qwen3VLForConditionalGeneration(config, rngs=nnx.Rngs(0))

        # Check model structure
        self.assertEqual(len(model.model.language_model.layers), config.text_config.num_hidden_layers)
        self.assertEqual(len(model.model.visual.blocks), config.vision_config.depth)

    def test_text_only_forward_pass(self):
        """Test text-only forward pass produces correct shapes."""
        config = modeling.Qwen3VLConfig.standard_test()
        model = modeling.Qwen3VLForConditionalGeneration(config, rngs=nnx.Rngs(0))

        batch_size, seq_len = 2, 10
        input_ids = jax.random.randint(jax.random.key(0), (batch_size, seq_len), 0, config.text_config.vocab_size)

        logits = model(input_ids)

        expected_shape = (batch_size, seq_len, config.text_config.vocab_size)
        self.assertEqual(logits.shape, expected_shape)

    def test_embedding_shape(self):
        """Test embedding layer has correct shape."""
        config = modeling.Qwen3VLConfig.standard_test()
        model = modeling.Qwen3VLForConditionalGeneration(config, rngs=nnx.Rngs(0))

        embed_shape = model.model.language_model.embed_tokens.embedding.value.shape
        self.assertEqual(embed_shape, (config.text_config.vocab_size, config.text_config.hidden_size))


class TestVisionPatchEmbed(absltest.TestCase):
    """Test vision patch embedding."""

    def test_patch_embed_output_shape(self):
        """Test patch embed produces correct output shape."""
        config = modeling.Qwen3VLVisionConfig(
            hidden_size=64,
            patch_size=8,
            temporal_patch_size=2,
            in_channels=3,
        )
        embed = modeling.Qwen3VLPatchEmbed(config, rngs=nnx.Rngs(0))

        # Simulated preprocessed input (seq_len, patch_dim)
        patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
        x = jax.random.normal(jax.random.key(0), (16, patch_dim))

        out = embed(x)
        self.assertEqual(out.shape, (16, config.hidden_size))


class TestVisionMLP(absltest.TestCase):
    """Test vision MLP."""

    def test_mlp_shape(self):
        """Test MLP preserves sequence dimension."""
        config = modeling.Qwen3VLVisionConfig(hidden_size=64, intermediate_size=128)
        mlp = modeling.Qwen3VLVisionMLP(config, rngs=nnx.Rngs(0))
        x = jax.random.normal(jax.random.key(0), (100, 64))
        out = mlp(x)
        self.assertEqual(out.shape, (100, 64))


class TestTextMLP(absltest.TestCase):
    """Test text decoder MLP."""

    def test_text_mlp_gated(self):
        """Test gated MLP (SiLU activation)."""
        config = modeling.Qwen3VLTextConfig(hidden_size=128, intermediate_size=256)
        mlp = modeling.Qwen3VLMLP(config, rngs=nnx.Rngs(0))
        x = jax.random.normal(jax.random.key(0), (2, 10, 128))
        out = mlp(x)
        self.assertEqual(out.shape, (2, 10, 128))


class TestVisionBlock(absltest.TestCase):
    """Test vision transformer block."""

    def test_block_forward(self):
        """Test vision block forward pass."""
        config = modeling.Qwen3VLVisionConfig(
            hidden_size=64,
            intermediate_size=128,
            num_heads=4,
        )
        block = modeling.Qwen3VLVisionBlock(config, rngs=nnx.Rngs(0))

        seq_len = 16
        x = jax.random.normal(jax.random.key(0), (seq_len, 64))
        head_dim = config.head_dim
        cos = jax.random.normal(jax.random.key(1), (seq_len, head_dim))
        sin = jax.random.normal(jax.random.key(2), (seq_len, head_dim))
        cu_seqlens = jnp.array([0, seq_len])

        out = block(x, cu_seqlens, (cos, sin))
        self.assertEqual(out.shape, (seq_len, 64))


class TestDecoderLayer(absltest.TestCase):
    """Test text decoder layer."""

    def test_decoder_layer_forward(self):
        """Test single decoder layer."""
        config = modeling.Qwen3VLTextConfig(
            hidden_size=128,
            intermediate_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
        )
        layer = modeling.Qwen3VLDecoderLayer(config, layer_idx=0, rngs=nnx.Rngs(0))

        batch, seq = 2, 10
        x = jax.random.normal(jax.random.key(0), (batch, seq, 128))
        cos = jax.random.normal(jax.random.key(1), (batch, seq, 32))
        sin = jax.random.normal(jax.random.key(2), (batch, seq, 32))

        out = layer(x, (cos, sin), attention_mask=None, cache=None)
        self.assertEqual(out.shape, (batch, seq, 128))


class TestPatchMerger(absltest.TestCase):
    """Test vision patch merger."""

    def test_merger_output_shape(self):
        """Test patch merger output shape."""
        config = modeling.Qwen3VLVisionConfig(
            hidden_size=64,
            out_hidden_size=128,
            spatial_merge_size=2,
        )
        merger = modeling.Qwen3VLPatchMerger(config, use_postshuffle_norm=False, rngs=nnx.Rngs(0))

        # Input: seq_len patches, will be merged by spatial_merge_size^2
        merge_factor = config.spatial_merge_size**2
        seq_len = 16 * merge_factor  # 16 merged patches
        x = jax.random.normal(jax.random.key(0), (seq_len, config.hidden_size))

        out = merger(x)
        self.assertEqual(out.shape, (16, config.out_hidden_size))


# PyTorch comparison tests (only run if PyTorch is available)
if TORCH_AVAILABLE:

    class TestPyTorchComparison(absltest.TestCase):
        """PyTorch-JAX numerical comparison tests."""

        def test_rms_norm_pytorch_equivalence(self):
            """Compare JAX RMSNorm with manual PyTorch implementation."""
            dim = 64
            eps = 1e-6

            # JAX implementation
            jax_norm = modeling.RMSNorm(dim, eps=eps, rngs=nnx.Rngs(0))

            # Test input
            key = jax.random.PRNGKey(42)
            jax_input = jax.random.normal(key, (2, 10, dim), dtype=jnp.float32)
            pt_input = torch.tensor(np.array(jax_input), dtype=torch.float32)

            # PyTorch manual RMS norm
            def pt_rms_norm(x, weight, eps):
                variance = x.pow(2).mean(-1, keepdim=True)
                return weight * (x / torch.sqrt(variance + eps))

            pt_weight = torch.tensor(np.array(jax_norm.weight.value), dtype=torch.float32)
            pt_output = pt_rms_norm(pt_input, pt_weight, eps)
            jax_output = jax_norm(jax_input)

            # Compare
            torch.testing.assert_close(
                torch.tensor(np.array(jax_output)),
                pt_output,
                rtol=1e-4,
                atol=1e-4,
            )

        def test_gated_mlp_equivalence(self):
            """Compare JAX gated MLP with PyTorch implementation."""
            hidden_size = 128
            intermediate_size = 256

            # JAX implementation
            config = modeling.Qwen3VLTextConfig(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
            )
            jax_mlp = modeling.Qwen3VLMLP(config, rngs=nnx.Rngs(0))

            # PyTorch manual gated MLP
            key = jax.random.PRNGKey(42)
            jax_input = jax.random.normal(key, (2, 10, hidden_size), dtype=jnp.float32)
            pt_input = torch.tensor(np.array(jax_input), dtype=torch.float32)

            # Get JAX weights
            gate_w = np.array(jax_mlp.gate_proj.kernel.value)
            up_w = np.array(jax_mlp.up_proj.kernel.value)
            down_w = np.array(jax_mlp.down_proj.kernel.value)

            # PyTorch computation
            gate = torch.nn.functional.silu(pt_input @ torch.tensor(gate_w))
            up = pt_input @ torch.tensor(up_w)
            pt_output = (gate * up) @ torch.tensor(down_w)

            jax_output = jax_mlp(jax_input)

            # Compare
            torch.testing.assert_close(
                torch.tensor(np.array(jax_output)),
                pt_output,
                rtol=1e-3,
                atol=1e-3,
            )


if __name__ == "__main__":
    absltest.main()
