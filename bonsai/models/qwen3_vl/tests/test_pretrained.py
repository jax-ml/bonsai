"""Tests for actual pretrained Qwen3-VL models.

Uses rtol/atol comparisons for numerical equivalence verification.
"""

import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from huggingface_hub import snapshot_download

from bonsai.models.qwen3_vl import modeling as model_lib
from bonsai.models.qwen3_vl import params


MODEL_2B_ID = "Qwen/Qwen3-VL-2B-Instruct"


class TestPretrainedComponents(absltest.TestCase):
    """Layer-by-layer comparison between PyTorch and Flax."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.model_path = snapshot_download(MODEL_2B_ID)

        cls.pt_model = Qwen3VLForConditionalGeneration.from_pretrained(
            cls.model_path,
            torch_dtype=torch.float32,
        ).to("cpu")
        cls.pt_model.eval()

        cls.flax_config = model_lib.Qwen3VLConfig.qwen3vl_2b()
        cls.flax_model = params.create_model_from_safe_tensors(cls.model_path, cls.flax_config)

        cls.processor = AutoProcessor.from_pretrained(cls.model_path)

        cls.test_text = "Hello"
        inputs = cls.processor(text=cls.test_text, return_tensors="pt")
        cls.input_ids_pt = inputs["input_ids"]
        cls.input_ids_jax = jnp.array(inputs["input_ids"].numpy())

    def test_01_embedding_weights_match(self):
        """Check that embedding weights loaded correctly."""
        pt_embed = self.pt_model.model.language_model.embed_tokens.weight.detach().numpy()
        flax_embed = np.array(self.flax_model.model.language_model.embed_tokens.embedding[:])

        self.assertEqual(pt_embed.shape, flax_embed.shape)
        np.testing.assert_allclose(pt_embed, flax_embed, rtol=1e-5, atol=1e-5)

    def test_02_embedding_output_match(self):
        """Check that embedding outputs match."""
        with torch.inference_mode():
            pt_embed_out = self.pt_model.model.language_model.embed_tokens(self.input_ids_pt)
        pt_out = pt_embed_out.numpy()

        flax_embed_out = self.flax_model.model.language_model.embed_tokens(self.input_ids_jax)
        flax_out = np.array(flax_embed_out)

        np.testing.assert_allclose(pt_out, flax_out, rtol=1e-5, atol=1e-5)

    def test_03_layer0_weights(self):
        """Check that layer 0 weights match."""
        pt_norm = self.pt_model.model.language_model.layers[0].input_layernorm.weight.detach().numpy()
        flax_norm = np.array(self.flax_model.model.language_model.layers[0].input_layernorm.weight[:])
        np.testing.assert_allclose(pt_norm, flax_norm, rtol=1e-5, atol=1e-5)

    def test_04_full_forward_pass(self):
        """Full forward pass comparison with cache."""
        # PyTorch
        with torch.inference_mode():
            pt_out = self.pt_model(input_ids=self.input_ids_pt)
        pt_logits = pt_out.logits.numpy()

        # Flax with cache
        batch, seq_len = self.input_ids_jax.shape
        cache = model_lib.init_cache(self.flax_config, batch, seq_len, generate_steps=10)
        flax_logits = np.array(self.flax_model(self.input_ids_jax, cache))

        print(f"PT logits: mean={pt_logits.mean():.4f}, std={pt_logits.std():.4f}")
        print(f"Flax logits: mean={flax_logits.mean():.4f}, std={flax_logits.std():.4f}")

        # Cosine similarity
        pt_flat = pt_logits.flatten()
        flax_flat = flax_logits.flatten()
        cos_sim = np.dot(pt_flat, flax_flat) / (np.linalg.norm(pt_flat) * np.linalg.norm(flax_flat))
        print(f"Cosine similarity: {cos_sim}")

        max_diff = np.abs(pt_logits - flax_logits).max()
        print(f"Max logits diff: {max_diff}")

        torch.testing.assert_close(
            torch.tensor(flax_logits),
            torch.tensor(pt_logits),
            rtol=0.05,
            atol=0.05,
        )


class TestKVCache(absltest.TestCase):
    """Test KV-cache functionality."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model_path = snapshot_download(MODEL_2B_ID)
        cls.flax_config = model_lib.Qwen3VLConfig.qwen3vl_2b()
        cls.flax_model = params.create_model_from_safe_tensors(cls.model_path, cls.flax_config)
        cls.processor = AutoProcessor.from_pretrained(cls.model_path)

    def test_cache_initialization(self):
        """Test KV-cache initialization."""
        cache = model_lib.init_cache(self.flax_config, batch_size=1, token_len=10, generate_steps=20)
        self.assertEqual(len(cache), self.flax_config.text_config.num_hidden_layers)
        self.assertEqual(cache[0].k_cache.value.shape[0], 1)

    def test_jit_forward(self):
        """Test JIT-compiled forward."""
        text = "Hello"
        inputs = self.processor(text=text, return_tensors="pt")
        input_ids = jnp.array(inputs["input_ids"].numpy())

        cache = model_lib.init_cache(self.flax_config, 1, input_ids.shape[1], 20)
        logits, _ = model_lib.forward(self.flax_model, cache, input_ids)

        self.assertEqual(logits.shape, (1, self.flax_config.text_config.vocab_size))

    def test_generation_step(self):
        """Test single token generation step."""
        cache = model_lib.init_cache(self.flax_config, 1, 5, 20)

        # Prefill
        input_ids = jnp.array([[1, 2, 3, 4, 5]])
        logits, cache = model_lib.forward(self.flax_model, cache, input_ids)

        # Check cache was updated
        self.assertEqual(int(cache[0].cur_ind.value), 5)

        # Decode step
        next_token = jnp.argmax(logits, axis=-1, keepdims=True)
        logits2, cache = model_lib.forward(self.flax_model, cache, next_token)

        self.assertEqual(int(cache[0].cur_ind.value), 6)
        self.assertEqual(logits2.shape, (1, self.flax_config.text_config.vocab_size))


if __name__ == "__main__":
    absltest.main()
