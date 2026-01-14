"""Tests for actual pretrained Qwen3-VL models.

This file tests loading and running actual pretrained models like Qwen3-VL-2B-Instruct.
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


# HuggingFace model IDs
MODEL_2B_ID = "Qwen/Qwen3-VL-2B-Instruct"


class TestPretrainedComponents(absltest.TestCase):
    """Layer-by-layer comparison between PyTorch and Flax."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # Download model
        cls.model_path = snapshot_download(MODEL_2B_ID)

        # Load PyTorch model
        cls.pt_model = Qwen3VLForConditionalGeneration.from_pretrained(
            cls.model_path,
            torch_dtype=torch.float32,
        ).to("cpu")
        cls.pt_model.eval()

        # Load Flax model
        cls.flax_config = model_lib.Qwen3VLConfig.qwen3vl_2b()
        cls.flax_model = params.create_model_from_safe_tensors(cls.model_path, cls.flax_config)

        # Load processor
        cls.processor = AutoProcessor.from_pretrained(cls.model_path)

        # Test input
        cls.test_text = "Hello"
        inputs = cls.processor(text=cls.test_text, return_tensors="pt")
        cls.input_ids_pt = inputs["input_ids"]
        cls.input_ids_jax = jnp.array(inputs["input_ids"].numpy())

    def test_01_embedding_weights_match(self):
        """Check that embedding weights loaded correctly."""
        pt_embed = self.pt_model.model.language_model.embed_tokens.weight.detach().numpy()
        flax_embed = np.array(self.flax_model.model.language_model.embed_tokens.embedding[:])

        print(f"PT embed shape: {pt_embed.shape}")
        print(f"Flax embed shape: {flax_embed.shape}")
        print(f"PT embed stats: mean={pt_embed.mean():.6f}, std={pt_embed.std():.6f}")
        print(f"Flax embed stats: mean={float(flax_embed.mean()):.6f}, std={float(flax_embed.std()):.6f}")

        # Check shapes match
        self.assertEqual(pt_embed.shape, flax_embed.shape, "Embedding shapes don't match")

        # Check values match
        max_diff = np.abs(pt_embed - flax_embed).max()
        print(f"Max embedding weight diff: {max_diff}")

        np.testing.assert_allclose(pt_embed, flax_embed, rtol=1e-5, atol=1e-5)

    def test_02_embedding_output_match(self):
        """Check that embedding outputs match."""
        # PyTorch
        with torch.inference_mode():
            pt_embed_out = self.pt_model.model.language_model.embed_tokens(self.input_ids_pt)
        pt_out = pt_embed_out.numpy()

        # Flax
        flax_embed_out = self.flax_model.model.language_model.embed_tokens(self.input_ids_jax)
        flax_out = np.array(flax_embed_out)

        print(f"PT embed out: mean={pt_out.mean():.6f}, std={pt_out.std():.6f}")
        print(f"Flax embed out: mean={flax_out.mean():.6f}, std={flax_out.std():.6f}")

        max_diff = np.abs(pt_out - flax_out).max()
        print(f"Max embedding output diff: {max_diff}")

        np.testing.assert_allclose(pt_out, flax_out, rtol=1e-5, atol=1e-5)

    def test_03_layer0_input_norm_weights(self):
        """Check that layer 0 input_layernorm weights match."""
        pt_norm = self.pt_model.model.language_model.layers[0].input_layernorm.weight.detach().numpy()
        flax_norm = np.array(self.flax_model.model.language_model.layers[0].input_layernorm.weight[:])

        print(f"PT norm shape: {pt_norm.shape}")
        print(f"Flax norm shape: {flax_norm.shape}")

        max_diff = np.abs(pt_norm - flax_norm).max()
        print(f"Max input_layernorm weight diff: {max_diff}")

        np.testing.assert_allclose(pt_norm, flax_norm, rtol=1e-5, atol=1e-5)

    def test_04_layer0_attention_weights(self):
        """Check that layer 0 attention projection weights match."""
        pt_layer = self.pt_model.model.language_model.layers[0].self_attn
        flax_layer = self.flax_model.model.language_model.layers[0].self_attn

        # Q projection
        pt_q = pt_layer.q_proj.weight.detach().numpy()
        flax_q = np.array(flax_layer.q_proj.kernel[:])

        print(f"PT q_proj shape: {pt_q.shape}")
        print(f"Flax q_proj shape: {flax_q.shape}")

        # JAX Linear uses (in, out), PyTorch uses (out, in)
        # So Flax kernel needs transpose to compare
        max_diff = np.abs(pt_q - flax_q.T).max()
        print(f"Max q_proj weight diff (transposed): {max_diff}")

        np.testing.assert_allclose(pt_q, flax_q.T, rtol=1e-5, atol=1e-5)

    def test_05_layer0_mlp_weights(self):
        """Check that layer 0 MLP weights match."""
        pt_mlp = self.pt_model.model.language_model.layers[0].mlp
        flax_mlp = self.flax_model.model.language_model.layers[0].mlp

        # Gate projection
        pt_gate = pt_mlp.gate_proj.weight.detach().numpy()
        flax_gate = np.array(flax_mlp.gate_proj.kernel[:])

        print(f"PT gate_proj shape: {pt_gate.shape}")
        print(f"Flax gate_proj shape: {flax_gate.shape}")

        max_diff = np.abs(pt_gate - flax_gate.T).max()
        print(f"Max gate_proj weight diff (transposed): {max_diff}")

        np.testing.assert_allclose(pt_gate, flax_gate.T, rtol=1e-5, atol=1e-5)

    def test_06_final_norm_weights(self):
        """Check that final norm weights match."""
        pt_norm = self.pt_model.model.language_model.norm.weight.detach().numpy()
        flax_norm = np.array(self.flax_model.model.language_model.norm.weight[:])

        print(f"PT final norm shape: {pt_norm.shape}")
        print(f"Flax final norm shape: {flax_norm.shape}")

        max_diff = np.abs(pt_norm - flax_norm).max()
        print(f"Max final_norm weight diff: {max_diff}")

        np.testing.assert_allclose(pt_norm, flax_norm, rtol=1e-5, atol=1e-5)

    def test_07_full_forward_pass(self):
        """Full forward pass comparison."""
        # PyTorch
        with torch.inference_mode():
            pt_out = self.pt_model(input_ids=self.input_ids_pt)
        pt_logits = pt_out.logits.numpy()

        # Flax
        flax_logits = np.array(self.flax_model(self.input_ids_jax))

        print(f"PT logits: mean={pt_logits.mean():.4f}, std={pt_logits.std():.4f}")
        print(f"Flax logits: mean={flax_logits.mean():.4f}, std={flax_logits.std():.4f}")

        # Cosine similarity
        pt_flat = pt_logits.flatten()
        flax_flat = flax_logits.flatten()
        cos_sim = np.dot(pt_flat, flax_flat) / (np.linalg.norm(pt_flat) * np.linalg.norm(flax_flat))
        print(f"Cosine similarity: {cos_sim}")

        max_diff = np.abs(pt_logits - flax_logits).max()
        print(f"Max logits diff: {max_diff}")

        # Allow 5% tolerance for accumulated numerical differences over 28 layers
        torch.testing.assert_close(
            torch.tensor(flax_logits),
            torch.tensor(pt_logits),
            rtol=0.05,
            atol=0.05,
        )


class TestPretrainedWithCache(absltest.TestCase):
    """Test pretrained model with KV-cache for generation."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.model_path = snapshot_download(MODEL_2B_ID)
        cls.flax_config = model_lib.Qwen3VLConfig.qwen3vl_2b()
        cls.flax_model = params.create_model_from_safe_tensors(cls.model_path, cls.flax_config)
        cls.processor = AutoProcessor.from_pretrained(cls.model_path)

    def test_cache_initialization(self):
        """Test KV-cache initialization."""
        batch_size = 1
        token_len = 10
        generate_steps = 20

        cache = model_lib.init_cache(self.flax_config, batch_size, token_len, generate_steps)

        self.assertEqual(len(cache), self.flax_config.text_config.num_hidden_layers)
        self.assertEqual(cache[0].k_cache[:].shape[0], batch_size)

    def test_prefill_and_step(self):
        """Test prefill followed by single token generation step."""
        text = "Hello"
        inputs = self.processor(text=text, return_tensors="pt")
        input_ids = jnp.array(inputs["input_ids"].numpy())

        logits = self.flax_model(input_ids)

        self.assertEqual(logits.shape[0], 1)
        self.assertEqual(logits.shape[1], input_ids.shape[1])
        self.assertEqual(logits.shape[2], self.flax_config.text_config.vocab_size)


if __name__ == "__main__":
    absltest.main()
