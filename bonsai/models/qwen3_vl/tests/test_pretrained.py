"""Tests for actual pretrained Qwen3-VL models.

This file tests loading and running actual pretrained models like Qwen3-VL-2B-Instruct.
For numerical equivalence testing with dummy configs, see test_outputs_qwen3vl.py.
"""

import os
import unittest

import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# Import Flax models
from bonsai.models.qwen3_vl import modeling as model_lib
from bonsai.models.qwen3_vl import params


# Model paths - update these to your local paths
MODEL_2B_PATH = "/home/LinuxGod/opensource/qwen3-vl/Qwen3-VL-2B-Instruct"


class TestPretrained2B(absltest.TestCase):
    """Test Qwen3-VL 2B pretrained model."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(MODEL_2B_PATH):
            raise unittest.SkipTest(f"Model not found at {MODEL_2B_PATH}")

        # Load PyTorch model
        cls.pt_model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_2B_PATH,
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        cls.pt_model.eval()

        # Load Flax model
        cls.flax_config = model_lib.Qwen3VLConfig.qwen3vl_2b()
        cls.flax_model = params.create_model_from_safe_tensors(MODEL_2B_PATH, cls.flax_config)

        # Load processor
        cls.processor = AutoProcessor.from_pretrained(MODEL_2B_PATH)

    def test_text_only_forward(self):
        """Test text-only forward pass."""
        # Simple input
        text = "Hello, how are you?"
        inputs = self.processor(text=text, return_tensors="pt")

        # PyTorch forward
        with torch.inference_mode():
            pt_out = self.pt_model(input_ids=inputs["input_ids"])
        pt_logits = pt_out.logits

        # Flax forward
        input_ids = jnp.array(inputs["input_ids"].numpy())
        flax_logits = self.flax_model(input_ids)

        # Compare
        pt_np = pt_logits.numpy()
        flax_np = np.array(flax_logits)

        # Cosine similarity
        pt_flat = pt_np.flatten()
        flax_flat = flax_np.flatten()
        cos_sim = np.dot(pt_flat, flax_flat) / (np.linalg.norm(pt_flat) * np.linalg.norm(flax_flat))

        print(f"Cosine similarity: {cos_sim}")
        print(f"PT logits: mean={pt_np.mean():.4f}, std={pt_np.std():.4f}")
        print(f"Flax logits: mean={flax_np.mean():.4f}, std={flax_np.std():.4f}")

        self.assertGreater(cos_sim, 0.95, f"Cosine similarity {cos_sim} below threshold")

    def test_next_token_prediction(self):
        """Test that next token predictions match."""
        text = "The capital of France is"
        inputs = self.processor(text=text, return_tensors="pt")

        # PyTorch forward
        with torch.inference_mode():
            pt_out = self.pt_model(input_ids=inputs["input_ids"])
        pt_next_token = pt_out.logits[0, -1].argmax().item()

        # Flax forward
        input_ids = jnp.array(inputs["input_ids"].numpy())
        flax_logits = self.flax_model(input_ids)
        flax_next_token = int(jnp.argmax(flax_logits[0, -1]))

        # Decode tokens
        pt_decoded = self.processor.decode([pt_next_token])
        flax_decoded = self.processor.decode([flax_next_token])

        print(f"PT next token: {pt_next_token} -> '{pt_decoded}'")
        print(f"Flax next token: {flax_next_token} -> '{flax_decoded}'")

        self.assertEqual(pt_next_token, flax_next_token, "Next token predictions differ")


class TestPretrainedWithCache(absltest.TestCase):
    """Test pretrained model with KV-cache for generation."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(MODEL_2B_PATH):
            raise unittest.SkipTest(f"Model not found at {MODEL_2B_PATH}")

        cls.flax_config = model_lib.Qwen3VLConfig.qwen3vl_2b()
        cls.flax_model = params.create_model_from_safe_tensors(MODEL_2B_PATH, cls.flax_config)
        cls.processor = AutoProcessor.from_pretrained(MODEL_2B_PATH)

    def test_cache_initialization(self):
        """Test KV-cache initialization."""
        batch_size = 1
        token_len = 10
        generate_steps = 20

        cache = model_lib.init_cache(self.flax_config, batch_size, token_len, generate_steps)

        self.assertEqual(len(cache), self.flax_config.text_config.num_hidden_layers)
        self.assertEqual(cache[0].k_cache.value.shape[0], batch_size)

    def test_prefill_and_step(self):
        """Test prefill followed by single token generation step."""
        text = "Hello"
        inputs = self.processor(text=text, return_tensors="pt")
        input_ids = jnp.array(inputs["input_ids"].numpy())

        # Prefill
        logits = self.flax_model(input_ids)

        self.assertEqual(logits.shape[0], 1)  # batch
        self.assertEqual(logits.shape[1], input_ids.shape[1])  # seq
        self.assertEqual(logits.shape[2], self.flax_config.text_config.vocab_size)


if __name__ == "__main__":
    absltest.main()
