import os

import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from huggingface_hub import constants, snapshot_download
from safetensors.torch import save_model
from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration
from PIL import Image

from bonsai.models.qwen3_vl import modeling as model_lib
from bonsai.models.qwen3_vl import params

# Set highest precision for testing
jax.config.update("jax_default_matmul_precision", "highest")


class TestForwardPass(absltest.TestCase):
    """Test forward pass using small manually-initialized models."""

    def setUp(self):
        super().setUp()
        self.save_dir = constants.default_cache_path
        os.makedirs(self.save_dir, exist_ok=True)

        # Create small PyTorch config for testing
        self.pt_config = Qwen3VLConfig(
            vision_config={
                "depth": 2,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_heads": 4,
                "in_channels": 3,
                "patch_size": 8,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "out_hidden_size": 128,
                "num_position_embeddings": 256,
                "deepstack_visual_indexes": [0, 1],
                "hidden_act": "gelu_pytorch_tanh",
            },
            text_config={
                "vocab_size": 1000,
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 32,
                "hidden_act": "silu",
                "rms_norm_eps": 1e-6,
                "rope_theta": 5_000_000,
                "rope_scaling": {
                    "mrope_interleaved": True,
                    "mrope_section": [12, 10, 10],
                    "rope_type": "default",
                },
                "attention_bias": False,
                "tie_word_embeddings": True,
            },
            image_token_id=151655,
            video_token_id=151656,
        )

        # Create PyTorch model and save weights
        self.pt_model = Qwen3VLForConditionalGeneration(config=self.pt_config)
        self.model_ckpt_path = os.path.join(self.save_dir, "qwen3vl_test.safetensors")
        save_model(self.pt_model, self.model_ckpt_path)

        # Create Flax config and load model
        self.flax_config = model_lib.Qwen3VLConfig.standard_test()
        self.flax_model = params.create_model_from_safe_tensors(
            self.save_dir, self.flax_config, model_filename="qwen3vl_test.safetensors"
        )

        if os.path.exists(self.model_ckpt_path):
            os.remove(self.model_ckpt_path)
        self.pt_model.eval()
        self.batch_size = 1
        self.seq_len = 10

    def test_text_embeddings(self):
        """Compare embedding layer output."""
        pt_embed = self.pt_model.model.language_model.embed_tokens
        flax_embed = self.flax_model.model.language_model.embed_tokens

        key = jax.random.PRNGKey(0)
        input_ids = jax.random.randint(key, (self.batch_size, self.seq_len), 0, 100)
        pt_ids = torch.tensor(np.asarray(input_ids), dtype=torch.long)

        with torch.inference_mode():
            pt_out = pt_embed(pt_ids)
        flax_out = flax_embed(input_ids)

        np.testing.assert_allclose(np.asarray(flax_out), pt_out.numpy(), rtol=1e-4, atol=1e-4)

    def test_rms_norm(self):
        """Compare RMSNorm layer."""
        pt_norm = self.pt_model.model.language_model.norm
        flax_norm = self.flax_model.model.language_model.norm

        key = jax.random.PRNGKey(0)
        hidden_size = self.flax_config.text_config.hidden_size
        jx = jax.random.normal(key, (self.batch_size, self.seq_len, hidden_size), dtype=jnp.float32)
        tx = torch.tensor(np.asarray(jx), dtype=torch.float32)

        with torch.inference_mode():
            pt_out = pt_norm(tx)
        flax_out = flax_norm(jx)

        np.testing.assert_allclose(np.asarray(flax_out), pt_out.numpy(), rtol=1e-4, atol=1e-4)

    def test_text_mlp(self):
        """Compare gated MLP layer."""
        pt_mlp = self.pt_model.model.language_model.layers[0].mlp
        flax_mlp = self.flax_model.model.language_model.layers[0].mlp

        key = jax.random.PRNGKey(0)
        hidden_size = self.flax_config.text_config.hidden_size
        jx = jax.random.normal(key, (self.batch_size, self.seq_len, hidden_size), dtype=jnp.float32)
        tx = torch.tensor(np.asarray(jx), dtype=torch.float32)

        with torch.inference_mode():
            pt_out = pt_mlp(tx)
        flax_out = flax_mlp(jx)

        np.testing.assert_allclose(np.asarray(flax_out), pt_out.numpy(), rtol=1e-3, atol=1e-3)


class TestVisionComponentsEquivalence(absltest.TestCase):
    """Test vision encoder components with PyTorch comparison."""

    def setUp(self):
        super().setUp()
        self.save_dir = constants.default_cache_path
        os.makedirs(self.save_dir, exist_ok=True)

        # Create small PyTorch config
        self.pt_config = Qwen3VLConfig(
            vision_config={
                "depth": 2,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_heads": 4,
                "in_channels": 3,
                "patch_size": 8,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "out_hidden_size": 128,
                "num_position_embeddings": 256,
                "deepstack_visual_indexes": [0, 1],
                "hidden_act": "gelu_pytorch_tanh",
            },
            text_config={
                "vocab_size": 1000,
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 32,
                "tie_word_embeddings": True,
                "rope_theta": 5_000_000,
                "rope_scaling": {
                    "mrope_interleaved": True,
                    "mrope_section": [12, 10, 10],
                    "rope_type": "default",
                },
            },
        )

        self.pt_model = Qwen3VLForConditionalGeneration(config=self.pt_config)
        self.model_ckpt_path = os.path.join(self.save_dir, "qwen3vl_vision_test.safetensors")
        save_model(self.pt_model, self.model_ckpt_path)

        self.flax_config = model_lib.Qwen3VLConfig.standard_test()
        self.flax_model = params.create_model_from_safe_tensors(
            self.save_dir, self.flax_config, model_filename="qwen3vl_vision_test.safetensors"
        )

        if os.path.exists(self.model_ckpt_path):
            os.remove(self.model_ckpt_path)
        self.pt_model.eval()

    def test_vision_mlp(self):
        """Compare vision MLP output."""
        pt_mlp = self.pt_model.model.visual.blocks[0].mlp
        flax_mlp = self.flax_model.model.visual.blocks[0].mlp

        hidden_size = 64  # from config
        key = jax.random.PRNGKey(0)
        jx = jax.random.normal(key, (16, hidden_size), dtype=jnp.float32)
        tx = torch.tensor(np.asarray(jx), dtype=torch.float32)

        with torch.inference_mode():
            pt_out = pt_mlp(tx)
        flax_out = flax_mlp(jx)

        np.testing.assert_allclose(np.asarray(flax_out), pt_out.numpy(), rtol=1e-3, atol=1e-3)

    def test_vision_layernorm(self):
        """Compare vision LayerNorm output."""
        pt_norm = self.pt_model.model.visual.blocks[0].norm1
        flax_norm = self.flax_model.model.visual.blocks[0].norm1

        hidden_size = 64
        key = jax.random.PRNGKey(0)
        jx = jax.random.normal(key, (16, hidden_size), dtype=jnp.float32)
        tx = torch.tensor(np.asarray(jx), dtype=torch.float32)

        with torch.inference_mode():
            pt_out = pt_norm(tx)
        flax_out = flax_norm(jx)

        np.testing.assert_allclose(np.asarray(flax_out), pt_out.numpy(), rtol=1e-4, atol=1e-4)


class TestKVCache(absltest.TestCase):
    """Test KV-cache functionality."""

    def test_cache_initialization(self):
        """Test cache is initialized with correct shapes."""
        config = model_lib.Qwen3VLConfig.standard_test()
        cache = model_lib.init_cache(config, batch_size=2, token_len=10, generate_steps=5)

        self.assertEqual(len(cache), config.text_config.num_hidden_layers)

        for layer_cache in cache:
            k_shape = layer_cache.k_cache[:].shape
            self.assertEqual(k_shape[0], 2)  # batch
            self.assertEqual(k_shape[2], config.text_config.num_key_value_heads)
            self.assertEqual(k_shape[3], config.text_config.head_dim)


class TestPretrained2B(absltest.TestCase):
    """Test pretrained Qwen3-VL-2B model."""

    MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model_path = snapshot_download(cls.MODEL_ID)

        cls.pt_model = Qwen3VLForConditionalGeneration.from_pretrained(cls.model_path, dtype=torch.float32).eval()

        cls.flax_config = model_lib.Qwen3VLConfig.qwen3vl_2b()
        cls.flax_model = params.create_model_from_safe_tensors(cls.model_path, cls.flax_config)

        cls.processor = AutoProcessor.from_pretrained(cls.model_path)

    def test_embedding_output(self):
        """Check embedding outputs match."""
        text = "Hello"
        inputs = self.processor(text=text, return_tensors="pt")
        input_ids_pt = inputs["input_ids"]
        input_ids_jax = jnp.array(inputs["input_ids"].numpy())

        with torch.inference_mode():
            pt_out = self.pt_model.model.language_model.embed_tokens(input_ids_pt).numpy()

        flax_out = np.array(self.flax_model.model.language_model.embed_tokens(input_ids_jax))

        np.testing.assert_allclose(flax_out, pt_out, rtol=1e-6, atol=1e-6)

    def test_text_forward_pass(self):
        """Full text forward pass comparison."""
        text = "Hello"
        inputs = self.processor(text=text, return_tensors="pt")
        input_ids_jax = jnp.array(inputs["input_ids"].numpy())

        with torch.inference_mode():
            pt_logits = self.pt_model(input_ids=inputs["input_ids"]).logits.numpy()

        batch, seq_len = input_ids_jax.shape
        cache = model_lib.init_cache(self.flax_config, batch, seq_len, generate_steps=10)
        flax_logits = np.array(self.flax_model(input_ids_jax, cache))

        np.testing.assert_allclose(flax_logits, pt_logits, rtol=1e-5, atol=1e-2)

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

        self.assertEqual(int(cache[0].cur_ind.get_value()), 5)

        # Decode step
        next_token = jnp.argmax(logits, axis=-1, keepdims=True)
        logits2, cache = model_lib.forward(self.flax_model, cache, next_token)

        self.assertEqual(int(cache[0].cur_ind.get_value()), 6)
        self.assertEqual(logits2.shape, (1, self.flax_config.text_config.vocab_size))


class TestVisionEncoderPretrained(absltest.TestCase):
    """Test vision encoder with pretrained weights."""

    MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model_path = snapshot_download(cls.MODEL_ID)

        cls.pt_model = Qwen3VLForConditionalGeneration.from_pretrained(cls.model_path, dtype=torch.float32).eval()

        cls.flax_config = model_lib.Qwen3VLConfig.qwen3vl_2b()
        cls.flax_model = params.create_model_from_safe_tensors(cls.model_path, cls.flax_config)

        cls.processor = AutoProcessor.from_pretrained(cls.model_path)

    def _create_dummy_image_input(self):
        """Create dummy image input for testing."""
        image = Image.new("RGB", (256, 256), color=(128, 128, 128))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "What?"},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )

    def test_patch_embed_output(self):
        """Check patch embedding output matches."""
        inputs = self._create_dummy_image_input()
        pixel_values_pt = inputs["pixel_values"]
        pixel_values_jax = jnp.array(pixel_values_pt.numpy())

        with torch.inference_mode():
            pt_out = self.pt_model.model.visual.patch_embed(pixel_values_pt).numpy()

        flax_out = np.array(self.flax_model.model.visual.patch_embed(pixel_values_jax))

        np.testing.assert_allclose(flax_out, pt_out, rtol=1e-6, atol=1e-6)

    def test_position_embedding_output(self):
        """Check position embedding interpolation output matches."""
        inputs = self._create_dummy_image_input()
        grid_thw_pt = inputs["image_grid_thw"]
        grid_thw_jax = jnp.array(grid_thw_pt.numpy())

        with torch.inference_mode():
            pt_pos = self.pt_model.model.visual.fast_pos_embed_interpolate(grid_thw_pt).numpy()

        flax_pos = np.array(self.flax_model.model.visual._fast_pos_embed_interpolate(grid_thw_jax))

        np.testing.assert_allclose(flax_pos, pt_pos, rtol=1e-5, atol=3e-5)

    def test_rope_embedding(self):
        """Check RoPE embedding output matches."""
        inputs = self._create_dummy_image_input()
        grid_thw_pt = inputs["image_grid_thw"]
        grid_thw_jax = jnp.array(grid_thw_pt.numpy())

        with torch.inference_mode():
            pt_rope = self.pt_model.model.visual.rot_pos_emb(grid_thw_pt)
            pt_emb = torch.cat([pt_rope, pt_rope], dim=-1)
            pt_cos = pt_emb.cos().numpy()
            pt_sin = pt_emb.sin().numpy()

        flax_cos, flax_sin = self.flax_model.model.visual._rot_pos_emb(grid_thw_jax)

        np.testing.assert_allclose(np.array(flax_cos), pt_cos, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(np.array(flax_sin), pt_sin, rtol=1e-6, atol=1e-6)

    def test_full_vision_output(self):
        """Check full vision encoder output matches."""
        inputs = self._create_dummy_image_input()
        pixel_values_pt = inputs["pixel_values"]
        grid_thw_pt = inputs["image_grid_thw"]
        pixel_values_jax = jnp.array(pixel_values_pt.numpy())
        grid_thw_jax = jnp.array(grid_thw_pt.numpy())

        with torch.inference_mode():
            pt_out = self.pt_model.model.visual(pixel_values_pt, grid_thw_pt)[0].numpy()

        flax_out, _ = self.flax_model.model.visual(pixel_values_jax, grid_thw_jax)

        np.testing.assert_allclose(np.array(flax_out), pt_out, rtol=1e-5, atol=2e-3)


if __name__ == "__main__":
    absltest.main()
