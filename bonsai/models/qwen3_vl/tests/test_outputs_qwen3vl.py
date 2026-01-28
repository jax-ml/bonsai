import os
import shutil
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from huggingface_hub import snapshot_download
from safetensors.torch import save_model
from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration
from PIL import Image

from bonsai.models.qwen3_vl import modeling as model_lib
from bonsai.models.qwen3_vl import params

# Set highest precision for testing
jax.config.update("jax_default_matmul_precision", "highest")


def get_test_config_torch() -> Qwen3VLConfig:
    return Qwen3VLConfig(
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


def get_test_config_flax() -> model_lib.Qwen3VLConfig:
    return model_lib.Qwen3VLConfig(
        vision_config=model_lib.Qwen3VLVisionConfig(
            depth=2,
            hidden_size=64,
            intermediate_size=128,
            num_heads=4,
            in_channels=3,
            patch_size=8,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=128,
            num_position_embeddings=256,
            deepstack_visual_indexes=(0, 1),
        ),
        text_config=model_lib.Qwen3VLTextConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            mrope_section=(12, 10, 10),
            tie_word_embeddings=True,
        ),
    )


class TestForwardPass(absltest.TestCase):
    """Test forward pass using small manually-initialized models."""

    def setUp(self):
        super().setUp()
        self.test_dir = tempfile.mkdtemp()

        self.pt_config = get_test_config_torch()
        self.pt_model = Qwen3VLForConditionalGeneration(config=self.pt_config)
        self.model_filename = "qwen3vl_test.safetensors"
        self.model_ckpt_path = os.path.join(self.test_dir, self.model_filename)
        save_model(self.pt_model, self.model_ckpt_path)

        self.flax_config = get_test_config_flax()
        self.flax_model = params.create_model_from_safe_tensors(
            self.test_dir, self.flax_config, model_filename=self.model_filename
        )

        self.pt_model.eval()
        self.batch_size = 1
        self.seq_len = 10

    def tearDown(self):
        # Recursively remove the temporary directory and all its contents
        shutil.rmtree(self.test_dir)
        super().tearDown()

    def test_text_embeddings(self):
        """Compare embedding layer output."""
        pt_embed = self.pt_model.model.language_model.embed_tokens
        flax_embed = self.flax_model.model.language_model.embed_tokens

        key = jax.random.PRNGKey(0)
        input_ids = jax.random.randint(key, (self.batch_size, self.seq_len), 0, 100)
        pt_ids = torch.tensor(np.asarray(input_ids), dtype=torch.long)

        with torch.inference_mode():
            pt_out = pt_embed(pt_ids)
        flax_out = flax_embed(input_ids, out_sharding=model_lib.P(None, None, None))

        np.testing.assert_allclose(np.asarray(flax_out), pt_out.numpy(), rtol=1e-7, atol=1e-7)

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

        np.testing.assert_allclose(np.asarray(flax_out), pt_out.numpy(), rtol=1e-6, atol=1e-6)

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
        # Reshape to (batch * seq, hidden) for ShardedLinear, then reshape back
        jx_flat = jx.reshape(-1, hidden_size)
        flax_out_flat = flax_mlp(jx_flat)
        flax_out = flax_out_flat.reshape(self.batch_size, self.seq_len, -1)

        np.testing.assert_allclose(np.asarray(flax_out), pt_out.numpy(), rtol=1e-7, atol=1e-7)


class TestVisionComponentsEquivalence(absltest.TestCase):
    """Test vision encoder components with PyTorch comparison."""

    def setUp(self):
        super().setUp()
        self.test_dir = tempfile.mkdtemp()

        self.pt_config = get_test_config_torch()
        self.pt_model = Qwen3VLForConditionalGeneration(config=self.pt_config)
        self.model_filename = "qwen3vl_vision_test.safetensors"
        self.model_ckpt_path = os.path.join(self.test_dir, self.model_filename)
        save_model(self.pt_model, self.model_ckpt_path)

        self.flax_config = get_test_config_flax()
        self.flax_model = params.create_model_from_safe_tensors(
            self.test_dir, self.flax_config, model_filename=self.model_filename
        )

        self.pt_model.eval()

    def tearDown(self):
        # Recursively remove the temporary directory and all its contents
        shutil.rmtree(self.test_dir)
        super().tearDown()

    def test_vision_mlp(self):
        """Compare vision MLP output."""
        pt_mlp = self.pt_model.model.visual.blocks[0].mlp
        flax_mlp = self.flax_model.model.visual.blocks[0].mlp

        hidden_size = 64
        key = jax.random.PRNGKey(0)
        jx = jax.random.normal(key, (16, hidden_size), dtype=jnp.float32)
        tx = torch.tensor(np.asarray(jx), dtype=torch.float32)

        with torch.inference_mode():
            pt_out = pt_mlp(tx)
        flax_out = flax_mlp(jx)

        np.testing.assert_allclose(np.asarray(flax_out), pt_out.numpy(), rtol=1e-5, atol=1e-5)

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

        np.testing.assert_allclose(np.asarray(flax_out), pt_out.numpy(), rtol=1e-6, atol=1e-6)

    def test_vision_attention_qkv(self):
        """Compare vision attention QKV projections."""
        pt_attn = self.pt_model.model.visual.blocks[0].attn
        flax_attn = self.flax_model.model.visual.blocks[0].attn

        hidden_size = 64
        seq_len = 16
        key = jax.random.PRNGKey(0)
        jx = jax.random.normal(key, (seq_len, hidden_size), dtype=jnp.float32)
        tx = torch.tensor(np.asarray(jx), dtype=torch.float32)

        # Compare QKV projection output
        with torch.inference_mode():
            pt_qkv = pt_attn.qkv(tx).numpy()
        flax_qkv = np.array(flax_attn.qkv(jx, out_sharding=model_lib.P(None, None)))

        np.testing.assert_allclose(flax_qkv, pt_qkv, rtol=1e-6, atol=1e-6)

    def test_vision_block_components(self):
        """Compare vision block layer norm and MLP separately."""
        pt_block = self.pt_model.model.visual.blocks[0]
        flax_block = self.flax_model.model.visual.blocks[0]

        hidden_size = 64
        seq_len = 16
        key = jax.random.PRNGKey(42)
        jx = jax.random.normal(key, (seq_len, hidden_size), dtype=jnp.float32)
        tx = torch.tensor(np.asarray(jx), dtype=torch.float32)

        # Test norm1
        with torch.inference_mode():
            pt_norm1 = pt_block.norm1(tx).numpy()
        flax_norm1 = np.array(flax_block.norm1(jx))
        np.testing.assert_allclose(flax_norm1, pt_norm1, rtol=1e-6, atol=1e-6)

        # Test norm2
        with torch.inference_mode():
            pt_norm2 = pt_block.norm2(tx).numpy()
        flax_norm2 = np.array(flax_block.norm2(jx))
        np.testing.assert_allclose(flax_norm2, pt_norm2, rtol=1e-6, atol=1e-6)

        # Test MLP
        with torch.inference_mode():
            pt_mlp = pt_block.mlp(tx).numpy()
        flax_mlp = np.array(flax_block.mlp(jx))
        np.testing.assert_allclose(flax_mlp, pt_mlp, rtol=1e-5, atol=1e-5)

    def test_vision_attention_with_rope(self):
        """Compare vision attention output with RoPE applied."""
        pt_attn = self.pt_model.model.visual.blocks[0].attn
        flax_attn = self.flax_model.model.visual.blocks[0].attn

        hidden_size = 64
        seq_len = 16
        head_dim = hidden_size // 4  # 4 heads

        key = jax.random.PRNGKey(42)
        jx = jax.random.normal(key, (seq_len, hidden_size), dtype=jnp.float32)
        tx = torch.tensor(np.asarray(jx), dtype=torch.float32)

        # Create position embeddings (cos, sin) for RoPE
        # Using same formula as vision encoder
        rotary_dim = head_dim // 2
        inv_freq = 1.0 / (10000.0 ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim))
        positions = np.arange(seq_len, dtype=np.float32)
        freqs = np.outer(positions, inv_freq)  # (seq, rotary_dim//2)
        emb = np.concatenate([freqs, freqs], axis=-1)  # (seq, rotary_dim)
        emb = np.concatenate([emb, emb], axis=-1)  # (seq, head_dim)
        cos_np, sin_np = np.cos(emb), np.sin(emb)

        cos_jax = jnp.array(cos_np)
        sin_jax = jnp.array(sin_np)
        cos_pt = torch.tensor(cos_np)
        sin_pt = torch.tensor(sin_np)

        with torch.inference_mode():
            cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32)
            pt_out = pt_attn(tx, cu_seqlens=cu_seqlens, position_embeddings=(cos_pt, sin_pt)).numpy()
        flax_out = np.array(flax_attn(jx, position_embeddings=(cos_jax, sin_jax)))

        np.testing.assert_allclose(flax_out, pt_out, rtol=1e-4, atol=1e-4)

    def test_patch_embed(self):
        """Compare patch embedding output with dummy config."""
        pt_patch_embed = self.pt_model.model.visual.patch_embed
        flax_patch_embed = self.flax_model.model.visual.patch_embed

        # Create input matching the expected format
        # patch_size=8, temporal_patch_size=2, so input per patch is: 3 * 2 * 8 * 8 = 384
        cfg = self.flax_config.vision_config
        per_patch_size = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
        num_patches = 16  # Arbitrary number of patches

        key = jax.random.PRNGKey(0)
        jx = jax.random.normal(key, (num_patches, per_patch_size), dtype=jnp.float32)
        tx = torch.tensor(np.asarray(jx), dtype=torch.float32)

        with torch.inference_mode():
            pt_out = pt_patch_embed(tx).numpy()
        flax_out = np.array(flax_patch_embed(jx))

        np.testing.assert_allclose(flax_out, pt_out, rtol=1e-5, atol=1e-5)

    def test_rope_embedding(self):
        """Compare RoPE embedding computation with dummy config."""
        pt_visual = self.pt_model.model.visual
        flax_visual = self.flax_model.model.visual

        # Create grid_thw for a small image: 1 frame, 16x16 grid
        grid_thw_np = np.array([[1, 16, 16]], dtype=np.int64)
        grid_thw_pt = torch.tensor(grid_thw_np)
        grid_thw_jax = jnp.array(grid_thw_np)

        with torch.inference_mode():
            pt_rope = pt_visual.rot_pos_emb(grid_thw_pt)
            pt_emb = torch.cat([pt_rope, pt_rope], dim=-1)
            pt_cos = pt_emb.cos().numpy()
            pt_sin = pt_emb.sin().numpy()

        flax_cos, flax_sin = flax_visual._rot_pos_emb(grid_thw_jax)

        np.testing.assert_allclose(np.array(flax_cos), pt_cos, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.array(flax_sin), pt_sin, rtol=1e-5, atol=1e-5)

    def test_full_vision_encoder(self):
        """Compare full vision encoder output with dummy config."""
        pt_visual = self.pt_model.model.visual
        flax_visual = self.flax_model.model.visual

        # Create dummy vision input
        # grid_thw: 1 frame, 16x16 grid (spatial_merge_size=2, so 8x8 merged patches = 64)
        cfg = self.flax_config.vision_config
        per_patch_size = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
        grid_t, grid_h, grid_w = 1, 16, 16
        num_patches = grid_t * grid_h * grid_w  # 256 patches before merge

        grid_thw_np = np.array([[grid_t, grid_h, grid_w]], dtype=np.int64)
        grid_thw_pt = torch.tensor(grid_thw_np)
        grid_thw_jax = jnp.array(grid_thw_np)

        key = jax.random.PRNGKey(42)
        jx = jax.random.normal(key, (num_patches, per_patch_size), dtype=jnp.float32)
        tx = torch.tensor(np.asarray(jx), dtype=torch.float32)

        with torch.inference_mode():
            pt_result = pt_visual(tx, grid_thw_pt)
            # Handle both tuple and BaseModelOutputWithDeepstackFeatures
            # HF now returns pre-merger output in last_hidden_state, so we apply merger manually
            if hasattr(pt_result, "last_hidden_state"):
                pt_hidden = pt_result.last_hidden_state
                # Apply merger to get merged output matching Flax
                pt_out = pt_visual.merger(pt_hidden).numpy()
            else:
                pt_out = pt_result[0].numpy()

        flax_out, flax_deepstack = flax_visual(jx, grid_thw_jax)
        flax_out = np.array(flax_out)

        # After spatial merge (2x2), 256 patches become 64
        expected_merged_patches = num_patches // (cfg.spatial_merge_size**2)
        self.assertEqual(flax_out.shape[0], expected_merged_patches)
        self.assertEqual(pt_out.shape[0], expected_merged_patches)
        self.assertEqual(flax_out.shape[1], cfg.out_hidden_size)
        self.assertEqual(pt_out.shape[1], cfg.out_hidden_size)

        np.testing.assert_allclose(flax_out, pt_out, rtol=1e-3, atol=5e-3)


class TestKVCache(absltest.TestCase):
    """Test KV-cache functionality."""

    def test_cache_initialization(self):
        """Test cache is initialized with correct shapes."""
        config = get_test_config_flax()
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
        flax_out = np.array(
            self.flax_model.model.language_model.embed_tokens(input_ids_jax, out_sharding=model_lib.P(None, None, None))
        )

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

    def test_text_decoder_layer_output(self):
        """Compare text decoder layer 0 output."""
        text = "Hello world"
        inputs = self.processor(text=text, return_tensors="pt")
        input_ids_pt = inputs["input_ids"]
        input_ids_jax = jnp.array(input_ids_pt.numpy())

        # Get hidden states after layer 0 using hooks
        pt_hidden_after_layer0 = []

        def hook(module, input, output):
            pt_hidden_after_layer0.append(output[0].detach())

        handle = self.pt_model.model.language_model.layers[0].register_forward_hook(hook)

        with torch.inference_mode():
            _ = self.pt_model(input_ids=input_ids_pt)

        handle.remove()
        pt_layer0_out = pt_hidden_after_layer0[0].numpy()

        # Flax: run through embed + layer 0 manually
        batch, seq_len = input_ids_jax.shape
        cache = model_lib.init_cache(self.flax_config, batch, seq_len, 10)

        flax_embeds = self.flax_model.model.language_model.embed_tokens(
            input_ids_jax, out_sharding=model_lib.P(None, None, None)
        )

        # Generate position embeddings
        positions = jnp.arange(seq_len)[None, :]
        sin, cos = model_lib._generate_rope(
            positions, self.flax_config.text_config.head_dim, self.flax_config.text_config.rope_theta
        )

        # Run through layer 0
        mask = model_lib.make_causal_mask(cache[0], seq_len)
        flax_layer0_out = self.flax_model.model.language_model.layers[0](flax_embeds, cache[0], sin, cos, mask)

        np.testing.assert_allclose(np.array(flax_layer0_out.squeeze(0)), pt_layer0_out, rtol=1e-4, atol=5e-3)

    def test_jit_forward(self):
        """Test JIT-compiled forward."""
        text = "Hello"
        inputs = self.processor(text=text, return_tensors="pt")
        input_ids = jnp.array(inputs["input_ids"].numpy())

        cache = model_lib.init_cache(self.flax_config, 1, input_ids.shape[1], 20)
        logits, _ = model_lib.forward(self.flax_model, cache, input_ids)

        self.assertEqual(logits.shape, (1, self.flax_config.text_config.vocab_size))

    def test_generation_step_with_numeric_check(self):
        """Test generation step with numeric output verification."""
        # Use processor to get proper token ids
        text = "Hello"
        inputs = self.processor(text=text, return_tensors="pt")
        input_ids_pt = inputs["input_ids"]
        input_ids_jax = jnp.array(input_ids_pt.numpy())

        seq_len = input_ids_jax.shape[1]
        cache = model_lib.init_cache(self.flax_config, 1, seq_len, 20)

        # Prefill - compare with PyTorch
        with torch.inference_mode():
            pt_logits = self.pt_model(input_ids=input_ids_pt).logits[:, -1, :].numpy()

        flax_logits, cache = model_lib.forward(self.flax_model, cache, input_ids_jax)
        flax_logits_np = np.array(flax_logits)

        # Verify numeric equivalence
        np.testing.assert_allclose(flax_logits_np, pt_logits, rtol=1e-4, atol=1e-2)

        # Verify cache position
        self.assertEqual(int(cache[0].cur_ind.get_value()), seq_len)

        # Decode step
        next_token_flax = jnp.argmax(flax_logits, axis=-1, keepdims=True)
        next_token_pt = torch.tensor(np.array(next_token_flax), dtype=torch.long)

        flax_logits2, cache = model_lib.forward(self.flax_model, cache, next_token_flax)

        # Verify cache position updated
        self.assertEqual(int(cache[0].cur_ind.get_value()), seq_len + 1)

        # Verify logits are valid (not NaN/Inf)
        self.assertFalse(np.any(np.isnan(np.array(flax_logits2))))
        self.assertFalse(np.any(np.isinf(np.array(flax_logits2))))


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

        np.testing.assert_allclose(flax_out, pt_out, rtol=1e-5, atol=1e-5)

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

        np.testing.assert_allclose(np.array(flax_cos), pt_cos, rtol=1e-6, atol=2e-5)
        np.testing.assert_allclose(np.array(flax_sin), pt_sin, rtol=1e-6, atol=2e-5)

    def test_vision_patch_plus_pos_output(self):
        """Check patch + position embedding output matches."""
        inputs = self._create_dummy_image_input()
        pixel_values_pt = inputs["pixel_values"]
        grid_thw_pt = inputs["image_grid_thw"]
        pixel_values_jax = jnp.array(pixel_values_pt.numpy())
        grid_thw_jax = jnp.array(grid_thw_pt.numpy())

        # Get patch + pos embeddings
        with torch.inference_mode():
            pt_patches = self.pt_model.model.visual.patch_embed(pixel_values_pt)
            pt_pos = self.pt_model.model.visual.fast_pos_embed_interpolate(grid_thw_pt)
            pt_hidden = (pt_patches + pt_pos).numpy()

        flax_patches = self.flax_model.model.visual.patch_embed(pixel_values_jax)
        flax_pos = self.flax_model.model.visual._fast_pos_embed_interpolate(grid_thw_jax)
        flax_hidden = np.array(flax_patches + flax_pos)

        np.testing.assert_allclose(flax_hidden, pt_hidden, rtol=1e-5, atol=1e-5)

    def test_full_vision_output(self):
        """Check full vision encoder output matches."""
        inputs = self._create_dummy_image_input()
        pixel_values_pt = inputs["pixel_values"]
        grid_thw_pt = inputs["image_grid_thw"]
        pixel_values_jax = jnp.array(pixel_values_pt.numpy())
        grid_thw_jax = jnp.array(grid_thw_pt.numpy())

        with torch.inference_mode():
            pt_result = self.pt_model.model.visual(pixel_values_pt, grid_thw_pt)
            # Handle both tuple and BaseModelOutputWithDeepstackFeatures
            # HF now returns pre-merger output in last_hidden_state, so we apply merger manually
            if hasattr(pt_result, "last_hidden_state"):
                pt_hidden = pt_result.last_hidden_state
                pt_out = self.pt_model.model.visual.merger(pt_hidden).numpy()
            else:
                pt_out = pt_result[0].numpy()
        flax_out, _ = self.flax_model.model.visual(pixel_values_jax, grid_thw_jax)

        self.assertEqual(
            np.array(flax_out).shape,
            pt_out.shape,
            f"Shape mismatch: Flax {np.array(flax_out).shape} vs PT {pt_out.shape}.",
        )
        np.testing.assert_allclose(np.array(flax_out), pt_out, rtol=1e-4, atol=5e-3)


class TestVisionTextGeneration(absltest.TestCase):
    """Test vision + text generation with pretrained model."""

    MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model_path = snapshot_download(cls.MODEL_ID)
        cls.pt_model = Qwen3VLForConditionalGeneration.from_pretrained(cls.model_path, dtype=torch.float32).eval()
        cls.flax_config = model_lib.Qwen3VLConfig.qwen3vl_2b()
        cls.flax_model = params.create_model_from_safe_tensors(cls.model_path, cls.flax_config)
        cls.processor = AutoProcessor.from_pretrained(cls.model_path)

    def test_vision_forward_with_numeric_check(self):
        """Test vision forward pass with numeric verification."""
        image = Image.new("RGB", (256, 256), color=(100, 150, 200))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe"},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )

        pixel_values_pt = inputs["pixel_values"]
        grid_thw_pt = inputs["image_grid_thw"]
        input_ids_pt = inputs["input_ids"]

        pixel_values_jax = jnp.array(pixel_values_pt.numpy())
        grid_thw_jax = jnp.array(grid_thw_pt.numpy())
        input_ids_jax = jnp.array(input_ids_pt.numpy())

        # Create token_type_ids: 1 for image tokens, 0 for text
        image_token_id = self.flax_config.image_token_id
        token_type_ids_jax = (input_ids_jax == image_token_id).astype(jnp.int32)

        # PyTorch forward
        with torch.inference_mode():
            pt_out = self.pt_model(
                input_ids=input_ids_pt,
                pixel_values=pixel_values_pt,
                image_grid_thw=grid_thw_pt,
            )
            pt_logits = pt_out.logits[:, -1, :].numpy()

        # Flax forward
        batch, seq_len = input_ids_jax.shape
        cache = model_lib.init_cache(self.flax_config, batch, seq_len, 20)

        flax_logits, cache = model_lib.forward_vision(
            self.flax_model, cache, input_ids_jax, pixel_values_jax, grid_thw_jax, token_type_ids_jax
        )

        # Vision+text forward has larger tolerance due to accumulated numerical diffs
        # Top logits should still match for generation correctness
        flax_logits_np = np.array(flax_logits)

        # Check top-k token predictions match
        pt_top5 = np.argsort(pt_logits[0])[-5:]
        flax_top5 = np.argsort(flax_logits_np[0])[-5:]
        overlap = len(set(pt_top5) & set(flax_top5))
        self.assertTrue(overlap >= 4, f"Out of Top-5 tokens, top 4 must overlap, overlap: {overlap}")

        # Check overall correlation
        corr = np.corrcoef(pt_logits.flatten(), flax_logits_np.flatten())[0, 1]
        self.assertGreater(corr, 0.95, f"Logits should be highly correlated, got {corr}")

    def test_generation_with_vision_input(self):
        """Test generation step with vision input."""
        image = Image.new("RGB", (256, 256), color=(50, 100, 150))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "What is this?"},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )

        pixel_values_jax = jnp.array(inputs["pixel_values"].numpy())
        grid_thw_jax = jnp.array(inputs["image_grid_thw"].numpy())
        input_ids_jax = jnp.array(inputs["input_ids"].numpy())

        # Create token_type_ids: 1 for image tokens, 0 for text
        image_token_id = self.flax_config.image_token_id
        token_type_ids_jax = (input_ids_jax == image_token_id).astype(jnp.int32)

        batch, seq_len = input_ids_jax.shape
        cache = model_lib.init_cache(self.flax_config, batch, seq_len, 30)

        # Prefill with vision
        logits, cache = model_lib.forward_vision(
            self.flax_model, cache, input_ids_jax, pixel_values_jax, grid_thw_jax, token_type_ids_jax
        )

        # Verify cache position
        self.assertEqual(int(cache[0].cur_ind.get_value()), seq_len)

        # Generate a few tokens
        for i in range(3):
            next_token = jnp.argmax(logits, axis=-1, keepdims=True)
            logits, cache = model_lib.forward(self.flax_model, cache, next_token)

            # Verify logits are valid
            self.assertFalse(np.any(np.isnan(np.array(logits))))
            self.assertFalse(np.any(np.isinf(np.array(logits))))

            # Verify cache position updates
            self.assertEqual(int(cache[0].cur_ind.get_value()), seq_len + i + 1)


if __name__ == "__main__":
    absltest.main()
