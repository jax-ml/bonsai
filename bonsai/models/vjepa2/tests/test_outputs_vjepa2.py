import os

import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from huggingface_hub import constants
from safetensors.torch import save_file
from transformers import VJEPA2Config, VJEPA2Model

from bonsai.models.vjepa2 import modeling as model_lib
from bonsai.models.vjepa2 import params


class TestForwardPass(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.save_dir = constants.default_cache_path
        os.makedirs(self.save_dir, exist_ok=True)

        # Small config for testing - matching standard_test
        self.hfconfig = VJEPA2Config(
            crop_size=64,
            frames_per_clip=4,
            hidden_size=64,
            num_attention_heads=2,
            num_hidden_layers=2,
            mlp_ratio=4.0,
            pred_hidden_size=32,
            pred_num_attention_heads=2,
            pred_num_hidden_layers=2,
            patch_size=16,
            tubelet_size=2,
            in_chans=3,
            num_pooler_layers=1,
        )
        self.baseline_model = VJEPA2Model(config=self.hfconfig)
        self.model_ckpt_path = os.path.join(self.save_dir, "model.safetensors")
        save_file(self.baseline_model.state_dict(), self.model_ckpt_path)

        self.config = model_lib.VJEPA2FlaxConfig.standard_test()
        self.bonsai_model = params.create_model_from_safe_tensors(self.save_dir, self.config, classifier=False)

        self.bonsai_model.eval()
        self.baseline_model.eval()

        self.batch_size = 1
        # PyTorch format: (B, T, C, H, W)
        self.num_frames = self.config.frames_per_clip
        self.height = self.config.crop_size
        self.width = self.config.crop_size
        self.channels = self.config.in_chans

    def test_input_embeddings(self):
        """Test embeddings output match between PyTorch and Flax."""
        torch_emb = self.baseline_model.encoder.embeddings
        nnx_emb = self.bonsai_model.encoder.embeddings

        key = jax.random.PRNGKey(0)
        # PyTorch format: (B, T, C, H, W)
        torch_shape = (self.batch_size, self.num_frames, self.channels, self.height, self.width)
        jx = jax.random.normal(key, torch_shape, dtype=jnp.float32)

        np_x = np.asarray(jax.device_get(jx))
        tx = torch.tensor(np_x, dtype=torch.float32)

        # Flax format: (B, T, H, W, C)
        jx_flax = jnp.transpose(jx, (0, 1, 3, 4, 2))

        with torch.inference_mode():
            ty = torch_emb(tx)
        jy = nnx_emb(jx_flax)

        np_y = np.asarray(jax.device_get(jy))
        ty_bonsai = torch.tensor(np_y, dtype=torch.float32)

        torch.testing.assert_close(ty_bonsai, ty, rtol=1e-5, atol=1e-3)

    def test_first_layer(self):
        """Test first transformer layer output."""
        torch_emb = self.baseline_model.encoder.embeddings
        nnx_emb = self.bonsai_model.encoder.embeddings
        torch_layer = self.baseline_model.encoder.layer[0]
        nnx_layer = self.bonsai_model.encoder.layer[0]

        key = jax.random.PRNGKey(0)
        torch_shape = (self.batch_size, self.num_frames, self.channels, self.height, self.width)
        jx = jax.random.normal(key, torch_shape, dtype=jnp.float32)
        np_x = np.asarray(jax.device_get(jx))
        tx = torch.tensor(np_x, dtype=torch.float32)

        # Flax format: (B, T, H, W, C)
        jx_flax = jnp.transpose(jx, (0, 1, 3, 4, 2))

        jhs = nnx_emb(jx_flax)

        with torch.inference_mode():
            ths = torch_emb(tx)
            ty = torch_layer(ths, position_mask=None, output_attentions=False)[0]

        jy = nnx_layer(jhs, None)

        np_y = np.asarray(jax.device_get(jy))
        ty_bonsai = torch.tensor(np_y, dtype=torch.float32)

        torch.testing.assert_close(ty_bonsai, ty, rtol=1e-5, atol=3e-3)

    def test_last_hidden_state(self):
        """Test encoder last hidden state output."""
        key = jax.random.PRNGKey(0)
        torch_shape = (self.batch_size, self.num_frames, self.channels, self.height, self.width)
        jx = jax.random.normal(key, torch_shape, dtype=jnp.float32)

        np_x = np.asarray(jax.device_get(jx))
        tx = torch.tensor(np_x, dtype=torch.float32)

        # Flax format: (B, T, H, W, C)
        jx_flax = jnp.transpose(jx, (0, 1, 3, 4, 2))

        with torch.inference_mode():
            ty = self.baseline_model(tx, skip_predictor=True).last_hidden_state
        jy = self.bonsai_model(jx_flax, skip_predictor=True).last_hidden_state

        np_y = np.asarray(jax.device_get(jy))
        ty_bonsai = torch.tensor(np_y, dtype=torch.float32)

        torch.testing.assert_close(ty_bonsai, ty, rtol=1e-5, atol=1e-2)


class TestVideoClassification(absltest.TestCase):
    """Test video classification model."""

    def setUp(self):
        super().setUp()
        self.save_dir = constants.default_cache_path
        os.makedirs(self.save_dir, exist_ok=True)

        # Import PyTorch classification model
        from transformers import VJEPA2ForVideoClassification as VJEPA2ForVideoClassificationTorch

        # Small config for testing - matching standard_test
        self.hfconfig = VJEPA2Config(
            crop_size=64,
            frames_per_clip=4,
            hidden_size=64,
            num_attention_heads=2,
            num_hidden_layers=2,
            mlp_ratio=4.0,
            pred_hidden_size=32,
            pred_num_attention_heads=2,
            pred_num_hidden_layers=2,
            patch_size=16,
            tubelet_size=2,
            in_chans=3,
            num_pooler_layers=1,
            num_labels=10,  # Small number of labels for testing
        )
        self.baseline_model = VJEPA2ForVideoClassificationTorch(config=self.hfconfig)
        self.model_ckpt_path = os.path.join(self.save_dir, "classifier_model.safetensors")
        save_file(self.baseline_model.state_dict(), self.model_ckpt_path)

        self.config = model_lib.VJEPA2FlaxConfig(
            crop_size=64,
            frames_per_clip=4,
            hidden_size=64,
            num_attention_heads=2,
            num_hidden_layers=2,
            mlp_ratio=4.0,
            pred_hidden_size=32,
            pred_num_attention_heads=2,
            pred_num_hidden_layers=2,
            num_pooler_layers=1,
            num_labels=10,
        )
        self.bonsai_model = params.create_model_from_safe_tensors(self.save_dir, self.config, classifier=True)

        self.bonsai_model.eval()
        self.baseline_model.eval()

        self.batch_size = 1
        self.num_frames = self.config.frames_per_clip
        self.height = self.config.crop_size
        self.width = self.config.crop_size
        self.channels = self.config.in_chans

    def test_classification_logits(self):
        """Test classification logits output."""
        key = jax.random.PRNGKey(0)
        torch_shape = (self.batch_size, self.num_frames, self.channels, self.height, self.width)
        jx = jax.random.normal(key, torch_shape, dtype=jnp.float32)

        np_x = np.asarray(jax.device_get(jx))
        tx = torch.tensor(np_x, dtype=torch.float32)

        # Flax format: (B, T, H, W, C)
        jx_flax = jnp.transpose(jx, (0, 1, 3, 4, 2))

        with torch.inference_mode():
            ty = self.baseline_model(tx).logits
        jy = self.bonsai_model(jx_flax).logits

        np_y = np.asarray(jax.device_get(jy))
        ty_bonsai = torch.tensor(np_y, dtype=torch.float32)

        # Check shapes match
        self.assertEqual(ty_bonsai.shape, ty.shape)
        self.assertEqual(ty_bonsai.shape[-1], self.config.num_labels)

        torch.testing.assert_close(ty_bonsai, ty, rtol=1e-5, atol=1e-2)


if __name__ == "__main__":
    absltest.main()
