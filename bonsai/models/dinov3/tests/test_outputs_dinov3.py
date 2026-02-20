import os
import tempfile
import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from safetensors.torch import save_file
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from bonsai.models.dinov3 import modeling as model_lib
from bonsai.models.dinov3 import params


class TestForwardPass(absltest.TestCase):
    def setUp(self):
        super().setUp()

        self.hfconfig = DINOv3ViTConfig(
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            hidden_act="gelu",
            use_gated_mlp=False,
            num_register_tokens=4,
        )
        self.baseline_model = DINOv3ViTModel(config=self.hfconfig)
        self.config = model_lib.ModelConfig.dinov3_vitb16()

        with tempfile.TemporaryDirectory() as temp_dir:
            filename = os.path.join(temp_dir, "ref.safetensors")
            save_file(self.baseline_model.state_dict(), filename)
            self.bonsai_model = params.create_model_from_safe_tensors(temp_dir, self.config)

        self.bonsai_model.eval()
        self.baseline_model.eval()

        self.batch_size = 2
        self.image_shape = (self.batch_size, 3, 224, 224)

    def test_input_embeddings(self):
        torch_emb = self.baseline_model.embeddings
        nnx_emb = self.bonsai_model.embeddings

        jx = jax.random.normal(jax.random.key(0), self.image_shape, dtype=jnp.float32)
        tx = torch.tensor(jx, dtype=torch.float32)

        with torch.inference_mode():
            ty = torch_emb(tx)
        jy = nnx_emb(jx)

        np.testing.assert_allclose(jy, ty.detach().cpu().numpy(), rtol=1e-5, atol=1e-5)

    def test_first_layer(self):
        torch_emb = self.baseline_model.embeddings
        nnx_emb = self.bonsai_model.embeddings
        torch_pe = self.baseline_model.rope_embeddings
        nnx_pe = self.bonsai_model.rope_embeddings
        torch_layer = self.baseline_model.layer[0]
        nnx_layer = self.bonsai_model.layer[0]

        jx = jax.random.normal(jax.random.key(0), self.image_shape, dtype=jnp.float32)
        tx = torch.tensor(jx, dtype=torch.float32)

        jhs = nnx_emb(jx)
        jpe = nnx_pe(jx)

        ths = torch_emb(tx)
        tpe = torch_pe(tx)

        with torch.inference_mode():
            ty = torch_layer(ths, position_embeddings=tpe)
        jy = nnx_layer(jhs, jpe)

        np.testing.assert_allclose(jy, ty.detach().cpu().numpy(), rtol=1e-5, atol=3e-3)

    def test_last_hidden_state(self):
        jx = jax.random.normal(jax.random.key(0), self.image_shape, dtype=jnp.float32)
        tx = torch.tensor(jx, dtype=torch.float32)

        with torch.inference_mode():
            ty = self.baseline_model(tx).last_hidden_state
        jy = self.bonsai_model(jx)["last_hidden_state"]

        np.testing.assert_allclose(jy, ty.detach().cpu().numpy(), rtol=1e-5, atol=3e-2)

    def test_pooled_output_embeddings(self):
        jx = jax.random.normal(jax.random.key(0), self.image_shape, dtype=jnp.float32)
        tx = torch.tensor(jx, dtype=torch.float32)

        with torch.inference_mode():
            ty = self.baseline_model(tx).pooler_output
        jy = self.bonsai_model(jx)["pooler_output"]

        np.testing.assert_allclose(jy, ty.detach().cpu().numpy(), rtol=1e-5, atol=2e-2)


class TestRopePositionEmbedding(absltest.TestCase):
    def test_non_square_patch_size_uses_width_patch_dimension(self):
        config = model_lib.ModelConfig(
            patch_size=(16, 8),
            hidden_size=64,
            num_attention_heads=1,
            image_size=64,
        )
        rope = model_lib.Dinov3ViTRopePositionEmbedding(config)

        x = jnp.zeros((1, 3, 64, 40), dtype=jnp.float32)
        cos, sin = rope(x)

        _, _, height, width = x.shape
        expected_num_patches = (height // config.patch_size[0]) * (width // config.patch_size[1])
        expected_head_dim = config.hidden_size // config.num_attention_heads
        self.assertEqual(cos.shape, (expected_num_patches, expected_head_dim))
        self.assertEqual(sin.shape, (expected_num_patches, expected_head_dim))


if __name__ == "__main__":
    absltest.main()
