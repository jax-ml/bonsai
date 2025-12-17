import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from huggingface_hub import snapshot_download
from transformers import DINOv3ViTModel

from bonsai.models.dinov3 import modeling as model_lib
from bonsai.models.dinov3 import params


class TestForwardPass(absltest.TestCase):
    def setUp(self):
        super().setUp()
        model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
        model_ckpt_path = snapshot_download(model_name, allow_patterns=["*.safetensors", "*.json"])

        self.config = model_lib.DINOv3ViTFlaxConfig.dinov3_vitl16()
        self.bonsai_model = params.create_model_from_safe_tensors(model_ckpt_path, self.config)
        self.baseline_model = DINOv3ViTModel.from_pretrained(model_name)

        self.bonsai_model.eval()
        self.baseline_model.eval()

        self.batch_size = 1
        self.image_shape = (self.batch_size, 3, 224, 224)

    def test_input_embeddings(self):
        torch_emb = self.baseline_model.embeddings
        nnx_emb = self.bonsai_model.embeddings

        key = jax.random.PRNGKey(0)
        jx = jax.random.normal(key, self.image_shape, dtype=jnp.float32)

        np_x = np.asarray(jax.device_get(jx))
        tx = torch.tensor(np_x, dtype=torch.float32)

        with torch.inference_mode():
            ty = torch_emb(tx)
        jy = nnx_emb(jx)

        np_y = np.asarray(jax.device_get(jy))
        ty_bonsai = torch.tensor(np_y, dtype=torch.float32)

        torch.testing.assert_close(ty_bonsai, ty, rtol=1e-5, atol=1e-5)

    def test_first_layer(self):
        torch_emb = self.baseline_model.embeddings
        nnx_emb = self.bonsai_model.embeddings
        torch_pe = self.baseline_model.rope_embeddings
        nnx_pe = self.bonsai_model.rope_embeddings
        torch_layer = self.baseline_model.layer[0]
        nnx_layer = self.bonsai_model.layer[0]

        key = jax.random.PRNGKey(0)
        jx = jax.random.normal(key, self.image_shape, dtype=jnp.float32)
        np_x = np.asarray(jax.device_get(jx))
        tx = torch.tensor(np_x, dtype=torch.float32)

        jhs = nnx_emb(jx)
        jpe = nnx_pe(jx)

        ths = torch_emb(tx)
        tpe = torch_pe(tx)

        with torch.inference_mode():
            ty = torch_layer(ths, position_embeddings=tpe)
        jy = nnx_layer(jhs, jpe)

        np_y = np.asarray(jax.device_get(jy))
        ty_bonsai = torch.tensor(np_y, dtype=torch.float32)

        torch.testing.assert_close(ty_bonsai, ty, rtol=3e-1, atol=3e-1)


    def test_last_hidden_state(self):
        key = jax.random.PRNGKey(0)
        jx = jax.random.normal(key, self.image_shape, dtype=jnp.float32)

        np_x = np.asarray(jax.device_get(jx))
        tx = torch.tensor(np_x, dtype=torch.float32)

        with torch.inference_mode():
            ty = self.baseline_model(tx).last_hidden_state
        jy = self.bonsai_model(jx).last_hidden_state

        np_y = np.asarray(jax.device_get(jy))
        ty_bonsai = torch.tensor(np_y, dtype=torch.float32)

        torch.testing.assert_close(ty_bonsai, ty, rtol=1e-5, atol=5e-2)

    def test_pooled_output_embeddings(self):
        key = jax.random.PRNGKey(0)
        jx = jax.random.normal(key, self.image_shape, dtype=jnp.float32)

        np_x = np.asarray(jax.device_get(jx))
        tx = torch.tensor(np_x, dtype=torch.float32)

        with torch.inference_mode():
            ty = self.baseline_model(tx).pooler_output
        jy = self.bonsai_model(jx).pooler_output

        np_y = np.asarray(jax.device_get(jy))
        ty_bonsai = torch.tensor(np_y, dtype=torch.float32)

        torch.testing.assert_close(ty_bonsai, ty, rtol=1e-5, atol=2e-2)

if __name__ == "__main__":
    absltest.main()
