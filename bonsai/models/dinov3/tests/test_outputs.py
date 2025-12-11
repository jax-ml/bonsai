import jax 
import jax.numpy as jnp 

import torch 
from huggingface_hub import snapshot_download
from transformers import DINOv3ViTModel 
import numpy as np

from bonsai.models.dinov3 import params
from absl.testing import absltest

class TestForwardPass(absltest.TestCase):
    def setUp(self):
        super().setUp()
        model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
        model_ckpt_path = snapshot_download(model_name, allow_patterns=["*.safetensors", "*.json"])

        self.config = params.load_model_config(model_ckpt_path)
        self.bonsai_model = params.create_model_from_safe_tensors(model_ckpt_path, self.config)
        self.baseline_model = DINOv3ViTModel.from_pretrained(model_name)

        self.bonsai_model.eval()
        self.baseline_model.eval()

        self.batch_size = 1
        self.image_shape = (self.batch_size, 3, 224, 224)
    
    def test_output_embeddings(self):
        key = jax.random.PRNGKey(0)
        jx = jax.random.normal(key, self.image_shape, dtype=jnp.float32)

        np_x = np.asarray(jax.device_get(jx))
        tx = torch.tensor(np_x, dtype=torch.float32)

        with torch.inference_mode():
            ty = self.baseline_model(tx).pooler_output
        jy = self.bonsai_model(jx)

        np_y = np.asarray(jax.device_get(jy))
        ty_bonsai = torch.tensor(np_y, dtype=torch.float32)

        npty = np.asarray(ty)

        def cosine_similarity(A, B):
            dot_product = np.sum(A * B, axis=1)
            norm_a = np.linalg.norm(A, axis=1)
            norm_b = np.linalg.norm(B, axis=1)
            similarity = dot_product / (norm_a * norm_b)
            return similarity

        print(f"Cosine similarity score between torch and jax embeddings: {cosine_similarity(np_y, npty)}")

        torch.testing.assert_close(ty_bonsai, ty, rtol=1e-2, atol=1e-2)

if __name__ == "__main__":
    absltest.main()