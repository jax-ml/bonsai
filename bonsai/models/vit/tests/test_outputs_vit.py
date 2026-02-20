import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from flax import nnx
from transformers import ViTForImageClassification
from bonsai.models.vit.modeling import ViTClassificationModel


class TestModuleForwardPasses(absltest.TestCase):
    def setUp(self):
        super().setUp()
        jax.config.update("jax_default_matmul_precision", "float32")
        model_name = "google/vit-base-patch16-224"

        # Cast JAX model to float32 for precision matching with PyTorch CPU
        graph_def, state = nnx.split(ViTClassificationModel.from_pretrained(model_name))
        state = jax.tree.map(lambda x: x.astype(jnp.float32) if isinstance(x, jax.Array) else x, state)
        self.bonsai_model = nnx.merge(graph_def, state)

        self.baseline_model = ViTForImageClassification.from_pretrained(model_name)
        self.bonsai_model.eval()
        self.baseline_model.eval()

        self.batch_size = 32
        self.image_shape = (self.batch_size, 224, 224, 3)

    def test_embeddings(self):
        torch_emb = self.baseline_model.vit.embeddings
        nnx_emb = self.bonsai_model.pos_embeddings

        jx = jax.random.normal(jax.random.key(0), self.image_shape, dtype=jnp.float32)
        tx = torch.tensor(np.array(jx)).permute(0, 3, 1, 2)

        with torch.no_grad():
            ty = torch_emb(tx)
        jy = nnx_emb(jx, rngs=None)

        torch.testing.assert_close(torch.tensor(np.array(jy)), ty, rtol=1e-5, atol=1e-5)

    def test_first_layer(self):
        torch_layer = self.baseline_model.vit.encoder.layer[0]
        nnx_layer = self.bonsai_model.layers[0]

        hidden_shape = (self.batch_size, 197, 768)
        jx = jax.random.normal(jax.random.key(0), hidden_shape, dtype=jnp.float32)
        tx = torch.tensor(np.array(jx))

        with torch.no_grad():
            ty = torch_layer(tx)
        jy = nnx_layer(jx, rngs=None)

        torch.testing.assert_close(torch.tensor(np.array(jy)), ty, rtol=1e-5, atol=1e-2)

    def test_full(self):
        jx = jax.random.normal(jax.random.key(0), self.image_shape, dtype=jnp.float32)
        tx = torch.tensor(np.array(jx)).permute(0, 3, 1, 2)

        with torch.no_grad():
            ty = self.baseline_model(tx).logits
        jy = self.bonsai_model(jx, rngs=None)

        torch.testing.assert_close(torch.tensor(np.array(jy)), ty, rtol=1e-5, atol=5e-2)

    def test_full_interpolation(self):
        image_shape_384 = (self.batch_size, 384, 384, 3)

        jx = jax.random.normal(jax.random.key(1), image_shape_384, dtype=jnp.float32)
        tx = torch.tensor(np.array(jx)).permute(0, 3, 1, 2)

        with torch.no_grad():
            ty = self.baseline_model(tx, interpolate_pos_encoding=True).logits
        jy = self.bonsai_model(jx, rngs=None)

        torch.testing.assert_close(torch.tensor(np.array(jy)), ty, rtol=1e-5, atol=1e-1)


if __name__ == "__main__":
    absltest.main()
