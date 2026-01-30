import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest, parameterized
from huggingface_hub import snapshot_download
from transformers import ConvNextForImageClassification

from bonsai.models.convnext import modeling as model_lib
from bonsai.models.convnext import params


class TestModuleForwardPasses(absltest.TestCase):
    """Tests forward passes for the convenext-large-224 model"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        model_name = "facebook/convnext-large-224"
        model_ckpt_path = snapshot_download(model_name)

        cls.bonsai_config = model_lib.ModelConfig.convnext_large_224()
        cls.bonsai_model = params.create_convnext_from_pretrained(model_ckpt_path, cls.bonsai_config)
        cls.baseline_model = ConvNextForImageClassification.from_pretrained(model_name)

        cls.baseline_model.eval()

        cls.image_shape = (2, 224, 224, 3)

    def test_embeddings(self):
        torch_emb = self.baseline_model.convnext.embeddings
        nnx_emb = self.bonsai_model.embedding_layer

        jx = jax.random.normal(jax.random.key(0), self.image_shape, dtype=jnp.float32)
        tx = torch.tensor(np.asarray(jx)).permute(0, 3, 1, 2)

        with torch.no_grad():
            ty = torch_emb(tx)
        jy = nnx_emb(jx)

        torch.testing.assert_close(torch.tensor(np.asarray(jy)).permute(0, 3, 1, 2), ty, rtol=1e-5, atol=1e-5)

    def test_blocks_isolated(self):
        jax_model = self.bonsai_model
        torch_model = self.baseline_model.convnext
        torch_model.eval()

        stage_img_sizes = [56, 28, 14, 7]
        stage_dims = self.bonsai_config.stage_dims
        key = jax.random.key(0)

        for stage_idx, (stage_blocks, h, dim) in enumerate(zip(jax_model.stages, stage_img_sizes, stage_dims)):
            jx_input = jax.random.normal(key, (1, h, h, dim), dtype=jnp.float32)
            tx_input = torch.tensor(np.asarray(jx_input)).permute(0, 3, 1, 2)
            for block_idx in range(len(stage_blocks.layers)):
                j_out = stage_blocks.layers[block_idx](jx_input, rngs=key, train=False)
                with torch.no_grad():
                    t_out = torch_model.encoder.stages[stage_idx].layers[block_idx](tx_input)

                torch.testing.assert_close(
                    torch.tensor(np.asarray(j_out)), t_out.permute(0, 2, 3, 1), rtol=5e-4, atol=5e-4
                )

    def test_full(self):
        jx = jax.random.normal(jax.random.key(0), self.image_shape, dtype=jnp.float32)
        tx = torch.tensor(np.asarray(jx)).permute(0, 3, 1, 2)
        with torch.no_grad():
            ty = self.baseline_model(tx).logits
        jy = self.bonsai_model(jx, rngs=jax.random.key(0))
        torch.testing.assert_close(torch.tensor(np.asarray(jy)), ty)


class TestModuleFullOtherConfigs(parameterized.TestCase):
    @parameterized.named_parameters(("tiny", "tiny"), ("small", "small"), ("base", "base"))
    def test_full(self, model_size):
        model_name = f"facebook/convnext-{model_size}-224"
        model_ckpt_path = snapshot_download(model_name)

        bonsai_config = getattr(model_lib.ModelConfig, f"convnext_{model_size}_224")()
        bonsai_model = params.create_convnext_from_pretrained(model_ckpt_path, bonsai_config)
        baseline_model = ConvNextForImageClassification.from_pretrained(model_name)
        baseline_model.eval()

        image_shape = (2, 224, 224, 3)
        jx = jax.random.normal(jax.random.key(0), image_shape, dtype=jnp.float32)
        tx = torch.tensor(np.asarray(jx)).permute(0, 3, 1, 2)

        with torch.no_grad():
            ty = baseline_model(tx).logits
        jy = bonsai_model(jx, rngs=jax.random.key(0))

        torch.testing.assert_close(torch.tensor(np.asarray(jy)), ty, rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    absltest.main()
