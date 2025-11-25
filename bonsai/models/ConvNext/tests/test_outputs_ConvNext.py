import jax
import jax.numpy as jnp
import torch
from absl.testing import absltest
from huggingface_hub import snapshot_download
from transformers import ConvNextForImageClassification

from bonsai.models.ConvNext import modeling as model_lib
from bonsai.models.ConvNext import params


class TestModuleForwardPasses(absltest.TestCase):
    def setUp(self):
        super().setUp()
        model_name = "facebook/convnext-large-224"
        model_ckpt_path = snapshot_download(model_name)

        self.bonsai_model = params._create_convnext_from_pretrained(model_lib.ConvNeXt, model_ckpt_path)
        self.baseline_model = ConvNextForImageClassification.from_pretrained(model_name)

        self.bonsai_model.eval()
        self.baseline_model.eval()

        self.batch_size = 1
        self.image_shape = (self.batch_size, 224, 224, 3)

    def test_embeddings(self):
        torch_emb = self.baseline_model.convnext.embeddings
        nnx_emb = self.bonsai_model.embeddings

        jx = jax.random.normal(jax.random.key(0), self.image_shape, dtype=jnp.float32)
        tx = torch.tensor(jx).permute(0, 3, 1, 2)

        with torch.no_grad():
            ty = torch_emb(tx)
        jy = nnx_emb(jx)

        torch.testing.assert_close(torch.tensor(jy), ty, rtol=1e-5, atol=1e-5)

    def test_blocks_isolated(self):
        """
        Compare every ConvNeXt block between HuggingFace PyTorch and the JAX/NNX model.
        Tests each block in ISOLATION with fresh random input to avoid error accumulation.
        """
        jax_model = self.bonsai_model
        torch_model = self.baseline_model.convnext
        torch_model.eval()

        # Dimensions for each stage (NHWC for JAX, NCHW for Torch)
        # Stage 0: 56x56, dim=192
        # Stage 1: 28x28, dim=384
        # Stage 2: 14x14, dim=768
        # Stage 3: 7x7,   dim=1536
        stage_dims = [(56, 192), (28, 384), (14, 768), (7, 1536)]

        key = jax.random.PRNGKey(42)

        for stage_idx, stage_blocks in enumerate(jax_model.stages):
            h, dim = stage_dims[stage_idx]

            for block_idx in range(len(stage_blocks)):
                key, sub = jax.random.split(key)

                jx_input = jax.random.normal(sub, (1, h, h, dim), dtype=jnp.float32)
                tx_input = torch.tensor(jx_input).permute(0, 3, 1, 2)

                key, sub = jax.random.split(key)
                j_out = jax_model.stages[stage_idx][block_idx](jx_input, rng=sub, train=False)

                with torch.no_grad():
                    t_out = torch_model.encoder.stages[stage_idx].layers[block_idx](tx_input)

                # Convert Torch output to NHWC for comparison
                t_out_nhwc = t_out.permute(0, 2, 3, 1)

                torch.testing.assert_close(torch.tensor(j_out), t_out_nhwc, rtol=5e-4, atol=5e-4)

    def test_full(self):
        jx = jax.random.normal(jax.random.key(0), self.image_shape, dtype=jnp.float32)
        tx = torch.tensor(jx).permute(0, 3, 1, 2)
        with torch.no_grad():
            ty = self.baseline_model(tx).logits
        jy = self.bonsai_model(jx)
        torch.testing.assert_close(torch.tensor(jy), ty, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    absltest.main()
