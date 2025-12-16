import jax
import jax.numpy as jnp
from absl.testing import absltest
from flax import nnx

from bonsai.models.unetr.modeling import UNETR, UNETRConfig


class TestUNETROutputs(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.config = UNETRConfig(
            out_channels=3,
            in_channels=3,
            img_size=96,
            patch_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            num_layers=12,
            feature_size=16,
        )
        self.model = UNETR(config=self.config)

    def test_forward_pass_shape(self):
        batch_size = 2
        x = jnp.ones((batch_size, self.config.img_size, self.config.img_size, self.config.in_channels))
        y = self.model(x)

        expected_shape = (batch_size, self.config.img_size, self.config.img_size, self.config.out_channels)
        self.assertEqual(y.shape, expected_shape)

    def test_model_initialization(self):
        self.assertIsInstance(self.model, nnx.Module)


if __name__ == "__main__":
    absltest.main()
