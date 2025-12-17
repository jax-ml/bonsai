import jax
import jax.numpy as jnp
from absl.testing import absltest
from flax import nnx

from bonsai.models.unetr.modeling import UNETR
from bonsai.models.vit.modeling import ModelConfig


class TestUNETROutputs(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.config = ModelConfig.unetr()
        self.model = UNETR(config=self.config)

    def test_forward_pass_shape(self):
        batch_size = 2
        img_size = self.config.image_size[0]
        x = jnp.ones((batch_size, img_size, img_size, self.config.num_channels))
        y = self.model(x)

        expected_shape = (batch_size, img_size, img_size, self.config.out_channels)
        self.assertEqual(y.shape, expected_shape)

    def test_model_initialization(self):
        self.assertIsInstance(self.model, nnx.Module)


if __name__ == "__main__":
    absltest.main()
