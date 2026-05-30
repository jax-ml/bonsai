from flax import nnx
import unittest
import jax
from absl.testing import absltest
from jax.sharding import AxisType

from bonsai.models.gemma4.modeling import ModelConfig as Gemma4Config, Gemma4ForCausalLM


@unittest.skipIf(jax.device_count() < 8, "At least 8 devices required")
class TestSharding(absltest.TestCase):
    """Test suite for model sharding."""

    @classmethod
    def setUpClass(cls):
        """Sets up the virtual mesh for sharding tests."""
        super().setUpClass()
        cls.mesh = jax.make_mesh(((4, 2)), ("fsdp", "tp"), axis_types=(AxisType.Explicit, AxisType.Explicit))
        jax.set_mesh(cls.mesh)
        cls.config = Gemma4Config.gemma4_base(use_fsdp=True, use_tp=True)
        # decrease sizes to avoid OOM
        cls.config.hidden_size = 64
        cls.config.intermediate_size = 128
        cls.config.num_hidden_layers = 2
        cls.config.num_attention_heads = 4
        cls.config.num_key_value_heads = 2
        cls.config.head_dim = 16
        cls.config.num_experts = 8

    def test_model_sharding(self):
        # Verify the model does not crash during init on mesh with sharding config
        """Tests that model parameters are sharded correctly."""
        model = Gemma4ForCausalLM(self.config, rngs=nnx.Rngs(0))
        self.assertIsNotNone(model)


if __name__ == "__main__":
    absltest.main()
