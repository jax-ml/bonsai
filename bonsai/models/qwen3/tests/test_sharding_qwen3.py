import unittest

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from flax import nnx
from jax import P
from jax.sharding import AxisType, NamedSharding

from bonsai.models.qwen3 import modeling


@unittest.skipIf(jax.device_count() < 8, "At least 8 devices required")
class TestSharding(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        fsdp, tp = modeling.ShardMode.FSDP.value, modeling.ShardMode.TP.value

        cls.mesh = jax.make_mesh(((4, 2)), (fsdp, tp), axis_types=(AxisType.Explicit, AxisType.Explicit))
        jax.set_mesh(cls.mesh)

        cls.bonsai_config = modeling.ModelConfig.qwen3_0_6b(True, True)
        cls.bonsai_model = modeling.Qwen3(cls.bonsai_config, rngs=nnx.Rngs(0))

    def test_full(self):
        nm = self.bonsai_model
        fsdp = modeling.ShardMode.FSDP.value

        batch_size = 4  # should be evenly divisible to num devices for fsdp axis
        num_tokens = 128
        n_text = jax.device_put(
            np.arange(batch_size * num_tokens).reshape(batch_size, -1),
            device=P(fsdp),
        )
        segment_ids = jnp.ones((batch_size, num_tokens), out_sharding=P(fsdp))
        cache = modeling.init_cache(self.bonsai_config, batch_size, num_tokens, 1, jnp.float32)

        out = nm(n_text, cache=cache, segment_ids=segment_ids)
        assert isinstance(out.sharding, NamedSharding)
        assert out.sharding.spec == self.bonsai_config.shd_cfg.act_btd


if __name__ == "__main__":
    absltest.main()
