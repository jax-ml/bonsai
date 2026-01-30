import unittest

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from flax import nnx
from jax import P
from jax.sharding import AxisType

from bonsai.models.gemma3 import modeling


class TestSharding(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        fsdp, tp = modeling.ShardMode.FSDP.value, modeling.ShardMode.TP.value

        cls.mesh = jax.make_mesh(((1, 1)), (fsdp, tp), axis_types=(AxisType.Explicit, AxisType.Explicit))
        jax.set_mesh(cls.mesh)

        cls.bonsai_config = modeling.ModelConfig.gemma3_4b_it(True, True, norm_dtype=jnp.float32)
        cls.bonsai_model = modeling.Gemma3Model(cls.bonsai_config, rngs=nnx.Rngs(0))

    def test_full(self):
        nm = self.bonsai_model
        fsdp = modeling.ShardMode.FSDP.value

        key = jax.random.key(0)
        n_img = jax.random.uniform(key, (1, 896, 896, 3), dtype=jnp.float32, minval=-1, maxval=1, out_sharding=P(fsdp))
        n_text = jnp.arange(1781, out_sharding=P(fsdp)).reshape(1, -1)
        token_type_ids = np.zeros((1, 1781), dtype=int)
        token_type_ids[12:268] = 1
        n_tti = jax.device_put(token_type_ids, device=P(fsdp))

        batch_size, num_tokens = n_text.shape
        segment_ids = jnp.ones((batch_size, num_tokens), out_sharding=P(fsdp))
        cache = modeling.init_cache(self.bonsai_config, batch_size, num_tokens, 1, jnp.float32)

        nm(n_text, n_img, cache, segment_ids, n_tti)

    @unittest.skip("Only for viewing purposes")
    def test_view_model(self):
        state = nnx.state(self.bonsai_model)
        out = jax.tree_util.tree_map(lambda x: jax.typeof(x), state)

        # print(out)
        # print(out.vision_tower)
        # print(out.language_model)
        # print(out.embed_tokens)
        print(out.multi_modal_projector)


if __name__ == "__main__":
    absltest.main()
