import unittest

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from flax import nnx
from jax import P
from jax.sharding import AxisType, NamedSharding

from bonsai.models.gemma3 import modeling


@unittest.skipIf(jax.device_count() < 4, "At least 4 devices required")
class TestSharding(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        fsdp, tp = modeling.ShardMode.FSDP.value, modeling.ShardMode.TP.value

        cls.mesh = jax.make_mesh(((2, 2)), (fsdp, tp), axis_types=(AxisType.Explicit, AxisType.Explicit))
        jax.set_mesh(cls.mesh)

        # Define a small model
        cls.bonsai_config = modeling.ModelConfig(
            vision_config=modeling.VisionConfig(
                attention_dropout=0.0,
                hidden_size=256,
                image_size=224,
                intermediate_size=320,
                layer_norm_eps=1e-6,
                num_attention_heads=2,
                num_channels=3,
                num_hidden_layers=2,
                patch_size=14,
                vision_use_head=False,
                shd_cfg=modeling.VisionShardConfig.default(True, True),
            ),
            text_config=modeling.TextConfig(
                attention_bias=False,
                attention_dropout=0.0,  # TODO: unused
                head_dim=256,
                hidden_size=2560,
                intermediate_size=10240,
                layer_types=modeling._set_attention_modes(3, 4),
                max_position_embeddings=131072,  # TODO: unused
                num_attention_heads=4,
                num_hidden_layers=4,
                num_key_value_heads=2,
                rms_norm_eps=1e-6,
                rope_full_factor=8.0,
                rope_full_theta=1000000.0,
                rope_slide_factor=1.0,
                rope_slide_theta=10000.0,
                sliding_window=512,
                vocab_size=10_000,
                shd_cfg=modeling.TextShardConfig.default(True, True),
                norm_dtype=jnp.float32,
            ),
            mm_tokens_per_image=256,
            dtype="bfloat16",  # TODO: unused
            final_logit_softcapping=None,
            shd_cfg=modeling.ShardConfig.default(True),
        )
        cls.bonsai_model = modeling.Gemma3Model(cls.bonsai_config, rngs=nnx.Rngs(0))

    def test_full(self):
        nm = self.bonsai_model
        fsdp = modeling.ShardMode.FSDP.value

        batch_size = 2  # should be evenly divisible to num devices for fsdp axis
        num_tokens = 128
        key = jax.random.key(0)
        img_size = self.bonsai_config.vision_config.image_size
        n_img = jax.random.uniform(
            key, (batch_size, img_size, img_size, 3), dtype=jnp.float32, minval=-1, maxval=1, out_sharding=P(fsdp)
        )
        n_text = jax.device_put(
            np.arange(batch_size * num_tokens).reshape(batch_size, -1),
            device=P(fsdp),
        )
        token_type_ids = np.zeros((batch_size, num_tokens), dtype=int)
        token_type_ids[:, 12:98] = 1
        n_tti = jax.device_put(token_type_ids, device=P(fsdp))

        segment_ids = jnp.ones((batch_size, num_tokens), out_sharding=P(fsdp))
        cache = modeling.init_cache(self.bonsai_config, batch_size, num_tokens, 1, jnp.float32)

        out = nm(n_text, n_img, cache=cache, segment_ids=segment_ids, token_type_ids=n_tti)
        assert isinstance(out.sharding, NamedSharding)
        assert out.sharding.spec == self.bonsai_config.text_config.shd_cfg.activation

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
