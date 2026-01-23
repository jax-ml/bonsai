import jax
import jax.numpy as jnp
from absl.testing import absltest
from flax import nnx
from jax import P
from jax.sharding import AxisType

from bonsai.models.llama3_2 import modeling


def _tiny_sharded_config() -> modeling.ModelConfig:
    return modeling.ModelConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=8,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        rope_scaling=None,
        tie_word_embeddings=True,
        shd_cfg=modeling.LlamaShardCfg.default(use_fsdp=True, use_tp=True),
        dtype=jnp.float32,
    )


class TestShardingLlama32(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        fsdp, tp = modeling.ShardMode.FSDP.value, modeling.ShardMode.TP.value
        cls.mesh = jax.make_mesh((1, 1), (fsdp, tp), axis_types=(AxisType.Explicit, AxisType.Explicit))
        jax.set_mesh(cls.mesh)

        cls.cfg = _tiny_sharded_config()
        cls.model = modeling.Llama(cls.cfg, rngs=nnx.Rngs(params=0))

    def test_forward_sharded_inputs(self):
        fsdp = modeling.ShardMode.FSDP.value
        tokens = jnp.array(
            [
                [1, 2, 3, 0],
                [4, 5, 0, 0],
            ],
            dtype=jnp.int32,
            out_sharding=P(fsdp),
        )
        attention_mask = jnp.array(
            [
                [1, 1, 1, 0],
                [1, 1, 0, 0],
            ],
            dtype=jnp.int32,
            out_sharding=P(fsdp),
        )
        segment_ids = attention_mask.astype(jnp.int32)

        cache = self.model.init_cache(self.cfg, batch_size=tokens.shape[0], token_len=tokens.shape[1], generate_steps=1)
        _ = self.model(tokens, segment_ids, cache, attn_mask=None)


if __name__ == "__main__":
    absltest.main()
