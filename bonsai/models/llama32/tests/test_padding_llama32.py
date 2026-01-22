import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import AxisType

from bonsai.models.llama32 import modeling


def _tiny_config() -> modeling.ModelConfig:
    return modeling.ModelConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        head_dim=4,
        num_key_value_heads=1,
        max_position_embeddings=32,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        rope_scaling=None,
        tie_word_embeddings=True,
        shd_cfg=modeling.LlamaShardCfg.no_sharding(),
        dtype=jnp.float32,
    )


def test_forward_uses_per_sample_right_padding():
    cfg = _tiny_config()
    fsdp, tp = modeling.ShardMode.FSDP.value, modeling.ShardMode.TP.value
    mesh = jax.make_mesh((1, 1), (fsdp, tp), axis_types=(AxisType.Explicit, AxisType.Explicit))
    jax.set_mesh(mesh)
    model = modeling.Llama(cfg, rngs=nnx.Rngs(params=0))

    tokens = jnp.array(
        [
            [1, 2, 3, 0, 0],
            [4, 5, 6, 7, 0],
        ],
        dtype=jnp.int32,
    )
    attention_mask = jnp.array(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
        ],
        dtype=jnp.int32,
    )

    cache = model.init_cache(cfg, batch_size=tokens.shape[0], token_len=tokens.shape[1], generate_steps=1)
    logits, _ = modeling.forward(model, cache, tokens, pad_id=0, attention_mask=attention_mask)

    ref_cache = model.init_cache(cfg, batch_size=tokens.shape[0], token_len=tokens.shape[1], generate_steps=1)
    full_logits = model(tokens, attention_mask.astype(jnp.int32), ref_cache, attn_mask=None)
    target_ind = jnp.sum(attention_mask, axis=1) - 1
    expected = full_logits[jnp.arange(tokens.shape[0]), target_ind]

    np.testing.assert_allclose(np.array(logits), np.array(expected), rtol=1e-6, atol=1e-6)


def test_compute_positions_from_segment_ids_packed():
    seg_ids = jnp.array(
        [
            [1, 1, 1, 2, 2, 0, 0],
            [3, 3, 0, 4, 4, 4, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=jnp.int32,
    )
    positions = modeling.compute_positions_from_segment_ids(seg_ids)
    pad_val = 2**30
    expected = jnp.array(
        [
            [0, 1, 2, 0, 1, pad_val, pad_val],
            [0, 1, pad_val, 0, 1, 2, pad_val],
            [pad_val, pad_val, pad_val, pad_val, pad_val, pad_val, pad_val],
        ],
        dtype=jnp.int32,
    )
    np.testing.assert_array_equal(np.array(positions), np.array(expected))


def test_forward_accepts_segment_ids_packed():
    cfg = _tiny_config()
    fsdp, tp = modeling.ShardMode.FSDP.value, modeling.ShardMode.TP.value
    mesh = jax.make_mesh((1, 1), (fsdp, tp), axis_types=(AxisType.Explicit, AxisType.Explicit))
    jax.set_mesh(mesh)
    model = modeling.Llama(cfg, rngs=nnx.Rngs(params=0))

    tokens = jnp.array(
        [
            [1, 2, 3, 4, 0, 0],
            [5, 6, 7, 8, 9, 0],
        ],
        dtype=jnp.int32,
    )
    segment_ids = jnp.array(
        [
            [1, 1, 2, 2, 0, 0],
            [3, 3, 3, 4, 4, 0],
        ],
        dtype=jnp.int32,
    )

    cache = model.init_cache(cfg, batch_size=tokens.shape[0], token_len=tokens.shape[1], generate_steps=1)
    logits, _ = modeling.forward(model, cache, tokens, pad_id=0, segment_ids=segment_ids)

    ref_cache = model.init_cache(cfg, batch_size=tokens.shape[0], token_len=tokens.shape[1], generate_steps=1)
    full_logits = model(tokens, segment_ids, ref_cache, attn_mask=None)
    target_ind = jnp.sum(segment_ids != 0, axis=1) - 1
    expected = full_logits[jnp.arange(tokens.shape[0]), target_ind]

    np.testing.assert_allclose(np.array(logits), np.array(expected), rtol=1e-6, atol=1e-6)
