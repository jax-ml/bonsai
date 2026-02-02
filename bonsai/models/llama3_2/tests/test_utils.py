import jax.numpy as jnp

from bonsai.models.llama3_2 import modeling


def tiny_config(*, use_sharding: bool = False) -> modeling.ModelConfig:
    """Create a minimal Llama3.2 model configuration for testing purposes"""
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
        shd_cfg=(
            modeling.LlamaShardCfg.default(use_fsdp=True, use_tp=True)
            if use_sharding
            else modeling.LlamaShardCfg.no_sharding()
        ),
        dtype=jnp.float32,
    )
