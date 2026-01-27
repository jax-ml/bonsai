# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from enum import Enum
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.nnx.nn.linear import default_embed_init
from jax import P

from jax.sharding import PartitionSpec
from jaxtyping import Array, DTypeLike
from tqdm import trange

LARGE_NEGATIVE = jnp.finfo(jnp.float32).min


class ShardMode(Enum):
    FSDP = "fsdp"
    TP = "tp"


@dataclass(slots=True, frozen=True)
class ShardingConfig:
    attn_kernel: PartitionSpec
    attn_bias: PartitionSpec
    attn_qk_activation: PartitionSpec
    fc1_kernel: PartitionSpec
    fc1_bias: PartitionSpec
    fc2_kernel: PartitionSpec
    fc2_bias: PartitionSpec
    activation: PartitionSpec
    norm: PartitionSpec
    emb_kernel: PartitionSpec

    @staticmethod
    def no_sharding():
        return ShardingConfig.default(False, False)

    @staticmethod
    def default(use_fsdp: bool, use_tp: bool):
        fsdp = ShardMode.FSDP.value if use_fsdp else None
        tp = ShardMode.TP.value if use_tp else None
        return ShardingConfig(
            attn_kernel=P(tp, fsdp),
            attn_bias=P(tp),
            attn_qk_activation=P(fsdp, tp),
            fc1_kernel=P(fsdp, tp),
            fc1_bias=P(tp),
            fc2_kernel=P(tp, fsdp),
            fc2_bias=P(tp),
            activation=P(fsdp, None, tp),
            norm=P(tp),
            emb_kernel=P(None, tp),
        )


@dataclass
class ModelConfig:
    dtype: DTypeLike
    d_model: int
    n_heads: int
    n_kv_heads: int
    n_layers: int
    embedding_dropout: float
    max_sequence_length: int
    rope_theta: float
    include_qkv_bias: bool
    include_bias: bool
    vocab_size: int
    embedding_size: int
    mlp_hidden_size: int
    shd_cfg: ShardingConfig
    return_hidden_states: bool
    block_group_size: int = 1
    rms_norm_eps: float = 1e-5
    attn_drop: float = 0.0
    emb_drop: float = 0.0
    resid_drop: float = 0.0
    eos_idx: int = 126081
    eot_idx: int = 126348

    def llada_8b_it(use_fsdp: bool, use_tp: bool, dtype: DTypeLike = jnp.bfloat16, return_hidden_states: bool = False):
        return ModelConfig(
            dtype=dtype,
            d_model=4096,
            n_heads=32,
            n_kv_heads=32,
            n_layers=32,
            embedding_dropout=0.0,
            max_sequence_length=4096,
            rope_theta=500000.0,
            include_qkv_bias=False,
            include_bias=False,
            vocab_size=126464,
            embedding_size=126464,
            mlp_hidden_size=12288,
            shd_cfg=ShardingConfig.default(use_fsdp, use_tp),
            return_hidden_states=return_hidden_states,
        )


# TODO: Replace with nnx.Linear once explicit sharding is supported.
class ShardedLinear(nnx.Module):
    def __init__(
        self, in_dim: int, out_dim: int, *, use_bias: bool = True, kernel_sharding, bias_sharding, dtype=None, rngs
    ):
        kernel_initializer = jax.nn.initializers.lecun_normal()
        self.kernel = nnx.Param(
            kernel_initializer(rngs.params(), (in_dim, out_dim), dtype=dtype, out_sharding=kernel_sharding)
        )
        if use_bias:
            self.bias = nnx.Param(jnp.zeros((out_dim,), dtype=dtype, out_sharding=bias_sharding))
        else:
            self.bias = nnx.data(jnp.zeros((out_dim,), dtype=dtype, out_sharding=bias_sharding))

    def __call__(self, x, *, out_sharding):
        return jnp.matmul(x, self.kernel, out_sharding=out_sharding) + self.bias


# TODO: Replace with nnx.Embed once explicit sharding is supported.
class ShardedEmbedding(nnx.Embed):
    def __call__(self, inputs: Array, *, out_sharding) -> Array:
        # Modified from Flax NNX
        if not jnp.issubdtype(inputs.dtype, jnp.integer):
            raise ValueError("Input type must be an integer or unsigned integer.")
        (embedding,) = self.promote_dtype((self.embedding[...],), dtype=self.dtype, inexact=False)
        if self.num_embeddings == 1:
            return jnp.broadcast_to(embedding, (*inputs.shape, self.features))
        return embedding.at[inputs].get(out_sharding=out_sharding)

    def attend(self, query: Array, *, out_sharding) -> Array:
        query, embedding = self.promote_dtype(
            (
                query,
                self.embedding[...],
            ),
            dtype=self.dtype,
        )
        return jnp.dot(query, embedding.T, out_sharding=out_sharding)


def _generate_pos_embeddings(positions: Array, head_dim: int, rope_theta: int) -> tuple[Array, Array]:
    # Forked from: jax-llm-examples/qwen3/qwen3_jax/model.py;l=571
    fraction = jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim
    timescale = rope_theta**fraction
    rotational_frequency = 1.0 / timescale
    # Use high-precision einsum to prevent catastrophic bfloat16 rounding (ex: 257→256), as sin(257) differs from sin(256).
    sinusoid_inp = jnp.einsum("BT,k->BTk", positions, rotational_frequency, precision=jax.lax.Precision.HIGHEST)
    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


def apply_rope(x: Array, sin: Array, cos: Array) -> Array:
    assert x.ndim == 4 and sin.ndim == 3 and cos.ndim == 3
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    # [B, T, head_dim] -> [B, h, T, head_dim]
    sin, cos = sin[:, :, None, :], cos[:, :, None, :]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(x.dtype)


def count_left_pads(x: jax.Array) -> int:
    """Count left padding tokens."""
    return jnp.sum(jnp.cumsum(x != 0, axis=-1) == 0, -1)


def compute_positions_from_segment_ids(seg_ids: Array):
    return jax.vmap(lambda row: jnp.where(row != 0, jnp.arange(seg_ids.shape[1]) - jnp.argmax(row), 2**30))(seg_ids)


class RMSNorm(nnx.Module):
    def __init__(self, dim: int, eps: float, *, dtype: jnp.dtype, shd: PartitionSpec, rngs: nnx.Rngs):
        self.weight = nnx.Param(jax.nn.initializers.zeros(rngs.params(), dim, dtype=dtype, out_sharding=shd))
        self.eps = eps

    @jax.named_scope("rms_norm")
    def __call__(self, x: Array) -> Array:
        dtype = x.dtype
        xf32 = x.astype(jnp.float32)
        out = xf32 * jax.lax.rsqrt(jnp.square(xf32).mean(-1, keepdims=True) + self.eps)
        out = out * self.weight[...].astype(jnp.float32)
        return out.astype(dtype)


def sharded_attention(
    q,
    k,
    v,
    mask,
    scale=None,
    *,
    attn_logit_sharding: PartitionSpec,
    out_sharding: PartitionSpec,
    rate: float | None,
    key: Array,
):
    logits = jnp.einsum("BTNH,BSNH->BNTS", q, k, out_sharding=attn_logit_sharding)
    scale_val = (1.0 / jnp.sqrt(k.shape[-1])) if scale is None else scale
    logits *= jnp.array(scale_val, dtype=logits.dtype)

    if mask is None:
        probs = jax.nn.softmax(logits.astype(np.float32), axis=-1).astype(k.dtype)
    else:
        padded_logits = jnp.where(mask[:, None, :, :], logits.astype(np.float32), LARGE_NEGATIVE)
        probs = jax.nn.softmax(padded_logits, axis=-1).astype(k.dtype)

    if rate is not None and rate > 0.0:
        keep_prob = 1.0 - rate
        mask = jax.random.bernoulli(key, p=keep_prob, shape=probs.shape)
        probs = jax.lax.select(mask, probs / keep_prob, jnp.zeros_like(probs))

    attn_out = jnp.einsum("BNTS,BSNH->BTNH", probs, v, out_sharding=out_sharding)
    return attn_out


class LLaDALlamaBlock(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.config = cfg
        shd = cfg.shd_cfg
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps, dtype=jnp.float32, shd=shd.norm, rngs=rngs)
        self.ff_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps, dtype=jnp.float32, shd=shd.norm, rngs=rngs)

        self.dropout = nnx.Dropout(cfg.resid_drop)
        self.attn_drop = cfg.attn_drop

        # Attention
        self.head_dim = cfg.d_model // cfg.n_heads
        kv_proj_out_dim = cfg.n_kv_heads * self.head_dim
        attn_bias = cfg.include_bias | cfg.include_qkv_bias
        lin_kwargs = dict(
            use_bias=attn_bias, kernel_sharding=shd.attn_kernel, bias_sharding=shd.attn_bias, dtype=cfg.dtype
        )
        self.q_proj = ShardedLinear(cfg.d_model, cfg.d_model, rngs=rngs, **lin_kwargs)
        self.k_proj = ShardedLinear(cfg.d_model, kv_proj_out_dim, rngs=rngs, **lin_kwargs)
        self.v_proj = ShardedLinear(cfg.d_model, kv_proj_out_dim, rngs=rngs, **lin_kwargs)
        self.attn_out = ShardedLinear(cfg.d_model, cfg.d_model, rngs=rngs, **lin_kwargs)

        # MLPs
        hidden_size = cfg.mlp_hidden_size
        self.ff_proj = ShardedLinear(
            cfg.d_model,
            hidden_size,
            use_bias=cfg.include_bias,
            kernel_sharding=shd.fc1_kernel,
            bias_sharding=shd.fc1_bias,
            dtype=cfg.dtype,
            rngs=rngs,
        )
        self.up_proj = ShardedLinear(
            cfg.d_model,
            hidden_size,
            use_bias=cfg.include_bias,
            kernel_sharding=shd.fc1_kernel,
            bias_sharding=shd.fc1_bias,
            dtype=cfg.dtype,
            rngs=rngs,
        )
        self.ff_out = ShardedLinear(
            hidden_size,
            cfg.d_model,
            use_bias=cfg.include_bias,
            kernel_sharding=shd.fc2_kernel,
            bias_sharding=shd.fc2_bias,
            dtype=cfg.dtype,
            rngs=rngs,
        )

    def __call__(self, x: Array, sin: Array, cos: Array, attention_bias: Array | None, key: Array) -> Array:
        x_normed = self.attn_norm(x)
        new_shape = (*x.shape[:-1], -1, self.head_dim)
        shd = self.config.shd_cfg.activation
        q = self.q_proj(x_normed, out_sharding=shd).reshape(new_shape)
        k = self.k_proj(x_normed, out_sharding=shd).reshape(new_shape)
        v = self.v_proj(x_normed, out_sharding=shd).reshape(new_shape)

        q = apply_rope(q, sin, cos)
        k = apply_rope(k, sin, cos)

        intermediate_shd = self.config.shd_cfg.attn_qk_activation
        key, subkey = jax.random.split(key)
        attn = sharded_attention(
            q,
            k,
            v,
            attention_bias,
            attn_logit_sharding=intermediate_shd,
            out_sharding=shd,
            rate=self.attn_drop,
            key=subkey,
        )
        attn = self.attn_out(attn.reshape(x.shape), out_sharding=shd)

        key, subkey = jax.random.split(key)
        x = x + self.dropout(attn, rngs=subkey).reshape(x.shape)

        res = x
        x = self.ff_norm(x)
        x_ff, x_up = self.ff_proj(x, out_sharding=shd), self.up_proj(x, out_sharding=shd)
        x = jax.nn.silu(x_ff)
        x = x * x_up
        x = self.ff_out(x, out_sharding=shd)
        key, subkey = jax.random.split(key)
        x = self.dropout(x, rngs=subkey)
        x = res + x

        return x, key


class LLaDAModel(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.config = cfg
        shd = cfg.shd_cfg
        ei = partial(default_embed_init, out_sharding=cfg.shd_cfg.emb_kernel)
        self.wte = ShardedEmbedding(cfg.embedding_size, cfg.d_model, embedding_init=ei, rngs=rngs)

        self.emb_drop = nnx.Dropout(self.config.emb_drop)
        self.ln_f = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps, dtype=cfg.dtype, shd=shd.norm, rngs=rngs)
        self.blocks = nnx.List([LLaDALlamaBlock(cfg, rngs=rngs) for _ in range(cfg.n_layers)])

        self.ff_out = ShardedLinear(
            cfg.d_model,
            cfg.vocab_size,
            use_bias=cfg.include_bias,
            kernel_sharding=shd.fc1_kernel,
            bias_sharding=shd.fc1_bias,
            dtype=cfg.dtype,
            rngs=rngs,
        )

        self.head_dim = self.blocks[0].head_dim
        self.rope_theta = cfg.rope_theta

    def __call__(self, input_ids: Array, attention_mask: Array, key: Array) -> tuple[Array, Array, list[Array]]:
        shd = self.config.shd_cfg.activation
        left_pads = count_left_pads(attention_mask)
        start_ind = left_pads.reshape((-1, 1))
        position_ids = compute_positions_from_segment_ids(attention_mask) + start_ind
        sin, cos = _generate_pos_embeddings(position_ids, self.head_dim, self.rope_theta)

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, :]

        x = self.wte(input_ids, out_sharding=shd)

        key, subkey = jax.random.split(key)
        x = self.emb_drop(x, rngs=subkey)

        hiddden_states = []
        for b in self.blocks:
            if self.config.return_hidden_states:
                hiddden_states.append(x)
            x, key = b(x, sin, cos, attention_mask, key)
        x = self.ln_f(x)
        if self.config.return_hidden_states:
            hiddden_states.append(x)
        x = self.ff_out(x, out_sharding=shd)
        return x, key, hiddden_states


def add_gumbel_noise(logits: Array, temperature: float, key: Array) -> Array:
    if temperature == 0:
        return logits
    noise = jax.random.uniform(key, logits.shape, jnp.float32)
    gumbel_noise = jnp.pow(-jnp.log(noise), temperature)
    return (jnp.exp(logits.to(jnp.float32)) / gumbel_noise).to(logits.dtype)


def get_num_transfer_tokens(batch_size: int, block_length: int, steps_per_block: int):
    base_steps_per_iter = block_length // steps_per_block
    additional_steps = block_length % steps_per_block
    return np.full((batch_size, steps_per_block), base_steps_per_iter) + (np.arange(steps_per_block) < additional_steps)


def forward(model: LLaDAModel, x: Array, attn_mask: Array, key: Array) -> tuple[Array, Array, list[Array]]:
    return model(input_ids=x, attention_mask=attn_mask, key=key)


def generate(
    model: LLaDAModel,
    prompt: Array,
    attention_mask: Array | None = None,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    key: Array = jax.random.key(0),
):
    """
    Generates text multiple tokens at a time in blocks.

    model (LLaDAModel): The LLaDA model.
    prompt (Array): Tokenized prompt of shape (B, L) with left padding.
    attention_mask (Array | None): Attention mask of shape (B, L). Defaults to None.
        This is used to signal padding tokens.
    steps (int): Total number of model evaluations. Defaults to 128.
    gen_length (int): Desired number of tokens to generate.  Defaults to 128.
    block_length (int): Length of blocks used for generation.  Defaults to 128.
    temperature (float): Parameter controling randomness when using low_confidence remasking.  Defaults to 0.0.
    cfg_scale (float): Classifier-free guidance parameter. Defaults to 0.0.
        If 0.0 then just does a model evaluation.
        If > 0.0, then it uses classifier-free guidance.
    remasking (str): Remasking strategy. Defaults to "low_confidence".
        Can pick from "low_confidence" and "random".
    mask_id (int): Mask token id. Defaults to 126336.
    logits_eos_inf (bool): Boolean for preventing eos tokens. Defaults to False.
        If True, prevents eos tokens from being generated.
    confidence_eos_eot_inf (bool): Boolean for preventing eos and eot tokens. Defaults to False.
        If True, prevents eos and eot tokens from being generated.
    key
    """
    shd, cfg = model.config.shd_cfg.activation, model.config
    fsdp_enabled, tp_enabled = shd[0] is not None, shd[2] is not None

    # Functions to enable sharded computation or make it more efficient
    if fsdp_enabled:
        argsort = jax.shard_map(lambda x: jnp.argsort(x, axis=-1), in_specs=P(shd[0], None), out_specs=P(shd[0], None))
    else:
        argsort = lambda x: jnp.argsort(x, axis=-1)
    rank_fn = jax.jit(lambda x: argsort(argsort(x)))
    if tp_enabled:

        @jax.jit
        @partial(jax.shard_map, in_specs=(P(shd[0], None, shd[2]), P(shd[0], None)), out_specs=P(shd[0], None))
        def take_along_axis(p, x0):
            idx = jax.lax.axis_index(shd[2])
            shard_size = p.shape[-1]
            lower, upper = idx * shard_size, (idx + 1) * shard_size
            where_max = (x0 >= lower) & (x0 < upper)
            local_max = jnp.take_along_axis(p, x0[..., None] - lower, axis=-1).squeeze(-1)
            local_max = jnp.nan_to_num(local_max, nan=0.0)
            tmp = local_max * where_max
            return jax.lax.psum(tmp, axis_name=shd[2])
    else:
        take_along_axis = lambda p, x0: jnp.take_along_axis(p, x0[:, :, None], axis=-1).squeeze(-1)

    # Write mask_id into the generation part of the array
    x = jnp.concatenate([prompt, jnp.full((prompt.shape[0], gen_length), mask_id, dtype=jnp.int32)], axis=1)
    if fsdp_enabled:
        x = jax.device_put(x, P(ShardMode.FSDP.value))

    if attention_mask is not None:
        attention_mask = jnp.concat(
            [attention_mask, jnp.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype)], axis=-1
        )

    prompt_index = x != mask_id

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    # generate outputs in blocks instead of autoregressively
    for num_block in range(num_blocks):
        # This part is correct and much more simple
        num_transfer_tokens = get_num_transfer_tokens(x.shape[0], block_length, steps_per_block)

        # Do the generation for some number of steps
        for i in trange(steps_per_block):
            # Sample logits
            if cfg_scale > 0.0:
                # Batch inputs to run model twice
                un_x = x.at[prompt_index].set(mask_id)
                x_ = jnp.concat([x, un_x], dim=0)
                attention_mask_ = (
                    jnp.concat([attention_mask, attention_mask], dim=0) if attention_mask is not None else None
                )
                logits, key = forward(model, x_, attention_mask_, key)
                logits, un_logits = jnp.split(logits, 2, axis=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits, key, _ = forward(model, x, attention_mask, key)

            if logits_eos_inf:
                # Update eos logits to -inf so they are impossible
                logits, key, _ = logits.at[:, :, cfg.eos_idx].set(-jnp.inf)

            # Add noise to logits
            key, subkey = jax.random.split(key)
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature, key=subkey)
            x0 = jnp.argmax(logits_with_noise, axis=-1)

            if confidence_eos_eot_inf:
                # Update eos and eot logits so they are -inf
                logits = logits.at[:, :, cfg.eos_idx].set(-jnp.inf)
                logits = logits.at[:, :, cfg.eot_idx].set(-jnp.inf)

            # Compute remasking confidence for next step of diffusion
            if remasking == "low_confidence":
                p = jax.nn.softmax(logits, axis=-1)
                x0_p = take_along_axis(p, x0)
            elif remasking == "random":
                key, subkey = jax.random.split(key)
                x0_p = jax.random.uniform(subkey, x0.shape[:2])
            else:
                raise NotImplementedError(remasking)

            # Remove the future block tokens from this analysis
            x0_p = x0_p.at[:, prompt.shape[1] + (num_block + 1) * block_length :].set(-jnp.inf)

            # Only consider masked tokens
            mask_index = x == mask_id
            x0 = jnp.where(mask_index, x0, x)
            confidence = jnp.where(mask_index, x0_p, -jnp.inf)

            # Compute array ranks for topk mask
            ranks = rank_fn(confidence)
            rank_threshold = (confidence.shape[-1] - num_transfer_tokens[:, i])[:, None]
            topk_mask = ranks >= rank_threshold

            # Assign topk from x0 to x
            x = jnp.where(topk_mask, x0, x)

    return x
