# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mamba2 JAX/Flax NNX Implementation.

A pure JAX/Flax implementation of the Mamba2 architecture using the State Space Duality (SSD) mechanism.
Reference: "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality"
Paper: https://arxiv.org/abs/2405.21060
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import jax
import jax.numpy as jnp
import optax
from flax import nnx

# Configuration


@dataclasses.dataclass(frozen=True)
class Mamba2Config:
    """Configuration for Mamba2 models."""

    vocab_size: int = 50280
    pad_token_id: int = 0
    bos_token_id: int = 0
    eos_token_id: int = 0

    hidden_size: int = 768
    state_size: int = 128
    head_dim: int = 64
    chunk_size: int = 256
    expand: int = 2
    conv_kernel: int = 4
    num_hidden_layers: int = 24
    layer_norm_epsilon: float = 1e-5

    use_bias: bool = False
    use_conv_bias: bool = True
    hidden_act: Literal["silu", "gelu", "relu", "tanh"] = "silu"

    emb_initializer_range: float = 0.02
    A_initializer_range: tuple[float, float] = (1.0, 16.0)

    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_floor: float = 1e-4
    time_step_limit: tuple[float, float] = (0.0, float("inf"))

    residual_in_fp32: bool = True
    tie_word_embeddings: bool = True

    @property
    def intermediate_size(self) -> int:
        return int(self.expand * self.hidden_size)

    @property
    def num_heads(self) -> int:
        return self.intermediate_size // self.head_dim

    @classmethod
    def tiny(cls):
        """Tiny configuration for testing."""
        return cls(vocab_size=1000, hidden_size=64, state_size=16, head_dim=16, chunk_size=32, num_hidden_layers=2)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class Mamba2Cache:
    """Cache for Mamba2 SSM and convolution states."""

    ssm_states: list[jnp.ndarray]  # (batch, heads, head_dim, state_size) per layer
    conv_states: list[jnp.ndarray]  # (batch, conv_dim, kernel_size - 1) per layer

    def tree_flatten(self):
        return (self.ssm_states, self.conv_states), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(ssm_states=list(children[0]), conv_states=list(children[1]))


# SSD Core Algorithm


def _pad_seq_dim(x: jnp.ndarray, pad_size: int) -> jnp.ndarray:
    """Pad zeros at the end of the sequence dimension (axis=1)."""
    if pad_size == 0:
        return x
    pad_width = [(0, 0)] * x.ndim
    pad_width[1] = (0, pad_size)
    return jnp.pad(x, pad_width, mode="constant", constant_values=0.0)


def segsum(x: jnp.ndarray) -> jnp.ndarray:
    """Stable segment sum calculation. Input: (..., T) -> Output: (..., T, T)."""
    T = x.shape[-1]
    x_cumsum = jnp.cumsum(x, axis=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = jnp.tril(jnp.ones((T, T), dtype=bool), k=0)
    x_segsum = jnp.where(mask, x_segsum, -jnp.inf)
    return x_segsum


def ssd_forward(
    x: jnp.ndarray,  # (B, L, H, P)
    dt: jnp.ndarray,  # (B, L, H)
    A: jnp.ndarray,  # (H,)
    B_mat: jnp.ndarray,  # (B, L, H, N)
    C_mat: jnp.ndarray,  # (B, L, H, N)
    chunk_size: int,
    D: jnp.ndarray,  # (H,)
    dt_bias: jnp.ndarray,  # (H,)
    dt_min: float,
    dt_max: float,
    initial_states: jnp.ndarray | None = None,
    return_final_states: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """SSD (State Space Duality) forward pass with chunked computation.

    Args:
        x: Input tensor (batch_size, seq_len, num_heads, head_dim)
        dt: Time deltas (batch_size, seq_len, num_heads)
        A: State transition scalar per head (num_heads,)
        B_mat: Input-to-state matrix (batch_size, seq_len, num_heads, state_size)
        C_mat: State-to-output matrix (batch_size, seq_len, num_heads, state_size)
        chunk_size: Size of chunks for efficient computation
        D: Skip connection weights (num_heads,)
        dt_bias: Bias for time deltas (num_heads,)
        dt_min: Minimum time delta after clamping
        dt_max: Maximum time delta after clamping
        initial_states: Optional initial SSM states (batch, 1, heads, head_dim, state_size)
        return_final_states: Whether to return final SSM states

    Returns:
        y: Output tensor (batch_size, seq_len, num_heads, head_dim)
        final_state: Optional final states (batch_size, num_heads, head_dim, state_size)
    """
    _B_size, seq_len, num_heads, _head_dim = x.shape
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size

    # Apply dt bias with softplus and clamp
    dt = jax.nn.softplus(dt + dt_bias)
    dt = jnp.clip(dt, dt_min, dt_max)

    # Pad tensors along sequence dimension
    x_padded = _pad_seq_dim(x, pad_size)
    dt_padded = _pad_seq_dim(dt, pad_size)
    B_padded = _pad_seq_dim(B_mat, pad_size)
    C_padded = _pad_seq_dim(C_mat, pad_size)

    # D residual connection
    D_residual = D.reshape(1, 1, num_heads, 1) * x_padded

    # Discretize x and A
    x_disc = x_padded * dt_padded[..., None]
    A_disc = A.astype(x_disc.dtype) * dt_padded

    # Chunk everything
    def chunk_tensor(t):
        b, cl, *remaining = t.shape
        return t.reshape(b, cl // chunk_size, chunk_size, *remaining)

    x_blk = chunk_tensor(x_disc)
    A_blk = chunk_tensor(A_disc)
    B_blk = chunk_tensor(B_padded)
    C_blk = chunk_tensor(C_padded)

    # A cumsum over intra-chunk time dimension
    A_blk2 = jnp.transpose(A_blk, (0, 3, 1, 2))
    A_cumsum = jnp.cumsum(A_blk2, axis=-1)

    # 1. Intra-chunk (diagonal blocks)
    L_mat = jnp.exp(segsum(A_blk2))
    Y_diag = jnp.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C_blk, B_blk, L_mat, x_blk)

    # 2. States within each chunk
    decay_states = jnp.exp(A_cumsum[..., -1:] - A_cumsum)
    states = jnp.einsum("bclhn,bhcl,bclhp->bchpn", B_blk, decay_states, x_blk)

    # 3. Inter-chunk recurrence
    if initial_states is None:
        initial_states = jnp.zeros_like(states[:, :1, ...])
    states = jnp.concatenate([initial_states, states], axis=1)

    A_end = A_cumsum[..., -1]
    A_end_padded = jnp.pad(A_end, ((0, 0), (0, 0), (1, 0)))
    decay_chunk = jnp.exp(segsum(A_end_padded))
    new_states = jnp.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1, ...], new_states[:, -1, ...]

    # 4. Convert states -> outputs
    state_decay_out = jnp.exp(A_cumsum)
    Y_off = jnp.einsum("bclhn,bchpn,bhcl->bclhp", C_blk, states, state_decay_out)

    y = Y_diag + Y_off
    b, c, l, h, p = y.shape
    y = y.reshape(b, c * l, h, p)
    y = y + D_residual

    # Remove padding
    if pad_size > 0:
        y = y[:, :seq_len, :, :]

    return (y, final_state) if return_final_states else (y, None)


# Model Components
ACT2FN = {"silu": nnx.silu, "gelu": nnx.gelu, "relu": nnx.relu, "tanh": jnp.tanh}


class RMSNorm(nnx.Module):
    """RMSNorm with optional residual gating."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, gate_residual: bool = False, *, rngs: nnx.Rngs):
        self.hidden_size = hidden_size
        self.eps = eps
        self.gate_residual = gate_residual
        self.weight = nnx.Param(jnp.ones((hidden_size,)))

    @jax.named_scope("rms_norm")
    def __call__(self, hidden_states: jnp.ndarray, residual: jnp.ndarray | None = None) -> jnp.ndarray:
        x = hidden_states.astype(jnp.float32)
        if residual is not None and self.gate_residual:
            x = x * nnx.silu(residual.astype(jnp.float32))
        variance = jnp.mean(x**2, axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.eps) * self.weight[:]
        return x.astype(hidden_states.dtype)


class DepthwiseConv1d(nnx.Module):
    """Depthwise causal 1D convolution with state caching. Expects (batch, seq_len, channels)."""

    def __init__(self, features: int, kernel_size: int, use_bias: bool = True, *, rngs: nnx.Rngs):
        self.features = features
        self.kernel_size = kernel_size
        self.conv = nnx.Conv(
            in_features=features,
            out_features=features,
            kernel_size=(kernel_size,),
            padding=((0, 0),),
            feature_group_count=features,
            use_bias=use_bias,
            rngs=rngs,
        )

    @jax.named_scope("depthwise_conv1d")
    def __call__(self, x: jnp.ndarray, conv_state: jnp.ndarray | None = None) -> tuple[jnp.ndarray, jnp.ndarray]:
        cache_len = self.kernel_size - 1

        if conv_state is None:
            x_padded = jnp.pad(x, ((0, 0), (cache_len, 0), (0, 0)), mode="constant", constant_values=0.0)
        else:
            x_padded = jnp.concatenate([jnp.transpose(conv_state, (0, 2, 1)), x], axis=1)

        output = self.conv(x_padded)
        new_conv_state = jnp.transpose(x_padded[:, -cache_len:, :], (0, 2, 1))

        return output, new_conv_state


class Mamba2Mixer(nnx.Module):
    """Mamba2 mixer block using the SSD algorithm."""

    def __init__(self, cfg: Mamba2Config, layer_idx: int, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.hidden_size = cfg.hidden_size
        self.ssm_state_size = cfg.state_size
        self.intermediate_size = cfg.intermediate_size
        self.head_dim = cfg.head_dim
        self.num_heads = cfg.num_heads
        self.chunk_size = cfg.chunk_size
        self.dt_min, self.dt_max = cfg.time_step_limit
        self.act = ACT2FN[cfg.hidden_act]

        # Input projection
        proj_size = 2 * (self.intermediate_size + self.ssm_state_size) + self.num_heads
        self.in_proj = nnx.Linear(cfg.hidden_size, proj_size, use_bias=cfg.use_bias, rngs=rngs)

        # Depthwise conv
        conv1d_dim = self.intermediate_size + 2 * self.ssm_state_size
        self.conv1d = DepthwiseConv1d(conv1d_dim, cfg.conv_kernel, use_bias=cfg.use_conv_bias, rngs=rngs)

        # SSM parameters
        key = rngs.params()
        low, high = cfg.time_step_min, cfg.time_step_max
        floor = cfg.time_step_floor
        dt_init = jnp.exp(jax.random.uniform(key, (cfg.num_heads,)) * (jnp.log(high) - jnp.log(low)) + jnp.log(low))
        dt_init = jnp.maximum(dt_init, floor)
        self.dt_bias = nnx.Param(dt_init + jnp.log(-jnp.expm1(-dt_init)))  # inverse softplus

        key = rngs.params()
        A_low, A_high = cfg.A_initializer_range
        A_init = jax.random.uniform(key, (cfg.num_heads,), minval=A_low, maxval=A_high)
        self.A_log = nnx.Param(jnp.log(A_init))

        self.D = nnx.Param(jnp.ones((cfg.num_heads,)))

        # Internal norm and output projection
        self.norm = RMSNorm(self.intermediate_size, eps=1e-5, gate_residual=True, rngs=rngs)
        self.out_proj = nnx.Linear(self.intermediate_size, cfg.hidden_size, use_bias=cfg.use_bias, rngs=rngs)

    @jax.named_scope("mamba2_mixer")
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        conv_state: jnp.ndarray | None = None,
        ssm_state: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        B_size, L, _ = hidden_states.shape

        # 1) Parallel projection
        zxbcdt = self.in_proj(hidden_states)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.intermediate_size - 2 * self.ssm_state_size - self.num_heads) // 2

        z0, x0, z, xBC, dt = jnp.split(
            zxbcdt,
            [
                d_mlp,
                2 * d_mlp,
                2 * d_mlp + self.intermediate_size,
                2 * d_mlp + self.intermediate_size + self.intermediate_size + 2 * self.ssm_state_size,
            ],
            axis=-1,
        )

        # 2) Depthwise causal convolution with state caching
        xBC, new_conv_state = self.conv1d(xBC, conv_state=conv_state)
        xBC = self.act(xBC)
        x, B_t, C_t = jnp.split(xBC, [self.intermediate_size, self.intermediate_size + self.ssm_state_size], axis=-1)

        # 3) SSD forward with state caching
        init_state = ssm_state[:, None, ...] if ssm_state is not None else None
        A = -jnp.exp(self.A_log[:].astype(jnp.float32))

        B_exp = jnp.broadcast_to(jnp.expand_dims(B_t, 2), (B_size, L, self.num_heads, self.ssm_state_size))
        C_exp = jnp.broadcast_to(jnp.expand_dims(C_t, 2), (B_size, L, self.num_heads, self.ssm_state_size))

        y, new_ssm_state = ssd_forward(
            x=x.reshape(B_size, L, -1, self.head_dim),
            dt=dt,
            A=A,
            B_mat=B_exp,
            C_mat=C_exp,
            chunk_size=self.chunk_size,
            D=self.D[:],
            dt_bias=self.dt_bias[:],
            dt_min=self.dt_min,
            dt_max=self.dt_max,
            initial_states=init_state,
            return_final_states=True,
        )
        y = y.reshape(B_size, L, -1)

        # 4) Residual gate normalization
        y = self.norm(y, residual=z)
        if d_mlp > 0:
            y = jnp.concatenate([self.act(z0) * x0, y], axis=-1)

        # 5) Output projection
        return self.out_proj(y), new_conv_state, new_ssm_state


class Mamba2Block(nnx.Module):
    """Single Mamba2 block with pre-norm and residual connection."""

    def __init__(self, cfg: Mamba2Config, layer_idx: int, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.residual_in_fp32 = cfg.residual_in_fp32
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.layer_norm_epsilon, rngs=rngs)
        self.mixer = Mamba2Mixer(cfg, layer_idx=layer_idx, rngs=rngs)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        conv_state: jnp.ndarray | None = None,
        ssm_state: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        residual = hidden_states
        hs = self.norm(hidden_states.astype(jnp.float32))
        if self.residual_in_fp32:
            residual = residual.astype(jnp.float32)
        hs_out, new_conv_state, new_ssm_state = self.mixer(hs, conv_state=conv_state, ssm_state=ssm_state)
        return residual + hs_out, new_conv_state, new_ssm_state


class Mamba2Model(nnx.Module):
    """Mamba2 backbone model (no task-specific head)."""

    def __init__(self, cfg: Mamba2Config, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.embedder = nnx.Embed(num_embeddings=cfg.vocab_size, features=cfg.hidden_size, rngs=rngs)
        self.layers = nnx.List([Mamba2Block(cfg, layer_idx=i, rngs=rngs) for i in range(cfg.num_hidden_layers)])
        self.final_norm = RMSNorm(cfg.hidden_size, eps=cfg.layer_norm_epsilon, rngs=rngs)

    @jax.named_scope("mamba2_backbone")
    def __call__(
        self,
        input_ids: jnp.ndarray | None = None,
        inputs_embeds: jnp.ndarray | None = None,
        cache: Mamba2Cache | None = None,
        output_hidden_states: bool = False,
    ) -> dict[str, jnp.ndarray | Mamba2Cache | list[jnp.ndarray] | None]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        hidden_states = self.embedder(input_ids) if inputs_embeds is None else inputs_embeds

        # Extract per-layer states from cache or initialize
        if cache is None:
            conv_states = [None] * self.cfg.num_hidden_layers
            ssm_states = [None] * self.cfg.num_hidden_layers
        else:
            if len(cache.conv_states) != self.cfg.num_hidden_layers:
                raise ValueError("cache.conv_states length must equal num_hidden_layers")
            if len(cache.ssm_states) != self.cfg.num_hidden_layers:
                raise ValueError("cache.ssm_states length must equal num_hidden_layers")
            conv_states = cache.conv_states
            ssm_states = cache.ssm_states

        all_hidden_states = [] if output_hidden_states else None
        new_conv_states = []
        new_ssm_states = []

        for layer, conv_state, ssm_state in zip(self.layers, conv_states, ssm_states):
            hidden_states, new_conv_state, new_ssm_state = layer(
                hidden_states, conv_state=conv_state, ssm_state=ssm_state
            )
            new_conv_states.append(new_conv_state)
            new_ssm_states.append(new_ssm_state)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        hidden_states = self.final_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        # Build updated cache
        updated_cache = Mamba2Cache(ssm_states=new_ssm_states, conv_states=new_conv_states)

        return {
            "last_hidden_state": hidden_states,
            "cache": updated_cache,
            "hidden_states": all_hidden_states,
        }


class Mamba2ForCausalLM(nnx.Module):
    """Mamba2 model with causal language modeling head."""

    def __init__(self, cfg: Mamba2Config, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.backbone = Mamba2Model(cfg, rngs=rngs)
        if not cfg.tie_word_embeddings:
            self.lm_head = nnx.Linear(cfg.hidden_size, cfg.vocab_size, use_bias=False, rngs=rngs)
        else:
            self.lm_head = None

    @jax.named_scope("mamba2_causal_lm")
    def __call__(
        self,
        input_ids: jnp.ndarray,
        labels: jnp.ndarray | None = None,
        cache: Mamba2Cache | None = None,
    ) -> dict[str, jnp.ndarray | Mamba2Cache | None]:
        backbone_outputs = self.backbone(input_ids=input_ids, cache=cache)
        hidden_states = backbone_outputs["last_hidden_state"]

        if self.cfg.tie_word_embeddings:
            logits = hidden_states @ self.backbone.embedder.embedding[:].T
        else:
            logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
            shift_labels = labels[:, 1:].reshape(-1)
            loss = optax.softmax_cross_entropy_with_integer_labels(shift_logits, shift_labels).mean()

        return {"logits": logits, "loss": loss, "cache": backbone_outputs["cache"]}

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str,
        *,
        cfg: Mamba2Config | None = None,
        dtype: jnp.dtype = jnp.float32,
        seed: int = 0,
        revision: str = "main",
    ) -> "Mamba2ForCausalLM":
        # Local import to avoid hard dependency cycles
        from bonsai.models.mamba2 import params as mamba2_params

        # If cfg is None, params.create_model_from_huggingface already infers it.
        if "/" in model_id_or_path and not model_id_or_path.startswith((".", "/")):
            return mamba2_params.create_model_from_huggingface(
                model_id_or_path, cfg=cfg, dtype=dtype, seed=seed, revision=revision
            )

        return mamba2_params.create_model_from_torch_checkpoint(model_id_or_path, cfg=cfg, dtype=dtype, seed=seed)


class Mamba2Forecaster(nnx.Module):
    """Mamba2-based time series forecaster."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 768,
        n_layers: int = 4,
        output_dim: int = 1,
        forecast_horizon: int = 24,
        d_state: int = 128,
        headdim: int = 64,
        d_conv: int = 4,
        chunk_size: int = 256,
        *,
        rngs: nnx.Rngs,
    ):
        self.forecast_horizon = forecast_horizon
        self.output_dim = output_dim

        self.input_proj = nnx.Linear(input_dim, d_model, rngs=rngs)
        cfg = Mamba2Config(
            vocab_size=1,
            hidden_size=d_model,
            state_size=d_state,
            head_dim=headdim,
            conv_kernel=d_conv,
            chunk_size=chunk_size,
            num_hidden_layers=n_layers,
        )
        self.mamba2 = Mamba2Model(cfg, rngs=rngs)
        self.output_proj = nnx.Linear(d_model, output_dim * forecast_horizon, rngs=rngs)

    @jax.named_scope("mamba2_forecaster")
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass. Input: (batch, seq_len, input_dim) -> Output: (batch, forecast_horizon, output_dim)."""
        x_proj = self.input_proj(x)
        outputs = self.mamba2(input_ids=None, inputs_embeds=x_proj)
        last_hidden = outputs["last_hidden_state"][:, -1, :]
        out = self.output_proj(last_hidden)
        return out.reshape(x.shape[0], self.forecast_horizon, self.output_dim)


@jax.jit
def forward(
    model: Mamba2ForCausalLM,
    input_ids: jnp.ndarray,
    labels: jnp.ndarray | None = None,
    cache: Mamba2Cache | None = None,
):
    """JIT-compiled forward pass for Mamba2ForCausalLM with optional caching."""
    return model(input_ids, labels, cache)
