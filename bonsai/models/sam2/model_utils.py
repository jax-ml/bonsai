# Copyright 2025 Google LLC
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

from typing import Callable

import flax.nnx as nnx
import jax.numpy as jnp


def get_activation_fn(activation: str):
    if activation == "relu":
        return nnx.relu
    if activation == "gelu":
        return nnx.gelu
    if activation == "glu":
        return nnx.glu
    raise ValueError(f"activation should be relu/gelu, not {activation}.")


class Identity(nnx.Module):
    """A no-op layer that returns its input unchanged."""

    def __call__(self, x):
        return x


class MLP(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: Callable[[jnp.ndarray], jnp.ndarray] = nnx.relu,
        sigmoid_output: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        for i in range(num_layers):
            self.layers.append(nnx.Linear(dims[i], dims[i + 1], rngs=rngs))

        self.activation = activation
        self.sigmoid_output = sigmoid_output
        self.num_layers = num_layers

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = self.activation(x)

        if self.sigmoid_output:
            x = nnx.sigmoid(x)

        return x


class LayerNorm2d(nnx.Module):
    """
    PyTorch-style LayerNorm over channel dimension only.

    Args:
        num_channels: number of channels (C)
        eps: small constant for numerical stability
    Input:
        x: [B, H, W, C]
    Output:
        normalized: [B, H, W, C]
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        self.weight = nnx.Param(jnp.ones((num_channels,)))  # [C]
        self.bias = nnx.Param(jnp.zeros((num_channels,)))  # [C]
        self.eps = eps

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Mean and variance across channels (dim=1), shape [B, H, W, 1]
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        x_norm = (x - mean) / jnp.sqrt(var + self.eps)

        # Broadcast weight and bias: [C] → [1, 1, 1, C]
        weight = self.weight[None, None, None, :]
        bias = self.bias[None, None, None, :]
        return x_norm * weight + bias


def select_closest_cond_frames(frame_idx: int, cond_frame_outputs: dict, max_cond_frame_num: int) -> tuple[dict, dict]:
    """
    Pick up to `max_cond_frame_num` frames closest in time to `frame_idx`.

    Returns:
        selected: dict of selected {frame_idx: output}
        unselected: dict of the rest
    """
    # if unlimited or small enough, take all
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        return cond_frame_outputs, {}

    assert max_cond_frame_num >= 2, "max_cond_frame_num must be >= 2 for subsampling"
    selected = {}
    # closest before
    idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
    if idx_before is not None:
        selected[idx_before] = cond_frame_outputs[idx_before]
    # closest after
    idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
    if idx_after is not None:
        selected[idx_after] = cond_frame_outputs[idx_after]
    # add others by temporal distance
    remaining = max_cond_frame_num - len(selected)
    if remaining > 0:
        # sort by abs distance
        others = sorted(
            (t for t in cond_frame_outputs if t not in selected),
            key=lambda t: abs(t - frame_idx),
        )[:remaining]
        for t in others:
            selected[t] = cond_frame_outputs[t]
    # compute unselected
    unselected = {t: v for t, v in cond_frame_outputs.items() if t not in selected}
    return selected, unselected


class DropPath(nnx.Module):
    def __init__(
        self,
        drop_prob: float,
        scale_by_keep: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        self.drop_prob = drop_prob
        self.dropout = nnx.Dropout(
            rate=self.drop_prob,
            broadcast_dims=(1, 2, 3),
        )

        self.scale_by_keep = scale_by_keep
        self.rngs = rngs

    def __call__(self, x: jnp.ndarray, *, deterministic=None) -> jnp.ndarray:
        if self.drop_prob == 0.0 or deterministic:
            return x

        # Use nnx.Dropout with broadcast_dims = all except batch (dim 0)
        out = self.dropout(x)
        if not self.scale_by_keep and self.drop_prob < 1.0:
            # Dropout always scales by 1 / keep_prob, undo that
            return out * (1.0 - self.drop_prob)
        return out
