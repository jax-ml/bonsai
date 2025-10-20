# Copyright 2025 The JAX Authors.
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

from functools import partial
from typing import Callable

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from bonsai.models.sam2.model_utils import MLP, DropPath, Identity


def window_partition(x: jnp.ndarray, window_size: int) -> tuple[jnp.ndarray, tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x: input array of shape [B, H, W, C]
        window_size: size of each window
    Returns:
        windows: [B * num_windows, window_size, window_size, C]
        (Hp, Wp): padded height and width
    """
    B, H, W, C = x.shape
    pad_h = (window_size - (H % window_size)) % window_size
    pad_w = (window_size - (W % window_size)) % window_size
    if pad_h or pad_w:
        x = jnp.pad(x, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)))
    Hp, Wp = H + pad_h, W + pad_w
    # reshape and permute to windows
    # x: (B, Hp, Wp, C) -> (B, Hp // Ws, Ws, Wp // Ws, Ws, C)
    x = x.reshape(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = jnp.transpose(x, (0, 1, 3, 2, 4, 5)).reshape(
        -1, window_size, window_size, C
    )  # windows: (B * (Hp // Ws) * (Wp // Ws), Ws, Ws, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: jnp.ndarray,
    window_size: int,
    pad_hw: tuple[int, int],
    hw: tuple[int, int],
) -> jnp.ndarray:
    """
    Reconstruct original array from windows, removing padding.
    Args:
        windows: [B * num_windows, window_size, window_size, C]
        window_size: size of each window
        pad_hw: (Hp, Wp) padded height and width
        hw: (H, W) original height and width
    Returns:
        x: [B, H, W, C]
    """
    Hp, Wp = pad_hw
    H, W = hw
    num_windows = (Hp // window_size) * (Wp // window_size)
    B = windows.shape[0] // num_windows
    # reshape and permute back
    x = windows.reshape(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = jnp.transpose(x, (0, 1, 3, 2, 4, 5)).reshape(B, Hp, Wp, -1)
    # remove padding if any
    pad_h = Hp - H
    pad_w = Wp - W
    if pad_h or pad_w:
        x = x[:, :H, :W, :]
    return x


class PatchEmbed(nnx.Module):
    """
    Image to patch embedding using convolution.
    Converts [B, H, W, C] -> [B, H', W', embed_dim]
    """

    def __init__(
        self,
        kernel_size: tuple[int, ...] = (7, 7),
        stride: tuple[int, ...] = (4, 4),
        padding: tuple[int, ...] = (3, 3),
        in_chans: int = 3,
        embed_dim: int = 768,
        *,
        rngs: nnx.Rngs,
    ):
        self.proj = nnx.Conv(in_chans, embed_dim, kernel_size=kernel_size, strides=stride, padding=padding, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.proj(x)
        return x


def do_pool(x: jnp.ndarray, pool: nnx.Module, norm: nnx.Module | None = None) -> jnp.ndarray:
    if pool is None:
        return x
    x = pool(x)
    if norm:
        x = norm(x)
    return x


class MultiScaleAttention(nnx.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nnx.Module | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.q_pool = q_pool
        # projection layers
        self.qkv = nnx.Linear(dim, dim_out * 3, rngs=rngs)
        self.proj = nnx.Linear(dim_out, dim_out, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, H, W, _ = x.shape
        # linear to qkv and reshape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        q, k, v = jnp.split(qkv, 3, axis=2)
        q = q.squeeze(2)
        k = k.squeeze(2)
        v = v.squeeze(2)

        # optional Q pooling
        if self.q_pool:
            q = q.reshape(B, H, W, -1)
            q = do_pool(q, self.q_pool)
            H, W = q.shape[1], q.shape[2]
            q = q.reshape(B, H * W, self.num_heads, -1)

        attn = nnx.dot_product_attention(q, k, v)

        # [B, H*W, nheads, C] -> [B, H, W, C]
        attn = attn.reshape(B, H, W, -1)

        return self.proj(attn)


class MultiScaleBlock(nnx.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        q_stride: tuple[int, ...] | None = None,
        act_layer: Callable = nnx.gelu,
        window_size: int = 0,
        *,
        rngs: nnx.Rngs,
    ):
        self.dim = dim
        self.dim_out = dim_out
        self.window_size = window_size
        self.q_stride = q_stride

        self.norm1 = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim_out, epsilon=1e-6, rngs=rngs)

        self.pool = None
        if q_stride is not None:
            self.pool = partial(nnx.max_pool, window_shape=q_stride, strides=q_stride)

        self.attn = MultiScaleAttention(dim=dim, dim_out=dim_out, num_heads=num_heads, q_pool=self.pool, rngs=rngs)

        self.drop_path = DropPath(drop_prob=drop_path, rngs=rngs) if drop_path > 0.0 else Identity()

        self.mlp = MLP(dim_out, int(dim_out * mlp_ratio), dim_out, num_layers=2, activation=act_layer, rngs=rngs)

        self.proj = nnx.Linear(dim, dim_out, rngs=rngs) if dim != dim_out else Identity()

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        shortcut = x
        x = self.norm1(x)

        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1:3]
            x, pad_hw = window_partition(x, window_size)
        x = self.attn(x)

        if self.q_stride:
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]
            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Hiera(nnx.Module):
    """
    Reference: https://arxiv.org/abs/2306.00989
    """

    def __init__(
        self,
        embed_dim: int = 96,
        num_heads: int = 1,
        drop_path_rate: float = 0.0,
        q_pool: int = 3,
        q_stride: tuple[int, ...] = (2, 2),
        stages: tuple[int, ...] = (2, 3, 16, 3),
        dim_mul: float = 2.0,
        head_mul: float = 2.0,
        window_pos_embed_bkg_spatial_size: tuple[int, ...] = (14, 14),
        window_spec: tuple[int, ...] = (8, 4, 14, 7),
        global_att_blocks: tuple[int, ...] = (12, 16, 20),
        weight_path=None,
        return_interm_layers: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        if len(stages) != len(window_spec):
            raise ValueError("`stages` and `window_sizes` must have same length.")

        self.window_spec = window_spec
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers

        self.patch_embed = PatchEmbed(embed_dim=embed_dim, rngs=rngs)

        self.global_att_blocks = global_att_blocks

        # Positional embeddings
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nnx.Param(jnp.zeros((1, embed_dim, *self.window_pos_embed_bkg_spatial_size)))
        self.pos_embed_window = nnx.Param(jnp.zeros((1, embed_dim, window_spec[0], window_spec[0])))

        depth = sum(stages)
        dpr = np.linspace(0, drop_path_rate, depth)

        cur_stage = 1
        self.blocks = []

        for i in range(depth):
            dim_out = embed_dim
            window_size = window_spec[cur_stage - 1]

            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=float(dpr[i]),
                q_stride=q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
                rngs=rngs,
            )
            self.blocks.append(block)
            embed_dim = dim_out

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

    def _get_pos_embed(self, hw: tuple[int, int]) -> jnp.ndarray:
        h, w = hw
        pos_embed = jax.image.resize(
            self.pos_embed.value, shape=(1, self.pos_embed.value.shape[1], h, w), method="bicubic"
        )

        tile_factors = [1, h // self.pos_embed_window.value.shape[2], w // self.pos_embed_window.value.shape[3]]
        window_embed = jnp.tile(self.pos_embed_window.value, tile_factors)
        pos_embed = pos_embed + window_embed
        return jnp.transpose(pos_embed, (0, 2, 3, 1))  # BCHW -> BHWC

    def __call__(self, x: jnp.ndarray):
        x = self.patch_embed(x)  # (B, H, W, C)
        x = x + self._get_pos_embed(x.shape[1:3])

        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i == self.stage_ends[-1]) or (i in self.stage_ends and self.return_interm_layers):
                feats = x  # jnp.transpose(x, (0, 3, 1, 2))  # BHWC → BCHW
                outputs.append(feats)
        return outputs
