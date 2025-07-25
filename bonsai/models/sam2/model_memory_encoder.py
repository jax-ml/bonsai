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

import math
from typing import Any, Callable

import flax.nnx as nnx
import jax.numpy as jnp

from bonsai.models.sam2.model_utils import DropPath, Identity, LayerNorm2d


class MaskDownSampler(nnx.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        kernel_size: int = 4,
        stride: int = 4,
        padding: int = 0,
        total_stride: int = 16,
        activation: Callable = nnx.gelu,
        *,
        rngs: nnx.Rngs,
    ):
        num_layers = int(math.log(total_stride) / math.log(stride))
        assert stride**num_layers == total_stride

        layers = []
        in_ch = 1
        for _ in range(num_layers):
            out_ch = in_ch * (stride**2)
            layers.append(
                nnx.Conv(
                    in_ch,
                    out_ch,
                    kernel_size=(kernel_size, kernel_size),
                    strides=stride,
                    padding=padding,
                    rngs=rngs,
                )
            )
            layers.append(LayerNorm2d(out_ch))
            layers.append(activation)
            in_ch = out_ch

        layers.append(nnx.Conv(in_ch, embed_dim, kernel_size=(1, 1), strides=1, padding=0, rngs=rngs))
        self.encoder = nnx.Sequential(*layers)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.encoder(x)


class ConvNextBlock(nnx.Module):
    """
    ConvNext Block (https://github.com/facebookresearch/ConvNeXt)
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        padding: int = 3,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        use_dwconv: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        groups = dim if use_dwconv else 1
        self.dwconv = nnx.Conv(
            dim,
            dim,
            kernel_size=(kernel_size, kernel_size),
            strides=1,
            padding=padding,
            feature_group_count=groups,
            rngs=rngs,
        )
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nnx.Linear(dim, 4 * dim, rngs=rngs)
        self.act = nnx.gelu
        self.pwconv2 = nnx.Linear(4 * dim, dim, rngs=rngs)

        if layer_scale_init_value > 0:
            init = jnp.full((dim,), layer_scale_init_value)
            self.gamma = nnx.Param(init)
        else:
            self.gamma = None

        self.drop_path = DropPath(drop_prob=drop_path_rate, rngs=rngs) if drop_path_rate > 0 else Identity()

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        # to channels-last for MLP
        x = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x * self.gamma  # broadcast over spatial dims
        # back to channels-first
        x = jnp.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
        return residual + self.drop_path(x)


class Fuser(nnx.Module):
    def __init__(
        self,
        layer: nnx.Module,
        num_layers: int,
        dim: int | None = None,
        input_projection: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        if input_projection:
            assert dim is not None
            self.proj = nnx.Conv(dim, dim, kernel_size=1, strides=1, padding=0, rngs=rngs)
        else:
            self.proj = Identity()

        # clone layers
        self.layers = [nnx.clone(layer) for _ in range(num_layers)]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.proj(x)
        for lyr in self.layers:
            x = lyr(x)
        return x


class MemoryEncoder(nnx.Module):
    def __init__(
        self,
        out_dim: int,
        mask_downsampler: MaskDownSampler,
        fuser: Fuser,
        position_encoding: Callable,
        in_dim: int = 256,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.mask_downsampler = mask_downsampler
        self.pix_feat_proj = nnx.Conv(in_dim, in_dim, kernel_size=(1, 1), strides=1, padding=0, rngs=rngs)
        self.fuser = fuser
        self.position_encoding = position_encoding
        if out_dim != in_dim:
            self.out_proj = nnx.Conv(in_dim, out_dim, kernel_size=(1, 1), strides=1, padding=0, rngs=rngs)
        else:
            self.out_proj = Identity()

    def __call__(
        self,
        pix_feat: jnp.ndarray,
        masks: jnp.ndarray,
        skip_mask_sigmoid: bool = False,
    ) -> dict[str, Any]:
        if not skip_mask_sigmoid:
            masks = nnx.sigmoid(masks)
        masks = self.mask_downsampler(masks)

        # project and fuse
        x = self.pix_feat_proj(pix_feat.astype(masks.dtype))
        x = x + masks
        x = self.fuser(x)
        x = self.out_proj(x)

        pos = self.position_encoding(x).astype(x.dtype)
        return {"vision_features": x, "vision_pos_enc": [pos]}
