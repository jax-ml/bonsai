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

import flax.nnx as nnx
import jax
import jax.numpy as jnp


class FPNNeck(nnx.Module):
    def __init__(
        self,
        position_encoding: nnx.Module,
        d_model: int,
        backbone_channel_list: list[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        fpn_interp_model: str = "bilinear",
        fuse_type: str = "sum",
        fpn_top_down_levels: list[int] | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.position_encoding = position_encoding
        self.backbone_channel_list = backbone_channel_list
        self.d_model = d_model
        self.convs = [
            nnx.Conv(
                c,
                d_model,
                kernel_size=(kernel_size, kernel_size),
                strides=stride,
                padding=padding,
                rngs=rngs,
            )
            for c in backbone_channel_list
        ]
        self.fpn_interp_model = fpn_interp_model
        self.fuse_type = fuse_type
        self.fpn_top_down_levels = list(range(len(self.convs))) if fpn_top_down_levels is None else fpn_top_down_levels

    def __call__(self, xs: list[jax.Array]) -> tuple[list[jax.Array], list[jax.Array]]:
        out = [None] * len(xs)
        pos = [None] * len(xs)

        prev_features = None
        n = len(xs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral = self.convs[n - i](x)

            if i in self.fpn_top_down_levels and prev_features is not None:
                prev_upsampled = jax.image.resize(
                    prev_features.astype(jnp.float32),
                    shape=lateral.shape,
                    method=self.fpn_interp_model,
                )
                fused = lateral + prev_upsampled
                if self.fuse_type == "avg":
                    fused = fused / 2
            else:
                fused = lateral

            out[i] = fused
            pos[i] = self.position_encoding(fused).astype(fused.dtype)
            prev_features = fused

        return out, pos


class ImageEncoder(nnx.Module):
    def __init__(
        self,
        trunk: nnx.Module,
        neck: nnx.Module,
        scalp: int = 0,
    ):
        self.trunk = trunk
        self.neck = neck
        self.scalp = scalp

    def __call__(self, image: jax.Array) -> dict:
        xs = self.trunk(image)
        features, pos = self.neck(xs)

        if self.scalp > 0:
            features = features[: -self.scalp]
            pos = pos[: -self.scalp]

        return {
            "vision_features": features[-1],
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }
