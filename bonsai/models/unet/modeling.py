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

import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx


@dataclasses.dataclass(frozen=True)
class ModelCfg:
    """Configuration for the U-Net model."""

    in_channels: int = 1
    num_classes: int = 2
    features: tuple[int, int, int, int, int] = (64, 128, 256, 512, 1024)


class DoubleConv(nnx.Module):
    """(Convolution => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(in_channels, out_channels, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.conv2 = nnx.Conv(out_channels, out_channels, kernel_size=(3, 3), padding="SAME", rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.conv1(x)
        x = nnx.relu(x)
        x = self.conv2(x)
        x = nnx.relu(x)
        return x


class Down(nnx.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int, *, rngs: nnx.Rngs):
        self.maxpool = partial(nnx.max_pool, window_shape=(2, 2), strides=(2, 2))
        self.conv = DoubleConv(in_channels, out_channels, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.maxpool(x)
        x = self.conv(x)
        return x


class Up(nnx.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels: int, out_channels: int, *, rngs: nnx.Rngs):
        # Use ConvTranspose for upsampling
        self.up = nnx.ConvTranspose(
            in_channels, out_channels, kernel_size=(2, 2), strides=(2, 2), padding="VALID", rngs=rngs
        )
        self.conv = DoubleConv(in_channels, out_channels, rngs=rngs)

    def __call__(self, x1: jax.Array, x2: jax.Array) -> jax.Array:
        x1 = self.up(x1)
        # Concatenate skip connection
        x = jnp.concatenate([x2, x1], axis=-1)
        return self.conv(x)


class OutConv(nnx.Module):
    """Final 1x1 convolution"""

    def __init__(self, in_channels: int, out_channels: int, *, rngs: nnx.Rngs):
        self.conv = nnx.Conv(in_channels, out_channels, kernel_size=(1, 1), rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.conv(x)


class UNet(nnx.Module):
    """
    U-Net implementation based on the original paper.
    See: https://arxiv.org/abs/1505.04597
    """

    def __init__(self, cfg: ModelCfg, *, rngs: nnx.Rngs):
        features = cfg.features
        self.inc = DoubleConv(cfg.in_channels, features[0], rngs=rngs)
        self.down1 = Down(features[0], features[1], rngs=rngs)
        self.down2 = Down(features[1], features[2], rngs=rngs)
        self.down3 = Down(features[2], features[3], rngs=rngs)
        self.down4 = Down(features[3], features[4], rngs=rngs)
        self.up1 = Up(features[4], features[3], rngs=rngs)
        self.up2 = Up(features[3], features[2], rngs=rngs)
        self.up3 = Up(features[2], features[1], rngs=rngs)
        self.up4 = Up(features[1], features[0], rngs=rngs)
        self.outc = OutConv(features[0], cfg.num_classes, rngs=rngs)

    @partial(jax.jit, static_argnums=(0,))
    def forward(model, x):
        return model(x)

    def __call__(self, x: jax.Array) -> jax.Array:
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output convolution
        logits = self.outc(x)
        return logits
