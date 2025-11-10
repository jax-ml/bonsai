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
from flax.linen.pooling import max_pool


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    num_classes: int
    block_layers: list[int]
    channels: list[int]

    @classmethod
    def vgg_19(cls):
        return cls(num_classes=1000, block_layers=[2, 2, 4, 4, 4], channels=[64, 128, 256, 512, 512, 4096])


class ConvBlock(nnx.Module):
    def __init__(self, num_conv: int, in_channels: int, out_channels: int, *, rngs: nnx.Rngs):
        self.conv_layers = nnx.List()
        for i in range(num_conv):
            in_ch = in_channels if i == 0 else out_channels
            self.conv_layers.append(
                nnx.Conv(in_ch, out_channels, kernel_size=(3, 3), padding="SAME", use_bias=True, rngs=rngs)
            )

    def __call__(self, x):
        for conv in self.conv_layers:
            x = conv(x)
            x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        return x


class VGG(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.conv_block0 = ConvBlock(cfg.block_layers[0], in_channels=3, out_channels=cfg.channels[0], rngs=rngs)
        self.conv_block1 = ConvBlock(
            cfg.block_layers[1], in_channels=cfg.channels[0], out_channels=cfg.channels[1], rngs=rngs
        )
        self.conv_block2 = ConvBlock(
            cfg.block_layers[2], in_channels=cfg.channels[1], out_channels=cfg.channels[2], rngs=rngs
        )
        self.conv_block3 = ConvBlock(
            cfg.block_layers[3], in_channels=cfg.channels[2], out_channels=cfg.channels[3], rngs=rngs
        )
        self.conv_block4 = ConvBlock(
            cfg.block_layers[4], in_channels=cfg.channels[3], out_channels=cfg.channels[4], rngs=rngs
        )
        self.global_mean_pool = lambda x: jnp.mean(x, axis=(1, 2))
        self.classifier = nnx.Sequential(
            nnx.Conv(cfg.channels[4], cfg.channels[5], (7, 7), rngs=rngs),
            nnx.Conv(cfg.channels[5], cfg.channels[5], (1, 1), rngs=rngs),
            self.global_mean_pool,
            nnx.Linear(cfg.channels[5], cfg.num_classes, rngs=rngs),
        )

    def __call__(self, x):
        x = self.conv_block0(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.classifier(x)
        return x


@jax.jit
def forward(graphdef: nnx.GraphDef, state: nnx.State, x: jax.Array) -> jax.Array:
    model = nnx.merge(graphdef, state)
    return model(x)
