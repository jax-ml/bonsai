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
class ModelCfg:
    num_classes: int
    conv_sizes: list[int]

    @classmethod
    def vgg_19(cls):
        return cls(
            num_classes=1000,
            conv_sizes=[2, 2, 4, 4, 4],
        )


class ConvBlock(nnx.Module):
    def __init__(self, num_conv: int, in_channels: int, out_channels: int, *, rngs: nnx.Rngs):
        self.conv_layers = []
        for i in range(num_conv):
            in_ch = in_channels if i == 0 else out_channels
            self.conv_layers.append(
                nnx.Conv(in_ch, out_channels, kernel_size=(3, 3), padding="SAME", use_bias=True, rngs=rngs)
            )
        self.max_pool = partial(max_pool, window_shape=(2, 2), strides=(2, 2), padding="VALID")

    def __call__(self, x):
        for conv in self.conv_layers:
            x = conv(x)
            x = nnx.relu(x)
        x = self.max_pool(x)
        return x


class VGG(nnx.Module):
    def __init__(self, cfg: ModelCfg, *, rngs: nnx.Rngs):
        self.conv_block0 = ConvBlock(cfg.conv_sizes[0], in_channels=3, out_channels=64, rngs=rngs)
        self.conv_block1 = ConvBlock(cfg.conv_sizes[1], in_channels=64, out_channels=128, rngs=rngs)
        self.conv_block2 = ConvBlock(cfg.conv_sizes[2], in_channels=128, out_channels=256, rngs=rngs)
        self.conv_block3 = ConvBlock(cfg.conv_sizes[3], in_channels=256, out_channels=512, rngs=rngs)
        self.conv_block4 = ConvBlock(cfg.conv_sizes[4], in_channels=512, out_channels=512, rngs=rngs)
        self.global_mean_pool = lambda x: jnp.mean(x, axis=(1, 2))
        self.classifier = nnx.Sequential(
            nnx.Conv(512, 4096, (7, 7), rngs=rngs),
            nnx.Conv(4096, 4096, (1, 1), rngs=rngs),
            self.global_mean_pool,
            nnx.Linear(4096, cfg.num_classes, rngs=rngs),
        )

    def __call__(self, x):
        x = self.conv_block0(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.classifier(x)
        return x


@partial(jax.jit, donate_argnums=(1))
def forward(graphdef: nnx.GraphDef, state: nnx.State, x: jax.Array) -> jax.Array:
    model = nnx.merge(graphdef, state)
    return model(x)
