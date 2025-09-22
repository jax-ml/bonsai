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
    num_classes: int
    dense_block_layers: list[int]
    growth_rate: int

    @classmethod
    def densenet_121(cls):
        return cls(
            num_classes=1000,
            dense_block_layers=[6, 12, 24, 16],
            growth_rate=32,
        )


class DenseBlock(nnx.Module):
    def __init__(self, num_layers: int, in_channels: int, growth_rate: int, *, rngs: nnx.Rngs):
        self.conv_layers = []
        self.bn_layers = []

        for i in range(num_layers):
            self.bn_layers.append(
                nnx.BatchNorm(in_channels, use_running_average=True, rngs=rngs),
            )
            self.bn_layers.append(
                nnx.BatchNorm(4 * growth_rate, use_running_average=True, rngs=rngs),
            )

            self.conv_layers.append(
                nnx.Conv(in_channels, 4 * growth_rate, kernel_size=(1, 1), padding="SAME", use_bias=False, rngs=rngs),
            )
            self.conv_layers.append(
                nnx.Conv(4 * growth_rate, growth_rate, kernel_size=(3, 3), padding="SAME", use_bias=False, rngs=rngs),
            )

            in_channels += growth_rate

    def __call__(self, x1):
        for i in range(0, len(self.conv_layers), 2):
            x2 = self.bn_layers[i](x1)
            x2 = nnx.relu(x2)
            x2 = self.conv_layers[i](x2)

            x2 = self.bn_layers[i + 1](x2)
            x2 = nnx.relu(x2)
            x2 = self.conv_layers[i + 1](x2)

            x1 = jax.numpy.concatenate([x1, x2], axis=-1)

        return x1


class TransitionLayer(nnx.Module):
    def __init__(self, in_channels: int, out_channels: int, *, rngs: nnx.Rngs):
        self.bn = nnx.BatchNorm(in_channels, use_running_average=True, rngs=rngs)
        self.conv = nnx.Conv(in_channels, out_channels, kernel_size=(1, 1), use_bias=False, padding="SAME", rngs=rngs)
        self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))

    def __call__(self, x):
        x = self.bn(x)
        x = nnx.relu(x)
        x = self.conv(x)
        x = self.avg_pool(x)

        return x


class DenseNet(nnx.Module):
    def __init__(self, cfg: ModelCfg, *, rngs: nnx.Rngs):
        self.init_conv = nnx.Conv(
            3, 2 * cfg.growth_rate, kernel_size=(7, 7), strides=(2, 2), padding="SAME", use_bias=False, rngs=rngs
        )
        self.init_bn = nnx.BatchNorm(2 * cfg.growth_rate, use_running_average=True, rngs=rngs)
        self.init_pool = partial(nnx.max_pool, window_shape=(3, 3), strides=(2, 2), padding="SAME")

        channels = 2 * cfg.growth_rate
        self.dense_blocks = []
        self.transition_layers = []

        for i, num_layers in enumerate(cfg.dense_block_layers):
            self.dense_blocks.append(DenseBlock(num_layers, channels, cfg.growth_rate, rngs=rngs))
            channels += num_layers * cfg.growth_rate

            if i == len(cfg.dense_block_layers) - 1:
                continue

            out_channels = int(channels * 0.5)
            self.transition_layers.append(TransitionLayer(channels, out_channels, rngs=rngs))
            channels = out_channels

        self.final_bn = nnx.BatchNorm(channels, use_running_average=True, rngs=rngs)
        self.linear = nnx.Linear(channels, cfg.num_classes, rngs=rngs)

    def __call__(self, x):
        x = self.init_conv(x)
        x = self.init_bn(x)
        x = self.init_pool(x)

        for i, dense_block in enumerate(self.dense_blocks):
            x = dense_block(x)

            if i == len(self.dense_blocks) - 1:
                continue

            x = self.transition_layers[i](x)

        x = self.final_bn(x)
        x = nnx.relu(x)
        # Global Average Pooling
        x = jnp.mean(x, axis=(1, 2))
        x = self.linear(x)

        return x


@partial(jax.jit, donate_argnums=(1))
def forward(graphdef: nnx.GraphDef, state: nnx.State, x: jax.Array) -> jax.Array:
    model = nnx.merge(graphdef, state)
    return model(x)
