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

import jax
from flax import nnx
from flax.linen.pooling import max_pool


class Bottleneck(nnx.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample=None, *, rngs: nnx.Rngs):
        self.conv0 = nnx.Conv(
            in_channels, out_channels, kernel_size=(1, 1), strides=1, padding=0, use_bias=False, rngs=rngs
        )
        self.bn0 = nnx.BatchNorm(out_channels, use_running_average=True, rngs=rngs)

        self.conv1 = nnx.Conv(
            out_channels, out_channels, kernel_size=(3, 3), strides=stride, padding=1, use_bias=False, rngs=rngs
        )
        self.bn1 = nnx.BatchNorm(out_channels, use_running_average=True, rngs=rngs)

        self.conv2 = nnx.Conv(
            out_channels, out_channels * 4, kernel_size=(1, 1), strides=1, padding=0, use_bias=False, rngs=rngs
        )
        self.bn2 = nnx.BatchNorm(out_channels * 4, use_running_average=True, rngs=rngs)

        self.downsample = downsample
        self.relu = nnx.relu

    def __call__(self, x):
        identity = x

        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        return self.relu(x + identity)


class Downsample(nnx.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, *, rngs: nnx.Rngs):
        self.conv = nnx.Conv(
            in_channels, out_channels, kernel_size=(1, 1), strides=stride, padding=0, use_bias=False, rngs=rngs
        )
        self.bn = nnx.BatchNorm(out_channels, use_running_average=True, rngs=rngs)

    def __call__(self, x):
        x = self.conv(x)
        return self.bn(x)


class BlockGroup(nnx.Module):
    def __init__(self, in_channels: int, out_channels: int, blocks, stride: int, *, rngs: nnx.Rngs):
        self.blocks = nnx.List()

        downsample = None
        if stride != 1 or in_channels != out_channels * 4:
            downsample = Downsample(in_channels, out_channels * 4, stride, rngs=rngs)

        self.blocks.append(Bottleneck(in_channels, out_channels, stride, downsample, rngs=rngs))
        for _ in range(1, blocks):
            self.blocks.append(Bottleneck(out_channels * 4, out_channels, stride=1, downsample=None, rngs=rngs))

    def __call__(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Stem(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.conv = nnx.Conv(3, 64, kernel_size=(7, 7), strides=2, padding=3, use_bias=False, rngs=rngs)
        self.bn = nnx.BatchNorm(64, use_running_average=True, rngs=rngs)
        self.relu = nnx.relu
        self.pool = partial(max_pool, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class ResNet(nnx.Module):
    def __init__(self, block_sizes: list[int], num_classes: int = 1000, *, rngs: nnx.Rngs):
        self.stem = Stem(rngs=rngs)

        self.layer0 = BlockGroup(64, 64, block_sizes[0], stride=1, rngs=rngs)
        self.layer1 = BlockGroup(256, 128, block_sizes[1], stride=2, rngs=rngs)
        self.layer2 = BlockGroup(512, 256, block_sizes[2], stride=2, rngs=rngs)
        self.layer3 = BlockGroup(1024, 512, block_sizes[3], stride=2, rngs=rngs)

        self.pool = partial(lambda x: x.mean(axis=(1, 2)))
        self.fc = nnx.Linear(2048, num_classes, rngs=rngs)

    def __call__(self, x):
        x = self.stem(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        return self.fc(x)


def ResNet50(num_classes=1000, *, rngs: nnx.Rngs):
    return ResNet([3, 4, 6, 3], num_classes=num_classes, rngs=rngs)


def ResNet152(num_classes=1000, *, rngs: nnx.Rngs):
    return ResNet([3, 8, 36, 3], num_classes=num_classes, rngs=rngs)


@jax.jit
def forward(model, x):
    return model(x)
