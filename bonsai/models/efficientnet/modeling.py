import dataclasses
import math
from functools import partial
from typing import Literal, Sequence

import jax
import jax.numpy as jnp
from flax import nnx


@dataclasses.dataclass(frozen=True)
class BlockConfig:
    input_filters: int
    output_filters: int
    kernel_size: int
    num_repeat: int
    expand_ratio: int
    strides: int
    se_ratio: float
    padding: int | Literal["SAME"]


@dataclasses.dataclass(frozen=True)
class BlockConfigs:
    items: Sequence[BlockConfig]

    @classmethod
    def default_block_config(cls):
        return cls(
            [
                BlockConfig(32, 16, 3, 1, 1, 1, 0.25, 1),
                BlockConfig(16, 24, 3, 2, 6, 2, 0.25, 1),
                BlockConfig(24, 40, 5, 2, 6, 2, 0.25, 2),
                BlockConfig(40, 80, 3, 3, 6, 2, 0.25, 1),
                BlockConfig(80, 112, 5, 3, 6, 1, 0.25, 2),
                BlockConfig(112, 192, 5, 4, 6, 2, 0.25, 2),
                BlockConfig(192, 320, 3, 1, 6, 1, 0.25, 1),
            ]
        )

    @classmethod
    def tf_block_config(cls):
        return cls(
            [
                BlockConfig(32, 16, 3, 1, 1, 1, 0.25, "SAME"),
                BlockConfig(16, 24, 3, 2, 6, 2, 0.25, "SAME"),
                BlockConfig(24, 40, 5, 2, 6, 2, 0.25, "SAME"),
                BlockConfig(40, 80, 3, 3, 6, 2, 0.25, "SAME"),
                BlockConfig(80, 112, 5, 3, 6, 1, 0.25, "SAME"),
                BlockConfig(112, 192, 5, 4, 6, 2, 0.25, "SAME"),
                BlockConfig(192, 320, 3, 1, 6, 1, 0.25, "SAME"),
            ]
        )


@dataclasses.dataclass(frozen=True)
class ModelCfg:
    width_coefficient: float
    depth_coefficient: float
    resolution: int
    dropout_rate: float
    stem_conv_padding: int | Literal["SAME"]
    bn_momentum: float
    bn_epsilon: float
    num_classes: int = 1000

    @classmethod
    def b0(cls, num_classes=1000):
        return cls(1.0, 1.0, 224, 0.2, 1, 0.99, 1e-5, num_classes)

    @classmethod
    def b1(cls, num_classes=1000):
        return cls(1.0, 1.1, 240, 0.2, 1, 0.99, 1e-5, num_classes)

    @classmethod
    def b2(cls, num_classes=1000):
        return cls(1.1, 1.2, 260, 0.3, 1, 0.99, 1e-5, num_classes)

    @classmethod
    def b3(cls, num_classes=1000):
        return cls(1.2, 1.4, 300, 0.3, 1, 0.99, 1e-5, num_classes)

    @classmethod
    def b4(cls, num_classes=1000):
        return cls(1.4, 1.8, 380, 0.4, 1, 0.99, 1e-5, num_classes)

    @classmethod
    def b5(cls, num_classes=1000):
        return cls(1.6, 2.2, 456, 0.4, "SAME", 1e-1, 1e-3, num_classes)

    @classmethod
    def b6(cls, num_classes=1000):
        return cls(1.8, 2.6, 528, 0.5, "SAME", 1e-1, 1e-3, num_classes)

    @classmethod
    def b7(cls, num_classes=1000):
        return cls(2.0, 3.1, 600, 0.5, "SAME", 1e-1, 1e-3, num_classes)


def round_filters(filters: int, width_coefficient: float, divisor: int = 8) -> int:
    """Round number of filters based on width multiplier."""
    filters *= width_coefficient
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats: int, depth_coefficient: float) -> int:
    """Round number of repeats based on depth multiplier."""
    return math.ceil(depth_coefficient * repeats)


class SqueezeAndExcitation(nnx.Module):
    """Squeeze-and-Excitation block"""

    def __init__(self, in_channels: int, se_channels: int, *, rngs: nnx.Rngs):
        # conv_reduce
        self.conv1 = nnx.Conv(
            in_channels,
            se_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            use_bias=True,
            rngs=rngs,
        )
        # conv_expand
        self.conv2 = nnx.Conv(
            se_channels,
            in_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            use_bias=True,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # 1. Squeeze
        squeeze = jnp.mean(x, axis=(1, 2), keepdims=True)

        # 2. Excitation
        excitation = self.conv1(squeeze)
        excitation = nnx.silu(excitation)
        excitation = self.conv2(excitation)
        excitation = nnx.sigmoid(excitation)

        # 3. Scale
        return x * excitation


class MBConv(nnx.Module):
    """Mobile Inverted Bottleneck Convolution (MBConv) block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        strides: int,
        expand_ratio: int,
        se_ratio: float,
        padding: int | Literal["SAME"],
        *,
        cfg: ModelCfg,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strides = strides
        self.expand_ratio = expand_ratio
        self.has_skip = strides == 1 and in_channels == out_channels

        # Expansion phase (1x1 Conv) - skipped if expand_ratio is 1
        expanded_channels = in_channels * expand_ratio
        if expand_ratio != 1:
            self.expand_conv = nnx.Conv(in_channels, expanded_channels, kernel_size=(1, 1), use_bias=False, rngs=rngs)
            self.bn0 = nnx.BatchNorm(
                expanded_channels, momentum=cfg.bn_momentum, epsilon=cfg.bn_epsilon, use_running_average=True, rngs=rngs
            )
        else:
            self.expand_conv = None
            self.bn0 = None

        # Depthwise convolution
        self.depthwise_conv = nnx.Conv(
            expanded_channels,
            expanded_channels,
            kernel_size=(kernel_size, kernel_size),
            strides=(strides, strides),
            feature_group_count=expanded_channels,
            padding=padding,
            use_bias=False,
            rngs=rngs,
        )
        self.bn1 = nnx.BatchNorm(
            expanded_channels, momentum=cfg.bn_momentum, epsilon=cfg.bn_epsilon, use_running_average=True, rngs=rngs
        )

        # Squeeze-and-Excitation layer
        if 0 < se_ratio and se_ratio <= 1:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se = SqueezeAndExcitation(expanded_channels, se_channels, rngs=rngs)
        else:
            self.se = None

        # Projection phase (1x1 Conv)
        self.project_conv = nnx.Conv(expanded_channels, out_channels, kernel_size=(1, 1), use_bias=False, rngs=rngs)
        self.bn2 = nnx.BatchNorm(
            out_channels, momentum=cfg.bn_momentum, epsilon=cfg.bn_epsilon, use_running_average=True, rngs=rngs
        )

    def __call__(self, x: jax.Array, training: bool) -> jax.Array:
        identity = x

        is_inference = not training

        if self.expand_conv is not None:
            x = self.expand_conv(x)
            x = self.bn0(x, use_running_average=is_inference)
            x = nnx.silu(x)

        x = self.depthwise_conv(x)
        x = self.bn1(x, use_running_average=is_inference)
        x = nnx.silu(x)

        if self.se is not None:
            x = self.se(x)

        x = self.project_conv(x)
        x = self.bn2(x, use_running_average=is_inference)

        if self.has_skip:
            x += identity

        return x


class EfficientNet(nnx.Module):
    """
    EfficientNet implementation.
    See: https://arxiv.org/abs/1905.11946
    """

    def __init__(
        self,
        cfg: ModelCfg,
        block_configs: BlockConfigs,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.cfg = cfg
        out_channels = round_filters(32, cfg.width_coefficient)
        self.stem_conv = nnx.Conv(
            3,
            out_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=cfg.stem_conv_padding,
            use_bias=False,
            rngs=rngs,
        )
        self.stem_bn = nnx.BatchNorm(
            out_channels, momentum=cfg.bn_momentum, epsilon=cfg.bn_epsilon, use_running_average=True, rngs=rngs
        )

        # Build blocks
        self.blocks = nnx.List()
        for bc in block_configs.items:
            input_filters = round_filters(bc.input_filters, cfg.width_coefficient)
            output_filters = round_filters(bc.output_filters, cfg.width_coefficient)
            num_repeat = round_repeats(bc.num_repeat, cfg.depth_coefficient)

            for i in range(num_repeat):
                strides = bc.strides if i == 0 else 1
                in_ch = input_filters if i == 0 else output_filters

                self.blocks.append(
                    MBConv(
                        in_ch,
                        output_filters,
                        kernel_size=bc.kernel_size,
                        strides=strides,
                        expand_ratio=bc.expand_ratio,
                        se_ratio=bc.se_ratio,
                        padding=bc.padding,
                        cfg=cfg,
                        rngs=rngs,
                    )
                )
        # Head
        in_channels = round_filters(block_configs.items[-1].output_filters, cfg.width_coefficient)
        out_channels = round_filters(1280, cfg.width_coefficient)
        self.head_conv = nnx.Conv(
            in_channels, out_channels, kernel_size=(1, 1), padding="SAME", use_bias=False, rngs=rngs
        )

        self.head_bn = nnx.BatchNorm(out_channels, use_running_average=True, rngs=rngs)

        self.gap = partial(jnp.mean, axis=(1, 2))
        self.dropout = nnx.Dropout(rate=cfg.dropout_rate)
        self.classifier = nnx.Linear(out_channels, cfg.num_classes, rngs=rngs)

    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        # Stem
        x = self.stem_conv(x)
        x = self.stem_bn(x, use_running_average=not training)
        x = nnx.silu(x)

        # Blocks
        for block in self.blocks:
            x = block(x, training=training)

        # Head
        x = self.head_conv(x)
        x = self.head_bn(x, use_running_average=not training)
        x = nnx.silu(x)

        x = self.gap(x)
        x = self.dropout(x, deterministic=not training)
        x = self.classifier(x)
        return x
