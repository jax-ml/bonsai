from dataclasses import dataclass
from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


@dataclass
class ModelConfig:
    stage_depths: Sequence[int]
    stage_dims: Sequence[int]
    drop_path_rate: float
    num_classes: int
    in_channels: int = 3
    patch_size: tuple[int, int] = (4, 4)
    layernorm_eps: float = 1e-12
    layer_scale_init_value: float = 1e-6

    @classmethod
    def convnext_tiny_224(cls, *, drop_path_rate: float = 0.0, num_classes: int = 1000):
        return cls(
            stage_depths=(3, 3, 9, 3),
            stage_dims=(96, 192, 384, 768),
            drop_path_rate=drop_path_rate,
            num_classes=num_classes,
        )

    @classmethod
    def convnext_small_224(cls, *, drop_path_rate: float = 0.0, num_classes: int = 1000):
        return cls(
            stage_depths=(3, 3, 27, 3),
            stage_dims=(96, 192, 384, 768),
            drop_path_rate=drop_path_rate,
            num_classes=num_classes,
        )

    @classmethod
    def convnext_base_224(cls, *, drop_path_rate: float = 0.0, num_classes: int = 1000):
        return cls(
            stage_depths=(3, 3, 27, 3),
            stage_dims=(128, 256, 512, 1024),
            drop_path_rate=drop_path_rate,
            num_classes=num_classes,
        )

    @classmethod
    def convnext_large_224(cls, *, drop_path_rate: float = 0.0, num_classes: int = 1000):
        return cls(
            stage_depths=(3, 3, 27, 3),
            stage_dims=(192, 384, 768, 1536),
            drop_path_rate=drop_path_rate,
            num_classes=num_classes,
        )


def drop_path(x: jax.Array, drop_prob: float, *, rngs: jax.Array, train: bool):
    if drop_prob < 1e-8 or not train:
        return x

    keep_prob = jnp.asarray(1.0) - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = jax.random.bernoulli(rngs, p=keep_prob, shape=shape)
    return (x * mask) / keep_prob


class Block(nnx.Module):
    def __init__(self, cfg: ModelConfig, stage_idx: int, drop_path_rate: float, *, rngs: nnx.Rngs):
        dim = cfg.stage_dims[stage_idx]
        self.dwconv = nnx.Conv(dim, dim, (7, 7), padding=(3, 3), feature_group_count=dim, rngs=rngs)
        self.norm = nnx.LayerNorm(dim, epsilon=cfg.layernorm_eps, rngs=rngs)
        self.pwconv1 = nnx.Linear(dim, 4 * dim, rngs=rngs)
        self.pwconv2 = nnx.Linear(4 * dim, dim, rngs=rngs)

        self.gamma = nnx.Param(cfg.layer_scale_init_value * jnp.ones((dim))) if cfg.layer_scale_init_value > 0 else None
        self.drop_path_rate = drop_path_rate

    def __call__(self, x: jax.Array, *, rngs: jax.Array, train: bool):
        res = x
        x = self.norm(self.dwconv(x))
        x = jax.nn.gelu(self.pwconv1(x), approximate=False)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma[...] * x

        return res + drop_path(x, self.drop_path_rate, rngs=rngs, train=train)


class Stage(nnx.Module):
    def __init__(self, cfg: ModelConfig, stage_idx: int, drop_path_rates: Sequence[int], *, rngs: nnx.Rngs):
        in_ch = cfg.stage_dims[max(0, stage_idx - 1)]
        out_ch = cfg.stage_dims[stage_idx]
        s = 2 if stage_idx > 0 else 1

        self.downsample_layers = nnx.List()
        if in_ch != out_ch or s > 1:
            self.downsample_layers.append(nnx.LayerNorm(in_ch, epsilon=cfg.layernorm_eps, rngs=rngs))
            self.downsample_layers.append(nnx.Conv(in_ch, out_ch, kernel_size=(2, 2), strides=(s, s), rngs=rngs))

        self.layers = nnx.List(
            [Block(cfg, stage_idx, drop_path_rates[i], rngs=rngs) for i in range(cfg.stage_depths[stage_idx])]
        )

    def __call__(self, x: jax.Array, *, rngs: jax.Array, train: bool):
        for l in self.downsample_layers:
            x = l(x)
        for l in self.layers:
            x = l(x, rngs=rngs, train=train)
        return x


class ConvNeXt(nnx.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        *,
        rngs: nnx.Rngs,
    ):
        self.embedding_layer = nnx.Sequential(
            nnx.Conv(cfg.in_channels, cfg.stage_dims[0], cfg.patch_size, cfg.patch_size, rngs=rngs),
            nnx.LayerNorm(cfg.stage_dims[0], epsilon=cfg.layernorm_eps, rngs=rngs),
        )

        splits = np.cumsum(cfg.stage_depths)
        dp_rates = np.split(np.linspace(0, cfg.drop_path_rate, splits[-1]), splits[:-1])
        self.stages = nnx.List([Stage(cfg, i, dpr, rngs=rngs) for i, dpr in enumerate(dp_rates)])

        self.norm = nnx.LayerNorm(cfg.stage_dims[-1], epsilon=cfg.layernorm_eps, rngs=rngs)
        self.head = nnx.Linear(cfg.stage_dims[-1], cfg.num_classes, rngs=rngs)

    def __call__(self, x: jax.Array, *, rngs: jax.Array, train: bool = False):
        """x is a batch of images of shape (B, H, W, C)"""
        x = self.embedding_layer(x)
        for l in self.stages:
            x = l(x, rngs=rngs, train=train)

        x = jnp.mean(x, axis=(1, 2))
        x = self.norm(x)
        return self.head(x)


@partial(jax.jit, static_argnames=["graph_def", "train"])
def forward(
    graph_def: nnx.GraphDef,
    state: nnx.State,
    x: jax.Array,
    *,
    rngs: jax.Array,
    train: bool = False,
):
    model = nnx.merge(graph_def, state)
    return model(x, rngs=rngs, train=train)
