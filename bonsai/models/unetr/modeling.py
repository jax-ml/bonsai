from dataclasses import dataclass, field
from typing import Callable, List, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from flax import nnx


@dataclass
class UNETRConfig:
    out_channels: int
    in_channels: int = 3
    img_size: int = 256
    patch_size: int = 16
    hidden_size: int = 768
    mlp_dim: int = 3072
    num_heads: int = 12
    dropout_rate: float = 0.0
    num_layers: int = 12
    feature_size: int = 16


class PatchEmbeddingBlock(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        img_size: int,
        patch_size: int,
        hidden_size: int,
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        n_patches = (img_size // patch_size) ** 2
        self.patch_embeddings = nnx.Conv(
            in_channels,
            hidden_size,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            use_bias=True,
            rngs=rngs,
        )
        initializer = jax.nn.initializers.truncated_normal(stddev=0.02)
        self.position_embeddings = nnx.Param(initializer(rngs.params(), (1, n_patches, hidden_size), jnp.float32))
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.patch_embeddings(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class MLPBlock(nnx.Sequential):
    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        dropout_rate: float = 0.0,
        activation_layer: Callable = nnx.gelu,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        layers = [
            nnx.Linear(hidden_size, mlp_dim, rngs=rngs),
            activation_layer,
            nnx.Dropout(dropout_rate, rngs=rngs),
            nnx.Linear(mlp_dim, hidden_size, rngs=rngs),
            nnx.Dropout(dropout_rate, rngs=rngs),
        ]
        super().__init__(*layers)


class ViTEncoderBlock(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ) -> None:
        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate, rngs=rngs)
        self.norm1 = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_size,
            dropout_rate=dropout_rate,
            broadcast_dropout=False,
            decode=False,
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(hidden_size, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        img_size: int,
        patch_size: int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            rngs=rngs,
        )
        self.blocks = nnx.Sequential(
            *[ViTEncoderBlock(hidden_size, mlp_dim, num_heads, dropout_rate, rngs=rngs) for i in range(num_layers)]
        )
        self.norm = nnx.LayerNorm(hidden_size, rngs=rngs)

    def __call__(self, x: jax.Array) -> Tuple[jax.Array, List[jax.Array]]:
        x = self.patch_embedding(x)
        hidden_states_out = []
        for blk in self.blocks.layers:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        return x, hidden_states_out


class Conv2dNormActivation(nnx.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Union[int, None] = None,
        groups: int = 1,
        norm_layer: Callable[..., nnx.Module] = nnx.BatchNorm,
        activation_layer: Callable = nnx.relu,
        dilation: int = 1,
        bias: Union[bool, None] = None,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.out_channels = out_channels
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None

        padding = ((padding, padding), (padding, padding))

        layers = [
            nnx.Conv(
                in_channels,
                out_channels,
                kernel_size=(kernel_size, kernel_size),
                strides=(stride, stride),
                padding=padding,
                kernel_dilation=(dilation, dilation),
                feature_group_count=groups,
                use_bias=bias,
                rngs=rngs,
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels, rngs=rngs))

        if activation_layer is not None:
            layers.append(activation_layer)

        super().__init__(*layers)


class InstanceNorm(nnx.GroupNorm):
    def __init__(self, num_features, **kwargs):
        num_groups, group_size = num_features, None
        super().__init__(
            num_features,
            num_groups=num_groups,
            group_size=group_size,
            **kwargs,
        )


class UnetResBlock(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        norm_layer: Callable[..., nnx.Module] = InstanceNorm,
        activation_layer: Callable = nnx.leaky_relu,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.conv_norm_act1 = Conv2dNormActivation(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            rngs=rngs,
        )
        self.conv_norm2 = Conv2dNormActivation(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            norm_layer=norm_layer,
            activation_layer=None,
            rngs=rngs,
        )
        self.downsample = (in_channels != out_channels) or (stride != 1)
        if self.downsample:
            self.conv_norm3 = Conv2dNormActivation(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                norm_layer=norm_layer,
                activation_layer=None,
                rngs=rngs,
            )
        self.act = activation_layer

    def __call__(self, x: jax.Array) -> jax.Array:
        residual = x
        out = self.conv_norm_act1(x)
        out = self.conv_norm2(out)
        if self.downsample:
            residual = self.conv_norm3(residual)
        out += residual
        out = self.act(out)
        return out


class UnetrBasicBlock(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        norm_layer: Callable[..., nnx.Module] = InstanceNorm,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.layer = UnetResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            norm_layer=norm_layer,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.layer(x)


class UnetrPrUpBlock(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layer: int,
        kernel_size: int,
        stride: int,
        upsample_kernel_size: int = 2,
        norm_layer: Callable[..., nnx.Module] = InstanceNorm,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        upsample_stride = upsample_kernel_size
        self.transp_conv_init = nnx.ConvTranspose(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(upsample_kernel_size, upsample_kernel_size),
            strides=(upsample_stride, upsample_stride),
            padding="VALID",
            rngs=rngs,
        )
        self.blocks = nnx.Sequential(
            *[
                nnx.Sequential(
                    nnx.ConvTranspose(
                        in_features=out_channels,
                        out_features=out_channels,
                        kernel_size=(upsample_kernel_size, upsample_kernel_size),
                        strides=(upsample_stride, upsample_stride),
                        rngs=rngs,
                    ),
                    UnetResBlock(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        norm_layer=norm_layer,
                        rngs=rngs,
                    ),
                )
                for i in range(num_layer)
            ]
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.transp_conv_init(x)
        x = self.blocks(x)
        return x


class UnetrUpBlock(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        upsample_kernel_size: int = 2,
        norm_layer: Callable[..., nnx.Module] = InstanceNorm,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ) -> None:
        upsample_stride = upsample_kernel_size
        self.transp_conv = nnx.ConvTranspose(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(upsample_kernel_size, upsample_kernel_size),
            strides=(upsample_stride, upsample_stride),
            padding="VALID",
            rngs=rngs,
        )
        self.conv_block = UnetResBlock(
            out_channels + out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            norm_layer=norm_layer,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, skip: jax.Array) -> jax.Array:
        out = self.transp_conv(x)
        out = jnp.concat((out, skip), axis=-1)
        out = self.conv_block(out)
        return out


class UNETR(nnx.Module):
    def __init__(
        self,
        config: UNETRConfig,
        norm_layer: Callable[..., nnx.Module] = InstanceNorm,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        if config.hidden_size % config.num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_layers = config.num_layers
        self.patch_size = config.patch_size
        self.feat_size = config.img_size // self.patch_size
        self.hidden_size = config.hidden_size
        self.feature_size = config.feature_size

        self.vit = ViT(
            in_channels=config.in_channels,
            img_size=config.img_size,
            patch_size=config.patch_size,
            hidden_size=config.hidden_size,
            mlp_dim=config.mlp_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout_rate=config.dropout_rate,
            rngs=rngs,
        )
        self.encoder1 = UnetrBasicBlock(
            in_channels=config.in_channels,
            out_channels=config.feature_size,
            kernel_size=3,
            stride=1,
            norm_layer=norm_layer,
            rngs=rngs,
        )
        self.encoder2 = UnetrPrUpBlock(
            in_channels=config.hidden_size,
            out_channels=config.feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_layer=norm_layer,
            rngs=rngs,
        )
        self.encoder3 = UnetrPrUpBlock(
            in_channels=config.hidden_size,
            out_channels=config.feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_layer=norm_layer,
            rngs=rngs,
        )
        self.encoder4 = UnetrPrUpBlock(
            in_channels=config.hidden_size,
            out_channels=config.feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_layer=norm_layer,
            rngs=rngs,
        )
        self.decoder5 = UnetrUpBlock(
            in_channels=config.hidden_size,
            out_channels=config.feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_layer=norm_layer,
            rngs=rngs,
        )
        self.decoder4 = UnetrUpBlock(
            in_channels=config.feature_size * 8,
            out_channels=config.feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_layer=norm_layer,
            rngs=rngs,
        )
        self.decoder3 = UnetrUpBlock(
            in_channels=config.feature_size * 4,
            out_channels=config.feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_layer=norm_layer,
            rngs=rngs,
        )
        self.decoder2 = UnetrUpBlock(
            in_channels=config.feature_size * 2,
            out_channels=config.feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_layer=norm_layer,
            rngs=rngs,
        )
        self.out = nnx.Conv(
            in_features=config.feature_size,
            out_features=config.out_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            use_bias=True,
            rngs=rngs,
        )
        self.proj_axes = (0, 1, 2, 3)
        self.proj_view_shape = [self.feat_size, self.feat_size, self.hidden_size]

    def proj_feat(self, x: jax.Array) -> jax.Array:
        new_view = [x.shape[0], *self.proj_view_shape]
        x = x.reshape(new_view)
        x = jnp.permute_dims(x, self.proj_axes)
        return x

    def __call__(self, x_in: jax.Array) -> jax.Array:
        x, hidden_states_out = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3))
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4))
        dec4 = self.proj_feat(x)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        return self.out(out)
