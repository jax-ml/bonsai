from typing import Callable, List, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from bonsai.models.vit.modeling import ModelConfig
from bonsai.models.vit.modeling import TransformerEncoder as TransformerEncoderBlock


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


class ViT(nnx.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        img_size = config.image_size[0]
        patch_size = config.patch_size[0]

        if config.hidden_dim % config.num_heads != 0:
            raise ValueError("hidden_dim should be divisible by num_heads.")

        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=config.num_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=config.hidden_dim,
            dropout_rate=config.dropout_prob,
            rngs=rngs,
        )

        self.blocks = nnx.Sequential(*[TransformerEncoderBlock(config, rngs=rngs) for i in range(config.num_layers)])
        self.norm = nnx.LayerNorm(config.hidden_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> Tuple[jax.Array, List[jax.Array]]:
        x = self.patch_embedding(x)
        hidden_states_out = []
        for blk in self.blocks.layers:
            x = blk(x, rngs=None)
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
        padding: int | None = None,
        groups: int = 1,
        norm_layer: Callable[..., nnx.Module] = nnx.BatchNorm,
        activation_layer: Callable = nnx.relu,
        dilation: int = 1,
        bias: bool | None = None,
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
        config: ModelConfig,
        norm_layer: Callable[..., nnx.Module] = InstanceNorm,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        if config.hidden_dim % config.num_heads != 0:
            raise ValueError("hidden_dim should be divisible by num_heads.")

        img_size = config.image_size[0]
        patch_size = config.patch_size[0]

        self.num_layers = config.num_layers
        self.patch_size = patch_size
        self.feat_size = img_size // patch_size
        self.hidden_size = config.hidden_dim
        self.feature_size = config.feature_size

        self.vit = ViT(config=config, rngs=rngs)

        self.encoder1 = UnetrBasicBlock(
            in_channels=config.num_channels,
            out_channels=config.feature_size,
            kernel_size=3,
            stride=1,
            norm_layer=norm_layer,
            rngs=rngs,
        )

        self.encoders = nnx.List(
            [
                UnetrPrUpBlock(
                    in_channels=config.hidden_dim,
                    out_channels=config.feature_size * ch_mult,
                    num_layer=num_layer,
                    kernel_size=3,
                    stride=1,
                    upsample_kernel_size=2,
                    norm_layer=norm_layer,
                    rngs=rngs,
                )
                for ch_mult, num_layer in zip(config.encoder_channels[1:], config.encoder_num_layers[1:])
            ]
        )

        self.decoders = nnx.List(
            [
                UnetrUpBlock(
                    in_channels=config.feature_size * in_mult if i > 0 else config.hidden_dim,
                    out_channels=config.feature_size * out_mult,
                    kernel_size=3,
                    upsample_kernel_size=2,
                    norm_layer=norm_layer,
                    rngs=rngs,
                )
                for i, (in_mult, out_mult) in enumerate(config.decoder_channels)
            ]
        )
        self.final_conv = nnx.Conv(
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
        hidden_state_indices = [3, 6, 9]
        encs = [enc1] + [
            encoder(self.proj_feat(hidden_states_out[idx])) for encoder, idx in zip(self.encoders, hidden_state_indices)
        ]
        dec = self.proj_feat(x)
        for decoder, enc in zip(self.decoders, reversed(encs)):
            dec = decoder(dec, enc)
        return self.final_conv(dec)
