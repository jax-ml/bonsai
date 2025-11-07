from typing import Optional

import jax
import jax.image
import jax.numpy as jnp
from flax import nnx


class ResnetBlock(nnx.Module):
    conv_shortcut: nnx.Data[Optional[nnx.Conv]]

    def __init__(self, in_channels: int, out_channels: int, groups: int, rngs: nnx.Rngs):
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = nnx.Conv(
                in_features=in_channels,
                out_features=out_channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="VALID",
                use_bias=True,
                rngs=rngs,
            )
        self.norm1 = nnx.GroupNorm(num_groups=groups, num_features=in_channels, epsilon=1e-6, rngs=rngs)
        self.conv1 = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            rngs=rngs,
        )
        self.norm2 = nnx.GroupNorm(num_groups=groups, num_features=out_channels, epsilon=1e-6, rngs=rngs)
        self.conv2 = nnx.Conv(
            in_features=out_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            rngs=rngs,
        )

    def __call__(self, input_tensor):
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = nnx.silu(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = nnx.silu(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / 1.0

        return output_tensor


class DownEncoderBlock2D(nnx.Module):
    downsamplers: nnx.Data[Optional[nnx.Conv]]

    def __init__(self, in_channels: int, out_channels: int, groups: int, is_final_block: bool, rngs: nnx.Rngs):
        self.resnets = nnx.List([])

        for i in range(2):
            current_in_channels = in_channels if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock(in_channels=current_in_channels, out_channels=out_channels, groups=groups, rngs=rngs)
            )

        self.downsamplers = None

        if not is_final_block:
            self.downsamplers = nnx.Conv(
                in_features=out_channels,
                out_features=out_channels,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="SAME",
                rngs=rngs,
            )

    def __call__(self, x):
        for resnet in self.resnets:
            x = resnet(x)

        if self.downsamplers is not None:
            x = self.downsamplers(x)

        return x


def scaled_dot_product_attention(query, key, value):
    d_k = query.shape[-1]
    scale_factor = 1.0 / jnp.sqrt(d_k)

    attention_scores = jnp.einsum("bhld,bhsd->bhls", query, key)

    attention_scores *= scale_factor
    attention_weights = jax.nn.softmax(attention_scores, axis=-1)

    output = jnp.einsum("bhls,bhsd->bhld", attention_weights, value)

    return output


class Attention(nnx.Module):
    def __init__(self, channels: int, groups: int, rngs: nnx.Rngs):
        self.group_norm = nnx.GroupNorm(num_groups=groups, num_features=channels, epsilon=1e-6, rngs=rngs)

        self.to_q = nnx.Linear(in_features=channels, out_features=channels, use_bias=True, rngs=rngs)
        self.to_k = nnx.Linear(in_features=channels, out_features=channels, use_bias=True, rngs=rngs)
        self.to_v = nnx.Linear(in_features=channels, out_features=channels, use_bias=True, rngs=rngs)

        self.to_out = nnx.Linear(in_features=channels, out_features=channels, use_bias=True, rngs=rngs)

    def __call__(self, hidden_states):
        heads = 1
        rescale_output_factor = 1
        residual = hidden_states

        batch_size, height, width, channel = None, None, None, None

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, height, width, channel = hidden_states.shape
            hidden_states = hidden_states.reshape(batch_size, height * width, channel)

        batch_size, _, _ = hidden_states.shape
        hidden_states = self.group_norm(hidden_states)

        query = self.to_q(hidden_states)

        encoder_hidden_states = hidden_states

        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // heads

        query = query.reshape(batch_size, -1, heads, head_dim)
        query = jnp.transpose(query, (0, 2, 1, 3))

        key = key.reshape(batch_size, -1, heads, head_dim)
        key = jnp.transpose(key, (0, 2, 1, 3))
        value = value.reshape(batch_size, -1, heads, head_dim)
        value = jnp.transpose(value, (0, 2, 1, 3))

        hidden_states = scaled_dot_product_attention(query, key, value)

        hidden_states = jnp.transpose(hidden_states, (0, 2, 1, 3))
        B, L, H, D = hidden_states.shape
        hidden_states = hidden_states.reshape(B, L, H * D)

        hidden_states = self.to_out(hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.reshape(batch_size, height, width, channel)

        hidden_states = hidden_states + residual
        hidden_states = hidden_states / rescale_output_factor

        return hidden_states


class UNetMidBlock2D(nnx.Module):
    def __init__(self, channels: int, groups: int, num_res_blocks: int, rngs: nnx.Rngs):
        self.resnets = nnx.List([])

        for i in range(num_res_blocks):
            self.resnets.append(ResnetBlock(in_channels=channels, out_channels=channels, groups=groups, rngs=rngs))

        self.attentions = nnx.List([Attention(channels=channels, groups=groups, rngs=rngs)])

    def __call__(self, x):
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)

        return x


class Encoder(nnx.Module):
    def __init__(self, block_out_channels, rngs: nnx.Rngs):
        groups = 32

        self.conv_in = nnx.Conv(
            in_features=3,
            out_features=block_out_channels[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            rngs=rngs,
        )

        self.down_blocks = nnx.List([])

        in_channels = block_out_channels[0]

        for i, out_channels in enumerate(block_out_channels):
            is_final_block = i == len(block_out_channels) - 1

            self.down_blocks.append(
                DownEncoderBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    groups=groups,
                    is_final_block=is_final_block,
                    rngs=rngs,
                )
            )

            in_channels = out_channels

        self.mid_block = UNetMidBlock2D(channels=in_channels, groups=groups, num_res_blocks=2, rngs=rngs)
        self.conv_norm_out = nnx.GroupNorm(
            num_groups=groups, num_features=block_out_channels[-1], epsilon=1e-6, rngs=rngs
        )

        conv_out_channels = 2 * 4

        self.conv_out = nnx.Conv(
            in_features=block_out_channels[-1],
            out_features=conv_out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            rngs=rngs,
        )

    def __call__(self, x):
        x = self.conv_in(x)

        for down_block in self.down_blocks:
            x = down_block(x)

        x = self.mid_block(x)
        x = self.conv_norm_out(x)
        x = nnx.silu(x)
        x = self.conv_out(x)

        return x


def upsample_nearest2d(input_tensor, scale_factors):
    # (N, C, H_in, W_in) -> (N, H_in, W_in, C)
    input_permuted = jnp.transpose(input_tensor, (0, 2, 3, 1))

    # Nearest neighbor interpolation using jax.image.resize
    output_permuted = jax.image.resize(
        input_permuted,
        shape=(
            input_permuted.shape[0],
            int(input_permuted.shape[1] * scale_factors[0]),  # H_out
            int(input_permuted.shape[2] * scale_factors[1]),  # W_out
            input_permuted.shape[3],  # C
        ),
        method="nearest",
    )

    # (N, C, H_out, W_out)
    output_tensor = jnp.transpose(output_permuted, (0, 3, 1, 2))

    return output_tensor


def interpolate(input, scale_factor):
    dim = input.ndim - 2  # 4 - 2
    scale_factors = [scale_factor for _ in range(dim)]  # 2.0, 2.0
    return upsample_nearest2d(input, scale_factors)


class Upsample2D(nnx.Module):
    def __init__(self, channel: int, scale_factor: int, rngs: nnx.Rngs):
        self.scale_factor = scale_factor
        self.conv = nnx.Conv(
            in_features=channel,
            out_features=channel,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=True,
            rngs=rngs,
        )

    def __call__(self, x):
        b, h, w, c = x.shape
        new_shape = (b, int(h * self.scale_factor), int(w * self.scale_factor), c)
        x = jax.image.resize(x, shape=new_shape, method="nearest")
        x = self.conv(x)

        return x


class UpDecoderBlock2D(nnx.Module):
    upsamplers = nnx.Data[Optional["Upsample2D"]]

    def __init__(self, in_channels: int, out_channels: int, groups: int, is_final_block: bool, rngs: nnx.Rngs):
        self.resnets = nnx.List([])

        for i in range(3):
            current_in_channels = in_channels if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock(in_channels=current_in_channels, out_channels=out_channels, groups=groups, rngs=rngs)
            )

        if not is_final_block:
            self.upsamplers = Upsample2D(channel=out_channels, scale_factor=2.0, rngs=rngs)
        else:
            self.upsamplers = None

    def __call__(self, x):
        for resnet in self.resnets:
            x = resnet(x)

        if self.upsamplers is not None:
            x = self.upsamplers(x)

        return x


class Decoder(nnx.Module):
    def __init__(self, latent_channels, block_out_channels, rngs: nnx.Rngs):
        groups = 32

        self.conv_in = nnx.Conv(
            in_features=latent_channels,
            out_features=block_out_channels[-1],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            rngs=rngs,
        )
        self.mid_block = UNetMidBlock2D(channels=block_out_channels[-1], groups=groups, num_res_blocks=2, rngs=rngs)
        self.up_blocks = nnx.List([])

        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]

        for i, out_channels in enumerate(block_out_channels):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            self.up_blocks.append(
                UpDecoderBlock2D(
                    in_channels=prev_output_channel,
                    out_channels=output_channel,
                    groups=groups,
                    is_final_block=is_final_block,
                    rngs=rngs,
                )
            )

            prev_output_channel = output_channel

        self.conv_norm_out = nnx.GroupNorm(
            num_groups=groups, num_features=block_out_channels[0], epsilon=1e-6, rngs=rngs
        )

        self.conv_out = nnx.Conv(block_out_channels[0], 3, kernel_size=(3, 3), strides=1, padding=1, rngs=rngs)

    def __call__(self, x):
        x = self.conv_in(x)
        x = self.mid_block(x)
        for up_block in self.up_blocks:
            x = up_block(x)
        x = self.conv_norm_out(x)
        x = nnx.silu(x)
        x = self.conv_out(x)

        return x


class VAE(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        block_out_channels = [128, 256, 512, 512]
        latent_channels = 4

        self.encoder = Encoder(block_out_channels, rngs)
        self.quant_conv = nnx.Conv(
            in_features=2 * latent_channels,
            out_features=2 * latent_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            rngs=rngs,
        )
        self.post_quant_conv = nnx.Conv(
            in_features=latent_channels,
            out_features=latent_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            rngs=rngs,
        )
        self.decoder = Decoder(latent_channels=latent_channels, block_out_channels=block_out_channels, rngs=rngs)

    def __call__(self, x):
        x = self.encoder(x)
        x = self.quant_conv(x)
        mean, _ = jnp.split(x, 2, axis=-1)
        x = self.post_quant_conv(mean)
        x = self.decoder(x)

        return x


@jax.jit
def forward(model, x):
    return model(x)
