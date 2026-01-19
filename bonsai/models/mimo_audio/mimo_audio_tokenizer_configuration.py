"""Configuration classes for MiMo Audio Tokenizer."""

from dataclasses import dataclass
from typing import Optional
import jax.numpy as jnp
from transformers import PretrainedConfig
from jax.sharding import PartitionSpec as P

Array = jnp.ndarray
ShardingSpec = P


@dataclass(slots=True, frozen=True)
class MiMoShardingCfg:
    """Sharding configuration for MiMo Audio Tokenizer.

    Controls how model parameters and activations are distributed across devices.
    """
    # Conv layer weight sharding
    conv_weight: ShardingSpec  # (in_channels, out_channels, kernel_size)
    conv_bias: ShardingSpec  # (out_channels,)

    # Transformer weight sharding (shared by Encoder/Decoder/Vocoder)
    attn_qkvo_weight: ShardingSpec  # (d_model, d_model)
    attn_qkv_bias: ShardingSpec  # (d_model,)
    attn_out_bias: ShardingSpec  # (d_model,)

    # FFN weight sharding
    ffn_weight_in: ShardingSpec  # (d_model, ffn_dim)
    ffn_weight_out: ShardingSpec  # (ffn_dim, d_model)
    ffn_bias: ShardingSpec  # (ffn_dim,) or (d_model,)

    # LayerNorm/GroupNorm sharding
    norm_scale: ShardingSpec  # (dim,)
    norm_bias: ShardingSpec  # (dim,)

    # Quantizer codebook sharding
    codebook: ShardingSpec  # (codebook_size, d_model)

    # ConvTranspose1d weight sharding
    conv_transpose_weight: ShardingSpec  # (in_ch, out_ch, kernel)
    conv_transpose_bias: ShardingSpec  # (out_ch,)

    # ISTFT related sharding
    istft_linear_weight: ShardingSpec  # (dim, n_fft+2)
    istft_linear_bias: ShardingSpec  # (n_fft+2,)
    istft_window: ShardingSpec  # (win_length,)

    # Activation sharding
    act_btd: ShardingSpec  # [batch, time, d_model]
    act_btnh: ShardingSpec  # [batch, time, num_heads, head_dim]
    act_btc: ShardingSpec  # [batch, time, channels]

    @staticmethod
    def no_sharding():
        """Configuration with no sharding (all None)."""
        return MiMoShardingCfg(
            conv_weight=P(None, None, None),
            conv_bias=P(None),
            attn_qkvo_weight=P(None, None),
            attn_qkv_bias=P(None),
            attn_out_bias=P(None),
            ffn_weight_in=P(None, None),
            ffn_weight_out=P(None, None),
            ffn_bias=P(None),
            norm_scale=P(None),
            norm_bias=P(None),
            codebook=P(None, None),
            conv_transpose_weight=P(None, None, None),
            conv_transpose_bias=P(None),
            istft_linear_weight=P(None, None),
            istft_linear_bias=P(None),
            istft_window=P(None),
            act_btd=P(None, None, None),
            act_btnh=P(None, None, None, None),
            act_btc=P(None, None, None),
        )

    @staticmethod
    def default():
        """Default sharding configuration for distributed training."""
        return MiMoShardingCfg(
            conv_weight=P(None, "tp", None),
            conv_bias=P("tp"),
            attn_qkvo_weight=P("fsdp", "tp"),
            attn_qkv_bias=P("tp"),
            attn_out_bias=P("tp"),
            ffn_weight_in=P("fsdp", "tp"),
            ffn_weight_out=P("tp", "fsdp"),
            ffn_bias=P("tp"),
            norm_scale=P("tp"),
            norm_bias=P("tp"),
            codebook=P("tp", "fsdp"),
            conv_transpose_weight=P(None, "tp", None),
            conv_transpose_bias=P("tp"),
            istft_linear_weight=P("fsdp", "tp"),
            istft_linear_bias=P("tp"),
            istft_window=P(None),  # replicated
            act_btd=P("fsdp", None, "tp"),
            act_btnh=P("fsdp", None, "tp", None),
            act_btc=P("fsdp", None, "tp"),
        )


class MiMoAudioTokenizerConfig(PretrainedConfig):
    model_type = "mimo_audio_tokenizer"

    def __init__(
            self,
            max_audio_seconds: int = 1800,
            stride_size: int = 2,
            avg_pooler: int = 2,
            d_model: int = 1280,
            scale_embedding: bool = False,
            kernel_size: int = 3,
            activation_function: str = "gelu",
            encoder_layers: int = 32,
            encoder_skip_layer_id: int = 3,
            encoder_attention_heads: int = 20,
            encoder_ffn_dim: int = 5120,
            encoder_causal: bool = False,
            encoder_attn_window_size: list[int] = None,  # [-1,-1]
            decoder_layers: int = 32,
            decoder_attention_heads: int = 20,
            decoder_ffn_dim: int = 5120,
            decoder_kernel_size: int = 3,
            decoder_stride_size: int = 2,
            decoder_causal: bool = True,
            decoder_attn_window_size: list[int] = None,  # [-1,-1]
            nfft: int = 960,
            vocoder_dim: int = 256,
            vocoder_intermediate_dim: int = 1024,
            vocoder_num_layers: int = 16,
            n_mels: int = 128,
            sampling_rate: int = 24000,
            hop_length: int = 240,
            window_size: int = 960,
            vocoder_padding: str = "same",
            fmin: int = 0,
            fmax: int = None,
            num_quantizers: int = 20,
            codebook_size: list[int] = None,
            # [1024,1024,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128]
            threshold_ema_dead_code: int = 2,
            position_embedding_type: str = "rope",
            rope_theta: int = 10000,
            rope_type: str = "default",
            ln_type: str = "LayerNorm",
            vocoder_attention_heads: int = 16,
            vocoder_attn_window_size: list[int] = None,  # [40,10]
            use_sharding: bool = False,
            shd_cfg: MiMoShardingCfg | None = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_audio_seconds = max_audio_seconds
        self.stride_size = stride_size
        self.avg_pooler = avg_pooler
        self.d_model = d_model
        self.scale_embedding = scale_embedding
        self.kernel_size = kernel_size
        self.activation_function = activation_function
        self.encoder_layers = encoder_layers
        self.encoder_skip_layer_id = encoder_skip_layer_id
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_causal = encoder_causal
        self.encoder_attn_window_size = (
            encoder_attn_window_size
            if encoder_attn_window_size is not None
            else [-1, -1]
        )
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_kernel_size = decoder_kernel_size
        self.decoder_stride_size = decoder_stride_size
        self.decoder_causal = decoder_causal
        self.decoder_attn_window_size = (
            decoder_attn_window_size
            if decoder_attn_window_size is not None
            else [-1, -1]
        )
        self.nfft = nfft
        self.vocoder_dim = vocoder_dim
        self.vocoder_intermediate_dim = vocoder_intermediate_dim
        self.vocoder_num_layers = vocoder_num_layers
        self.n_mels = n_mels
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.window_size = window_size
        self.vocoder_padding = vocoder_padding
        self.fmin = fmin
        self.fmax = fmax
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size if codebook_size is not None else [1024]
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.position_embedding_type = position_embedding_type
        self.rope_theta = rope_theta
        self.rope_type = rope_type
        self.ln_type = ln_type
        self.vocoder_attention_heads = vocoder_attention_heads
        self.vocoder_attn_window_size = (
            vocoder_attn_window_size
            if vocoder_attn_window_size is not None
            else [40, 10]
        )

        # Sharding configuration
        if shd_cfg is None:
            self.shd_cfg = MiMoShardingCfg.default() if use_sharding else MiMoShardingCfg.no_sharding()
        else:
            self.shd_cfg = shd_cfg


@dataclass
class EncoderOutput:
    hidden_states: Array
    packed_states: Array
    output_lengths: Array
    codes: Optional[Array]


@dataclass
class VocoderOutput:
    wav: Array
    wav_lengths: Array
