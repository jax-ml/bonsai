# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
from functools import partial
from typing import Any, Optional, Tuple, TypeAlias

import flax
import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, ArrayLike

Array: TypeAlias = jnp.ndarray


@dataclasses.dataclass(frozen=True)
class WhisperConfig:
    """Configuration for Whisper model."""
    vocab_size: int = 51865
    n_mels: int = 80
    n_audio_ctx: int = 1500
    n_audio_state: int = 1280
    n_audio_head: int = 20
    n_audio_layer: int = 32
    n_text_ctx: int = 448
    n_text_state: int = 1280
    n_text_head: int = 20
    n_text_layer: int = 32
    n_vocab: int = 51865
    n_langs: int = 99
    dtype: jnp.dtype = jnp.float32

    @classmethod
    def whisper_tiny(cls):
        return cls(
            n_audio_state=384,
            n_audio_head=6,
            n_audio_layer=4,
            n_text_state=384,
            n_text_head=6,
            n_text_layer=4,
        )

    @classmethod
    def whisper_base(cls):
        return cls(
            n_audio_state=512,
            n_audio_head=8,
            n_audio_layer=6,
            n_text_state=512,
            n_text_head=8,
            n_text_layer=6,
        )

    @classmethod
    def whisper_small(cls):
        return cls(
            n_audio_state=768,
            n_audio_head=12,
            n_audio_layer=12,
            n_text_state=768,
            n_text_head=12,
            n_text_layer=12,
        )

    @classmethod
    def whisper_medium(cls):
        return cls(
            n_audio_state=1024,
            n_audio_head=16,
            n_audio_layer=24,
            n_text_state=1024,
            n_text_head=16,
            n_text_layer=24,
        )

    @classmethod
    def whisper_large(cls):
        return cls(
            n_audio_state=1280,
            n_audio_head=20,
            n_audio_layer=32,
            n_text_state=1280,
            n_text_head=20,
            n_text_layer=32,
        )


def sinusoids(length: int, channels: int, max_timescale: float = 10000.0) -> Array:
    """Generate sinusoids for positional encoding."""
    log_timescale_increment = jnp.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = jnp.exp(-log_timescale_increment * jnp.arange(channels // 2))
    scaled_time = jnp.arange(length)[:, None] * inv_timescales[None, :]
    return jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=1)


class MultiHeadAttention(nnx.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, config: WhisperConfig, is_cross_attention: bool = False):
        self.n_head = config.n_audio_head if not is_cross_attention else config.n_text_head
        self.n_state = config.n_audio_state if not is_cross_attention else config.n_text_state
        self.head_dim = self.n_state // self.n_head
        self.is_cross_attention = is_cross_attention
        
        # Linear projections
        self.query = nnx.Linear(self.n_state, self.n_state, use_bias=False)
        self.key = nnx.Linear(self.n_state, self.n_state, use_bias=False)
        self.value = nnx.Linear(self.n_state, self.n_state, use_bias=False)
        self.out = nnx.Linear(self.n_state, self.n_state, use_bias=False)

    def __call__(self, x: Array, xa: Optional[Array] = None, mask: Optional[Array] = None) -> Array:
        q = self.query(x)
        k = self.key(xa if xa is not None else x)
        v = self.value(xa if xa is not None else x)
        
        # Reshape for multi-head attention
        batch_size, seq_len, _ = x.shape
        q = q.reshape(batch_size, seq_len, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, -1, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, -1, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        
        # Compute attention
        scale = 1.0 / jnp.sqrt(self.head_dim)
        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        
        if mask is not None:
            scores = scores + mask
            
        attn_weights = jax.nn.softmax(scores, axis=-1)
        attn_output = jnp.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.n_state)
        return self.out(attn_output)


class ResidualAttentionBlock(nnx.Module):
    """Residual attention block with layer normalization."""
    
    def __init__(self, config: WhisperConfig, is_cross_attention: bool = False):
        self.attn = MultiHeadAttention(config, is_cross_attention)
        self.attn_ln = nnx.LayerNorm(config.n_audio_state if not is_cross_attention else config.n_text_state)
        
        # Cross attention for decoder
        if is_cross_attention:
            self.cross_attn = MultiHeadAttention(config, is_cross_attention=True)
            self.cross_attn_ln = nnx.LayerNorm(config.n_text_state)
        
        # MLP
        n_state = config.n_audio_state if not is_cross_attention else config.n_text_state
        self.mlp = nnx.Sequential([
            nnx.Linear(n_state, n_state * 4),
            lambda x: jax.nn.gelu(x),
            nnx.Linear(n_state * 4, n_state),
        ])
        self.mlp_ln = nnx.LayerNorm(n_state)

    def __call__(self, x: Array, xa: Optional[Array] = None, mask: Optional[Array] = None) -> Array:
        # Self attention
        x = x + self.attn(self.attn_ln(x), mask=mask)
        
        # Cross attention (for decoder)
        if hasattr(self, 'cross_attn') and xa is not None:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa)
        
        # MLP
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nnx.Module):
    """Audio encoder that processes mel spectrogram features."""
    
    def __init__(self, config: WhisperConfig):
        self.config = config
        
        # Convolutional layers
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), strides=(1, 1), padding='SAME')
        self.conv2 = nnx.Conv(32, 32, kernel_size=(3, 3), strides=(1, 1), padding='SAME')
        self.conv3 = nnx.Conv(32, 64, kernel_size=(3, 3), strides=(2, 1), padding='SAME')
        self.conv4 = nnx.Conv(64, 64, kernel_size=(3, 3), strides=(1, 1), padding='SAME')
        self.conv5 = nnx.Conv(64, 128, kernel_size=(3, 3), strides=(2, 1), padding='SAME')
        self.conv6 = nnx.Conv(128, 128, kernel_size=(3, 3), strides=(1, 1), padding='SAME')
        self.conv7 = nnx.Conv(128, 256, kernel_size=(3, 3), strides=(2, 1), padding='SAME')
        self.conv8 = nnx.Conv(256, 256, kernel_size=(3, 3), strides=(1, 1), padding='SAME')
        self.conv9 = nnx.Conv(256, 512, kernel_size=(3, 3), strides=(2, 1), padding='SAME')
        self.conv10 = nnx.Conv(512, 512, kernel_size=(3, 3), strides=(1, 1), padding='SAME')
        self.conv11 = nnx.Conv(512, 512, kernel_size=(3, 3), strides=(2, 1), padding='SAME')
        self.conv12 = nnx.Conv(512, 512, kernel_size=(3, 3), strides=(1, 1), padding='SAME')
        
        # Projection to transformer dimension
        self.projection = nnx.Linear(512, config.n_audio_state)
        
        # Positional encoding
        self.positional_embedding = nnx.Variable(
            sinusoids(config.n_audio_ctx, config.n_audio_state)
        )
        
        # Transformer blocks
        self.blocks = [
            ResidualAttentionBlock(config, is_cross_attention=False)
            for _ in range(config.n_audio_layer)
        ]
        
        # Final layer norm
        self.ln_post = nnx.LayerNorm(config.n_audio_state)

    def __call__(self, x: Array) -> Array:
        # x shape: (batch_size, n_mels, time)
        x = x[..., None]  # Add channel dimension
        
        # Convolutional layers
        x = jax.nn.gelu(self.conv1(x))
        x = jax.nn.gelu(self.conv2(x))
        x = jax.nn.gelu(self.conv3(x))
        x = jax.nn.gelu(self.conv4(x))
        x = jax.nn.gelu(self.conv5(x))
        x = jax.nn.gelu(self.conv6(x))
        x = jax.nn.gelu(self.conv7(x))
        x = jax.nn.gelu(self.conv8(x))
        x = jax.nn.gelu(self.conv9(x))
        x = jax.nn.gelu(self.conv10(x))
        x = jax.nn.gelu(self.conv11(x))
        x = jax.nn.gelu(self.conv12(x))
        
        # Global average pooling over mel dimension
        x = jnp.mean(x, axis=2)  # (batch_size, time, 512)
        
        # Project to transformer dimension
        x = self.projection(x)
        
        # Add positional encoding
        x = x + self.positional_embedding[:x.shape[1]]
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_post(x)
        
        return x


class TextDecoder(nnx.Module):
    """Text decoder that generates text tokens."""
    
    def __init__(self, config: WhisperConfig):
        self.config = config
        
        # Token embedding
        self.token_embedding = nnx.Embed(config.n_vocab, config.n_text_state)
        
        # Positional encoding
        self.positional_embedding = nnx.Variable(
            sinusoids(config.n_text_ctx, config.n_text_state)
        )
        
        # Transformer blocks
        self.blocks = [
            ResidualAttentionBlock(config, is_cross_attention=True)
            for _ in range(config.n_text_layer)
        ]
        
        # Final layer norm and output projection
        self.ln = nnx.LayerNorm(config.n_text_state)
        self.output_projection = nnx.Linear(config.n_text_state, config.n_vocab, use_bias=False)

    def __call__(self, x: Array, xa: Array, mask: Optional[Array] = None) -> Array:
        # Token embedding
        x = self.token_embedding(x)
        
        # Add positional encoding
        x = x + self.positional_embedding[:x.shape[1]]
        
        # Apply transformer blocks with cross attention to audio features
        for block in self.blocks:
            x = block(x, xa, mask)
        
        # Final layer norm and projection
        x = self.ln(x)
        x = self.output_projection(x)
        
        return x


class WhisperModel(nnx.Module):
    """Complete Whisper model for speech recognition."""
    
    def __init__(self, config: WhisperConfig):
        self.config = config
        self.encoder = AudioEncoder(config)
        self.decoder = TextDecoder(config)

    def encode(self, mel: Array) -> Array:
        """Encode audio mel spectrogram to features."""
        return self.encoder(mel)

    def decode(self, tokens: Array, audio_features: Array, mask: Optional[Array] = None) -> Array:
        """Decode tokens using audio features."""
        return self.decoder(tokens, audio_features, mask)

    def __call__(self, mel: Array, tokens: Array, mask: Optional[Array] = None) -> Array:
        """Forward pass: encode audio and decode text."""
        audio_features = self.encode(mel)
        return self.decode(tokens, audio_features, mask)


def create_causal_mask(seq_len: int) -> Array:
    """Create causal mask for autoregressive generation."""
    mask = jnp.triu(jnp.ones((seq_len, seq_len)), k=1)
    return mask * -1e9


@jax.jit
def forward(model: WhisperModel, mel: Array, tokens: Array, mask: Optional[Array] = None) -> Array:
    """Jitted forward pass for the Whisper model."""
    return model(mel, tokens, mask)


def generate(model: WhisperModel, mel: Array, max_length: int = 448, temperature: float = 0.0) -> Array:
    """Generate text tokens from audio features."""
    batch_size = mel.shape[0]
    
    # Start with BOS token (usually 50258 for Whisper)
    tokens = jnp.full((batch_size, 1), 50258, dtype=jnp.int32)
    
    # Encode audio features
    audio_features = model.encode(mel)
    
    for i in range(max_length - 1):
        # Create causal mask
        mask = create_causal_mask(tokens.shape[1])
        
        # Get logits
        logits = model.decode(tokens, audio_features, mask)
        next_token_logits = logits[:, -1, :]
        
        # Apply temperature
        if temperature > 0:
            next_token_logits = next_token_logits / temperature
        
        # Sample next token (greedy for temperature=0)
        if temperature == 0:
            next_tokens = jnp.argmax(next_token_logits, axis=-1, keepdims=True)
        else:
            next_tokens = jax.random.categorical(
                jax.random.PRNGKey(0), next_token_logits, axis=-1, keepdims=True
            )
        
        # Append to sequence
        tokens = jnp.concatenate([tokens, next_tokens], axis=-1)
        
        # Stop if EOS token is generated
        if jnp.any(next_tokens == 50257):  # EOS token
            break
    
    return tokens
