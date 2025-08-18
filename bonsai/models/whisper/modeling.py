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
import numpy as np
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


def extract_mel_features(audio: jnp.ndarray, sample_rate: int = 16000, n_mels: int = 80) -> jnp.ndarray:
    """Extract mel spectrogram features from audio."""
    try:
        import librosa
        # Convert to numpy for librosa
        audio_np = np.array(audio) if hasattr(audio, 'device') else audio
        mel_spec = librosa.feature.melspectrogram(
            y=audio_np, 
            sr=sample_rate, 
            n_mels=n_mels,
            hop_length=160,
            win_length=400,
            window='hann'
        )
        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return jnp.array(mel_spec.T)  # Transpose to (time, n_mels) and convert to jnp
    except ImportError:
        print("librosa not available, using dummy mel features")
        # Generate dummy mel features for testing using JAX
        time_steps = len(audio) // 160  # Approximate time steps
        return jax.random.normal(jax.random.PRNGKey(0), (time_steps, n_mels))


class MultiHeadAttention(nnx.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, config: WhisperConfig, is_cross_attention: bool = False, rngs=None):
        self.n_head = config.n_audio_head if not is_cross_attention else config.n_text_head
        self.n_state = config.n_audio_state if not is_cross_attention else config.n_text_state
        self.head_dim = self.n_state // self.n_head
        self.is_cross_attention = is_cross_attention
        
        # Linear projections
        self.query = nnx.Linear(self.n_state, self.n_state, use_bias=False, rngs=rngs)
        self.key = nnx.Linear(self.n_state, self.n_state, use_bias=False, rngs=rngs)
        self.value = nnx.Linear(self.n_state, self.n_state, use_bias=False, rngs=rngs)
        self.out = nnx.Linear(self.n_state, self.n_state, use_bias=False, rngs=rngs)

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
    
    def __init__(self, config: WhisperConfig, is_cross_attention: bool = False, rngs=None):
        self.attn = MultiHeadAttention(config, is_cross_attention, rngs=rngs)
        self.attn_ln = nnx.LayerNorm(config.n_audio_state if not is_cross_attention else config.n_text_state, rngs=rngs)
        
        # Cross attention for decoder
        if is_cross_attention:
            self.cross_attn = MultiHeadAttention(config, is_cross_attention=True, rngs=rngs)
            self.cross_attn_ln = nnx.LayerNorm(config.n_text_state, rngs=rngs)
        
        # MLP
        n_state = config.n_audio_state if not is_cross_attention else config.n_text_state
        self.mlp_ln = nnx.LayerNorm(n_state, rngs=rngs)
        self.mlp_fc1 = nnx.Linear(n_state, n_state * 4, rngs=rngs)
        self.mlp_fc2 = nnx.Linear(n_state * 4, n_state, rngs=rngs)

    def __call__(self, x: Array, xa: Optional[Array] = None, mask: Optional[Array] = None) -> Array:
        # Self attention
        x = x + self.attn(self.attn_ln(x), mask=mask)
        
        # Cross attention (for decoder)
        if hasattr(self, 'cross_attn') and xa is not None:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa)
        
        # MLP
        mlp_in = self.mlp_ln(x)
        mlp_hidden = jax.nn.gelu(self.mlp_fc1(mlp_in))
        mlp_out = self.mlp_fc2(mlp_hidden)
        x = x + mlp_out
        return x


class AudioEncoder(nnx.Module):
    """Audio encoder that processes mel spectrogram features.

    Matches Whisper: 2 Conv1D layers with stride 2, then Transformer blocks.
    """

    def __init__(self, config: WhisperConfig, rngs=None):
        self.config = config

        # Two 1D conv layers with stride 2 (time downsample by 4 total)
        self.conv1 = nnx.Conv(
            in_features=config.n_mels,
            out_features=config.n_audio_state,
            kernel_size=(3,),
            strides=(2,),
            padding="SAME",
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            in_features=config.n_audio_state,
            out_features=config.n_audio_state,
            kernel_size=(3,),
            strides=(2,),
            padding="SAME",
            rngs=rngs,
        )

        # Learned positional embeddings (length n_audio_ctx, dim n_audio_state)
        pos_key = rngs.params() if rngs is not None else jax.random.PRNGKey(0)
        self.positional_embedding = nnx.Variable(
            jax.random.normal(pos_key, (config.n_audio_ctx, config.n_audio_state), dtype=config.dtype)
        )

        # Transformer blocks
        self.blocks = [
            ResidualAttentionBlock(config, is_cross_attention=False, rngs=rngs)
            for _ in range(config.n_audio_layer)
        ]

        # Final layer norm
        self.ln_post = nnx.LayerNorm(config.n_audio_state, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        # x shape: (batch, n_mels, time) -> (batch, time, n_mels) for Conv1D
        x = x.transpose(0, 2, 1)

        # Conv stack with GELU
        x = jax.nn.gelu(self.conv1(x))
        x = jax.nn.gelu(self.conv2(x))

        # Add positional embeddings (trim to sequence length)
        seq_len = x.shape[1]
        x = x + self.positional_embedding[:seq_len]

        # Transformer encoder
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_post(x)

        return x


class TextDecoder(nnx.Module):
    """Text decoder that generates text tokens."""
    
    def __init__(self, config: WhisperConfig, rngs=None):
        self.config = config
        
        # Token embedding
        self.token_embedding = nnx.Embed(config.n_vocab, config.n_text_state, rngs=rngs)
        
        # Positional encoding
        self.positional_embedding = nnx.Variable(
            sinusoids(config.n_text_ctx, config.n_text_state)
        )
        
        # Transformer blocks
        self.blocks = [
            ResidualAttentionBlock(config, is_cross_attention=True, rngs=rngs)
            for _ in range(config.n_text_layer)
        ]
        
        # Final layer norm and output projection
        self.ln = nnx.LayerNorm(config.n_text_state, rngs=rngs)
        self.output_projection = nnx.Linear(config.n_text_state, config.n_vocab, use_bias=False, rngs=rngs)

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
    
    def __init__(self, config: WhisperConfig, rngs=None):
        self.config = config
        self.encoder = AudioEncoder(config, rngs=rngs)
        self.decoder = TextDecoder(config, rngs=rngs)

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
