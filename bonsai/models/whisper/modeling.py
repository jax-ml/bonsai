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

"""Whisper model implementation in JAX NNX."""

from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jaxtyping import Array


def sinusoids(length: int, channels: int) -> Array:
    """Generate sinusoidal positional embeddings."""
    position = jnp.arange(length, dtype=jnp.float32)[:, None]
    div_term = jnp.exp(jnp.arange(0, channels, 2, dtype=jnp.float32) * -(jnp.log(10000.0) / channels))
    pos_emb = jnp.zeros((length, channels), dtype=jnp.float32)
    pos_emb = pos_emb.at[:, 0::2].set(jnp.sin(position * div_term))
    pos_emb = pos_emb.at[:, 1::2].set(jnp.cos(position * div_term))
    return pos_emb


class WhisperConfig:
    """Configuration for Whisper model."""
    
    def __init__(
        self,
        vocab_size: int = 51865,
        n_mels: int = 80,
        n_audio_ctx: int = 1500,
        n_audio_state: int = 384,
        n_audio_head: int = 6,
        n_audio_layer: int = 4,
        n_text_ctx: int = 448,
        n_text_state: int = 384,
        n_text_head: int = 6,
        n_text_layer: int = 4,
        n_vocab: int = 51865,
        n_langs: int = 99,
        dtype=jnp.float32,
    ):
        self.vocab_size = vocab_size
        self.n_mels = n_mels
        self.n_audio_ctx = n_audio_ctx
        self.n_audio_state = n_audio_state
        self.n_audio_head = n_audio_head
        self.n_audio_layer = n_audio_layer
        self.n_text_ctx = n_text_ctx
        self.n_text_state = n_text_state
        self.n_text_head = n_text_head
        self.n_text_layer = n_text_layer
        self.n_vocab = n_vocab
        self.n_langs = n_langs
        self.dtype = dtype
    
    @classmethod
    def whisper_tiny(cls):
        return cls(
            vocab_size=51865,
            n_mels=80,
            n_audio_ctx=1500,
            n_audio_state=384,
            n_audio_head=6,
            n_audio_layer=4,
            n_text_ctx=448,
            n_text_state=384,
            n_text_head=6,
            n_text_layer=4,
            n_vocab=51865,
            n_langs=99,
        )
    
    @classmethod
    def whisper_base(cls):
        return cls(
            vocab_size=51865,
            n_mels=80,
            n_audio_ctx=1500,
            n_audio_state=512,
            n_audio_head=8,
            n_audio_layer=6,
            n_text_ctx=448,
            n_text_state=512,
            n_text_head=8,
            n_text_layer=6,
            n_vocab=51865,
            n_langs=99,
        )
    
    @classmethod
    def whisper_small(cls):
        return cls(
            vocab_size=51865,
            n_mels=80,
            n_audio_ctx=1500,
            n_audio_state=768,
            n_audio_head=12,
            n_audio_layer=12,
            n_text_ctx=448,
            n_text_state=768,
            n_text_head=12,
            n_text_layer=12,
            n_vocab=51865,
            n_langs=99,
        )
    
    @classmethod
    def whisper_medium(cls):
        return cls(
            vocab_size=51865,
            n_mels=80,
            n_audio_ctx=1500,
            n_audio_state=1024,
            n_audio_head=16,
            n_audio_layer=24,
            n_text_ctx=448,
            n_text_state=1024,
            n_text_head=16,
            n_text_layer=24,
            n_vocab=51865,
            n_langs=99,
        )
    
    @classmethod
    def whisper_large(cls):
        return cls(
            vocab_size=51865,
            n_mels=80,
            n_audio_ctx=1500,
            n_audio_state=1280,
            n_audio_head=20,
            n_audio_layer=32,
            n_text_ctx=448,
            n_text_state=1280,
            n_text_head=20,
            n_text_layer=32,
            n_vocab=51865,
            n_langs=99,
        )


class MultiHeadAttention(nnx.Module):
    """Multi-head attention layer."""
    
    def __init__(self, config: WhisperConfig, is_cross_attention: bool = False, rngs=None):
        self.config = config
        self.n_head = config.n_text_head if is_cross_attention else config.n_audio_head
        self.n_state = config.n_text_state if is_cross_attention else config.n_audio_state
        self.is_cross_attention = is_cross_attention
        
        # Projection matrices - match HF structure exactly
        self.q_proj = nnx.Linear(self.n_state, self.n_state, rngs=rngs)  # has bias
        self.k_proj = nnx.Linear(self.n_state, self.n_state, use_bias=False, rngs=rngs)  # no bias
        self.v_proj = nnx.Linear(self.n_state, self.n_state, rngs=rngs)  # has bias
        self.out_proj = nnx.Linear(self.n_state, self.n_state, rngs=rngs)  # has bias
    
    def __call__(self, x: Array, xa: Optional[Array] = None, mask: Optional[Array] = None) -> Array:
        batch_size, seq_len, n_state = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq, n_state)
        k = self.k_proj(xa if xa is not None else x)  # (batch, seq, n_state)
        v = self.v_proj(xa if xa is not None else x)  # (batch, seq, n_state)
        
        # Reshape to multi-head
        q = q.reshape(batch_size, seq_len, self.n_head, n_state // self.n_head).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, -1, self.n_head, n_state // self.n_head).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, -1, self.n_head, n_state // self.n_head).transpose(0, 2, 1, 3)
        
        # Attention
        scale = (n_state // self.n_head) ** -0.5
        attn_weights = jnp.einsum('bhid,bhjd->bhij', q, k) * scale
        
        if mask is not None:
            attn_weights = attn_weights + mask
        
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_output = jnp.einsum('bhij,bhjd->bhid', attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, n_state)
        
        # Output projection
        return self.out_proj(attn_output)


class ResidualAttentionBlock(nnx.Module):
    """Residual attention block with self-attention and MLP."""
    
    def __init__(self, config: WhisperConfig, is_decoder: bool = False, rngs=None):
        self.config = config
        self.is_decoder = is_decoder
        
        # Self-attention
        n_state = config.n_text_state if is_decoder else config.n_audio_state
        self.self_attn_layer_norm = nnx.LayerNorm(n_state, rngs=rngs)
        self.self_attn = MultiHeadAttention(config, is_cross_attention=False, rngs=rngs)
        
        # Cross attention (for decoder only)
        if is_decoder:
            self.encoder_attn_layer_norm = nnx.LayerNorm(n_state, rngs=rngs)
            self.encoder_attn = MultiHeadAttention(config, is_cross_attention=True, rngs=rngs)
        
        # MLP
        self.final_layer_norm = nnx.LayerNorm(n_state, rngs=rngs)
        self.fc1 = nnx.Linear(n_state, n_state * 4, rngs=rngs)
        self.fc2 = nnx.Linear(n_state * 4, n_state, rngs=rngs)
    
    def __call__(self, x: Array, xa: Optional[Array] = None, mask: Optional[Array] = None) -> Array:
        # Self-attention
        attn_in = self.self_attn_layer_norm(x)
        attn_out = self.self_attn(attn_in, mask=mask)
        x = x + attn_out
        
        # Cross attention (for decoder only)
        if self.is_decoder and xa is not None:
            cross_attn_in = self.encoder_attn_layer_norm(x)
            cross_attn_out = self.encoder_attn(cross_attn_in, xa)
            x = x + cross_attn_out
        
        # MLP
        mlp_in = self.final_layer_norm(x)
        mlp_hidden = jax.nn.gelu(self.fc1(mlp_in))
        mlp_out = self.fc2(mlp_hidden)
        x = x + mlp_out
        
        return x


class AudioEncoder(nnx.Module):
    """Audio encoder that processes mel spectrogram features."""
    
    def __init__(self, config: WhisperConfig, rngs=None):
        self.config = config
        
        # Two 1D conv layers with stride 2
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
        
        # Learned positional embeddings
        pos_key = rngs.params() if rngs is not None else jax.random.PRNGKey(0)
        self.embed_positions = nnx.Variable(
            jax.random.normal(pos_key, (config.n_audio_ctx, config.n_audio_state), dtype=config.dtype)
        )
        
        # Transformer layers
        self.layers = [
            ResidualAttentionBlock(config, is_decoder=False, rngs=rngs)
            for _ in range(config.n_audio_layer)
        ]
        
        # Final layer norm
        self.layer_norm = nnx.LayerNorm(config.n_audio_state, rngs=rngs)
    
    def __call__(self, x: Array) -> Array:
        # x shape: (batch, n_mels, time) -> (batch, time, n_mels) for Conv1D
        x = x.transpose(0, 2, 1)
        
        # Conv stack with GELU
        x = jax.nn.gelu(self.conv1(x))
        x = jax.nn.gelu(self.conv2(x))
        
        # Add positional embeddings
        seq_len = x.shape[1]
        x = x + self.embed_positions[:seq_len]
        
        # Transformer encoder
        for layer in self.layers:
            x = layer(x)
        
        # Final layer norm
        x = self.layer_norm(x)
        
        return x


class TextDecoder(nnx.Module):
    """Text decoder that generates text tokens."""
    
    def __init__(self, config: WhisperConfig, rngs=None):
        self.config = config
        
        # Token embedding
        self.embed_tokens = nnx.Embed(config.n_vocab, config.n_text_state, rngs=rngs)
        
        # Positional encoding
        self.embed_positions = nnx.Variable(
            sinusoids(config.n_text_ctx, config.n_text_state)
        )
        
        # Transformer layers
        self.layers = [
            ResidualAttentionBlock(config, is_decoder=True, rngs=rngs)
            for _ in range(config.n_text_layer)
        ]
        
        # Final layer norm
        self.layer_norm = nnx.LayerNorm(config.n_text_state, rngs=rngs)
    
    def __call__(self, x: Array, xa: Array, mask: Optional[Array] = None) -> Array:
        # Token embedding
        x = self.embed_tokens(x)
        
        # Add positional embeddings
        seq_len = x.shape[1]
        x = x + self.embed_positions[:seq_len]
        
        # Transformer decoder
        for layer in self.layers:
            x = layer(x, xa, mask)
        
        # Final layer norm
        x = self.layer_norm(x)
        
        # Weight tying: use token embeddings for output projection
        logits = jnp.dot(x, self.embed_tokens.embedding.T)
        
        return logits


class WhisperModel(nnx.Module):
    """Whisper model for speech recognition."""
    
    def __init__(self, config: WhisperConfig, rngs=None):
        self.config = config
        
        # Audio encoder
        self.encoder = AudioEncoder(config, rngs=rngs)
        
        # Text decoder
        self.decoder = TextDecoder(config, rngs=rngs)
    
    def __call__(self, mel_features: Array, tokens: Array, mask: Optional[Array] = None) -> Array:
        # Encode audio
        xa = self.encoder(mel_features)
        
        # Decode text
        logits = self.decoder(tokens, xa, mask)
        
        return logits


def create_causal_mask(size: int) -> Array:
    """Create causal mask for decoder attention."""
    mask = jnp.triu(jnp.ones((size, size)), k=1)
    return mask * -1e9


def forward(model: WhisperModel, mel_features: Array, tokens: Array) -> Array:
    """Forward pass through the model."""
    # Create causal mask for decoder
    seq_len = tokens.shape[1]
    mask = create_causal_mask(seq_len)
    
    return model(mel_features, tokens, mask)


def generate(model: WhisperModel, mel_features: Array, max_length: int = 448, temperature: float = 0.0) -> Array:
    """Generate text tokens from audio features."""
    batch_size = mel_features.shape[0]
    
    # Start with HF Whisper prompt tokens: <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
    # HF uses forced_decoder_ids: position 1=<|en|>, 2=<|transcribe|>, 3=<|notimestamps|>
    prompt_tokens = jnp.array([[50258, 50259, 50359, 50363]])
    tokens = jnp.repeat(prompt_tokens, batch_size, axis=0)
    
    # Encode audio once
    xa = model.encoder(mel_features)
    
    for _ in range(max_length - len(prompt_tokens[0])):
        # Create causal mask
        seq_len = tokens.shape[1]
        mask = create_causal_mask(seq_len)
        
        # Get logits
        logits = model.decoder(tokens, xa, mask)
        
        # Get next token (greedy decoding)
        next_logits = logits[:, -1, :]
        next_token = jnp.argmax(next_logits, axis=-1, keepdims=True)
        
        # Append to sequence
        tokens = jnp.concatenate([tokens, next_token], axis=1)
        
        # Stop if EOS token
        if jnp.any(next_token == 50257):  # EOS token
            break
    
    return tokens


def extract_mel_features(audio: Array, sample_rate: int = 16000, n_mels: int = 80) -> Array:
    """Extract mel spectrogram features from audio."""
    # This is a simplified version - in practice you'd use librosa
    # For now, return random features for testing
    import numpy as np
    try:
        import librosa
        audio_np = np.array(audio)
        mel_spec = librosa.feature.melspectrogram(
            y=audio_np, 
            sr=sample_rate, 
            n_mels=n_mels,
            hop_length=160,
            win_length=400,
            window='hann'
        )
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return jnp.array(mel_spec.T)  # Transpose to (time, n_mels)
    except ImportError:
        # Fallback to random features
        time_steps = len(audio) // 160
        return jax.random.normal(jax.random.PRNGKey(0), (time_steps, n_mels))
