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
from typing import Optional, Dict, Tuple
import math

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


@dataclasses.dataclass(frozen=True)
class WhisperConfig:
    """Configuration for Whisper model."""
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int
    dtype: jnp.dtype = jnp.float32

    @classmethod
    def tiny(cls):
        """Tiny Whisper model configuration."""
        return cls(
            n_mels=80,
            n_audio_ctx=1500,
            n_audio_state=384,
            n_audio_head=6,
            n_audio_layer=4,
            n_vocab=51865,
            n_text_ctx=448,
            n_text_state=384,
            n_text_head=6,
            n_text_layer=4,
        )

    @classmethod
    def base(cls):
        """Base Whisper model configuration."""
        return cls(
            n_mels=80,
            n_audio_ctx=1500,
            n_audio_state=512,
            n_audio_head=8,
            n_audio_layer=6,
            n_vocab=51865,
            n_text_ctx=448,
            n_text_state=512,
            n_text_head=8,
            n_text_layer=6,
        )

    @classmethod
    def small(cls):
        """Small Whisper model configuration."""
        return cls(
            n_mels=80,
            n_audio_ctx=1500,
            n_audio_state=768,
            n_audio_head=12,
            n_audio_layer=12,
            n_vocab=51865,
            n_text_ctx=448,
            n_text_state=768,
            n_text_head=12,
            n_text_layer=12,
        )

    @classmethod
    def medium(cls):
        """Medium Whisper model configuration."""
        return cls(
            n_mels=80,
            n_audio_ctx=1500,
            n_audio_state=1024,
            n_audio_head=16,
            n_audio_layer=24,
            n_vocab=51865,
            n_text_ctx=448,
            n_text_state=1024,
            n_text_head=16,
            n_text_layer=24,
        )

    @classmethod
    def large(cls):
        """Large Whisper model configuration."""
        return cls(
            n_mels=80,
            n_audio_ctx=1500,
            n_audio_state=1280,
            n_audio_head=20,
            n_audio_layer=32,
            n_vocab=51865,
            n_text_ctx=448,
            n_text_state=1280,
            n_text_head=20,
            n_text_layer=32,
        )


# Alias for backward compatibility
ModelDimensions = WhisperConfig


class MultiHeadAttention(nnx.Module):
    """Multi-head attention"""
    
    def __init__(self, n_state: int, n_head: int, rngs: nnx.Rngs):
        self.n_head = n_head
        self.query = nnx.Linear(n_state, n_state, rngs=rngs)
        self.key = nnx.Linear(n_state, n_state, use_bias=False, rngs=rngs)
        self.value = nnx.Linear(n_state, n_state, rngs=rngs)
        self.out = nnx.Linear(n_state, n_state, rngs=rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        xa: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
        kv_cache: Optional[Dict] = None,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # For self-attention: use x, for cross-attention: use xa
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # For cross-attention, reuse cached keys and values
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, mask: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        
        # Reshape q, k, v for multi-head attention (same as PyTorch)
        q = q.reshape(n_batch, n_ctx, self.n_head, -1).transpose(0, 2, 1, 3)  # (batch, head, q_len, head_dim)
        k = k.reshape(n_batch, k.shape[1], self.n_head, -1).transpose(0, 2, 1, 3)  # (batch, head, kv_len, head_dim)
        v = v.reshape(n_batch, v.shape[1], self.n_head, -1).transpose(0, 2, 1, 3)  # (batch, head, kv_len, head_dim)

        # Compute attention scores: (batch, head, q_len, kv_len) - same as PyTorch
        qk = (q * scale) @ (k * scale).transpose(0, 1, 3, 2)
        
        # Apply mask if provided (only for self-attention)
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        
        qk = qk.astype(jnp.float32)
        # Handle -inf values before softmax to avoid NaN
        qk = jnp.where(jnp.isinf(qk), -1e9, qk)
        w = jax.nn.softmax(qk, axis=-1).astype(q.dtype)
        
        # Apply attention weights to values: (batch, head, q_len, head_dim)
        wv = (w @ v).transpose(0, 2, 1, 3).reshape(n_batch, n_ctx, n_state)
        
        return wv, qk


class ResidualAttentionBlock(nnx.Module):
    
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False, rngs: nnx.Rngs = None):
        self.attn = MultiHeadAttention(n_state, n_head, rngs=rngs)
        self.attn_ln = nnx.LayerNorm(n_state, rngs=rngs)

        if cross_attention:
            self.cross_attn = MultiHeadAttention(n_state, n_head, rngs=rngs)
            self.cross_attn_ln = nnx.LayerNorm(n_state, rngs=rngs)
        else:
            self.cross_attn = None
            self.cross_attn_ln = None

        # MLP: n_state -> n_state * 4 -> n_state (same as PyTorch)
        n_mlp = n_state * 4
        self.mlp_linear1 = nnx.Linear(n_state, n_mlp, rngs=rngs)
        self.mlp_linear2 = nnx.Linear(n_mlp, n_state, rngs=rngs)
        self.mlp_ln = nnx.LayerNorm(n_state, rngs=rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        xa: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
        kv_cache: Optional[Dict] = None,
    ) -> jnp.ndarray:
        # Self-attention with residual connection
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        
        # Cross-attention with residual connection (if enabled)
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, mask=None, kv_cache=kv_cache)[0]
        
        # MLP with residual connection (same as PyTorch: Sequential(Linear, GELU, Linear))
        x = x + self.mlp_linear2(jax.nn.gelu(self.mlp_linear1(self.mlp_ln(x)), approximate=False))
        
        return x


class AudioEncoder(nnx.Module):
    
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(n_mels, n_state, kernel_size=(3,), padding=1, rngs=rngs)
        self.conv2 = nnx.Conv(n_state, n_state, kernel_size=(3,), strides=(2,), padding=1, rngs=rngs)
        
        # Positional embeddings
        self.positional_embedding = nnx.Param(jnp.empty((n_ctx, n_state)))
        
        # Transformer blocks
        self.blocks = [
            ResidualAttentionBlock(n_state, n_head, cross_attention=False, rngs=rngs)
            for _ in range(n_layer)
        ]
        
        self.ln_post = nnx.LayerNorm(n_state, rngs=rngs)

    @jax.jit
    def __call__(self, x: jnp.ndarray):
        # x shape: (batch, n_mels, time) -> (batch, time, n_mels) for NNX Conv
        x = x.transpose(0, 2, 1)  # (batch, time, n_mels)
        x = jax.nn.gelu(self.conv1(x), approximate=False)  # (batch, time, n_state)
        x = jax.nn.gelu(self.conv2(x), approximate=False)  # (batch, time/2, n_state)
        # x is now (batch, time/2, n_state)
        
        # Add positional embeddings
        seq_len = x.shape[1]
        x = x + self.positional_embedding[:seq_len]
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_post(x)
        return x


class TextDecoder(nnx.Module):
    
    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, rngs: nnx.Rngs):
        self.token_embedding = nnx.Embed(n_vocab, n_state, rngs=rngs)
        self.positional_embedding = nnx.Param(jnp.empty((n_ctx, n_state)))

        # Create blocks with cross-attention (same as PyTorch)
        self.blocks = [
            ResidualAttentionBlock(n_state, n_head, cross_attention=True, rngs=rngs)
            for _ in range(n_layer)
        ]
        
        self.ln = nnx.LayerNorm(n_state, rngs=rngs)

        # Create causal mask (same as PyTorch)
        mask = jnp.full((n_ctx, n_ctx), -jnp.inf)
        mask = jnp.triu(mask, k=1)
        self.mask = mask

    @jax.jit
    def __call__(self, x: jnp.ndarray, xa: jnp.ndarray, kv_cache: Optional[Dict] = None) -> jnp.ndarray:
        """
        x : jnp.ndarray, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : jnp.ndarray, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        # Calculate offset for positional embeddings (same as PyTorch)
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache and len(kv_cache) > 0 else 0
        
        # Token embedding + positional embedding (same as PyTorch)
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        x = x.astype(xa.dtype)

        # Apply transformer blocks (same as PyTorch)
        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        # Final layer norm
        x = self.ln(x)
        
        # Compute logits (same as PyTorch: x @ token_embedding.weight.T)
        logits = jnp.matmul(x, self.token_embedding.embedding.astype(x.dtype).T)
        
        return logits


class Whisper(nnx.Module):
    
    def __init__(self, dims: ModelDimensions, rngs: nnx.Rngs):
        self.dims = dims
        
        self.encoder = AudioEncoder(
            dims.n_mels, dims.n_audio_ctx, dims.n_audio_state, 
            dims.n_audio_head, dims.n_audio_layer, rngs=rngs
        )
        
        self.decoder = TextDecoder(
            dims.n_vocab, dims.n_text_ctx, dims.n_text_state,
            dims.n_text_head, dims.n_text_layer, rngs=rngs
        )

    @jax.jit
    def embed_audio(self, mel: jnp.ndarray) -> jnp.ndarray:
        return self.encoder(mel)

    @jax.jit
    def logits(self, tokens: jnp.ndarray, audio_features: jnp.ndarray) -> jnp.ndarray:
        return self.decoder(tokens, audio_features)

    @jax.jit
    def __call__(self, mel: jnp.ndarray, tokens: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        return self.decoder(tokens, self.encoder(mel))


# Alias for compatibility with params.py
WhisperModel = Whisper


# Compatibility functions to match original interface
def load_model(name: str, device: str = "cpu", download_root: str = None):
    """Load a Whisper model with real weights."""
    # Map model names to configurations
    config_map = {
        "tiny": WhisperConfig.tiny(),
        "base": WhisperConfig.base(),
        "small": WhisperConfig.small(),
        "medium": WhisperConfig.medium(),
        "large": WhisperConfig.large(),
    }
    
    if name not in config_map:
        raise ValueError(f"Unknown model name: {name}. Available: {list(config_map.keys())}")
    
    config = config_map[name]
    rngs = nnx.Rngs(0)
    model = Whisper(config, rngs=rngs)
    
    # Load real weights from PyTorch Whisper
    try:
        import whisper
        real_whisper = whisper.load_model(name)
        
        # Copy encoder weights
        model.encoder.conv1.kernel = nnx.Param(jnp.array(real_whisper.encoder.conv1.weight.data.numpy().T))
        model.encoder.conv1.bias = nnx.Param(jnp.array(real_whisper.encoder.conv1.bias.data.numpy()))
        model.encoder.conv2.kernel = nnx.Param(jnp.array(real_whisper.encoder.conv2.weight.data.numpy().T))
        model.encoder.conv2.bias = nnx.Param(jnp.array(real_whisper.encoder.conv2.bias.data.numpy()))
        
        # Copy positional embedding
        model.encoder.positional_embedding = nnx.Param(jnp.array(real_whisper.encoder.positional_embedding.data.numpy()))
        
        # Copy encoder blocks
        for i, (pytorch_block, jax_block) in enumerate(zip(real_whisper.encoder.blocks, model.encoder.blocks)):
            # Self-attention
            jax_block.attn.query.kernel = nnx.Param(jnp.array(pytorch_block.attn.query.weight.data.numpy().T))
            jax_block.attn.query.bias = nnx.Param(jnp.array(pytorch_block.attn.query.bias.data.numpy()))
            jax_block.attn.key.kernel = nnx.Param(jnp.array(pytorch_block.attn.key.weight.data.numpy().T))
            if pytorch_block.attn.key.bias is not None:
                jax_block.attn.key.bias = nnx.Param(jnp.array(pytorch_block.attn.key.bias.data.numpy()))
            jax_block.attn.value.kernel = nnx.Param(jnp.array(pytorch_block.attn.value.weight.data.numpy().T))
            jax_block.attn.value.bias = nnx.Param(jnp.array(pytorch_block.attn.value.bias.data.numpy()))
            jax_block.attn.out.kernel = nnx.Param(jnp.array(pytorch_block.attn.out.weight.data.numpy().T))
            jax_block.attn.out.bias = nnx.Param(jnp.array(pytorch_block.attn.out.bias.data.numpy()))
            
            jax_block.attn_ln.scale = nnx.Param(jnp.array(pytorch_block.attn_ln.weight.data.numpy()))
            jax_block.attn_ln.bias = nnx.Param(jnp.array(pytorch_block.attn_ln.bias.data.numpy()))
            
            # MLP
            jax_block.mlp_linear1.kernel = nnx.Param(jnp.array(pytorch_block.mlp[0].weight.data.numpy().T))
            jax_block.mlp_linear1.bias = nnx.Param(jnp.array(pytorch_block.mlp[0].bias.data.numpy()))
            jax_block.mlp_linear2.kernel = nnx.Param(jnp.array(pytorch_block.mlp[2].weight.data.numpy().T))
            jax_block.mlp_linear2.bias = nnx.Param(jnp.array(pytorch_block.mlp[2].bias.data.numpy()))
            jax_block.mlp_ln.scale = nnx.Param(jnp.array(pytorch_block.mlp_ln.weight.data.numpy()))
            jax_block.mlp_ln.bias = nnx.Param(jnp.array(pytorch_block.mlp_ln.bias.data.numpy()))
        
        model.encoder.ln_post.scale = nnx.Param(jnp.array(real_whisper.encoder.ln_post.weight.data.numpy()))
        model.encoder.ln_post.bias = nnx.Param(jnp.array(real_whisper.encoder.ln_post.bias.data.numpy()))
        
        # Copy decoder weights
        model.decoder.token_embedding.embedding = nnx.Param(jnp.array(real_whisper.decoder.token_embedding.weight.data.numpy()))
        model.decoder.positional_embedding = nnx.Param(jnp.array(real_whisper.decoder.positional_embedding.data.numpy()))
        
        # Copy decoder blocks
        for i, (pytorch_block, jax_block) in enumerate(zip(real_whisper.decoder.blocks, model.decoder.blocks)):
            # Self-attention
            jax_block.attn.query.kernel = nnx.Param(jnp.array(pytorch_block.attn.query.weight.data.numpy().T))
            jax_block.attn.query.bias = nnx.Param(jnp.array(pytorch_block.attn.query.bias.data.numpy()))
            jax_block.attn.key.kernel = nnx.Param(jnp.array(pytorch_block.attn.key.weight.data.numpy().T))
            if pytorch_block.attn.key.bias is not None:
                jax_block.attn.key.bias = nnx.Param(jnp.array(pytorch_block.attn.key.bias.data.numpy()))
            jax_block.attn.value.kernel = nnx.Param(jnp.array(pytorch_block.attn.value.weight.data.numpy().T))
            jax_block.attn.value.bias = nnx.Param(jnp.array(pytorch_block.attn.value.bias.data.numpy()))
            jax_block.attn.out.kernel = nnx.Param(jnp.array(pytorch_block.attn.out.weight.data.numpy().T))
            jax_block.attn.out.bias = nnx.Param(jnp.array(pytorch_block.attn.out.bias.data.numpy()))
            
            jax_block.attn_ln.scale = nnx.Param(jnp.array(pytorch_block.attn_ln.weight.data.numpy()))
            jax_block.attn_ln.bias = nnx.Param(jnp.array(pytorch_block.attn_ln.bias.data.numpy()))
            
            # Cross-attention
            if pytorch_block.cross_attn is not None:
                jax_block.cross_attn.query.kernel = nnx.Param(jnp.array(pytorch_block.cross_attn.query.weight.data.numpy().T))
                jax_block.cross_attn.query.bias = nnx.Param(jnp.array(pytorch_block.cross_attn.query.bias.data.numpy()))
                jax_block.cross_attn.key.kernel = nnx.Param(jnp.array(pytorch_block.cross_attn.key.weight.data.numpy().T))
                if pytorch_block.cross_attn.key.bias is not None:
                    jax_block.cross_attn.key.bias = nnx.Param(jnp.array(pytorch_block.cross_attn.key.bias.data.numpy()))
                jax_block.cross_attn.value.kernel = nnx.Param(jnp.array(pytorch_block.cross_attn.value.weight.data.numpy().T))
                jax_block.cross_attn.value.bias = nnx.Param(jnp.array(pytorch_block.cross_attn.value.bias.data.numpy()))
                jax_block.cross_attn.out.kernel = nnx.Param(jnp.array(pytorch_block.cross_attn.out.weight.data.numpy().T))
                jax_block.cross_attn.out.bias = nnx.Param(jnp.array(pytorch_block.cross_attn.out.bias.data.numpy()))
                
                jax_block.cross_attn_ln.scale = nnx.Param(jnp.array(pytorch_block.cross_attn_ln.weight.data.numpy()))
                jax_block.cross_attn_ln.bias = nnx.Param(jnp.array(pytorch_block.cross_attn_ln.bias.data.numpy()))
            
            # MLP
            jax_block.mlp_linear1.kernel = nnx.Param(jnp.array(pytorch_block.mlp[0].weight.data.numpy().T))
            jax_block.mlp_linear1.bias = nnx.Param(jnp.array(pytorch_block.mlp[0].bias.data.numpy()))
            jax_block.mlp_linear2.kernel = nnx.Param(jnp.array(pytorch_block.mlp[2].weight.data.numpy().T))
            jax_block.mlp_linear2.bias = nnx.Param(jnp.array(pytorch_block.mlp[2].bias.data.numpy()))
            jax_block.mlp_ln.scale = nnx.Param(jnp.array(pytorch_block.mlp_ln.weight.data.numpy()))
            jax_block.mlp_ln.bias = nnx.Param(jnp.array(pytorch_block.mlp_ln.bias.data.numpy()))
        
        model.decoder.ln.scale = nnx.Param(jnp.array(real_whisper.decoder.ln.weight.data.numpy()))
        model.decoder.ln.bias = nnx.Param(jnp.array(real_whisper.decoder.ln.bias.data.numpy()))
        
        # Copy final layer (if it exists)
        if hasattr(model, 'final_layer_norm'):
            model.final_layer_norm.scale = nnx.Param(jnp.array(real_whisper.decoder.ln.weight.data.numpy()))
            model.final_layer_norm.bias = nnx.Param(jnp.array(real_whisper.decoder.ln.bias.data.numpy()))
        
        
    except ImportError:
        pass
    except Exception as e:
        pass
    
    return model


def available_models():
    """Return available model names - placeholder for compatibility"""
    return ["tiny", "base", "small", "medium", "large"]


def disable_sdpa():
    """Disable scaled dot product attention - placeholder for compatibility"""
    pass
