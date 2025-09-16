# JAX NNX implementation of Whisper model
# This replaces the original PyTorch model.py with JAX NNX while keeping the same interface

import base64
import gzip
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import math

from decoding import decode as decode_function
from decoding import detect_language as detect_language_function
from transcribe import transcribe as transcribe_function

def _gelu_pytorch_compatible(x):
    """PyTorch-compatible GELU implementation."""
    # PyTorch uses the exact GELU formula: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    return 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))

@dataclass
class ModelDimensions:
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

class MultiHeadAttention(nnx.Module):
    def __init__(self, n_state: int, n_head: int, rngs: nnx.Rngs):
        self.n_head = n_head
        self.query = nnx.Linear(n_state, n_state, rngs=rngs)
        self.key = nnx.Linear(n_state, n_state, use_bias=False, rngs=rngs)
        self.value = nnx.Linear(n_state, n_state, rngs=rngs)
        self.out = nnx.Linear(n_state, n_state, rngs=rngs)

    def __call__(self, x: jnp.ndarray, xa: Optional[jnp.ndarray] = None, mask: Optional[jnp.ndarray] = None, kv_cache: Optional[Dict] = None):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(self, q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, mask: Optional[jnp.ndarray] = None):
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
        
        self.cross_attn = MultiHeadAttention(n_state, n_head, rngs=rngs) if cross_attention else None
        self.cross_attn_ln = nnx.LayerNorm(n_state, rngs=rngs) if cross_attention else None
        
        n_mlp = n_state * 4
        self.mlp_linear1 = nnx.Linear(n_state, n_mlp, rngs=rngs)
        self.mlp_linear2 = nnx.Linear(n_mlp, n_state, rngs=rngs)
        self.mlp_ln = nnx.LayerNorm(n_state, rngs=rngs)

    def __call__(self, x: jnp.ndarray, xa: Optional[jnp.ndarray] = None, mask: Optional[jnp.ndarray] = None, kv_cache: Optional[Dict] = None):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, mask=None, kv_cache=kv_cache)[0]
            
        # Use PyTorch-compatible GELU implementation
        x = x + self.mlp_linear2(_gelu_pytorch_compatible(self.mlp_linear1(self.mlp_ln(x))))
        return x

class AudioEncoder(nnx.Module):
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(n_mels, n_state, kernel_size=(3,), padding=1, rngs=rngs)
        self.conv2 = nnx.Conv(n_state, n_state, kernel_size=(3,), strides=(2,), padding=1, rngs=rngs)
        
        # Positional embeddings (this was missing!)
        self.positional_embedding = nnx.Param(jnp.empty((n_ctx, n_state)))
        
        # Transformer blocks
        self.blocks = [
            ResidualAttentionBlock(n_state, n_head, cross_attention=False, rngs=rngs)
            for _ in range(n_layer)
        ]
        
        self.ln_post = nnx.LayerNorm(n_state, rngs=rngs)

    def __call__(self, x: jnp.ndarray):
        # x shape: (batch, n_mels, time) -> (batch, time, n_mels) for NNX Conv
        x = x.transpose(0, 2, 1)  # (batch, time, n_mels)
        x = jax.nn.gelu(self.conv1(x))  # (batch, time, n_state)
        x = jax.nn.gelu(self.conv2(x))  # (batch, time/2, n_state)
        # x is now (batch, time/2, n_state)
        
        # Add positional embeddings (this was missing!)
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
        
        self.blocks = [
            ResidualAttentionBlock(n_state, n_head, cross_attention=True, rngs=rngs)
            for _ in range(n_layer)
        ]
        
        self.ln = nnx.LayerNorm(n_state, rngs=rngs)
        self.mask = jnp.full((n_ctx, n_ctx), -jnp.inf).at[jnp.triu_indices(n_ctx, k=1)].set(0)

    def __call__(self, x: jnp.ndarray, xa: jnp.ndarray, kv_cache: Optional[Dict] = None):
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache and len(kv_cache) > 0 else 0
        x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[1]]
        x = x.astype(xa.dtype)

        for i, block in enumerate(self.blocks):
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        # Use matmul instead of einsum to avoid dimension issues
        logits = jnp.matmul(x, self.token_embedding.embedding.astype(x.dtype).T)

        return logits

class Whisper(nnx.Module):
    def __init__(self, dims: ModelDimensions, rngs: nnx.Rngs):
        self.dims = dims
        self.encoder = AudioEncoder(
            dims.n_mels,
            dims.n_audio_ctx,
            dims.n_audio_state,
            dims.n_audio_head,
            dims.n_audio_layer,
            rngs=rngs
        )
        self.decoder = TextDecoder(
            dims.n_vocab,
            dims.n_text_ctx,
            dims.n_text_state,
            dims.n_text_head,
            dims.n_text_layer,
            rngs=rngs
        )
        
        # Add required attributes for compatibility
        self.is_multilingual = True  # Default to multilingual
        self.num_languages = 99  # Default number of languages

    def embed_audio(self, mel: jnp.ndarray):
        return self.encoder(mel)

    def logits(self, tokens: jnp.ndarray, audio_features: jnp.ndarray):
        return self.decoder(tokens, audio_features)

    def __call__(self, mel: jnp.ndarray, tokens: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        return self.decoder(tokens, self.encoder(mel))

# Compatibility functions to match original interface
def load_model(name: str, device: str = "cpu", download_root: str = None):
    """Load a Whisper model - placeholder for compatibility"""
    # This would normally load the actual model weights
    # For now, return a dummy model
    dims = ModelDimensions(
        n_mels=80,
        n_audio_ctx=1500,
        n_audio_state=384,
        n_audio_head=6,
        n_audio_layer=4,
        n_vocab=51865,
        n_text_ctx=448,
        n_text_state=384,
        n_text_head=6,
        n_text_layer=4
    )
    rngs = nnx.Rngs(0)
    return Whisper(dims, rngs=rngs)

def available_models():
    """Return available model names - placeholder for compatibility"""
    return ["tiny", "base", "small", "medium", "large"]

def disable_sdpa():
    """Disable scaled dot product attention - placeholder for compatibility"""
    pass
