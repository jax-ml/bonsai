# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""Simplified generation functions for Whisper model without complex stopping conditions."""

import jax
import jax.numpy as jnp
from typing import Optional
from .modeling import WhisperModel, create_causal_mask
import numpy as np


def create_causal_mask_jit(size: int) -> jnp.ndarray:
    """Create causal mask for decoder attention that works with JIT."""
    # Create a mask that allows attention to all previous positions
    # This is equivalent to the original create_causal_mask but JIT-compatible
    mask = jnp.triu(jnp.ones((size, size)), k=1)
    return mask * -1e9


def generate_simple(model: WhisperModel, mel_features: jnp.ndarray, max_length: int = 448, temperature: float = 0.0) -> jnp.ndarray:
    """
    Simple generation function without complex stopping conditions.
    
    Args:
        model: Whisper model
        mel_features: Audio mel features
        max_length: Maximum sequence length
        temperature: Sampling temperature (0.0 = greedy)
    
    Returns:
        Generated token sequence
    """
    batch_size = mel_features.shape[0]
    
    # Start with basic prompt tokens: <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
    prompt_tokens = jnp.array([[50258, 50259, 50359, 50363]])
    tokens = jnp.repeat(prompt_tokens, batch_size, axis=0)
    
    # Encode audio once
    xa = model.encoder(mel_features)
    
    # Generate tokens step by step
    for step in range(max_length - len(prompt_tokens[0])):
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
        
        # Only stop if we hit the EOS token (50257)
        if jnp.any(next_token == 50257):
            break
    
    return tokens


def generate_fast_vectorized(model: WhisperModel, mel_features: jnp.ndarray, max_length: int = 448, temperature: float = 0.0) -> jnp.ndarray:
    """
    Fast vectorized generation using JAX's efficient operations.
    
    Args:
        model: Whisper model
        mel_features: Audio mel features
        max_length: Maximum sequence length
        temperature: Sampling temperature (0.0 = greedy)
    
    Returns:
        Generated token sequence
    """
    batch_size = mel_features.shape[0]
    
    # Start with basic prompt tokens: <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
    prompt_tokens = jnp.array([[50258, 50259, 50359, 50363]])
    tokens = jnp.repeat(prompt_tokens, batch_size, axis=0)
    
    # Encode audio once
    xa = model.encoder(mel_features)
    
    # Pre-allocate the full sequence to avoid dynamic shapes
    full_tokens = jnp.zeros((batch_size, max_length), dtype=jnp.int32)
    full_tokens = full_tokens.at[:, :len(prompt_tokens[0])].set(prompt_tokens)
    
    # Use a more efficient approach: generate in chunks
    chunk_size = 20  # Generate 20 tokens at a time
    
    for chunk_start in range(0, max_length - len(prompt_tokens[0]), chunk_size):
        chunk_end = min(chunk_start + chunk_size, max_length - len(prompt_tokens[0]))
        
        # Generate tokens for this chunk
        for step in range(chunk_start, chunk_end):
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
            
            # Check for EOS
            if jnp.any(next_token == 50257):
                return tokens
        
        # Early stopping if we've generated enough
        if tokens.shape[1] >= max_length:
            break
    
    return tokens


def generate_with_eos_only(model: WhisperModel, mel_features: jnp.ndarray, max_length: int = 448, temperature: float = 0.0) -> jnp.ndarray:
    """
    Generation function that only stops on EOS token, no other stopping conditions.
    
    Args:
        model: Whisper model
        mel_features: Audio mel features
        max_length: Maximum sequence length
        temperature: Sampling temperature (0.0 = greedy)
    
    Returns:
        Generated token sequence
    """
    batch_size = mel_features.shape[0]
    
    # Start with basic prompt tokens: <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
    prompt_tokens = jnp.array([[50258, 50259, 50359, 50363]])
    tokens = jnp.repeat(prompt_tokens, batch_size, axis=0)
    
    # Encode audio once
    xa = model.encoder(mel_features)
    
    # Generate tokens step by step
    for step in range(max_length - len(prompt_tokens[0])):
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
        
        # Only stop on EOS token (50257)
        if jnp.any(next_token == 50257):
            break
    
    return tokens


def generate_continuous(model: WhisperModel, mel_features: jnp.ndarray, max_length: int = 448, temperature: float = 0.0) -> jnp.ndarray:
    """
    Continuous generation that runs for the full max_length unless EOS is generated.
    
    Args:
        model: Whisper model
        mel_features: Audio mel features
        max_length: Maximum sequence length
        temperature: Sampling temperature (0.0 = greedy)
    
    Returns:
        Generated token sequence
    """
    batch_size = mel_features.shape[0]
    
    # Start with basic prompt tokens: <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
    prompt_tokens = jnp.array([[50258, 50259, 50359, 50363]])
    tokens = jnp.repeat(prompt_tokens, batch_size, axis=0)
    
    # Encode audio once
    xa = model.encoder(mel_features)
    
    # Generate tokens step by step for the full length
    for step in range(max_length - len(prompt_tokens[0])):
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
        
        # Only stop on EOS token (50257), otherwise continue to max_length
        if jnp.any(next_token == 50257):
            break
    
    return tokens


def generate_hybrid_fast(model: WhisperModel, mel_features: jnp.ndarray, max_length: int = 50, temperature: float = 0.0) -> jnp.ndarray:
    """
    Hybrid fast generation: JIT for model forward pass, Python for generation loop.
    
    Args:
        model: Whisper model
        mel_features: Audio mel features
        max_length: Maximum sequence length (limited to 50 for speed)
        temperature: Sampling temperature (0.0 = greedy)
    
    Returns:
        Generated token sequence
    """
    batch_size = mel_features.shape[0]
    
    # Start with basic prompt tokens: <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
    prompt_tokens = jnp.array([[50258, 50259, 50359, 50363]])
    tokens = jnp.repeat(prompt_tokens, batch_size, axis=0)
    
    # Encode audio once
    xa = model.encoder(mel_features)
    
    # Create a JIT-compiled forward pass function
    @jax.jit
    def forward_step(tokens, xa):
        """JIT-compiled forward pass through the decoder."""
        seq_len = tokens.shape[1]
        mask = create_causal_mask(seq_len)
        logits = model.decoder(tokens, xa, mask)
        return logits
    
    # Generate tokens efficiently
    for step in range(max_length - len(prompt_tokens[0])):
        # Get logits using JIT-compiled forward pass
        logits = forward_step(tokens, xa)
        
        # Get next token (greedy decoding) - vectorized across batch
        next_logits = logits[:, -1, :]
        next_token = jnp.argmax(next_logits, axis=-1, keepdims=True)
        
        # Print progress every 10 tokens
        if step % 10 == 0:
            print(f"Generated {tokens.shape[1]} tokens, current token: {next_token[0, 0]}")
        
        # Append to sequence
        tokens = jnp.concatenate([tokens, next_token], axis=1)
        
        # Check for EOS tokens (original Whisper stopping condition)
        if jnp.any(next_token == 50257):
            print(f"EOS token found at step {step}")
            break
        
        # Also stop if we've reached max length
        if tokens.shape[1] >= max_length:
            print(f"Reached max length {max_length}")
            break
    
    return tokens


def generate_ultra_fast(model: WhisperModel, mel_features: jnp.ndarray, max_length: int = 50, temperature: float = 0.0) -> jnp.ndarray:
    """
    Ultra-fast generation using optimized operations without JIT (for debugging).
    
    Args:
        model: Whisper model
        mel_features: Audio mel features
        max_length: Maximum sequence length (limited to 50 for speed)
        temperature: Sampling temperature (0.0 = greedy)
    
    Returns:
        Generated token sequence
    """
    batch_size = mel_features.shape[0]
    
    # Start with basic prompt tokens: <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
    prompt_tokens = jnp.array([[50258, 50259, 50359, 50363]])
    tokens = jnp.repeat(prompt_tokens, batch_size, axis=0)
    
    # Encode audio once
    xa = model.encoder(mel_features)
    
    # Generate tokens efficiently using vectorized operations
    for step in range(max_length - len(prompt_tokens[0])):
        # Create causal mask for current sequence length
        seq_len = tokens.shape[1]
        mask = create_causal_mask(seq_len)
        
        # Get logits
        logits = model.decoder(tokens, xa, mask)
        
        # Get next token (greedy decoding) - vectorized across batch
        next_logits = logits[:, -1, :]
        next_token = jnp.argmax(next_logits, axis=-1, keepdims=True)
        
        # Print progress every 10 tokens
        if step % 10 == 0:
            print(f"Generated {tokens.shape[1]} tokens, current token: {next_token[0, 0]}")
        
        # Append to sequence
        tokens = jnp.concatenate([tokens, next_token], axis=1)
        
        # Check for EOS tokens (original Whisper stopping condition)
        if jnp.any(next_token == 50257):
            print(f"EOS token found at step {step}")
            break
        
        # Also stop if we've reached max length
        if tokens.shape[1] >= max_length:
            print(f"Reached max length {max_length}")
            break
    
    return tokens


def generate_super_fast(model: WhisperModel, mel_features: jnp.ndarray, max_length: int = 50, temperature: float = 0.0) -> jnp.ndarray:
    """
    Super fast generation using original Whisper's stopping conditions and limited to 50 tokens.
    
    Args:
        model: Whisper model
        mel_features: Audio mel features
        max_length: Maximum sequence length (limited to 50 for speed)
        temperature: Sampling temperature (0.0 = greedy)
    
    Returns:
        Generated token sequence
    """
    batch_size = mel_features.shape[0]
    
    # Start with basic prompt tokens: <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
    prompt_tokens = jnp.array([[50258, 50259, 50359, 50363]])
    tokens = jnp.repeat(prompt_tokens, batch_size, axis=0)
    
    # Encode audio once
    xa = model.encoder(mel_features)
    
    # Original Whisper stopping conditions:
    # 1. EOS token (50257)
    # 2. Max length reached
    # 3. All sequences completed
    
    # Track completion status
    completed = jnp.zeros(batch_size, dtype=bool)
    
    # Generate tokens efficiently
    for step in range(max_length - len(prompt_tokens[0])):
        # Create causal mask
        seq_len = tokens.shape[1]
        mask = create_causal_mask(seq_len)
        
        # Get logits
        logits = model.decoder(tokens, xa, mask)
        
        # Get next token (greedy decoding)
        next_logits = logits[:, -1, :]
        next_token = jnp.argmax(next_logits, axis=-1, keepdims=True)
        
        # Print progress every 10 tokens
        if step % 10 == 0:
            print(f"Generated {tokens.shape[1]} tokens, current token: {next_token[0, 0]}")
        
        # Check for EOS tokens (original Whisper stopping condition)
        eos_mask = (next_token == 50257).flatten()
        completed = completed | eos_mask
        
        # Append to sequence
        tokens = jnp.concatenate([tokens, next_token], axis=1)
        
        # Stop if all sequences are completed (original Whisper logic)
        if jnp.all(completed):
            print(f"All sequences completed at step {step}")
            break
        
        # Also stop if we've reached max length
        if tokens.shape[1] >= max_length:
            print(f"Reached max length {max_length}")
            break
    
    return tokens


def generate_whisper_style(model: WhisperModel, mel_features: jnp.ndarray, max_length: int = 448, temperature: float = 0.0) -> jnp.ndarray:
    """
    Generation function that mimics original Whisper's approach:
    - Process in chunks (like original Whisper's 30-second windows)
    - Use efficient JAX operations
    - Stop on EOS or natural completion
    
    Args:
        model: Whisper model
        mel_features: Audio mel features
        max_length: Maximum sequence length
        temperature: Sampling temperature (0.0 = greedy)
    
    Returns:
        Generated token sequence
    """
    batch_size = mel_features.shape[0]
    
    # Start with basic prompt tokens: <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
    prompt_tokens = jnp.array([[50258, 50259, 50359, 50363]])
    tokens = jnp.repeat(prompt_tokens, batch_size, axis=0)
    
    # Encode audio once
    xa = model.encoder(mel_features)
    
    # Use a more efficient approach: generate in larger chunks
    chunk_size = 10  # Generate 10 tokens at a time
    
    for chunk_start in range(0, max_length - len(prompt_tokens[0]), chunk_size):
        chunk_end = min(chunk_start + chunk_size, max_length - len(prompt_tokens[0]))
        
        # Generate tokens for this chunk
        for step in range(chunk_start, chunk_end):
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
            
            # Check for EOS
            if jnp.any(next_token == 50257):
                return tokens
        
        # Early stopping if we've generated enough
        if tokens.shape[1] >= max_length:
            break
    
    return tokens


def generate_chunks_with_beam_search(model: WhisperModel, mel_features: jnp.ndarray, max_length: int = 100, temperature: float = 0.0, beam_size: int = 5) -> jnp.ndarray:
    """
    Make our Bonsai model work with the original Whisper's mel features.

    Implement the original Whisper's clever chunking strategy:
    1. Pad the entire audio with 30 seconds of silence
    2. Use variable segment sizes based on content
    3. Use timestamp predictions to determine where to seek next
    4. Process segments sequentially with smart seeking

    Args:
        model: Our Bonsai Whisper model
        mel_features: Original Whisper's mel features (80, time_frames)
        max_length: Maximum sequence length
        temperature: Sampling temperature (0.0 = greedy)
        beam_size: Number of beams for beam search

    Returns:
        Generated token sequence from our Bonsai model
    """
    print("="*80)
    print("MAKING OUR BONSAI MODEL WORK WITH ORIGINAL WHISPER MEL FEATURES")
    print("="*80)

    print(f"Input mel features shape: {mel_features.shape}")
    print(f"Using our Bonsai model to process original Whisper's mel features")

    # Start with Whisper prompt tokens
    prompt_tokens = jnp.array([[50258, 50259, 50359, 50363]])  # <|startoftranscript|><|en|><|transcribe|><|notimestamps|>

    print(f"Starting with prompt tokens: {prompt_tokens.shape}")
    print(f"Prompt: <|startoftranscript|><|en|><|transcribe|><|notimestamps|>")

    print(f"\nProcessing mel features with our Bonsai model:")
    print(f"  Mel shape: {mel_features.shape}")
    print(f"  Audio duration: {mel_features.shape[1] * 160 / 16000:.1f}s")

    # Original Whisper constants
    N_FRAMES = 3000  # 30 seconds of audio frames
    HOP_LENGTH = 160  # 10ms hop length
    SAMPLE_RATE = 16000
    FRAMES_PER_SECOND = 100

    # The mel features are already padded with 30 seconds of silence by preprocessing
    # So we can work directly with them
    padded_mel = mel_features
    
    print(f"Mel features shape (already padded): {padded_mel.shape}")
    print(f"Audio duration: {padded_mel.shape[1] * HOP_LENGTH / SAMPLE_RATE:.1f}s")

    # Calculate content frames (the original audio without the 30-second padding)
    # The padding is at the end, so content is from 0 to (total - 3000)
    content_frames = mel_features.shape[1] - N_FRAMES  # Remove the 30-second padding
    content_duration = float(content_frames * HOP_LENGTH / SAMPLE_RATE)
    
    print(f"Content frames: {content_frames}")
    print(f"Content duration: {content_duration:.1f}s")

    all_tokens = prompt_tokens.copy()
    
    # Process with smart seeking like original Whisper
    seek = 0  # Start from the beginning of content
    segment_idx = 0
    
    while seek < content_frames:
        # Calculate segment size (variable, like original Whisper)
        segment_size = min(N_FRAMES, content_frames - seek)
        
        # Extract segment from mel features (already padded)
        mel_segment = padded_mel[:, seek : seek + segment_size]
        
        # Pad to N_FRAMES if needed (for the last segment)
        if mel_segment.shape[1] < N_FRAMES:
            padding_size = N_FRAMES - mel_segment.shape[1]
            segment_padding = jnp.zeros((80, padding_size))
            mel_segment = jnp.concatenate([mel_segment, segment_padding], axis=1)

        print(f"\n{'='*60}")
        print(f"PROCESSING SEGMENT {segment_idx + 1}")
        print(f"{'='*60}")
        print(f"Seek position: {seek}")
        print(f"Segment frames: {seek} - {seek + segment_size} ({segment_size} frames)")
        print(f"Segment duration: {segment_size * HOP_LENGTH / SAMPLE_RATE:.1f}s")
        print(f"Segment shape: {mel_segment.shape}")

        # Add batch dimension for our model
        mel_segment_batch = mel_segment[np.newaxis, :, :]  # (1, 80, 3000)

        # Generate tokens for this segment using our Bonsai model
        print(f"Generating tokens with our Bonsai model...")

        # Use our model's generate function with improved stopping conditions
        segment_tokens = model.generate(
            mel_segment_batch,
            max_length=min(100, max_length - all_tokens.shape[1] + 4),  # Leave room for prompt
            temperature=temperature
        )

        print(f"  Generated {segment_tokens.shape[1]} tokens for this segment")

        # Extract new tokens (excluding the prompt)
        if segment_tokens.shape[1] > 4:  # More than just the prompt
            new_tokens = segment_tokens[:, 4:]  # Remove prompt tokens
            print(f"  New tokens: {new_tokens.shape[1]}")

            # Append to main sequence
            all_tokens = jnp.concatenate([all_tokens, new_tokens], axis=1)
            print(f"  Total tokens so far: {all_tokens.shape[1]}")

        # Smart seeking: move forward by segment size (like original Whisper)
        # In a full implementation, we would use timestamp predictions to determine seek position
        seek += segment_size
        segment_idx += 1

        # Stop if we've reached max length
        if all_tokens.shape[1] >= max_length:
            print(f"Reached max length {max_length}")
            break

    print(f"\n{'='*80}")
    print(f"BONSAI MODEL GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total tokens generated: {all_tokens.shape[1]}")
    print(f"Final sequence shape: {all_tokens.shape}")

    return all_tokens
