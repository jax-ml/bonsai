#!/usr/bin/env python3
"""Simple test of Whisper model with random weights."""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from bonsai.models.whisper import modeling as model_lib
from flax import nnx


def load_audio_file(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Load audio file and resample to target sample rate."""
    try:
        import librosa
        audio, _ = librosa.load(audio_path, sr=sample_rate)
        return audio
    except ImportError:
        raise RuntimeError("librosa is required for this demo; please install it.")


def extract_mel_features(audio: np.ndarray, sample_rate: int = 16000, n_mels: int = 80) -> np.ndarray:
    """Extract mel spectrogram features from audio."""
    try:
        import librosa
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sample_rate, 
            n_mels=n_mels,
            hop_length=160,
            win_length=400,
            window='hann'
        )
        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec.T  # Transpose to (time, n_mels)
    except ImportError:
        raise RuntimeError("librosa is required for this demo; please install it.")


def test_whisper_model():
    """Test Whisper model with random weights."""
    print("=== Testing Whisper Model with Random Weights ===")
    
    # Load audio
    audio_path = "bonsai/models/whisper/tests/audio_samples/bush_speech.wav"
    print(f"Loading audio: {audio_path}")
    
    audio = load_audio_file(audio_path)
    mel_features = extract_mel_features(audio)
    print(f"Mel features shape: {mel_features.shape}")
    
    # Pad or truncate to expected length
    max_time_steps = 1500
    if mel_features.shape[0] > max_time_steps:
        mel_features = mel_features[:max_time_steps]
    else:
        padding = np.zeros((max_time_steps - mel_features.shape[0], mel_features.shape[1]))
        mel_features = np.concatenate([mel_features, padding], axis=0)
    
    # Add batch dimension and transpose to (batch, n_mels, time)
    mel_features = mel_features[None, ...].transpose(0, 2, 1)
    mel_features = jnp.array(mel_features)
    
    # Create model with random weights
    config = model_lib.WhisperConfig.whisper_tiny()
    print(f"Model config: {config}")
    
    rngs = nnx.Rngs(0)
    model = model_lib.WhisperModel(config, rngs=rngs)
    print("Created model with random weights")
    
    # Test forward pass
    print("\n--- Testing Forward Pass ---")
    start_time = time.time()
    tokens = jnp.array([[50258]])  # BOS token
    logits = model(mel_features, tokens)
    forward_time = time.time() - start_time
    print(f"Forward pass: {forward_time:.3f}s")
    print(f"Output shape: {logits.shape}")
    print(f"Output dtype: {logits.dtype}")
    
    # Test generation
    print("\n--- Testing Generation ---")
    start_time = time.time()
    generated_tokens = model_lib.generate(model, mel_features, max_length=20, temperature=0.0)
    generation_time = time.time() - start_time
    print(f"Generation: {generation_time:.3f}s")
    print(f"Generated tokens shape: {generated_tokens.shape}")
    print(f"Generated tokens: {generated_tokens[0]}")
    
    # Test with different input lengths
    print("\n--- Testing Different Input Lengths ---")
    for length in [500, 1000, 1500]:
        test_features = mel_features[:, :, :length]
        print(f"Input length {length}: {test_features.shape}")
        
        start_time = time.time()
        logits = model(test_features, tokens)
        forward_time = time.time() - start_time
        print(f"  Forward pass: {forward_time:.3f}s")
    
    print("\n=== Test Complete ===")
    return model


if __name__ == "__main__":
    test_whisper_model()
