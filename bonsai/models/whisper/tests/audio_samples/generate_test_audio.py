#!/usr/bin/env python3
"""
Generate synthetic speech-like audio for testing Whisper model.
This creates audio files that simulate speech patterns for testing.
"""

import jax.numpy as jnp
import numpy as np  # Still needed for soundfile compatibility
import soundfile as sf
import librosa
from scipy import signal

def generate_synthetic_speech(duration=5.0, sample_rate=16000, filename="synthetic_speech.wav"):
    """Generate synthetic speech-like audio for testing."""
    
    # Create time array using JAX
    t = jnp.linspace(0, duration, int(sample_rate * duration))
    
    # Generate base frequency (speech-like fundamental frequency)
    base_freq = 150  # Hz (typical male voice)
    
    # Create harmonics (speech has multiple harmonics)
    harmonics = []
    for i in range(1, 6):  # First 5 harmonics
        harmonic = jnp.sin(2 * jnp.pi * base_freq * i * t)
        harmonics.append(harmonic * (1.0 / i))  # Decreasing amplitude
    
    # Combine harmonics
    speech_like = jnp.sum(jnp.stack(harmonics), axis=0)
    
    # Add some modulation (like speech intonation)
    modulation = 1 + 0.3 * jnp.sin(2 * jnp.pi * 0.5 * t)  # Slow modulation
    speech_like *= modulation
    
    # Add some noise (like breath and articulation) using JAX
    noise = jax.random.normal(jax.random.PRNGKey(0), (len(speech_like),)) * 0.1
    speech_like += noise
    
    # Normalize
    speech_like = speech_like / jnp.max(jnp.abs(speech_like)) * 0.8
    
    # Save as WAV file (convert to numpy for soundfile)
    sf.write(filename, np.array(speech_like), sample_rate)
    print(f"Generated {filename} ({duration}s, {sample_rate}Hz)")
    
    return speech_like

def generate_multilingual_samples():
    """Generate different speech-like samples for testing."""
    
    # English-like speech (higher pitch, faster modulation)
    t = jnp.linspace(0, 4.0, int(16000 * 4.0))
    english_like = jnp.sin(2 * jnp.pi * 200 * t)  # Higher pitch
    english_like += 0.5 * jnp.sin(2 * jnp.pi * 400 * t)
    english_like += 0.3 * jnp.sin(2 * jnp.pi * 600 * t)
    english_like *= (1 + 0.4 * jnp.sin(2 * jnp.pi * 1.0 * t))  # Faster modulation
    english_like += jax.random.normal(jax.random.PRNGKey(1), (len(english_like),)) * 0.05
    english_like = english_like / jnp.max(jnp.abs(english_like)) * 0.8
    sf.write("english_like.wav", np.array(english_like), 16000)
    
    # Spanish-like speech (different rhythm)
    t = jnp.linspace(0, 3.5, int(16000 * 3.5))
    spanish_like = jnp.sin(2 * jnp.pi * 180 * t)
    spanish_like += 0.6 * jnp.sin(2 * jnp.pi * 360 * t)
    spanish_like += 0.4 * jnp.sin(2 * jnp.pi * 540 * t)
    spanish_like *= (1 + 0.5 * jnp.sin(2 * jnp.pi * 0.8 * t))
    spanish_like += jax.random.normal(jax.random.PRNGKey(2), (len(spanish_like),)) * 0.06
    spanish_like = spanish_like / jnp.max(jnp.abs(spanish_like)) * 0.8
    sf.write("spanish_like.wav", np.array(spanish_like), 16000)
    
    # French-like speech (different intonation)
    t = jnp.linspace(0, 4.5, int(16000 * 4.5))
    french_like = jnp.sin(2 * jnp.pi * 190 * t)
    french_like += 0.7 * jnp.sin(2 * jnp.pi * 380 * t)
    french_like += 0.3 * jnp.sin(2 * jnp.pi * 570 * t)
    french_like *= (1 + 0.6 * jnp.sin(2 * jnp.pi * 0.6 * t))  # Slower modulation
    french_like += jax.random.normal(jax.random.PRNGKey(3), (len(french_like),)) * 0.04
    french_like = french_like / jnp.max(jnp.abs(french_like)) * 0.8
    sf.write("french_like.wav", np.array(french_like), 16000)
    
    print("Generated multilingual speech-like samples")

def generate_whisper_test_samples():
    """Generate samples specifically for Whisper testing."""
    
    # Short sample (1 second)
    generate_synthetic_speech(1.0, 16000, "short_speech.wav")
    
    # Medium sample (5 seconds)
    generate_synthetic_speech(5.0, 16000, "medium_speech.wav")
    
    # Long sample (10 seconds)
    generate_synthetic_speech(10.0, 16000, "long_speech.wav")
    
    # Different sample rates
    generate_synthetic_speech(3.0, 8000, "low_sample_rate.wav")
    generate_synthetic_speech(3.0, 44100, "high_sample_rate.wav")
    
    # Generate multilingual samples
    generate_multilingual_samples()
    
    print("Generated all test samples for Whisper model")

if __name__ == "__main__":
    generate_whisper_test_samples()
