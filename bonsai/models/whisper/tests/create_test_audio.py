#!/usr/bin/env python3
"""Create a simple test audio file for Whisper testing."""

import numpy as np
import soundfile as sf
from pathlib import Path

def create_test_audio():
    """Create a simple test audio file."""
    # Audio parameters
    sample_rate = 16000
    duration = 3.0  # 3 seconds
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create a simple speech-like signal (sine wave with some variation)
    # This simulates a basic speech pattern
    frequency = 220  # A3 note
    signal = np.sin(2 * np.pi * frequency * t)
    
    # Add some variation to make it more speech-like
    signal += 0.3 * np.sin(2 * np.pi * 440 * t)  # A4 harmonic
    signal += 0.1 * np.sin(2 * np.pi * 880 * t)  # A5 harmonic
    
    # Add some amplitude modulation to simulate speech patterns
    am_freq = 5  # 5 Hz modulation
    am_signal = 1 + 0.3 * np.sin(2 * np.pi * am_freq * t)
    signal *= am_signal
    
    # Normalize
    signal = signal / np.max(np.abs(signal))
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 0.01, signal.shape)
    signal += noise
    
    # Normalize again
    signal = signal / np.max(np.abs(signal))
    
    # Save the audio file
    output_path = Path("bonsai/models/whisper/tests/audio_samples/test_speech.wav")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    sf.write(str(output_path), signal, sample_rate)
    print(f"Created test audio file: {output_path}")
    print(f"Duration: {duration}s, Sample rate: {sample_rate}Hz")
    print(f"File size: {output_path.stat().st_size} bytes")

if __name__ == "__main__":
    create_test_audio()
