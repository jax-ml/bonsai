#!/usr/bin/env python3
"""
Simple test script for Whisper model structure.
This tests the model without loading weights to verify the implementation.
"""

import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

# Add the bonsai directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from bonsai.models.whisper import modeling

def test_model_structure():
    """Test the Whisper model structure without loading weights."""
    print("Testing Whisper model structure...")
    
    # Create a simple config
    config = modeling.WhisperConfig.whisper_tiny()
    print(f"Config: {config}")
    
    # Create model with RNGs
    rngs = nnx.Rngs(0)  # Use seed 0 for deterministic initialization
    
    # Create a simple model structure for testing
    model = modeling.WhisperModel(config, rngs=rngs)
    print("✓ Model created successfully")
    
    # Create dummy input
    batch_size = 1
    n_mels = 80
    time_steps = 500
    
    mel_features = jax.random.normal(jax.random.PRNGKey(0), (batch_size, n_mels, time_steps))
    tokens = jnp.array([[50258]])  # BOS token
    
    print(f"Input mel features shape: {mel_features.shape}")
    print(f"Input tokens shape: {tokens.shape}")
    
    # Test forward pass
    try:
        output = model(mel_features, tokens)
        print(f"✓ Forward pass successful")
        print(f"Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False

def test_audio_processing():
    """Test audio processing functions."""
    print("\nTesting audio processing...")
    
    try:
        # Test mel feature extraction
        audio = jax.random.normal(jax.random.PRNGKey(0), (16000 * 3,))  # 3 seconds of audio
        mel_features = modeling.extract_mel_features(audio)
        print(f"✓ Mel feature extraction successful")
        print(f"Mel features shape: {mel_features.shape}")
        return True
    except Exception as e:
        print(f"✗ Audio processing failed: {e}")
        return False

def test_configurations():
    """Test different model configurations."""
    print("\nTesting model configurations...")
    
    configs = [
        ("Tiny", modeling.WhisperConfig.whisper_tiny()),
        ("Base", modeling.WhisperConfig.whisper_base()),
        ("Small", modeling.WhisperConfig.whisper_small()),
    ]
    
    for name, config in configs:
        print(f"✓ {name} config: {config.n_audio_state} dims, {config.n_audio_layer} layers")
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Whisper Model Simple Tests")
    print("=" * 60)
    
    tests = [
        ("Model Structure", test_model_structure),
        ("Audio Processing", test_audio_processing),
        ("Configurations", test_configurations),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The Whisper model structure is working.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
