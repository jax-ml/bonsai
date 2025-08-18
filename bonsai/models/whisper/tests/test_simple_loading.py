#!/usr/bin/env python3
"""Simple test for weight loading."""

import jax
import jax.numpy as jnp
from flax import nnx

from bonsai.models.whisper import modeling as model_lib
from bonsai.models.whisper import params as P


def test_simple_loading():
    """Test simple weight loading without comparison."""
    config = model_lib.WhisperConfig.whisper_tiny()
    model_dir = "/tmp/models-bonsai/whisper-tiny"
    
    print("=== Simple Weight Loading Test ===")
    
    try:
        # Load the model
        model = P.create_model_from_safe_tensors(model_dir, config)
        print("✅ Model loaded successfully!")
        
        # Test forward pass
        mel_features = jax.random.normal(jax.random.PRNGKey(0), (1, 80, 500))
        tokens = jnp.array([[50258]])  # BOS token
        
        logits = model(mel_features, tokens)
        print(f"✅ Forward pass successful! Output shape: {logits.shape}")
        
        # Check if output makes sense
        print(f"Logits min: {logits.min():.4f}, max: {logits.max():.4f}")
        print(f"Logits mean: {logits.mean():.4f}, std: {logits.std():.4f}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    test_simple_loading()
