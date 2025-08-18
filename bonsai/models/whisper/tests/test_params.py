#!/usr/bin/env python3
"""Test Whisper parameter counts and loading."""

import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from etils import epath
from flax import nnx
from safetensors import safe_open

from bonsai.models.whisper import modeling as model_lib
from bonsai.models.whisper import params as P


def count_hf_params(model_dir: str) -> dict:
    """Count parameters in HuggingFace model."""
    model_path = Path(model_dir)
    safetensors_file = model_path / "model.safetensors"
    
    if not safetensors_file.exists():
        raise FileNotFoundError(f"No model.safetensors found in {model_dir}")
    
    with safe_open(str(safetensors_file), framework="pt") as f:
        keys = list(f.keys())
    
    # Count by component
    encoder_keys = [k for k in keys if "encoder" in k]
    decoder_keys = [k for k in keys if "decoder" in k]
    
    # Count by layer type
    conv_keys = [k for k in keys if "conv" in k]
    attn_keys = [k for k in keys if "attn" in k]
    mlp_keys = [k for k in keys if "fc" in k or "mlp" in k]
    norm_keys = [k for k in keys if "norm" in k or "ln" in k]
    embed_keys = [k for k in keys if "embed" in k]
    
    return {
        "total": len(keys),
        "encoder": len(encoder_keys),
        "decoder": len(decoder_keys),
        "conv": len(conv_keys),
        "attention": len(attn_keys),
        "mlp": len(mlp_keys),
        "norm": len(norm_keys),
        "embed": len(embed_keys),
        "encoder_keys": encoder_keys,
        "decoder_keys": decoder_keys,
    }


def count_nnx_params(config: model_lib.WhisperConfig) -> dict:
    """Count parameters in NNX model."""
    # Create model
    model = nnx.eval_shape(lambda: model_lib.WhisperModel(config, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(model)
    state_dict = abs_state.to_pure_dict()
    
    def flatten_and_count(d, prefix=""):
        count = 0
        keys = []
        for k, v in d.items():
            if isinstance(v, dict):
                sub_count, sub_keys = flatten_and_count(v, f"{prefix}.{k}" if prefix else k)
                count += sub_count
                keys.extend(sub_keys)
            else:
                count += 1
                keys.append(f"{prefix}.{k}" if prefix else k)
        return count, keys
    
    total_count, all_keys = flatten_and_count(state_dict)
    
    # Count by component
    encoder_keys = [k for k in all_keys if "encoder" in k]
    decoder_keys = [k for k in all_keys if "decoder" in k]
    
    # Count by layer type
    conv_keys = [k for k in all_keys if "conv" in k]
    attn_keys = [k for k in all_keys if "attn" in k]
    mlp_keys = [k for k in all_keys if "fc" in k]  # NNX uses fc1, fc2 for MLP
    norm_keys = [k for k in all_keys if "norm" in k or "scale" in k]  # NNX uses layer_norm and scale
    embed_keys = [k for k in all_keys if "embed" in k]
    
    return {
        "total": total_count,
        "encoder": len(encoder_keys),
        "decoder": len(decoder_keys),
        "conv": len(conv_keys),
        "attention": len(attn_keys),
        "mlp": len(mlp_keys),
        "norm": len(norm_keys),
        "embed": len(embed_keys),
        "encoder_keys": encoder_keys,
        "decoder_keys": decoder_keys,
        "all_keys": all_keys,
    }


def test_parameter_mapping():
    """Test that HF parameters map correctly to NNX."""
    config = model_lib.WhisperConfig.whisper_tiny()
    model_dir = "/tmp/models-bonsai/whisper-tiny"
    
    print("=== Parameter Count Analysis ===")
    print(f"Model config: {config}")
    print(f"Model directory: {model_dir}")
    
    # Count HF parameters
    print("\n--- HuggingFace Parameters ---")
    try:
        hf_counts = count_hf_params(model_dir)
        for key, value in hf_counts.items():
            if not key.endswith("_keys"):
                print(f"  {key}: {value}")
        print(f"  Total HF parameters: {hf_counts['total']}")
    except Exception as e:
        print(f"  Error counting HF params: {e}")
        return
    
    # Count NNX parameters
    print("\n--- NNX Parameters ---")
    try:
        nnx_counts = count_nnx_params(config)
        for key, value in nnx_counts.items():
            if not key.endswith("_keys"):
                print(f"  {key}: {value}")
        print(f"  Total NNX parameters: {nnx_counts['total']}")
    except Exception as e:
        print(f"  Error counting NNX params: {e}")
        return
    
    # Compare
    print("\n--- Comparison ---")
    print(f"HF total: {hf_counts['total']}")
    print(f"NNX total: {nnx_counts['total']}")
    print(f"Difference: {nnx_counts['total'] - hf_counts['total']}")
    
    if hf_counts['total'] == nnx_counts['total']:
        print("✅ Parameter counts match!")
    else:
        print("❌ Parameter counts don't match!")
        
        # Show extra NNX parameters
        hf_key_set = set(hf_counts['encoder_keys'] + hf_counts['decoder_keys'])
        nnx_key_set = set(nnx_counts['encoder_keys'] + nnx_counts['decoder_keys'])
        
        extra_nnx = nnx_key_set - hf_key_set
        missing_nnx = hf_key_set - nnx_key_set
        
        if extra_nnx:
            print(f"\nExtra NNX parameters ({len(extra_nnx)}):")
            for k in sorted(extra_nnx)[:10]:
                print(f"  + {k}")
            if len(extra_nnx) > 10:
                print(f"  ... and {len(extra_nnx) - 10} more")
        
        if missing_nnx:
            print(f"\nMissing NNX parameters ({len(missing_nnx)}):")
            for k in sorted(missing_nnx)[:10]:
                print(f"  - {k}")
            if len(missing_nnx) > 10:
                print(f"  ... and {len(missing_nnx) - 10} more")


def test_weight_loading():
    """Test actual weight loading."""
    config = model_lib.WhisperConfig.whisper_tiny()
    model_dir = "/tmp/models-bonsai/whisper-tiny"
    
    print("\n=== Weight Loading Test ===")
    
    try:
        # Try to load weights
        model = P.create_model_from_safe_tensors(model_dir, config)
        print("✅ Successfully loaded pretrained weights!")
        
        # Test forward pass
        mel_features = jax.random.normal(jax.random.PRNGKey(0), (1, 80, 500))
        tokens = jnp.array([[50258]])  # BOS token
        
        logits = model(mel_features, tokens)
        print(f"✅ Forward pass successful! Output shape: {logits.shape}")
        
        return model
        
    except Exception as e:
        print(f"❌ Weight loading failed: {e}")
        return None


if __name__ == "__main__":
    test_parameter_mapping()
    test_weight_loading()
