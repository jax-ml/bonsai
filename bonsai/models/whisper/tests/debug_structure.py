#!/usr/bin/env python3
"""Debug the exact parameter structure differences."""

import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from safetensors import safe_open

from bonsai.models.whisper import modeling as model_lib
from flax import nnx


def analyze_hf_structure():
    """Analyze HuggingFace model structure."""
    model_dir = "/tmp/models-bonsai/whisper-tiny"
    safetensors_file = Path(model_dir) / "model.safetensors"
    
    with safe_open(str(safetensors_file), framework="pt") as f:
        keys = list(f.keys())
    
    # Group by component
    encoder_keys = [k for k in keys if "encoder" in k and "decoder" not in k]
    decoder_keys = [k for k in keys if "decoder" in k]
    
    print("=== HF Model Structure ===")
    print(f"Total parameters: {len(keys)}")
    print(f"Encoder parameters: {len(encoder_keys)}")
    print(f"Decoder parameters: {len(decoder_keys)}")
    
    # Show all keys with their types
    print("\n--- All HF Keys with Types ---")
    for key in sorted(keys):
        with safe_open(str(safetensors_file), framework="pt") as f:
            tensor = f.get_tensor(key)
            print(f"  {key}: {tensor.shape}, {tensor.dtype}")
    
    return keys


def analyze_nnx_structure():
    """Analyze NNX model structure."""
    config = model_lib.WhisperConfig.whisper_tiny()
    
    # Create model
    model = nnx.eval_shape(lambda: model_lib.WhisperModel(config, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(model)
    state_dict = abs_state.to_pure_dict()
    
    def flatten_dict(d, prefix=""):
        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                result.update(flatten_dict(v, f"{prefix}.{k}" if prefix else k))
            else:
                result[f"{prefix}.{k}" if prefix else k] = v
        return result
    
    flat_state = flatten_dict(state_dict)
    
    # Group by component
    encoder_keys = [k for k in flat_state.keys() if "encoder" in k]
    decoder_keys = [k for k in flat_state.keys() if "decoder" in k]
    
    print("=== NNX Model Structure ===")
    print(f"Total parameters: {len(flat_state)}")
    print(f"Encoder parameters: {len(encoder_keys)}")
    print(f"Decoder parameters: {len(decoder_keys)}")
    
    # Show all keys with their types
    print("\n--- All NNX Keys with Types ---")
    for key in sorted(flat_state.keys()):
        value = flat_state[key]
        if hasattr(value, 'shape') and hasattr(value, 'dtype'):
            print(f"  {key}: {value.shape}, {value.dtype}")
        else:
            print(f"  {key}: {type(value)}")
    
    return flat_state


if __name__ == "__main__":
    hf_keys = analyze_hf_structure()
    print("\n" + "="*50 + "\n")
    nnx_keys = analyze_nnx_structure()
    
    # Compare
    print("\n=== Comparison ===")
    hf_set = set(hf_keys)
    nnx_set = set(nnx_keys.keys())
    
    extra_nnx = nnx_set - hf_set
    missing_nnx = hf_set - nnx_set
    
    print(f"Extra NNX parameters: {len(extra_nnx)}")
    for param in sorted(extra_nnx)[:10]:
        print(f"  + {param}")
    
    print(f"\nMissing NNX parameters: {len(missing_nnx)}")
    for param in sorted(missing_nnx)[:10]:
        print(f"  - {param}")
    
    # Show matching parameters
    matching = hf_set & nnx_set
    print(f"\nMatching parameters: {len(matching)}")
    for param in sorted(matching)[:10]:
        print(f"  ✓ {param}")
