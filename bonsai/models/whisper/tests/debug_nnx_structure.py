#!/usr/bin/env python3
"""Debug NNX state structure."""

import jax
import jax.numpy as jnp
from flax import nnx

from bonsai.models.whisper import modeling as model_lib


def debug_nnx_structure():
    """Debug the NNX model structure."""
    config = model_lib.WhisperConfig.whisper_tiny()
    
    # Create model
    model = nnx.eval_shape(lambda: model_lib.WhisperModel(config, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(model)
    state_dict = abs_state.to_pure_dict()
    
    print("=== NNX State Structure ===")
    print(f"Top-level keys: {list(state_dict.keys())}")
    
    def explore_dict(d, prefix="", max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return
        
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                print(f"{'  ' * current_depth}{k} (dict): {list(v.keys())}")
                explore_dict(v, full_key, max_depth, current_depth + 1)
            else:
                if hasattr(v, 'shape') and hasattr(v, 'dtype'):
                    print(f"{'  ' * current_depth}{k}: {v.shape}, {v.dtype}")
                else:
                    print(f"{'  ' * current_depth}{k}: {type(v)}")
    
    explore_dict(state_dict)
    
    # Check for integer keys
    def find_integer_keys(d, prefix=""):
        integer_keys = []
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(k, int):
                integer_keys.append(full_key)
            if isinstance(v, dict):
                integer_keys.extend(find_integer_keys(v, full_key))
        return integer_keys
    
    int_keys = find_integer_keys(state_dict)
    if int_keys:
        print(f"\n⚠️  Found integer keys: {int_keys}")
    else:
        print("\n✅ No integer keys found")


if __name__ == "__main__":
    debug_nnx_structure()
