#!/usr/bin/env python3
"""Analyze exact HuggingFace Whisper structure."""

from pathlib import Path
from safetensors import safe_open

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
    
    # Show all encoder keys
    print("\n--- All Encoder Keys ---")
    for key in sorted(encoder_keys):
        print(f"  {key}")
    
    # Show all decoder keys  
    print("\n--- All Decoder Keys ---")
    for key in sorted(decoder_keys):
        print(f"  {key}")
    
    return keys

if __name__ == "__main__":
    analyze_hf_structure()
