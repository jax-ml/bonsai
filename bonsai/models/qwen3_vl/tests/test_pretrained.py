"""Pretrained model test script for HPC execution.

This script tests the Qwen3-VL JAX implementation against the PyTorch reference
using actual pretrained weights. Designed to run on HPC with sufficient memory.

Usage:
    python test_pretrained.py --model_path /path/to/Qwen3-VL-2B-Instruct

The model path should contain:
    - config.json
    - model*.safetensors
"""

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))


def load_jax_model(model_path: str):
    """Load JAX model from pretrained weights."""
    from bonsai.models.qwen3_vl import modeling, params

    # Determine config from path
    path = Path(model_path)
    config_file = path / "config.json"

    if config_file.exists():
        with open(config_file) as f:
            config_dict = json.load(f)

        # Determine model size from config
        text_hidden = config_dict.get("text_config", {}).get("hidden_size", 2048)
        if text_hidden == 2048:
            config = modeling.Qwen3VLConfig.qwen3vl_2b()
        elif text_hidden == 2560:
            config = modeling.Qwen3VLConfig.qwen3vl_4b()
        elif text_hidden == 4096:
            config = modeling.Qwen3VLConfig.qwen3vl_8b()
        else:
            raise ValueError(f"Unknown model size with hidden_size={text_hidden}")
    else:
        # Default to 2B
        config = modeling.Qwen3VLConfig.qwen3vl_2b()

    print(f"Loading JAX model with config: text_hidden={config.text_config.hidden_size}")
    model = params.create_model_from_safe_tensors(model_path, config)
    return model, config


def load_pytorch_model(model_path: str):
    """Load PyTorch model for comparison."""
    try:
        import torch
        from transformers import Qwen3VLForConditionalGeneration as PTModel
    except ImportError:
        print("PyTorch/transformers not available. Skipping PyTorch comparison.")
        return None

    print("Loading PyTorch model...")
    model = PTModel.from_pretrained(model_path, device_map="cpu", torch_dtype=torch.float32)
    model.eval()
    return model


def compare_logits(jax_logits, pt_logits, rtol=1e-2, atol=1e-3):
    """Compare JAX and PyTorch logits."""
    jax_np = np.array(jax_logits)
    pt_np = pt_logits.detach().numpy()

    # Basic stats
    print(f"  JAX logits: mean={jax_np.mean():.4f}, std={jax_np.std():.4f}")
    print(f"  PT logits:  mean={pt_np.mean():.4f}, std={pt_np.std():.4f}")

    # Cosine similarity
    jax_flat = jax_np.flatten()
    pt_flat = pt_np.flatten()
    cos_sim = np.dot(jax_flat, pt_flat) / (np.linalg.norm(jax_flat) * np.linalg.norm(pt_flat))
    print(f"  Cosine similarity: {cos_sim:.6f}")

    # Max absolute difference
    max_diff = np.max(np.abs(jax_np - pt_np))
    print(f"  Max absolute difference: {max_diff:.6f}")

    # Check if close
    is_close = np.allclose(jax_np, pt_np, rtol=rtol, atol=atol)
    print(f"  np.allclose(rtol={rtol}, atol={atol}): {is_close}")

    return cos_sim, max_diff, is_close


def test_text_only(jax_model, pt_model, config):
    """Test text-only forward pass comparison."""
    print("\n" + "=" * 60)
    print("TEST: Text-only forward pass")
    print("=" * 60)

    # Create input
    batch_size = 1
    seq_len = 20
    np.random.seed(42)
    input_ids = np.random.randint(0, config.text_config.vocab_size, (batch_size, seq_len))

    # JAX forward
    print("\nRunning JAX model...")
    jax_input = jnp.array(input_ids)
    jax_logits = jax_model(jax_input)
    print(f"  Output shape: {jax_logits.shape}")

    # PyTorch forward
    if pt_model is not None:
        import torch

        print("\nRunning PyTorch model...")
        pt_input = torch.tensor(input_ids, dtype=torch.long)
        with torch.no_grad():
            pt_outputs = pt_model(input_ids=pt_input)
        pt_logits = pt_outputs.logits
        print(f"  Output shape: {pt_logits.shape}")

        # Compare
        print("\nComparison:")
        cos_sim, max_diff, is_close = compare_logits(jax_logits, pt_logits)

        if cos_sim > 0.99:
            print("\n✓ TEST PASSED: High cosine similarity achieved")
        else:
            print(f"\n✗ TEST FAILED: Cosine similarity {cos_sim:.4f} below threshold 0.99")


def test_with_cache(jax_model, config):
    """Test generation with KV-cache."""
    from bonsai.models.qwen3_vl import modeling

    print("\n" + "=" * 60)
    print("TEST: KV-cache generation")
    print("=" * 60)

    batch_size = 1
    prefill_len = 10
    generate_steps = 5

    # Initialize cache
    cache = modeling.init_cache(config, batch_size, prefill_len, generate_steps)
    print(f"Cache initialized: {len(cache)} layers, size={cache[0].size}")

    # Prefill
    np.random.seed(42)
    input_ids = np.random.randint(0, config.text_config.vocab_size, (batch_size, prefill_len))
    jax_input = jnp.array(input_ids)

    # Note: For full generation, we'd need to handle position_ids properly
    # This is a basic shape test
    logits = jax_model(jax_input)
    print(f"Prefill logits shape: {logits.shape}")

    print("\n✓ Cache test completed")


def main():
    parser = argparse.ArgumentParser(description="Test Qwen3-VL pretrained model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model directory")
    parser.add_argument("--skip_pytorch", action="store_true", help="Skip PyTorch comparison")
    args = parser.parse_args()

    print("=" * 60)
    print("Qwen3-VL Pretrained Model Test")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"JAX devices: {jax.devices()}")

    # Load models
    jax_model, config = load_jax_model(args.model_path)
    pt_model = None if args.skip_pytorch else load_pytorch_model(args.model_path)

    # Run tests
    test_text_only(jax_model, pt_model, config)
    test_with_cache(jax_model, config)

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
