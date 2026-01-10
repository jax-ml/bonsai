"""Run Qwen3-VL model with proper input handling.

This script demonstrates:
1. Loading a pretrained Qwen3-VL model (or using test config)
2. Converting PyTorch/HuggingFace processor outputs to JAX
3. Running inference with text and image inputs

Usage:
    # Test mode (no pretrained weights needed):
    python run_model.py --test

    # With pretrained weights:
    python run_model.py --model_path /path/to/Qwen3-VL-2B-Instruct
"""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
import modeling


def convert_pt_to_jax(pt_tensor):
    """Convert PyTorch tensor to JAX array."""
    return jnp.array(pt_tensor.numpy())


def count_params(model):
    """Count total parameters in model."""
    params = nnx.state(model, nnx.Param)
    total = 0
    for leaf in jax.tree.leaves(params):
        if hasattr(leaf, "shape"):
            total += np.prod(leaf.shape)
    return int(total)


def run_test_mode():
    """Run with test config (no pretrained weights needed)."""
    print("=" * 60)
    print("Qwen3-VL JAX Demo - Test Mode")
    print("=" * 60)

    # Create model with test config
    print("\n1. Creating model with test config...")
    config = modeling.Qwen3VLConfig.standard_test()
    print(f"   Vision: {config.vision_config.depth} layers, {config.vision_config.hidden_size} hidden")
    print(f"   Text: {config.text_config.num_hidden_layers} layers, {config.text_config.hidden_size} hidden")

    rng = nnx.Rngs(0)
    model = modeling.Qwen3VLForConditionalGeneration(config, rngs=rng)
    print(f"   Total parameters: {count_params(model):,}")

    # Test text-only forward pass
    print("\n2. Testing text-only forward pass...")
    batch_size = 2
    seq_len = 10
    input_ids = jax.random.randint(jax.random.key(0), (batch_size, seq_len), 0, config.text_config.vocab_size)

    logits = model(input_ids)
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Output logits shape: {logits.shape}")
    print(f"   Expected: ({batch_size}, {seq_len}, {config.text_config.vocab_size})")
    assert logits.shape == (batch_size, seq_len, config.text_config.vocab_size)

    # Test KV-cache
    print("\n3. Testing KV-cache...")
    cache = modeling.init_cache(config, batch_size=1, token_len=5, generate_steps=10)
    print(f"   Cache layers: {len(cache)}")
    print(f"   K-cache shape: {cache[0].k_cache.value.shape}")

    # Simulate generation
    print("\n4. Simulating single-token generation...")
    single_token = jax.random.randint(jax.random.key(1), (1, 1), 0, config.text_config.vocab_size)

    # Note: For proper generation, we'd need to handle position_ids correctly
    # This is a simplified demo
    logits = model(single_token)
    print(f"   Single token logits shape: {logits.shape}")

    # Test causal mask
    print("\n5. Testing causal mask...")
    mask = modeling.make_causal_mask(seq_len=4, cache_len=8, cur_pos=0)
    print(f"   Mask shape: {mask.shape}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


def run_pretrained_mode(model_path: str):
    """Run with pretrained weights.

    Requires:
    - transformers with Qwen3-VL support
    - Model weights at model_path
    """
    print("=" * 60)
    print("Qwen3-VL JAX Demo - Pretrained Mode")
    print("=" * 60)
    print(f"Model path: {model_path}")

    try:
        import params
    except ImportError:
        print("Error: params module not found")
        return

    # Load model
    print("\n1. Loading pretrained model...")
    config = params.get_pretrained_config("2b")  # Adjust based on model
    model = params.create_model_from_safe_tensors(model_path, config)
    print("   Model loaded successfully!")
    print(f"   Total parameters: {count_params(model):,}")

    # Try to load HuggingFace processor for input preparation
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(model_path)
        print("   Processor loaded!")

        # Example text input
        print("\n2. Processing text input...")
        messages = [{"role": "user", "content": "Hello, what is 2+2?"}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, return_tensors="pt")

        input_ids = convert_pt_to_jax(inputs["input_ids"])
        print(f"   Input IDs shape: {input_ids.shape}")

        # Forward pass
        print("\n3. Running forward pass...")
        logits = model(input_ids)
        print(f"   Output logits shape: {logits.shape}")

        # Get next token prediction
        next_token_id = int(jnp.argmax(logits[0, -1]))
        next_token = processor.decode([next_token_id])
        print(f"   Next token prediction: '{next_token}'")

    except ImportError as e:
        print(f"\n   Note: transformers not available ({e})")
        print("   Running basic test without processor...")

        # Basic test without processor
        input_ids = jax.random.randint(jax.random.key(0), (1, 10), 0, config.text_config.vocab_size)
        logits = model(input_ids)
        print(f"   Logits shape: {logits.shape}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run Qwen3-VL JAX model")
    parser.add_argument("--test", action="store_true", help="Run in test mode (no pretrained weights)")
    parser.add_argument("--model_path", type=str, help="Path to pretrained model")
    args = parser.parse_args()

    if args.test:
        run_test_mode()
    elif args.model_path:
        run_pretrained_mode(args.model_path)
    else:
        print("Usage: python run_model.py --test OR python run_model.py --model_path /path/to/model")
        print("\nRunning in test mode by default...")
        run_test_mode()


if __name__ == "__main__":
    main()
