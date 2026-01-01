# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test runner for Qwen3-VL Flax NNX model.

Includes:
- Basic forward pass verification
- Text generation demo (with minimal model)
- Full pipeline demonstration
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from ..modeling import Qwen3VLConfig, Qwen3VLModel, TextConfig, VisionConfig


def create_minimal_test_config():
    """Create a very small config for testing."""
    vision_config = VisionConfig(
        depth=2,
        hidden_size=64,
        intermediate_size=256,
        num_heads=4,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=64,
        num_position_embeddings=16,
        deepstack_visual_indexes=(0, 1),
    )
    text_config = TextConfig(
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=1000,
        mrope_section=(8, 4, 4),
    )
    return Qwen3VLConfig(text_config=text_config, vision_config=vision_config)


def run_forward_pass():
    """Run a basic forward pass and verify output shapes."""
    print("=" * 60)
    print("Running Qwen3-VL Forward Pass Test")
    print("=" * 60)

    config = create_minimal_test_config()

    print("\nModel Configuration:")
    print(f"  Vision depth: {config.vision_config.depth}")
    print(f"  Vision hidden: {config.vision_config.hidden_size}")
    print(f"  Text layers: {config.text_config.num_hidden_layers}")
    print(f"  Text hidden: {config.text_config.hidden_size}")
    print(f"  Vocab size: {config.text_config.vocab_size}")

    print("\nInitializing model...")
    rngs = nnx.Rngs(0)
    model = Qwen3VLModel(config, rngs=rngs)

    # Count parameters
    def count_params(state):
        total = 0
        for leaf in jax.tree.leaves(state):
            if hasattr(leaf, "value"):
                total += leaf.value.size
            elif hasattr(leaf, "size"):
                total += leaf.size
        return total

    num_params = count_params(nnx.state(model, nnx.Param))
    print(f"Total parameters: {num_params:,}")

    # Dummy Inputs
    B, S = 1, 10
    input_ids = jnp.zeros((B, S), dtype=jnp.int32)
    pixel_values = jnp.zeros((B, 3, 2, 32, 32), dtype=jnp.float32)
    grid_thw = jnp.array([[1, 2, 2]], dtype=jnp.int32)
    visual_pos_masks = jnp.zeros((B, S), dtype=jnp.bool_)

    print("\nInput shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  pixel_values: {pixel_values.shape}")
    print(f"  grid_thw: {grid_thw.shape}")
    print(f"  visual_pos_masks: {visual_pos_masks.shape}")

    print("\nRunning forward pass...")
    hidden_states = model(input_ids, pixel_values, grid_thw, visual_pos_masks)

    print(f"\nOutput shape: {hidden_states.shape}")
    expected_shape = (B, S, config.text_config.hidden_size)
    assert hidden_states.shape == expected_shape, f"Expected {expected_shape}, got {hidden_states.shape}"

    # Check for NaN/Inf
    has_nan = jnp.any(jnp.isnan(hidden_states))
    has_inf = jnp.any(jnp.isinf(hidden_states))
    print(f"Contains NaN: {has_nan}")
    print(f"Contains Inf: {has_inf}")

    assert not has_nan, "Output contains NaN!"
    assert not has_inf, "Output contains Inf!"

    print("\n✓ Forward pass test PASSED!")
    return True


def run_text_only():
    """Test text-only forward pass (no vision inputs)."""
    print("\n" + "=" * 60)
    print("Running Text-Only Forward Pass")
    print("=" * 60)

    config = create_minimal_test_config()
    rngs = nnx.Rngs(42)
    model = Qwen3VLModel(config, rngs=rngs)

    B, S = 2, 8
    key = jax.random.PRNGKey(0)
    input_ids = jax.random.randint(key, (B, S), 0, config.text_config.vocab_size)

    print(f"Input ids shape: {input_ids.shape}")
    print("Running text-only forward...")

    output = model(input_ids)

    print(f"Output shape: {output.shape}")
    expected = (B, S, config.text_config.hidden_size)
    assert output.shape == expected, f"Expected {expected}, got {output.shape}"

    print("✓ Text-only test PASSED!")
    return True


def run_generation_demo():
    """Demo simple token generation loop (argmax sampling)."""
    print("\n" + "=" * 60)
    print("Running Generation Demo (argmax sampling)")
    print("=" * 60)

    config = Qwen3VLConfig.standard_test()
    rngs = nnx.Rngs(123)
    model = Qwen3VLModel(config, rngs=rngs)

    # Add a simple LM head for generation
    lm_head_w = jax.random.normal(rngs.params(), (config.text_config.hidden_size, config.text_config.vocab_size))
    lm_head_w = lm_head_w * 0.02  # Small init

    def get_logits(hidden_states):
        return jnp.einsum("bsh,hv->bsv", hidden_states, lm_head_w)

    # Start with [BOS] token (0)
    B = 1
    generated = [0]
    max_new_tokens = 10

    print(f"Starting generation (max {max_new_tokens} tokens)...")
    print(f"Initial token: {generated}")

    for step in range(max_new_tokens):
        # Current sequence
        input_ids = jnp.array([generated], dtype=jnp.int32)

        # Forward pass
        hidden_states = model(input_ids)

        # Get logits for last position
        logits = get_logits(hidden_states)[:, -1, :]

        # Argmax sampling
        next_token = int(jnp.argmax(logits, axis=-1)[0])
        generated.append(next_token)

        # Early stop at EOS (token 1)
        if next_token == 1:
            print(f"  Step {step + 1}: Generated EOS token, stopping")
            break

    print(f"\nGenerated sequence: {generated}")
    print(f"Total tokens: {len(generated)}")
    print("✓ Generation demo PASSED!")
    return True


def run_vision_text_forward():
    """Test vision + text multimodal forward pass."""
    print("\n" + "=" * 60)
    print("Running Vision + Text Forward Pass")
    print("=" * 60)

    config = create_minimal_test_config()
    rngs = nnx.Rngs(0)
    model = Qwen3VLModel(config, rngs=rngs)

    B, S = 1, 12
    key = jax.random.PRNGKey(0)

    # Random inputs
    input_ids = jax.random.randint(key, (B, S), 0, config.text_config.vocab_size)
    pixel_values = jax.random.normal(key, (B, 3, 2, 32, 32))
    grid_thw = jnp.array([[1, 2, 2]], dtype=jnp.int32)

    # Mark first 4 positions as visual tokens
    visual_pos_masks = jnp.zeros((B, S), dtype=jnp.bool_)
    visual_pos_masks = visual_pos_masks.at[0, :4].set(True)

    print(f"Input shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  pixel_values: {pixel_values.shape}")
    print(f"  visual_pos_masks sum: {visual_pos_masks.sum()}")

    output = model(input_ids, pixel_values, grid_thw, visual_pos_masks)

    print(f"Output shape: {output.shape}")
    expected = (B, S, config.text_config.hidden_size)
    assert output.shape == expected

    # Verify output statistics
    mean_val = float(jnp.mean(output))
    std_val = float(jnp.std(output))
    print(f"Output stats: mean={mean_val:.4f}, std={std_val:.4f}")

    print("✓ Vision + Text test PASSED!")
    return True


def run_all_tests():
    """Run all test functions."""
    print("\n" + "=" * 60)
    print("QWEN3-VL FLAX NNX MODEL TESTS")
    print("=" * 60)

    tests = [
        ("Forward Pass", run_forward_pass),
        ("Text Only", run_text_only),
        ("Vision + Text", run_vision_text_forward),
        ("Generation Demo", run_generation_demo),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ {name} FAILED: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")

    all_passed = all(p for _, p in results)
    print("\n" + ("ALL TESTS PASSED!" if all_passed else "SOME TESTS FAILED"))
    return all_passed


if __name__ == "__main__":
    run_all_tests()


__all__ = ["run_all_tests", "run_forward_pass", "run_text_only", "run_generation_demo"]
