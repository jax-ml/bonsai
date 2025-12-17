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

"""Smoke test for Mamba2 model."""

import time

import jax
import jax.numpy as jnp

from bonsai.models.mamba2 import modeling, params


def run_model():
    """Run a simple forward pass with a tiny Mamba2 model."""
    print("=" * 60)
    print("Mamba2 Smoke Test")
    print("=" * 60)

    cfg = modeling.Mamba2Config.tiny()
    print(f"\nConfig: hidden_size={cfg.hidden_size}, layers={cfg.num_hidden_layers}, vocab={cfg.vocab_size}")

    print("\n1. Creating model...")
    model = params.create_random_model(cfg, seed=42)

    batch_size, seq_len = 2, 64
    input_ids = jax.random.randint(jax.random.PRNGKey(0), (batch_size, seq_len), 0, cfg.vocab_size)
    labels = jax.random.randint(jax.random.PRNGKey(1), (batch_size, seq_len), 0, cfg.vocab_size)
    print(f"\n2. Input shape: {input_ids.shape}")

    print("\n3. Running forward pass (with JIT compilation)...")
    start = time.perf_counter()
    outputs = modeling.forward(model, input_ids, labels)
    jax.block_until_ready(outputs["logits"])
    first_time = time.perf_counter() - start
    print(f"   First forward pass: {first_time:.3f}s")

    print("\n4. Output validation:")
    print(f"   Logits shape: {outputs['logits'].shape}")
    print(f"   Loss: {outputs['loss']:.4f}")
    print(f"   Contains NaN: {bool(jnp.any(jnp.isnan(outputs['logits'])))}")

    print("\n5. Running second forward pass (JIT cached)...")
    start = time.perf_counter()
    outputs2 = modeling.forward(model, input_ids, labels)
    jax.block_until_ready(outputs2["logits"])
    second_time = time.perf_counter() - start
    print(f"   Second forward pass: {second_time:.4f}s")
    print(f"   Speedup: {first_time / second_time:.1f}x")

    print("\n6. Consistency check:")
    max_diff = jnp.max(jnp.abs(outputs["logits"] - outputs2["logits"]))
    print(f"   Max difference between runs: {max_diff}")
    assert max_diff < 1e-5, "Outputs should be deterministic"
    print("   ✓ Outputs are consistent")

    print("\n" + "=" * 60)
    print("✓ Mamba2 smoke test PASSED")
    print("=" * 60)


def run_forecaster():
    """Run a simple forward pass with a Mamba2Forecaster."""
    print("\n" + "=" * 60)
    print("Mamba2Forecaster Smoke Test")
    print("=" * 60)

    print("\n1. Creating forecaster...")
    model = params.create_random_forecaster(
        input_dim=10,
        d_model=64,
        n_layers=2,
        output_dim=1,
        forecast_horizon=24,
        seed=42,
    )

    batch_size, seq_len, input_dim = 4, 100, 10
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, input_dim))
    print(f"\n2. Input shape: {x.shape}")

    print("\n3. Running forward pass...")

    @jax.jit
    def forward_forecaster(model, x):
        return model(x)

    start = time.perf_counter()
    predictions = forward_forecaster(model, x)
    jax.block_until_ready(predictions)
    elapsed = time.perf_counter() - start
    print(f"   Forward pass time: {elapsed:.3f}s")

    print("\n4. Output validation:")
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Expected shape: ({batch_size}, 24, 1)")
    print(f"   Contains NaN: {bool(jnp.any(jnp.isnan(predictions)))}")

    assert predictions.shape == (batch_size, 24, 1), "Output shape mismatch"
    assert not jnp.any(jnp.isnan(predictions)), "Output contains NaN"

    print("\n" + "=" * 60)
    print("✓ Mamba2Forecaster smoke test PASSED")
    print("=" * 60)


def print_device_info():
    """Print JAX device information."""
    print("\nJAX Device Information:")
    print(f"  Backend: {jax.default_backend()}")
    print(f"  Devices: {jax.devices()}")
    print(f"  Device count: {jax.device_count()}")


if __name__ == "__main__":
    print_device_info()
    run_model()
    run_forecaster()
    print("\n" + "=" * 60)
    print("All smoke tests PASSED! ✓")
    print("=" * 60)


__all__ = ["run_forecaster", "run_model"]
