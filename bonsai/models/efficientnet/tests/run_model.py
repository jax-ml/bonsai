import time

import jax
import jax.numpy as jnp
from flax import nnx

from bonsai.models.efficientnet import modeling, params


def run_model():
    # 1. Create model and PRNG keys
    rngs = nnx.Rngs(params=0, dropout=1)
    config = modeling.ModelCfg.b0()
    model = params.create_model(cfg=config, rngs=rngs)

    # 2. Prepare dummy input
    batch_size = 4
    image_size = config.resolution
    dummy_input = jnp.ones((batch_size, image_size, image_size, 3), dtype=jnp.float32)

    # 3. Warmup (triggers JIT compilation)
    print("Starting JIT compilation (warmup)...")
    _ = model(dummy_input, training=False)
    jax.block_until_ready(_)
    print("Warmup complete.")

    # 4. Timed execution for inference
    num_runs = 10
    t0 = time.perf_counter()
    for _ in range(num_runs):
        logits = model(dummy_input, training=False)
        jax.block_until_ready(logits)
    t1 = time.perf_counter()
    print(f"{num_runs} inference runs took {t1 - t0:.4f} s")
    print(f"Average inference time: {(t1 - t0) / num_runs * 1000:.2f} ms")

    # 5. Show output shape
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output logits shape: {logits.shape}")


if __name__ == "__main__":
    run_model()
