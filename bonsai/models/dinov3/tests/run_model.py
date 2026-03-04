import time

import jax
import jax.numpy as jnp

from bonsai.models.dinov3 import modeling


def run_model():
    # 1. Create model
    model = modeling.Dinov3ViTModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
    config = model.config

    # 2. Prepare dummy input
    batch_size = 4
    image_size = 224
    dummy_input = jnp.ones((batch_size, 3, image_size, image_size), dtype=jnp.float32)

    # 3. Warmup (triggers JIT compilation)
    modeling.forward(model, dummy_input)["pooler_output"].block_until_ready()

    # Profile a few steps
    jax.profiler.start_trace("/tmp/profile-dinov3")
    for _ in range(5):
        logits = modeling.forward(model, dummy_input)["pooler_output"]
        jax.block_until_ready(logits)
    jax.profiler.stop_trace()

    # 4. Timed execution for inference
    num_runs = 10
    t0 = time.perf_counter()
    for _ in range(num_runs):
        logits = modeling.forward(model, dummy_input)["pooler_output"]
        jax.block_until_ready(logits)
    t1 = time.perf_counter()
    print(f"{num_runs} inference runs took {t1 - t0:.4f} s")
    print(f"Average inference time: {(t1 - t0) / num_runs * 1000:.2f} ms")

    # 5. Show output shape
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output logits shape: {logits.shape}")


if __name__ == "__main__":
    run_model()
