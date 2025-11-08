import os
import time

import jax
import jax.numpy as jnp
from flax import nnx
from huggingface_hub import snapshot_download

from bonsai.models.ConvNext import modeling as model_lib
from bonsai.models.ConvNext import params


def run_model():
    # 1. Download Model Weights
    model_name = "facebook/convnext-large-224"
    model_ckpt_path = snapshot_download(
        repo_id=model_name,
        allow_patterns="*.h5" 
    )
    print(f"Downloaded Keras weights from: {model_name}")

    # 2. Load Pretrained Model

    model = params._create_convnext_from_pretrained(
        model_cls=model_lib.ConvNeXt,
        file_dir=model_ckpt_path,
        num_classes=1000  # ImageNet-1K
    )
    
    graphdef, state = nnx.split(model)
    
    # 3. Prepare dummy input
    batch_size, channels, image_size = 8, 3, 224
    dummy_input = jnp.ones((batch_size, image_size, image_size, channels), dtype=jnp.float32)

    key = jax.random.PRNGKey(0)
    key, warmup_key, prof_key, time_key = jax.random.split(key, 4)

    # 4. Warmup + profiling 
    
    _ = model_lib.forward(
        graphdef, 
        state, 
        dummy_input, 
        rng=warmup_key, 
        train=False
    ).block_until_ready()

    # Profile a Few Steps 
    
    prof_keys = jax.random.split(prof_key, 5)
    
    jax.profiler.start_trace("/tmp/profile-convnext")
    for i in range(5):
        logits = model_lib.forward(
            graphdef, 
            state, 
            dummy_input, 
            rng=prof_keys[i], 
            train=False
        )
        jax.block_until_ready(logits)
    jax.profiler.stop_trace()
    print("Profiling complete. Trace saved to /tmp/profile-convnext")

    # 5. Timed execution
    
    time_keys = jax.random.split(time_key, 10)
    
    t0 = time.perf_counter()
    for i in range(10):
        logits = model_lib.forward(
            graphdef, 
            state, 
            dummy_input, 
            rng=time_keys[i], 
            train=False
        ).block_until_ready()
        
    step_time = (time.perf_counter() - t0) / 10
    print(f"Step time: {step_time:.4f} s")
    print(f"Throughput: {batch_size / step_time:.2f} images/s")

    # 6. Show Top-1 Predicted Class 
    
    pred = jnp.argmax(logits, axis=-1)
    print("Predicted classes (batch):", pred)


if __name__ == "__main__":
    run_model()

__all__ = ["run_model"]