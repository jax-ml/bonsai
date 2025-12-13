# Qwen3 in JAX

This directory contains a pure JAX implementation of the [Gemma3 model](https://deepmind.google/models/gemma/gemma-3/), using the [Flax NNX](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/index.html) API. Note that you need an access token to download the model weights. 


**This is currently in progress but passing numerics checks. Working on cleaning up the code and optimizing before the official PR.**


## Model Configuration Support Status


### Running this model


```sh
python3 -m bonsai.models.gemma3.tests.run_model
```


## How to contribute to this model

### Remaining Tasks

1. Properly implement sharding (vision, then text)
2. Implement with batching. Need this for FSDP. 
3. Optimize based on the profiling. 
4. Clean up code (variable names, etc.). Simplify unused configs (marked these with TODO)
5. Update to include other model sizes
