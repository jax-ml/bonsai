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

1. Implement KV caching to speed up inference
2. JIT Compile forward pass
3. Finish the `run_model.py` example. Add timing and profiling. 
4. Optimize based on the profiling. 
5. Implement sharding. 
6. Get the `lm_head` from the weights. 
7. Update to include other model sizes


### Implementation Notes
The implementation matches the HF one pretty well. To get KV caching working, we have to pad things. We also have to pad the token_type_ids on the right with 0's. 



