# Olmo3 in JAX

This directory contains a pure JAX implementation of the [Olmo3 model](https://allenai.org/olmo), using the [Flax NNX](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/index.html) API. 



## Model Configuration Support Status


### Running this model


```sh
python3 -m bonsai.models.olmo3.tests.run_model
```


## How to contribute to this model

### Remaining Tasks

1. Test batching
2. Profile model for best shardings. 
3. Update to include kv cache memory reduction benefits from window attention. Currently, decode generation is not performance optimized.

