# Gemma3 in JAX

This directory contains a pure JAX implementation of the [Gemma3 model](https://deepmind.google/models/gemma/gemma-3/), using the [Flax NNX](https://flax.readthedocs.io/en/stable/index.html) API. 

Note that you need an access token to download the model weights. In order to run the scripts, make sure to save an environment variable `HF_TOKEN` with your huggingface access token. 


## Model Configuration Support Status


### Running this model


```sh
python3 -m bonsai.models.gemma3.tests.run_model
```


## How to contribute to this model

### Remaining Tasks

1. Update to include kv cache memory reduction benefits from local attention. Currently, decode generation is not performance optimized.
2. Update to optimize parameter loading for larger models.
