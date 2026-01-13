# LLaDA in JAX

This directory contains a pure JAX implementation of the [LLaDA model](https://ml-gsai.github.io/LLaDA-demo/), using the [Flax NNX](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/index.html) API. 


## Model Configuration Support Status


### Running this model


```sh
python3 -m bonsai.models.llada.tests.run_model
```


## How to contribute to this model

### Remaining Tasks

1. Update to optimize parameter loading for larger models.
2. Add dropout layers to support training. 
3. Enable access to hidden features. 
