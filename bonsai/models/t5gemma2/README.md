# T5Gemma2 in JAX

This directory contains a pure JAX implementation of the [T5Gemma2 model](https://huggingface.co/collections/google/t5gemma-2-release-6839e38ad1e09ed3703c47e7), using the [Flax NNX](https://flax.readthedocs.io/en/stable/index.html) API.

### Running this model

```sh
python -m bonsai.models.t5gemma2.tests.run_model
python -m bonsai.models.t5gemma2.tests.run_model --demo image
python -m bonsai.models.t5gemma2.tests.run_model --demo translate
```
