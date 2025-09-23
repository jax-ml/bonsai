# U-Net in JAX

This directory contains a pure JAX implementation of the [VAE - Variational Autoencoder](https://arxiv.org/abs/1312.6114), using the [Flax NNX](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/index.html) API, including procedures to train, save and load weights, and make inference.


## Tested on:  
*(Last Updated: 2025-09-19)*

 

| Model Name | Config | CPU | GPU A100 (1x) | GPU H100 (1x) | GPU A100 (8x) | GPU H100 (8x) | TPU v2 (8x) | TPU v5e (1x) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model** | | | | | | | | |
| [VAE - Variational Autoencoder](https://arxiv.org/abs/1312.6114) | ✅ Supported | ✅ Runs | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check |❔ Needs check | ❔ Needs check |


### Running this model

Run U-Net in action, implemented in [76 lines of code](modeling.py) in JAX.

```sh
python -m bonsai.models.vae.tests.run_model
```


## How to contribute to this model

We welcome contributions! You can contribute to this model via the following:
* Add a model config variant from the above `🟡 Not started` to `class ModelConfig` in [modeling.py](modeling.py). Make sure your code is runnable on at least one hardware before creating a PR.
* Got some hardware? Run [run_model.py](tests/run_model.py) the existing configs above on hardwares marked `❔ Needs check`. Mark as `✅ Runs` or `⛔️ Not supported`.
