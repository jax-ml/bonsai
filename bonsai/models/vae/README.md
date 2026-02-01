# Variational Autoencoders (VAEs) in JAX

This directory contains a pure JAX implementation of the [VAE - Variational Autoencoder](https://arxiv.org/abs/1312.6114), using the [Flax NNX](https://flax.readthedocs.io/en/stable/index.html) API, including procedures to train, save and load weights, and make inference.


## Model Configuration Support Status

| Model Name | Config Support Status |
| :--- | :--- |
| **Model** | |
| [VAE - Variational Autoencoder](https://arxiv.org/abs/1312.6114) | **✅ Supported** |


### Running this model

Run Variational Autoencoders (VAEs) in action, implemented in [76 lines of code](modeling.py) in JAX.

```sh
python -m bonsai.models.vae.tests.run_model
```


## How to contribute to this model

We welcome contributions! You can contribute to this model via the following:
* Add a model config variant from the above `🟡 Not started` to `class ModelConfig` in [modeling.py](modeling.py). Make sure your code is runnable on at least one hardware before creating a PR.
* Got some hardware? Run [run_model.py](tests/run_model.py) the existing configs above on hardwares marked `❔ Needs check`. Mark as `✅ Runs` or `⛔️ Not supported`.
