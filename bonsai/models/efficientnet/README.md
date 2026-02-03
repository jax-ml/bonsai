# Efficientnet in JAX

This directory contains a pure JAX implementation of the [Efficientnet](https://arxiv.org/abs/1905.11946), using the [Flax NNX](https://flax.readthedocs.io/en/stable/index.html) API.


## Model Configuration Support Status

| Model Name | Config Support Status |
| :--- | :--- |
| **Model** | |
| [Efficientnet](https://arxiv.org/abs/1905.11946) | **✅ Supported** |

### Running this model

Run Efficientnet in JAX [see code](modeling.py).

```sh
python3 -m bonsai.models.efficientnet.tests.run_model
```


## How to contribute to this model

We welcome contributions! You can contribute to this model via the following:
* Add a model config variant from the above `🟡 Not started` to `class ModelConfig` in [modeling.py](modeling.py). Make sure your code is runnable on at least one hardware before creating a PR.
* Got some hardware? Run [run_model.py](tests/run_model.py) the existing configs above on hardwares marked `❔ Needs check`. Mark as `✅ Runs` or `⛔️ Not supported`.
