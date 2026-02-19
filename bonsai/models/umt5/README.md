# UMT5 in JAX

This directory contains a pure JAX implementation of the [UMT5 model](https://arxiv.org/abs/2304.09151), using the [Flax NNX](https://flax.readthedocs.io/en/stable/index.html) API.


## Model Configuration Support Status

| Model Name | Config Support Status |
| :--- | :--- |
| **Dense Models** | |
| [umt5-small](https://huggingface.co/google/umt5-small) | **✅ Supported** |
| [umt5-base](https://huggingface.co/google/umt5-base) | **✅ Supported** |
| [umt5-xl](https://huggingface.co/google/umt5-xl) | **✅ Supported** |
| [umt5-xxl](https://huggingface.co/google/umt5-xxl) | **✅ Supported** |


### Running this model

Run UMT5 in action, implemented in [300 lines of code](modeling.py) in JAX.

```sh
python3 -m bonsai.models.umt5.tests.run_model
```


## How to contribute to this model

We welcome contributions! You can contribute to this model via the following:
* Add a model config variant from the above `🟡 Not started` to `class ModelConfig` in [modeling.py](modeling.py). Make sure your code is runnable on at least one hardware before creating a PR.
* Got some hardware? Run [run_model.py](tests/run_model.py) the existing configs above on hardwares marked `❔ Needs check`. Mark as `✅ Runs` or `⛔️ Not supported`.
