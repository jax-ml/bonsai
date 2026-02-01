# Qwen3 in JAX

This directory contains a pure JAX implementation of the [Qwen3 language model](https://qwenlm.github.io/blog/qwen3/), using the [Flax NNX](https://flax.readthedocs.io/en/stable/index.html) API.


## Model Configuration Support Status

| Model Name | Config Support Status |
| :--- | :--- |
| **Dense Models** | |
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | **✅ Supported** |
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | **✅ Supported** |
| [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | **✅ Supported** |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | **✅ Supported** |
| [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) | **✅ Supported** |
| [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | **✅ Supported** |
| **MoE Models** | |
| [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) | **🟡 Not started** |
| [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B) | **🟡 Not started** |


### Running this model

Run Qwen3 in action, implemented in [300 lines of code](bonsai/models/qwen3/modeling.py) in JAX.

```sh
python3 -m bonsai.models.qwen3.tests.run_model
```


## How to contribute to this model

We welcome contributions! You can contribute to this model via the following:
* Add a model config variant from the above `🟡 Not started` to `class ModelConfig` in [modeling.py](modeling.py). Make sure your code is runnable on at least one hardware before creating a PR.
* Got some hardware? Run [run_model.py](tests/run_model.py) the existing configs above on hardwares marked `❔ Needs check`. Mark as `✅ Runs` or `⛔️ Not supported`.
