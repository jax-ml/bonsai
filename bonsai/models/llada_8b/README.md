# LLaDA in JAX

This directory contains a pure JAX implementation of the [LLaDA diffusion model](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct), using the [Flax NNX](flax.readthedocs.io/en/stable/index.html) API.

## Model Configuration Support Status

| Model Name | Config Support Status |
| :--- | :--- |
| **Dense Models** | |
| [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) | **✅ Supported** |

## How to contribute to this model

We welcome contributions! You can contribute to this model via the following:
* Add a model config variant from the above `🟡 Not started` to `class ModelConfig` in [modeling.py](modeling.py). Make sure your code is runnable on at least one hardware before creating a PR.
* Got some hardware? Run [run_model.py](tests/run_model.py) the existing configs above on hardwares marked `❔ Needs check`. Mark as `✅ Runs` or `⛔️ Not supported`.
