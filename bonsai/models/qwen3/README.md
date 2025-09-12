# Qwen3 in JAX

This directory contains a pure JAX implementation of the [Qwen3 language model](https://qwenlm.github.io/blog/qwen3/), using the [Flax NNX](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/index.html) API.

> [!IMPORTANT]
> For large-scale high performance use case, please see the [MaxText](https://github.com/AI-Hypercomputer/maxtext?tab=readme-ov-file#getting-started) version.


## Tested on:  
*(Last Updated: 2025-07-02)*

 

| Model Name | Config | CPU | GPU A100 (1x) | GPU H100 (1x) | GPU A100 (8x) | GPU H100 (8x) | TPU v2 (8x) | TPU v5e (1x) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dense Models** | | | | | | | | |
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | ✅ Supported | ✅ Runs | ✅ Runs | ✅ Runs | ✅ Runs | ✅ Runs | ✅ Runs | ✅ Runs |
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | ✅ Supported | ✅ Runs | ✅ Runs | ✅ Runs | ✅ Runs | ✅ Runs | ✅ Runs | ✅ Runs |
| [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | ✅ Supported | ❔ Needs check | ❔ Needs check | ✅ Runs | ❔ Needs check | ❔ Needs check| ❔ Needs check | ❔ Needs check|
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | ✅ Supported | ❔ Needs check | ❔ Needs check| ✅ Runs | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check |
| [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) | ✅ Supported | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check |
| [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | ✅ Supported | ❔ Needs check | ❔ Needs check | ⛔️ OOM | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check |
| **MoE Models** | | | | | | | | |
| [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) | 🟡 Not started | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check |
| [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B) | 🟡 Not started | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check |


### Running this model

Run Qwen3 in action, implemented in [300 lines of code](bonsai/models/qwen3/modeling.py) in JAX.

```sh
python3 -m bonsai.models.qwen3.tests.run_model
```


## How to contribute to this model

We welcome contributions! You can contribute to this model via the following:
* Add a model config variant from the above `🟡 Not started` to `class ModelConfig` in [modeling.py](modeling.py). Make sure your code is runnable on at least one hardware before creating a PR.
* Got some hardware? Run [run_model.py](tests/run_model.py) the existing configs above on hardwares marked `❔ Needs check`. Mark as `✅ Runs` or `⛔️ Not supported`.
