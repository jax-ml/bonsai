# Qwen3-VL in JAX

This directory contains a pure JAX implementation of the [Qwen3-VL SOTA Vision Language Model](https://github.com/QwenLM/Qwen3-VL), using the [Flax NNX](https://flax.readthedocs.io/en/stable/index.html) API.


## Model Configuration Support Status

| Model Name | Config Support Status |
| :--- | :--- |
| **Dense Models** | |
| [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | **✅ Supported** |
| [Qwen3-VL-2B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-2B-Thinking) | **✅ Supported** |
| [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | **✅ Supported** |
| [Qwen3-VL-4B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-4B-Thinking) | **✅ Supported** |
| [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | **✅ Supported** |
| [Qwen3-VL-8B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking) | **✅ Supported** |
| [Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) | **✅ Supported** |
| [Qwen3-VL-32B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-32B-Thinking) | **✅ Supported** |
| [NVIDIA-Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B) | **✅ Supported** |
| [NVIDIA-Cosmos-Reason2-8B](https://huggingface.co/nvidia/Cosmos-Reason2-8B) | **✅ Supported** |
| **MoE Models** | |
| [Qwen3-VL-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) | **🟡 Not started** |
| [Qwen3-VL-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B) | **🟡 Not started** |


### Running this model

Run Qwen3 in action, implemented in [900 lines of code](modeling.py) in JAX.

```sh
python3 -m bonsai.models.qwen3_vl.tests.run_model
```


## How to contribute to this model

We welcome contributions! You can contribute to this model via the following:
* Add a model config variant from the above `🟡 Not started` to `class ModelConfig` in [modeling.py](modeling.py). Make sure your code is runnable on at least one hardware before creating a PR.
* Got some hardware? Run [run_model.py](tests/run_model.py) the existing configs above on hardwares marked `❔ Needs check`. Mark as `✅ Runs` or `⛔️ Not supported`.
