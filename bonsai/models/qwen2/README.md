# Qwen2 in JAX

This directory contains a pure JAX implementation of the [Qwen2 language model](https://qwenlm.github.io/blog/qwen2/), using the [Flax NNX](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/index.html) API.

## Model Configuration Support Status

| Model Name | Config Support Status |
| :--- | :--- |
| **Dense Models** | |
| [Qwen2-0.5B](https://huggingface.co/Qwen/Qwen2-0.5B) | **✅ Supported** |
| [Qwen2-1.5B](https://huggingface.co/Qwen/Qwen2-1.5B) | **✅ Supported** |
| [Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B) | **✅ Supported** |
| [Qwen2-72B](https://huggingface.co/Qwen/Qwen2-72B) | **✅ Supported** |


### Running this model

Run Qwen2 in action, implemented in pure JAX.

```sh
python3 -m bonsai.models.qwen2.tests.run_model
```


## Usage Example

## Model Configurations

The implementation supports all Qwen2 model sizes:

- **0.5B**: 24 layers, 896 hidden size, 14 attention heads, 2 key-value heads
- **1.5B**: 28 layers, 1536 hidden size, 12 attention heads, 2 key-value heads
- **7B**: 28 layers, 3584 hidden size, 28 attention heads, 4 key-value heads
- **72B**: 80 layers, 8192 hidden size, 64 attention heads, 8 key-value heads