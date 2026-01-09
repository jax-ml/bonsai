# VJEPA-2 in JAX

This directory contains a pure JAX implementation of the [VJEPA-2 foundation world model](https://ai.meta.com/vjepa/), using the [Flax NNX](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/index.html) API.


## Model Configuration Support Status

| Model Name | Size | Config Support Status |
| :--- | :--- | :--- |
| **Foundation models** | | |
| [vitl-fpc64-256](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) | 0.3B | **✅ Supported** |
| [vith-fpc64-256](https://huggingface.co/facebook/vjepa2-vith-fpc64-256) | 0.7B | **✅ Supported** |
| [vitg-fpc64-256](https://huggingface.co/facebook/vjepa2-vitg-fpc64-256) | 1B   | **✅ Supported** |
| [vitg-fpc64-384](https://huggingface.co/facebook/vjepa2-vitg-fpc64-384) | 1B   | **✅ Supported** |
| **Video Classifiers** | | |
| [vitl-fpc16-256-ssv2](https://huggingface.co/facebook/vjepa2-vitl-fpc16-256-ssv2) | 0.4B | **✅ Supported** |
| [vitg-fpc64-384-ssv2](https://huggingface.co/facebook/vjepa2-vitg-fpc64-384-ssv2) | 1B   | **✅ Supported** |
| [vitl-fpc32-256-diving48](https://huggingface.co/facebook/vjepa2-vitl-fpc32-256-diving48) | 0.4B | **✅ Supported** |
| [vitg-fpc32-384-diving48](https://huggingface.co/facebook/vjepa2-vitg-fpc32-384-diving48) | 1B   | **✅ Supported** |


### Running this model

Run Qwen3 in action, implemented in [300 lines of code](bonsai/models/qwen3/modeling.py) in JAX.

```sh
python3 -m bonsai.models.vjepa2.tests.run_model
```


## How to contribute to this model

We welcome contributions! You can contribute to this model via the following:
* Add a model config variant from the above `🟡 Not started` to `class ModelConfig` in [modeling.py](modeling.py). Make sure your code is runnable on at least one hardware before creating a PR.
* Got some hardware? Run [run_model.py](tests/run_model.py) the existing configs above on hardwares marked `❔ Needs check`. Mark as `✅ Runs` or `⛔️ Not supported`.