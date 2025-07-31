## ResNet-50 in JAX

This directory contains a pure JAX implementation of the [ResNet-50](https://huggingface.co/microsoft/resnet-50) model, using the Flax NNX API.

## Tested on

| Model Name | Config | CPU | GPU A100 (1x) | GPU H100 (1x) | GPU A100 (8x) | GPU H100 (8x) | TPU v2 (8x) | TPU v5e (1x) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| [ResNet-50 v1.5](https://huggingface.co/microsoft/resnet-50) | ✅ Supported | ✅ Runs |  ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check   | ❔ Needs check  | ❔ Needs check  |

### Running this model

Run ResNet model inference in action:

```python
python bonsai/models/resnet50/tests/run_model.py
```


## How to contribute to this model

We welcome contributions! You can contribute to this model via the following:
* Add a model config variant from the above `🟡 Not started` to `class ModelConfig` in [modeling.py](modeling.py). Make sure your code is runnable on at least one hardware before creating a PR.
* Got some hardware? Run [run_model.py](tests/run_model.py) the existing configs above on hardwares marked `❔ Needs check`. Mark as `✅ Runs` or `⛔️ Not supported`.
