## ResNet-50 in JAX

This directory contains a pure JAX implementation of the [ResNet-50](https://huggingface.co/microsoft/resnet-50) model, using the Flax NNX API.

## Tested on

| Model Name | Config | CPU | GPU A100 (1x) | GPU H100 (1x) | GPU A100 (8x) | GPU H100 (8x) | TPU v2 (8x) | TPU v5e (1x) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| [ResNet-50 v1.5](https://huggingface.co/microsoft/resnet-50) | ✅ Supported | ✅ Runs |  ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check   | ❔ Needs check  | ❔ Needs check  |
