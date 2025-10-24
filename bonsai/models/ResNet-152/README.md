## ResNet-152 in JAX

This directory contains a pure JAX implementation of the [ResNet-152](https://huggingface.co/microsoft/resnet-152) model, using the Flax NNX API and built upon the generic ResNet components available in the Bonsai library.

## Tested on

| Model Name | Config | CPU | GPU A100 (1x) | GPU H100 (1x) | GPU A100 (8x) | GPU H100 (8x) | TPU v2 (8x) | TPU v5e (1x) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| [ResNet-152 v1.5](https://huggingface.co/microsoft/resnet-152) | ✅ Supported | ❔ Needs check |  ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check   | ❔ Needs check  | ❔ Needs check  |

### Running this model

Run ResNet-152 model inference in action:

```sh
# Assuming you saved the run script as run_model_jax_resnet152.py
python3 run_model_jax_resnet152.py


##Content for PyTorch ResNet-152

```markdown
## ResNet-152 in PyTorch (TorchVision)

This directory contains examples for using the standard [ResNet-152](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet152.html) model available in the `torchvision` library.

## Tested on

| Model Name | Config | CPU | GPU A100 (1x) | GPU H100 (1x) | GPU A100 (8x) | GPU H100 (8x) | TPU v2 (8x) | TPU v5e (1x) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| [ResNet-152 (TorchVision)](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet152.html) | ✅ Supported | ❔ Needs check |  ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check   | ❔ Needs check  | ❔ Needs check  |

### Running this model

Run ResNet-152 model inference in action:

```sh
# Assuming you saved the run script as run_model_pytorch_resnet152.py
python3 run_model_pytorch_resnet152.py