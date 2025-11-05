# U-Net in JAX (Work in progress)

This directory contains a pure JAX implementation of the [U-Net](https://arxiv.org/abs/1505.04597) with padded convolutions in its DoubleConv module, using the [Flax NNX](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/index.html) API.


## Model Status

Complete:
1. Model implementation and colab testing.

Needs work on:
1. Loading of pretrained checkpoints in `params.py` (Example [checkpoints](https://huggingface.co/models?sort=downloads&search=unet)).
2. Logit correctness test in `test_outputs_unet.py`.
3. Clean up `UNet_segmentation_example.ipynb` to demonstrate proper parameter loading.



## Tested on:  

 

| Model Name | Config | CPU | GPU A100 (1x) | GPU H100 (1x) | GPU A100 (8x) | GPU H100 (8x) | TPU v2 (8x) | TPU v5e (1x) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model** | | | | | | | | |
| [U-Net](https://arxiv.org/abs/1505.04597) | ✅ Supported | ✅ Runs | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check |❔ Needs check | ❔ Needs check |


### Running this model

Run U-Net in action, implemented in [121 lines of code](modeling.py) in JAX.

```sh
python3 -m bonsai.models.unet.tests.run_model
```


## How to contribute to this model

We welcome contributions! You can contribute to this model via the following:
* Add a model config variant from the above `🟡 Not started` to `class ModelConfig` in [modeling.py](modeling.py). Make sure your code is runnable on at least one hardware before creating a PR.
* Got some hardware? Run [run_model.py](tests/run_model.py) the existing configs above on hardwares marked `❔ Needs check`. Mark as `✅ Runs` or `⛔️ Not supported`.
