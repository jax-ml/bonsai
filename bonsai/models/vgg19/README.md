## VGG-19 in JAX

This directory contains a pure JAX implementation of the [VGG-19](https://huggingface.co/keras/vgg_19_imagenet) model, using the Flax NNX API. 

## Tested on
| Model Name                                              | Config | CPU | GPU A100 (1x) | GPU H100 (1x) | GPU A100 (8x) | GPU H100 (8x) | TPU v2 (8x) | TPU v5e (1x) |
|:--------------------------------------------------------| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| [VGG-19](https://huggingface.co/keras/vgg_19_imagenet) | ✅ Supported | ✅ Runs |  ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check   | ❔ Needs check  | ❔ Needs check  |

### Running this model

Run VGG model inference in action:

```python
python bonsai/models/vgg19/tests/run_model.py
```


### Validation with Real Images

For comprehensive validation of the VGG-19 model on actual images, see the [ImageNet validation colab](tests/VGG19_ImageNet_validation_example.ipynb). This notebook demonstrates:

- ImageNet classification on real downloaded images
- Proper image preprocessing and normalization
- Top-k predictions with confidence scores
- Batch processing for efficiency
- Performance benchmarking across batch sizes



## How to contribute to this model

We welcome contributions! You can contribute to this model via the following:
* Add a model config variant from the above `🟡 Not started` to `class ModelConfig` in [modeling.py](modeling.py). Make sure your code is runnable on at least one hardware before creating a PR.
* Got some hardware? Run [run_model.py](tests/run_model.py) the existing configs above on hardwares marked `❔ Needs check`. Mark as `✅ Runs` or `⛔️ Not supported`.
