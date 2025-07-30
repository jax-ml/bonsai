# SAM2 (Segment Anything 2) in Jax

A minimal, readable JAX + Flax NNX re-implementation of Meta’s [Segment Anything 2](https://github.com/facebookresearch/sam2), enabling promptable image and video segmentation.

## Tested on:

| Model Name       | Config       | CPU      | GPU A100 (1x) | GPU H100 (1x) | GPU A100 (8x) | GPU H100 (8x) | TPU v2 (8x) | TPU v5e (1x) |
|------------------|--------------|:--------:|:-------------:|:-------------:|:-------------:|:-------------:|:-----------:|:------------:|
| **SAM2 Variants**|              |          |               |               |               |               |             |              |
| `sam2_tiny`      | ✅ Supported | ✅ Runs  | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check |
| `sam2_small`     | ✅ Supported | ✅ Runs  | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check |
| `sam2_baseplus`  | ✅ Supported | ✅ Runs  | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check |
| `sam2_large`     | ✅ Supported | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check |


### Running this model

Run SAM2 model inference in action:

```python
python bonsai/models/sam2/tests/run_model.py
```

## How to contribute to this model

We welcome contributions! You can contribute to this model via the following:
* Add a model config variant from the above `🟡 Not started` to `class ModelConfig` in [modeling.py](modeling.py). Make sure your code is runnable on at least one hardware before creating a PR.
* Got some hardware? Run [run_model.py](tests/run_model.py) the existing configs above on hardwares marked `❔ Needs check`. Mark as `✅ Runs` or `⛔️ Not supported`.
