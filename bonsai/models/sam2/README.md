# SAM2 (Segment Anything 2) in Jax

A minimal, readable JAX + Flax NNX re-implementation of Meta’s [Segment Anything 2](https://github.com/facebookresearch/sam2), enabling promptable image and video segmentation.

## Model Configuration Support Status: SAM2 Variants

| Model Name | Config Support Status |
| :--- | :--- |
| **SAM2 Variants** | |
| `sam2_tiny` | **✅ Supported** |
| `sam2_small` | **✅ Supported** |
| `sam2_baseplus` | **✅ Supported** |
| `sam2_large` | **✅ Supported** |


### Running this model

Run SAM2 model inference in action:

```sh
python3 -m bonsai.models.sam2.tests.run_model
```

## How to contribute to this model

We welcome contributions! You can contribute to this model via the following:
* Add a model config variant from the above `🟡 Not started` to `class ModelConfig` in [modeling.py](modeling.py). Make sure your code is runnable on at least one hardware before creating a PR.
* Got some hardware? Run [run_model.py](tests/run_model.py) the existing configs above on hardwares marked `❔ Needs check`. Mark as `✅ Runs` or `⛔️ Not supported`.
