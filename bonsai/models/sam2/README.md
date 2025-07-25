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
