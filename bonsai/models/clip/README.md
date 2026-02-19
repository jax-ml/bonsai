## CLIP in JAX

This directory contains a pure JAX implementation of the **CLIP (Contrastive Language–Image Pretraining)** model, implemented using **Flax**.

The model consists of:
- A Vision Transformer (ViT-style) image encoder
- A Transformer-based text encoder
- A shared embedding space trained with a contrastive objective

This implementation focuses on correctness, modularity, and testability, and is designed to integrate cleanly with the rest of the Bonsai model zoo.

---

## Tested on

| Model Name | Config | CPU | GPU A100 (1x) | GPU H100 (1x) | GPU A100 (8x) | GPU H100 (8x) | TPU v2 (8x) | TPU v5e (1x) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CLIP (ViT + Text Transformer) | ✅ Supported | ✅ Runs | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check |

> **Note**  
> This model is tested and supported on **Python 3.11**.  
> Python 3.13 is currently **not supported** due to upstream JAX/Flax incompatibilities.

---

## Running this model

### Forward pass test (recommended)

You can verify that the model runs correctly by executing the pytest forward test:

```sh
python -m pytest bonsai/models/clip/tests/test_model.py -vv
