# DINOv3 in Jax
This directory contains a pure jax implementation of the [Dinov3 collection of VIT models](https://huggingface.co/collections/facebook/dinov3) using the [Flax NNX](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/index.html) API.

## Model configuration support status
| Model Name | Size |Config Support Status |
| :--- | :--- | :--- |
| **Web (LVD) Models** | | |
| [ViT-S](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) | 21M | **✅ Supported** |
| [ViT-S+](https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m) | 29M | **✅ Supported** |
| [ViT-B](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) | 86M | **✅ Supported** |
| [ViT-L](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) | 0.3B |**✅ Supported** |
| [ViT-H+](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m) | 0.84B |**✅ Supported** |
| [ViT-7B](https://huggingface.co/facebook/dinov3-vit7b16-pretrain-lvd1689m) | 7B |**Needs sharding** |
| **Satellite (SAT) Models** | | |
| [ViT-L](https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m) | 0.3B | **✅ Supported** |
| [ViT-7B](https://huggingface.co/facebook/dinov3-vit7b16-pretrain-sat493m) | 7B |**Needs sharding** |

* Note: Hf login and approval required. 