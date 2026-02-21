#  Bonsai

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Bonsai is a minimal, lightweight JAX implementation of popular models.

We're committed to making popular models accessible in JAX through simple, hackable, and concise code. Our aim is to lower the barrier to entry for JAX and promote academic innovation.


> [!TIP]
> For large-scale or industry use on Google Cloud, see [MaxText](https://github.com/AI-Hypercomputer/maxtext) and [MaxDiffusion](https://github.com/AI-Hypercomputer/maxdiffusion).


## Models

The following models are part of Bonsai. We have included the current model status here to easily convey which models are ready for full use. We categorize them as follows:
1. ✅ Ready with broad support
2. ⚙️ Adding additional features
4. 🟡 In progress
5. ⏳ Coming soon (has open PR)

These are listed based on status and then alphabetically.


| Model                                         | Type                 | Status | Details |
|-----------------------------------------------|----------------------|--------|---------|
| [ConvNeXT](bonsai/models/convnext/)           | Image classification | ✅     |         |
| [Densenet](bonsai/models/densenet121/)        | Image classification | ✅     |         |
| [EfficientNet](bonsai/models/efficientnet/)   | Image classification | ✅     |         |
| [Qwen 3](bonsai/models/qwen3)                 | LLM                  | ✅     |         |
| [ResNet50](bonsai/models/resnet)              | Image classification | ✅     |         |
| [VGG](bonsai/models/vgg19)                    | Image classification | ✅     |         |
| [Dinov3](bonsai/models/dinov3)                | Vision FM            | ✅     |         |
| [VJEPA-2](bonsai/models/vjepa2)               | Video FWM            | ✅     |         |
| [Qwen3VL](bonsai/models/qwen3_vl)             | VLM                  | ✅     |         |
| [Nvidia-Cosmos-Reason2](bonsai/models/qwen3_vl)| VLM                  | ✅     |         |
| [Gemma3](bonsai/models/gemma3)                | VLM                  | ⚙️     | Local attention cache and todos in file        |
| [Mamba2](bonsai/models/mamba2)                | Language SSM         | ⚙️     | Caching and sharding        |
| [umT5](bonsai/models/umt5)                    | LLM                  | ⚙️     | Caching and sharding        |
| [ViT](bonsai/models/vit)                      | Image classification | ⚙️     | Sharding        |
| [LLaDa](bonsai/models/llada_8b/)              | Diffusion LLM        | 🟡     | Need more numerical testing        |
| [Sam2](bonsai/models/sam2/)                   | Image segmentation   | 🟡     | Need more numerical testing        |
| [UNet](bonsai/models/unet/)                   | Image                | 🟡     | Need a reference implementation and numerical testing        |
| [VAE](bonsai/models/vae/)                     | Generative model     | 🟡     | Need a reference implementation and numerical testing         |
| [Whisper](bonsai/models/whisper/)             | Speech recognition   | 🟡     | Need more numerical testing and not all call methods implemented        |
| CLIP                                          |                      | ⏳     |         |


Got models you'd like to see in JAX? [Add a request](https://github.com/jax-ml/bonsai/issues) or [contribute](CONTRIBUTING.md). Please refer to the open issues and PRs before creating a new one to see if a feature is already being addressed.

## 🏁 Getting Started

To get started with JAX Bonsai, follow these steps to set up your development environment and run the models.

### Installation

Clone the JAX Bonsai repository to your local machine.

```bash
git clone https://github.com/jax-ml/bonsai.git
cd bonsai
```

Install the latest repository.
```bash
pip install -e .
```

### Running models

Jump right into our [Qwen3](bonsai/models/qwen3) model, implemented in [400 lines of code](bonsai/models/qwen3/modeling.py) in JAX.

```python
python bonsai/models/qwen3/tests/run_model.py
```


## Contributing

We welcome contributions!
If you're interested in [adding new models](CONTRIBUTING.md#contributing-a-model), improving existing implementations, or enhancing documentation, please see our [Contributing Guidelines](CONTRIBUTING.md).

Join our [discord](https://discord.gg/9x62QwZXj7) to socialize with other JAX enthusiasts.

## Useful Links
* [JAX](https://docs.jax.dev/en/latest/quickstart.html): Learn more about JAX, a super fast NumPy-based ML framework with automatic differentiation.
* [The JAX ecosystem](https://docs.jaxstack.ai/en/latest/getting_started.html): Unlock unparalleled speed and scale for your next-generation models. Explore an incredible suite of tools and libraries that effortlessly extend JAX's capabilities, transforming how you build, train, and deploy.
* [MaxText](https://github.com/AI-Hypercomputer/maxtext) and [MaxDiffusion](https://github.com/AI-Hypercomputer/maxdiffusion): Industury solution for highly scalable, high-performant JAX model library via Google Cloud Platform.
* [JAX LLM Examples](https://github.com/jax-ml/jax-llm-examples): Example high-performant implementation of LLMs in pure JAX.
