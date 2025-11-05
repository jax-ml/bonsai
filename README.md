#  Bonsai

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Bonsai is a minimal, lightweight JAX implementation of popular models.

We're committed to making popular models accessible in JAX through simple, hackable, and concise code. Our aim is to lower the barrier to entry for JAX and promote academic innovation.


> [!TIP]
> For large-scale or industry use on Google Cloud, see [MaxText](https://github.com/AI-Hypercomputer/maxtext) and [MaxDiffusion](https://github.com/AI-Hypercomputer/maxdiffusion).


## Models

* **LLM (Large Language Models)**: [Qwen 3](bonsai/models/qwen3), ...
* **dLLM (diffusion-based Large Language Models)**: (Coming soon) Llada, ...
* **ASR (Automatic Speech Recognition)**: (Coming soon) Whisper, ...
* **Image segmentation**: [SAM2](bonsai/models/sam2), ...
* **Image classification**: [ResNet50](bonsai/models/resnet50), ...
* **Computational Biology**: (Coming soon) ESM, ...
* **WFM (World Foundation Model)**: (Coming soon) Cosmos, ...

Got models you'd like to see in JAX? [Add a request](https://github.com/jax-ml/bonsai/issues) or [contribute](CONTRIBUTING.md).

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

Jump right into our [Qwen3](bonsai/models/qwen3) model, implemented in [300 lines of code](bonsai/models/qwen3/modeling.py) in JAX.

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
