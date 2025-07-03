#  (Work-in-progress) Bonsai

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

This repository serves as a curated list of JAX [NNX](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/index.html) simple implementations of common machine learning models.


Bonsai supports integration with powerful JAX libraries.
* [Tunix](https://github.com/google/tunix/tree/main), a post-training library supporting Supervised Fine-Tuning, RL, Knoweldge Distillation.

> [!IMPORTANT]
> Bonsai is a simple, lightweight JAX implementation. For large-scale high performance pretraining on Google Cloud, see [MaxText](https://github.com/AI-Hypercomputer/maxtext) and [MaxDiffusion](https://github.com/AI-Hypercomputer/maxdiffusion).


## Current Models

* [Qwen 3](bonsai/models/qwen3)
* (Coming soon) Gemma 3
* (Coming soon) Llama 3
* (Coming soon) SAM2

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
If you're interested in adding new models, improving existing implementations, or enhancing documentation, please see our [Contributing Guidelines](CONTRIBUTING.md).

## Useful Links
* [JAX](https://docs.jax.dev/en/latest/quickstart.html): Learn more about JAX, a super fast NumPy-based ML framework with automatic differentiation.
* [The JAX ecosystem](https://docs.jaxstack.ai/en/latest/getting_started.html): Unlock unparalleled speed and scale for your next-generation models. Explore an incredible suite of tools and libraries that effortlessly extend JAX's capabilities, transforming how you build, train, and deploy.
