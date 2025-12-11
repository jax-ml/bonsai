# Mamba2 in JAX

This directory contains a pure JAX implementation of the [Mamba2](https://arxiv.org/abs/2405.21060) model, using the Flax NNX API.

## Model Configuration Support Status

| Model Name | Config Support Status |
| :--- | :--- |
| [Mamba2ForCausalLM](https://arxiv.org/abs/2405.21060) | **✅ Supported** |
| [Mamba2Forecaster](https://arxiv.org/abs/2405.21060) | **✅ Supported** |

### Running this model

Run Mamba2 model inference in action:

```bash
python bonsai/models/mamba2/tests/run_model.py
```

### Hardware Validation Status

| Hardware | Status |
| :--- | :--- |
| CPU | ✅ Runs |
| GPU (NVIDIA) | ❔ Needs check |
| TPU v5e | ❔ Needs check |

## How to contribute to this model

We welcome contributions! You can contribute to this model via the following:
* Add a model config variant from the above `🟡 Not started` to `class ModelConfig` in [modeling.py](modeling.py). Make sure your code is runnable on at least one hardware before creating a PR.
* Got some hardware? Run [run_model.py](tests/run_model.py) the existing configs above on hardwares marked `❔ Needs check`. Mark as `✅ Runs` or `⛔️ Not supported`.

## References

* **Paper**: [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060) (Dao & Gu, ICML 2024)
* **Reference PyTorch Implementation**: [vasqu/mamba2-torch](https://github.com/vasqu/mamba2-torch)
* **Original JAX Port**: [CosmoNaught/mamba2-jax](https://github.com/CosmoNaught/mamba2-jax)