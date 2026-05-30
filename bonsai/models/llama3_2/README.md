# Llama 3.2 in JAX

This directory contains a pure JAX implementation of the
[Llama 3.2 language model](https://huggingface.co/meta-llama),
using the [Flax NNX](https://flax.readthedocs.io/en/stable/index.html) API.

Note: You need a Hugging Face access token to download model weights.
Set an environment variable `HF_TOKEN` before running any scripts that fetch checkpoints.

```sh
export HF_TOKEN="your_hf_access_token"
```

Some Llama models are gated. Make sure you have accepted the license in the
Hugging Face UI for the specific model you want to use.

## Model Configuration Support Status

| Model Name | Config Support Status |
| :--- | :--- |
| [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) | **✅ Supported** |
| [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | **✅ Supported** |
| [Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) | **✅ Supported** |
| [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | **✅ Supported** |

## Running this model

```sh
# Instruct model (default: 1B)
python3 -m bonsai.models.llama3_2.tests.run_model

# Base model (1B)
python3 -m bonsai.models.llama3_2.tests.run_model --base

# Base model (3B)
python3 -m bonsai.models.llama3_2.tests.run_model --size 3B --base

# Instruct model (3B)
python3 -m bonsai.models.llama3_2.tests.run_model --size 3B
```

## Output parity tests

These tests compare JAX outputs against Hugging Face PyTorch outputs and require `HF_TOKEN`.

```sh
python3 -m bonsai.models.llama3_2.tests.test_outputs_llama3_2
```

## References

* Paper: [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
* Model code: [Hugging Face Transformers (LlamaModel)](https://github.com/huggingface/transformers/tree/main/src/transformers/models/llama)

## How to contribute to this model

We welcome contributions! You can contribute via the following:

* Add a model config variant to `ModelConfig` in [modeling.py](modeling.py).
* Run [run_model.py](tests/run_model.py) and report whether the variant runs on your hardware.
