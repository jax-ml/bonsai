## ConvNeXT in JAX

This directory contains a pure JAX implementation of the [ConvNeXT](https://huggingface.co/docs/transformers/en/model_doc/convnext) model, using the Flax NNX API.

## Model Configuration Support Status

| Model Name | Config Support Status |
| :--- | :--- |
| [ConvNeXT tiny 224](https://huggingface.co/facebook/convnext-tiny-224) | **✅ Supported** |
| [ConvNeXT small 224](https://huggingface.co/facebook/convnext-small-224) | **✅ Supported** |
| [ConvNeXT base 224](https://huggingface.co/facebook/convnext-base-224) | **✅ Supported** |
| [ConvNeXT large 224](https://huggingface.co/facebook/convnext-large-224) | **✅ Supported** |

### Running this model

Run ConvNeXT model inference in action:

```sh
python3 bonsai.models.convnext.tests.run_model.py
```
