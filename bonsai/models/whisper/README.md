# Whisper in JAX NNX

This directory contains a **pure JAX implementation** of the [OpenAI Whisper speech recognition model](https://github.com/openai/whisper), using the [Flax NNX](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/index.html) API.

Whisper is a general-purpose speech recognition model that can transcribe audio in multiple languages and perform various speech recognition tasks including transcription, translation, and language identification.

**🚀 Status: Prototype* - The model successfully loads pretrained weights and performs accurate speech transcription for 30 sec. 

## Model Architecture

The Whisper model consists of two main components:

1. **Audio Encoder**: A convolutional neural network followed by transformer layers that processes mel spectrogram features
2. **Text Decoder**: A transformer decoder that generates text tokens autoregressively with cross-attention to the audio features

**Whisper Models** | | 
| [Whisper Tiny](https://huggingface.co/openai/whisper-tiny) | ✅ | 
| [Whisper Base](https://huggingface.co/openai/whisper-base) | ✅ |
| [Whisper Small](https://huggingface.co/openai/whisper-small)| ✅ |
| [Whisper Medium](https://huggingface.co/openai/whisper-medium) | ✅ |
| [Whisper Large](https://huggingface.co/openai/whisper-large) | ✅ | 

### Running this model

Run Whisper in action, implemented in JAX NNX for high-performance speech recognition.


**🔧 Basic Model Runner:**
```sh
python3 -m bonsai.models.whisper.tests.run_model
```

