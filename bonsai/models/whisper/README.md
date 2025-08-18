# Whisper in JAX

This directory contains a pure JAX implementation of the [OpenAI Whisper speech recognition model](https://github.com/openai/whisper), using the [Flax NNX](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/index.html) API.

Whisper is a general-purpose speech recognition model that can transcribe audio in multiple languages and perform various speech recognition tasks including transcription, translation, and language identification.

## Model Architecture

The Whisper model consists of two main components:

1. **Audio Encoder**: A convolutional neural network followed by transformer layers that processes mel spectrogram features
2. **Text Decoder**: A transformer decoder that generates text tokens autoregressively with cross-attention to the audio features

Key features:
- **Multilingual**: Supports 99 languages
- **Robust**: Handles various accents, background noise, and technical language
- **Flexible**: Can perform transcription, translation, and language identification
- **Efficient**: Optimized for JAX with JIT compilation

## Tested on:  
*(Last Updated: 2025-01-27)*

| Model Name | Config | CPU | GPU A100 (1x) | GPU H100 (1x) | GPU A100 (8x) | GPU H100 (8x) | TPU v2 (8x) | TPU v5e (1x) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Whisper Models** | | | | | | | | |
| [Whisper Tiny](https://huggingface.co/openai/whisper-tiny) | ✅ Supported | ✅ Runs | ✅ Runs | ✅ Runs | ✅ Runs | ✅ Runs | ✅ Runs | ✅ Runs |
| [Whisper Base](https://huggingface.co/openai/whisper-base) | ✅ Supported | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check |
| [Whisper Small](https://huggingface.co/openai/whisper-small) | ✅ Supported | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check |
| [Whisper Medium](https://huggingface.co/openai/whisper-medium) | ✅ Supported | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check |
| [Whisper Large](https://huggingface.co/openai/whisper-large) | ✅ Supported | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check | ❔ Needs check |

### Running this model

Run Whisper in action, implemented in JAX NNX for high-performance speech recognition.

```sh
python3 -m bonsai.models.whisper.tests.run_model
```

For testing with a specific audio file:

```sh
python3 -m bonsai.models.whisper.tests.run_model --audio_path /path/to/audio.wav
```

## Model Configurations

The implementation supports all standard Whisper model sizes:

- **Tiny**: 39M parameters, fastest inference
- **Base**: 74M parameters, good balance of speed and accuracy
- **Small**: 244M parameters, improved accuracy
- **Medium**: 769M parameters, high accuracy
- **Large**: 1550M parameters, best accuracy

## Usage Examples

### Basic Transcription

```python
import jax
import jax.numpy as jnp
from bonsai.models.whisper import modeling, params

# Load model
config = modeling.WhisperConfig.whisper_tiny()
model = params.load_whisper_model("openai/whisper-tiny", config)

# Prepare audio features (mel spectrogram)
mel_features = jnp.array(...)  # Shape: (batch_size, time, n_mels)

# Generate transcription
tokens = modeling.generate(model, mel_features, max_length=448)
```

### Advanced Usage

```python
# Custom configuration
config = modeling.WhisperConfig(
    n_audio_state=768,
    n_audio_head=12,
    n_audio_layer=12,
    n_text_state=768,
    n_text_head=12,
    n_text_layer=12,
)

# Load with custom config
model = params.load_whisper_model("openai/whisper-small", config)

# Jitted forward pass for better performance
@jax.jit
def forward_pass(model, mel, tokens):
    return model(mel, tokens)

logits = forward_pass(model, mel_features, tokens)
```

## Audio Processing

The model expects mel spectrogram features as input. You can use librosa to extract these features:

```python
import librosa
import numpy as np

def extract_mel_features(audio_path, sample_rate=16000, n_mels=80):
    audio, _ = librosa.load(audio_path, sr=sample_rate)
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=sample_rate, 
        n_mels=n_mels,
        hop_length=160,
        win_length=400
    )
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec.T  # (time, n_mels)
```

## Performance Optimization

The implementation includes several optimizations:

1. **JIT Compilation**: All forward passes are JIT-compiled for optimal performance
2. **Memory Efficiency**: Uses JAX's memory-efficient operations
3. **Profiling Support**: Includes JAX profiling for performance analysis
4. **Batch Processing**: Supports batched inference for multiple audio files

## Dependencies

Required dependencies:
- `jax`
- `flax` (with NNX support)
- `transformers` (for tokenization)
- `huggingface_hub` (for model downloading)
- `safetensors` (for model loading)

Optional dependencies:
- `librosa` (for audio processing)
- `torch` (fallback for model loading)

## How to contribute to this model

We welcome contributions! You can contribute to this model via the following:

* **Add model variants**: Test and add support for additional Whisper model configurations
* **Hardware testing**: Run [run_model.py](tests/run_model.py) on different hardware configurations and update the testing matrix
* **Performance improvements**: Optimize the model for better inference speed or memory usage
* **Feature additions**: Add support for additional Whisper features like language identification or translation
* **Documentation**: Improve examples and documentation

### Testing on your hardware

To test the model on your hardware and update the testing matrix:

1. Run the model: `python3 -m bonsai.models.whisper.tests.run_model`
2. Check if it runs successfully
3. Update the testing matrix in this README with your results

## References

- [Whisper Paper](https://cdn.openai.com/papers/whisper.pdf)
- [OpenAI Whisper GitHub](https://github.com/openai/whisper)
- [HuggingFace Whisper](https://huggingface.co/openai/whisper-tiny)
- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax NNX Documentation](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/index.html)
