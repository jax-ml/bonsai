# MiMo-Audio in JAX

This directory contains a pure JAX implementation of the [MiMo-Audio multimodal language model](https://github.com/XiaomiMiMo/MiMo), using the [Flax NNX](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/index.html) API.

MiMo-Audio is a unified speech-text model that supports:
- **Text-to-Speech (TTS)**: Generate natural speech from text
- **Speech-to-Text (ASR)**: Transcribe speech to text
- **Speech-to-Speech**: Direct speech translation and conversion

## Model Configuration Support Status

| Model Name | Config Support Status |
| :--- | :--- |
| **Main Models** | |
| [MiMo-Audio-7B-Base](https://huggingface.co/XiaomiMiMo/MiMo-Audio-7B-Base) | **✅ Supported** |
| [MiMo-Audio-7B-Instruct](https://huggingface.co/XiaomiMiMo/MiMo-Audio-7B-Instruct) | **✅ Supported** |
| **Audio Tokenizer** | |
| [MiMo-Audio-Tokenizer](https://huggingface.co/XiaomiMiMo/MiMo-Audio-Tokenizer) | **✅ Supported** |


## Running this model

Run MiMo-Audio inference with a minimal example:

```sh
python3 -m bonsai.models.mimo_audio.test.run_model
```
## Model Architecture

MiMo-Audio consists of three main components:

### 1. Main Transformer (Qwen2-based)
- **Layers**: 36
- **Hidden size**: 4096
- **Attention heads**: 32 (8 KV heads)
- **Intermediate size**: 11008
- **Max position embeddings**: 8192

### 2. Local Transformer (for audio generation)
- **Layers**: 16
- **Hidden size**: 1024
- **Attention heads**: 64
- **FFN dimension**: 4096

### 3. Input Local Transformer (for audio encoding)
- **Layers**: 6
- **Hidden size**: 1024
- **Attention heads**: 64
- **Bidirectional attention**: Yes

### 4. Audio Tokenizer
- **Encoder layers**: 32
- **Decoder layers**: 32
- **Quantizers**: 20
- **Sampling rate**: 24000 Hz
- **Audio channels**: 8 (used for multi-codebook representation)

## Special Tokens

MiMo-Audio uses special tokens for controlling speech generation:

- `<|sostm|>` (151648): Start of speech/stream
- `<|eostm|>` (151649): End of speech/stream
- `<|sosp|>` (151646): Start of speech
- `<|eosp|>` (151647): End of speech
- `<|empty|>` (151645): Empty token (indicates audio generation)
- `<|eot|>` (151643): End of turn

## Input Format

MiMo-Audio uses an interleaved format where each group contains:
- 1 text token (repeated `group_size` times)
- 8 audio channel tokens (one per channel, repeated `group_size` times)

Shape: `[batch, audio_channels + 1, num_groups * group_size]`

For text-only input (TTS), audio channels are filled with channel-specific empty IDs.

## Audio Processing

### Encoding (Speech → Tokens)
1. Waveform → Mel spectrogram
2. Mel spectrogram → Encoder
3. Encoder → Quantizer → Audio tokens (8 channels)

### Decoding (Tokens → Speech)
1. Audio tokens → Decoder
2. Decoder → Vocoder
3. Vocoder → Waveform (24kHz)

## References

- [MiMo-Audio Paper](https://github.com/XiaomiMiMo/MiMo-Audio/blob/main/MiMo-Audio-Technical-Report.pdf)
- [Official Implementation](https://github.com/XiaomiMiMo/MiMo-Audio)
