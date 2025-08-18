#!/usr/bin/env python3
"""Test Whisper generation quality comparison."""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from bonsai.models.whisper import modeling as model_lib
from bonsai.models.whisper.tests.run_model_hf import run_model_hf


def load_audio_file(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Load audio file and resample to target sample rate."""
    try:
        import librosa
        audio, _ = librosa.load(audio_path, sr=sample_rate)
        return audio
    except ImportError:
        raise RuntimeError("librosa is required for this demo; please install it.")


def extract_mel_features(audio: np.ndarray, sample_rate: int = 16000, n_mels: int = 80) -> np.ndarray:
    """Extract mel spectrogram features from audio."""
    try:
        import librosa
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sample_rate, 
            n_mels=n_mels,
            hop_length=160,
            win_length=400,
            window='hann'
        )
        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec.T  # Transpose to (time, n_mels)
    except ImportError:
        raise RuntimeError("librosa is required for this demo; please install it.")


def test_nnx_generation(audio_path: str, use_random_weights: bool = True):
    """Test NNX Whisper generation."""
    print(f"\n=== NNX Whisper Generation ({'Random' if use_random_weights else 'Pretrained'} Weights) ===")
    
    # Load audio
    print(f"Loading audio: {audio_path}")
    audio = load_audio_file(audio_path)
    mel_features = extract_mel_features(audio)
    print(f"Mel features shape: {mel_features.shape}")
    
    # Pad or truncate to expected length
    max_time_steps = 1500
    if mel_features.shape[0] > max_time_steps:
        mel_features = mel_features[:max_time_steps]
    else:
        padding = np.zeros((max_time_steps - mel_features.shape[0], mel_features.shape[1]))
        mel_features = np.concatenate([mel_features, padding], axis=0)
    
    # Add batch dimension and transpose to (batch, n_mels, time)
    mel_features = mel_features[None, ...].transpose(0, 2, 1)
    mel_features = jnp.array(mel_features)
    
    # Create model
    config = model_lib.WhisperConfig.whisper_tiny()
    
    if use_random_weights:
        # Use random weights
        rngs = nnx.Rngs(0)
        model = model_lib.WhisperModel(config, rngs=rngs)
        print("Using random weights")
    else:
        # Try to load pretrained weights
        try:
            from bonsai.models.whisper import params as P
            model = P.create_model_from_safe_tensors("/tmp/models-bonsai/whisper-tiny", config)
            print("Using pretrained weights")
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")
            print("Falling back to random weights")
            rngs = nnx.Rngs(0)
            model = model_lib.WhisperModel(config, rngs=rngs)
    
    # Test forward pass
    print("Testing forward pass...")
    start_time = time.time()
    tokens = jnp.array([[50258]])  # BOS token
    logits = model(mel_features, tokens)
    forward_time = time.time() - start_time
    print(f"Forward pass: {forward_time:.3f}s, output shape: {logits.shape}")
    
    # Test generation
    print("Testing text generation...")
    start_time = time.time()
    generated_tokens = model_lib.generate(model, mel_features, max_length=50, temperature=0.0)
    generation_time = time.time() - start_time
    print(f"Generation: {generation_time:.3f}s")
    
    # Decode tokens (simple mapping for testing)
    try:
        # Try to use HF processor for decoding
        from transformers import WhisperProcessor
        processor = WhisperProcessor.from_pretrained("/tmp/models-bonsai/whisper-tiny")
        decoded_text = processor.batch_decode(generated_tokens, skip_special_tokens=True)
        print(f"Generated text: {decoded_text[0]}")
    except Exception as e:
        print(f"Could not decode with HF processor: {e}")
        print(f"Generated token IDs: {generated_tokens[0][:20]}...")
    
    return {
        "forward_time": forward_time,
        "generation_time": generation_time,
        "tokens": generated_tokens,
    }


def test_hf_generation(audio_path: str):
    """Test HuggingFace Whisper generation."""
    print(f"\n=== HuggingFace Whisper Generation ===")
    
    start_time = time.time()
    transcription = run_model_hf("openai/whisper-tiny", audio_path)
    total_time = time.time() - start_time
    
    print(f"HF generation time: {total_time:.3f}s")
    print(f"HF transcription: {transcription}")
    
    return {
        "generation_time": total_time,
        "transcription": transcription,
    }


def main():
    """Run generation comparison tests."""
    audio_path = "bonsai/models/whisper/tests/audio_samples/bush_speech.wav"
    
    if not Path(audio_path).exists():
        print(f"Audio file not found: {audio_path}")
        return
    
    print("=== Whisper Generation Quality Test ===")
    print(f"Audio file: {audio_path}")
    
    # Test NNX with random weights
    nnx_random = test_nnx_generation(audio_path, use_random_weights=True)
    
    # Test NNX with pretrained weights (if possible)
    nnx_pretrained = test_nnx_generation(audio_path, use_random_weights=False)
    
    # Test HF model
    hf_result = test_hf_generation(audio_path)
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"NNX Random - Forward: {nnx_random['forward_time']:.3f}s, Generation: {nnx_random['generation_time']:.3f}s")
    print(f"NNX Pretrained - Forward: {nnx_pretrained['forward_time']:.3f}s, Generation: {nnx_pretrained['generation_time']:.3f}s")
    print(f"HF - Total: {hf_result['generation_time']:.3f}s")
    print(f"HF Transcription: {hf_result['transcription']}")


if __name__ == "__main__":
    from flax import nnx
    main()
