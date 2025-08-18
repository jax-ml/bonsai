#!/usr/bin/env python3
"""Compare HF and NNX preprocessing to find differences."""

import librosa
import numpy as np
from transformers import WhisperProcessor

# Load audio
audio_path = "/Users/fleandr/Downloads/bonsai/bonsai/models/whisper/tests/audio_samples/bush_speech.wav"
audio, _ = librosa.load(audio_path, sr=16000)

print("🎵 Audio loaded, duration:", len(audio) / 16000, "seconds")

# HF preprocessing
print("\n🔍 HF Preprocessing:")
processor = WhisperProcessor.from_pretrained('openai/whisper-tiny')
inputs = processor(audio, sampling_rate=16000, return_tensors='np')
hf_features = inputs['input_features']
print(f"HF shape: {hf_features.shape}")
print(f"HF stats: mean={np.mean(hf_features):.6f}, std={np.std(hf_features):.6f}")
print(f"HF range: [{np.min(hf_features):.6f}, {np.max(hf_features):.6f}]")

# NNX preprocessing
print("\n🔍 NNX Preprocessing:")
mel_spec = librosa.feature.melspectrogram(
    y=audio, 
    sr=16000, 
    n_mels=80,
    hop_length=160,
    win_length=400,
    window='hann',
    fmin=0.0,
    fmax=8000.0
)
mel_spec = librosa.power_to_db(mel_spec, ref=1.0, amin=1e-10, top_db=80.0)

# Transpose to (time, n_mels) and pad/truncate
mel_features = mel_spec.T  # (time, 80)
max_time_steps = 3000  # HF Whisper default context length
original_length = mel_features.shape[0]

if mel_features.shape[0] > max_time_steps:
    mel_features = mel_features[:max_time_steps]
else:
    padding = np.zeros((max_time_steps - mel_features.shape[0], mel_features.shape[1]))
    mel_features = np.concatenate([mel_features, padding], axis=0)

# Apply HF's zero_mean_unit_var_norm (only normalize over non-padded regions)
# HF normalizes each feature vector to zero mean and unit variance
for i in range(mel_features.shape[1]):  # For each mel frequency bin
    feature_vector = mel_features[:original_length, i]  # Only non-padded region
    mean_val = feature_vector.mean()
    std_val = np.sqrt(feature_vector.var() + 1e-7)
    mel_features[:original_length, i] = (feature_vector - mean_val) / std_val
    # Padded regions remain zero (HF behavior)

nnx_features = mel_features[None, ...].transpose(0, 2, 1)
print(f"NNX shape: {nnx_features.shape}")
print(f"NNX stats: mean={np.mean(nnx_features):.6f}, std={np.std(nnx_features):.6f}")
print(f"NNX range: [{np.min(nnx_features):.6f}, {np.max(nnx_features):.6f}]")

# Compare
print("\n🔍 Comparison:")
print(f"Shape match: {hf_features.shape == nnx_features.shape}")
print(f"Mean diff: {abs(np.mean(hf_features) - np.mean(nnx_features)):.6f}")
print(f"Std diff: {abs(np.std(hf_features) - np.std(nnx_features)):.6f}")
print(f"Max diff: {np.max(np.abs(hf_features - nnx_features)):.6f}")

# Check if the difference is significant
if np.max(np.abs(hf_features - nnx_features)) > 0.1:
    print("❌ Significant difference detected!")
else:
    print("✅ Preprocessing looks similar")
