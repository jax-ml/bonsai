# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from huggingface_hub import snapshot_download
from transformers import WhisperProcessor

from bonsai.models.whisper import modeling, params


def load_audio_file(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Load audio file and resample to target sample rate."""
    try:
        import librosa
        audio, _ = librosa.load(audio_path, sr=sample_rate)
        return audio
    except ImportError:
        print("librosa not available, using dummy audio data")
        # Generate dummy audio for testing
        return np.random.randn(sample_rate * 3)  # 3 seconds of random audio


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
        print("librosa not available, using dummy mel features")
        # Generate dummy mel features for testing
        time_steps = len(audio) // 160  # Approximate time steps
        return np.random.randn(time_steps, n_mels)


def run_model(MODEL_CP_PATH: Optional[str] = None, audio_path: Optional[str] = None):
    """Run Whisper model for speech recognition."""
    model_name = "openai/whisper-tiny"
    if MODEL_CP_PATH is None:
        MODEL_CP_PATH = "/tmp/models-bonsai/" + model_name.split("/")[1]
    
    # Download model if not present
    if not os.path.isdir(MODEL_CP_PATH):
        print(f"Downloading {model_name} to {MODEL_CP_PATH}...")
        snapshot_download(model_name, local_dir=MODEL_CP_PATH)

    # Load processor for tokenization
    processor = WhisperProcessor.from_pretrained(MODEL_CP_PATH)
    
    # Load audio
    if audio_path is None:
        print("No audio file provided, using test speech sample")
        # Use one of our generated speech samples
        audio_path = "bonsai/models/whisper/tests/audio_samples/medium_speech.wav"
        print(f"Using default audio: {audio_path}")
    
    print(f"Loading audio from {audio_path}")
    audio = load_audio_file(audio_path)
    
    # Extract mel features (librosa expects numpy array)
    mel_features = extract_mel_features(audio)
    # Convert to JAX array after mel extraction
    mel_features = jnp.array(mel_features)
    print(f"Mel features shape: {mel_features.shape}")
    
    # Pad or truncate to expected length
    max_time_steps = 1500  # Whisper's default audio context length
    if mel_features.shape[0] > max_time_steps:
        mel_features = mel_features[:max_time_steps]
    else:
        # Pad with zeros using JAX
        padding = jnp.zeros((max_time_steps - mel_features.shape[0], mel_features.shape[1]))
        mel_features = jnp.concatenate([mel_features, padding], axis=0)
    
    # Add batch dimension and transpose to (batch, n_mels, time)
    mel_features = mel_features[None, ...].transpose(0, 2, 1)  # (1, n_mels, time)
    mel_features = jnp.array(mel_features)
    
    # Create model from pretrained weights
    config = modeling.WhisperConfig.whisper_tiny()
    model = params.create_model_from_safe_tensors(MODEL_CP_PATH, config)
    print("Loaded pretrained Whisper weights")
    
    # Create dummy tokens for testing (BOS token)
    tokens = jnp.array([[50258]])  # BOS token for Whisper
    
    print("Running Whisper model...")
    
    # Time the forward pass
    start_time = time.perf_counter()
    
    # Run forward pass
    logits = model(mel_features, tokens)
    
    # Block until computation is complete
    jax.block_until_ready(logits)
    
    end_time = time.perf_counter()
    print(f"Forward pass completed in {end_time - start_time:.4f} seconds")
    print(f"Output logits shape: {logits.shape}")
    
    # Test generation
    print("Testing text generation...")
    start_time = time.perf_counter()
    
    generated_tokens = modeling.generate(model, mel_features, max_length=50, temperature=0.0)
    
    jax.block_until_ready(generated_tokens)
    end_time = time.perf_counter()
    print(f"Generation completed in {end_time - start_time:.4f} seconds")
    
    # Decode tokens
    try:
        decoded_text = processor.batch_decode(generated_tokens, skip_special_tokens=True)
        print(f"Generated text: {decoded_text[0]}")
    except Exception as e:
        print(f"Could not decode tokens: {e}")
        print(f"Generated tokens: {generated_tokens[0]}")
    
    # Test with JAX profiling
    print("Running with JAX profiling...")
    jax.profiler.start_trace("/tmp/profile-data")
    
    # Run a few iterations for profiling
    for i in range(5):
        logits = model(mel_features, tokens)
        jax.block_until_ready(logits)
    
    jax.profiler.stop_trace()
    print("Profiling completed. Run 'xprof --port 8791 /tmp/profile-data' to view traces.")
    
    return model, logits, generated_tokens


if __name__ == "__main__":
    run_model()
