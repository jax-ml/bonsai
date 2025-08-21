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
from pathlib import Path
from huggingface_hub import snapshot_download

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


def extract_mel_features_whisper(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Extract mel spectrogram features from audio using Whisper's exact preprocessing."""
    try:
        import librosa
        
        # Whisper's exact mel spectrogram parameters
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_mels=80,
            hop_length=160,
            win_length=400,
            window='hann',
            fmin=0,
            fmax=8000,
            power=2.0
        )
        
        # Convert to log10 scale (Whisper's approach)
        log_spec = np.log10(mel_spec + 1e-10)
        
        # Clip values (Whisper's approach)
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        
        # Normalize (Whisper's approach)
        log_spec = (log_spec + 4.0) / 4.0
        
        return log_spec.T  # Transpose to (time, n_mels)
        
    except ImportError:
        print("librosa not available, using dummy mel features")
        # Generate dummy mel features for testing
        time_steps = len(audio) // 160  # Approximate time steps
        return np.random.randn(time_steps, 80)


def run_model(MODEL_CP_PATH: Optional[str] = None, audio_path: Optional[str] = None):
    """Run Whisper model for speech recognition."""
    model_name = "openai/whisper-tiny"
    if MODEL_CP_PATH is None:
        MODEL_CP_PATH = "/tmp/models-bonsai/" + model_name.split("/")[1]
    
    # Download model if not present
    if not os.path.isdir(MODEL_CP_PATH):
        print(f"Downloading {model_name} to {MODEL_CP_PATH}...")
        snapshot_download(model_name, local_dir=MODEL_CP_PATH)

    # Load audio
    if audio_path is None:
        print("No audio file provided, using test speech sample")
        # Use the Bush speech sample
        audio_path = Path(__file__).parent / "audio_samples" / "bush_speech.wav"
        print(f"Using default audio: {audio_path}")
    
    print(f"Loading audio from {audio_path}")
    audio = load_audio_file(str(audio_path))
    
    # Extract mel features using Whisper's exact preprocessing
    mel_features = extract_mel_features_whisper(audio)
    # Convert to JAX array after mel extraction
    mel_features = jnp.array(mel_features)
    print(f"Mel features shape: {mel_features.shape}")
    
    # Pad or truncate to expected length (Whisper uses 3000 for full context)
    max_time_steps = 3000  # Whisper's full audio context length
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
    
    # Simple token-to-text decoding (using Whisper vocabulary)
    print("Decoding transcription...")
    try:
        # Load vocabulary from the model files
        import json
        vocab_path = os.path.join(MODEL_CP_PATH, "vocab.json")
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                vocab_data = json.load(f)
            
            # Create reverse mapping: token_id -> text
            vocab = {int(token_id): text for text, token_id in vocab_data.items()}
            
            # Add special tokens
            special_tokens = {
                50258: "<|startoftranscript|>",
                50259: "<|en|>", 
                50359: "<|transcribe|>",
                50363: "<|notimestamps|>",
                50257: "<|endoftext|>",
            }
            vocab.update(special_tokens)
            
            # Decode tokens to text
            decoded_text = ""
            for token in generated_tokens[0]:
                token_id = int(token)
                if token_id in vocab:
                    text = vocab[token_id]
                    # Skip special tokens for clean output
                    if not (text.startswith("<|") and text.endswith("|>")):
                        # Replace BPE space marker with actual space
                        text = text.replace("Ġ", " ")
                        decoded_text += text
                else:
                    # For unknown tokens, show the token ID
                    decoded_text += f"[{token_id}]"
            
            print(f"Transcription: {decoded_text.strip()}")
        else:
            print("Vocabulary file not found, showing raw tokens:")
            print(f"Generated tokens: {generated_tokens[0]}")
            
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
