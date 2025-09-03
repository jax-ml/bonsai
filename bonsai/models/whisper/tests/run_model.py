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

from bonsai.models.whisper import modeling, params, preprocessing, generation


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
    # Use the preprocessing module which has the correct implementation
    return preprocessing.extract_mel_features_whisper(audio, sample_rate)


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
    
    # Set audio parameters
    sample_rate = 16000
    n_mels = 80
    
    # Extract mel features using original Whisper's preprocessing
    mel_features = preprocessing.extract_mel_features_whisper(audio, sample_rate, n_mels)
    
    print(f"Preprocessing returned mel features shape: {mel_features.shape}")
    print(f"Using original Whisper's exact mel features")
    
    # Create model from pretrained weights
    config = modeling.WhisperConfig.whisper_tiny()
    model = params.create_model_from_safe_tensors(MODEL_CP_PATH, config)
    print("Loaded pretrained Whisper weights")
    
    # Start with BOS token for Whisper
    tokens = jnp.array([[50258]])  # BOS token for Whisper
    
    print("Testing chunked generation with 30-second windows...")
    start_time = time.perf_counter()
    
    # Generate text using the windows
    generated_tokens = generation.generate_chunks_with_beam_search(model, mel_features, max_length=500, temperature=0.0, beam_size=5)
    
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
    
    return model, None, generated_tokens


if __name__ == "__main__":
    run_model()
