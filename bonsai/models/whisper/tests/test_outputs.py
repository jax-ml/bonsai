# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from transformers import WhisperTokenizer

from bonsai.models.whisper import modeling as model_lib
from bonsai.models.whisper import audio


class WhisperTest(absltest.TestCase):
    def test_full(self):
        # Download test audio if not exists
        audio_path = os.path.join(os.path.dirname(__file__), "audio_samples", "bush_moscow_speech.mp3")
        if not os.path.exists(audio_path):
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)
            audio_url = "https://bush41library.tamu.edu/files/audio/Remarks%20of%20Vice%20President%20Bush%20Upon%20Arrival%20in%20Moscow%20for%20the%20Funeral%20of%20Konstantin%20Chernenko%2012%20March%201985.mp3"
            subprocess.run(["wget", "-O", audio_path, audio_url], check=True, capture_output=True)

        # Load models
        jax_model = model_lib.load_model("tiny")
        
        # Load and process audio
        audio_tensor = audio.load_audio(audio_path)
        mel_tensor = audio.log_mel_spectrogram(audio_tensor)
        mel_jax = jnp.array(mel_tensor)
        
        # Test segment with speech (5-35 seconds)
        mel_segment = mel_jax[:, 500:3500][None, :, :]  # 5-35 seconds
        
        # Get audio features
        audio_features = jax_model.embed_audio(mel_segment)
        
        # Initialize tokenizer
        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")
        
        # Create initial tokens
        sot = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        en_token = tokenizer.convert_tokens_to_ids("<|en|>")
        transcribe = tokenizer.convert_tokens_to_ids("<|transcribe|>")
        no_timestamps = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        
        initial_tokens = jnp.array([[sot, en_token, transcribe, no_timestamps]], dtype=jnp.int32)
        
        # Get logits
        logits = jax_model.logits(initial_tokens, audio_features)
        
        # Basic sanity checks
        self.assertEqual(logits.shape, (1, 4, 51865))
        self.assertFalse(jnp.any(jnp.isnan(logits)))
        self.assertFalse(jnp.any(jnp.isinf(logits)))
        
        # Check probability distribution
        probs = jax.nn.softmax(logits, axis=-1)
        prob_sums = jnp.sum(probs, axis=-1)
        np.testing.assert_allclose(prob_sums, 1.0, rtol=1e-5)
        
        # Compare with PyTorch if available
        try:
            import whisper
            torch_model = whisper.load_model("tiny")
            
            # Extract same audio segment
            start_sample = 5 * 16000
            end_sample = 35 * 16000
            audio_segment = audio_tensor[start_sample:end_sample]
            
            # PyTorch transcription
            torch_audio_segment = torch.from_numpy(np.array(audio_segment)).float()
            with torch.no_grad():
                result = torch_model.transcribe(torch_audio_segment, language="en", task="transcribe", 
                                              initial_prompt=None, verbose=False)
            torch_text = result["text"].strip()
            
            # JAX transcription
            jax_tokens = self._greedy_decode_jax(jax_model, audio_features, tokenizer, max_length=200)
            jax_text = tokenizer.decode(jax_tokens[0].tolist(), skip_special_tokens=True)
            
            # Check similarity (should be high)
            jax_words = set(jax_text.lower().split())
            torch_words = set(torch_text.lower().split())
            common_words = jax_words & torch_words
            all_words = jax_words | torch_words
            similarity = len(common_words) / len(all_words) if all_words else 0
            
            # Should be at least 60% similar
            self.assertGreater(similarity, 0.6, f"Transcriptions too different: {similarity:.2f}")
            
        except ImportError:
            self.skipTest("PyTorch Whisper not available for comparison")

    def _greedy_decode_jax(self, model, audio_features, tokenizer, max_length=200):
        """Simple greedy decoding for JAX model."""
        sot = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        en_token = tokenizer.convert_tokens_to_ids("<|en|>")
        transcribe = tokenizer.convert_tokens_to_ids("<|transcribe|>")
        no_timestamps = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        eot = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        
        initial_tokens = [sot, en_token, transcribe, no_timestamps]
        tokens = jnp.array([initial_tokens], dtype=jnp.int32)
        
        for i in range(max_length):
            logits = model.logits(tokens, audio_features)
            last_logits = logits[0, -1, :]
            next_token = jnp.argmax(last_logits)
            tokens = jnp.concatenate([tokens, jnp.array([[next_token]], dtype=jnp.int32)], axis=1)
            
            if next_token == eot:
                break
        
        return tokens


if __name__ == "__main__":
    absltest.main()