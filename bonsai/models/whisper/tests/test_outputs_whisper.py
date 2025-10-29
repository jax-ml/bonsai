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

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from transformers import WhisperTokenizer

from bonsai.models.whisper import audio
from bonsai.models.whisper import modeling as model_lib


class WhisperTest(absltest.TestCase):
    def test_full(self):
        repo_root = Path(__file__).resolve().parents[4]
        audio_path = repo_root / "bonsai/tests/data/Speech_12dB_s16_trimmed.flac"
        if not audio_path.exists():
            self.fail(f"Test audio file not found at: {audio_path}")

        # Load models
        jax_model = model_lib.load_model("tiny")

        # Load and process audio
        audio_tensor = audio.load_audio(audio_path)
        mel_tensor = audio.log_mel_spectrogram(audio_tensor)
        mel_jax = jnp.array(mel_tensor)

        # Test segment.
        mel_segment = mel_jax[:, :3000][None, :, :]

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

            torch_audio_segment = torch.from_numpy(np.array(audio_tensor)).float()

            with torch.no_grad():
                result = torch_model.transcribe(
                    torch_audio_segment, language="en", task="transcribe", initial_prompt=None, verbose=False
                )
            torch_text = result["text"].strip()

            # JAX transcription
            jax_tokens = self._greedy_decode_jax(jax_model, audio_features, tokenizer, max_length=200)
            jax_text = tokenizer.decode(jax_tokens[0].tolist(), skip_special_tokens=True)

            # Check similarity
            jax_words = set(jax_text.lower().split())
            torch_words = set(torch_text.lower().split())
            common_words = jax_words & torch_words
            all_words = jax_words | torch_words
            similarity = len(common_words) / len(all_words) if all_words else 0

            # TODO(team): Improve model quality and test with higher expected similarity.
            self.assertGreater(similarity, 0.6, f"Transcriptions differ. JAX: '{jax_text}' vs Torch: '{torch_text}'")

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
