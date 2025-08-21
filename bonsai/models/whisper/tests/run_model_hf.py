import os
from typing import Optional
from pathlib import Path

import numpy as np

from transformers import WhisperProcessor, WhisperForConditionalGeneration


def load_audio_file(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
    try:
        import librosa
        audio, _ = librosa.load(audio_path, sr=sample_rate)
        return audio
    except ImportError:
        raise RuntimeError("librosa is required for this demo; please install it.")


def run_model_hf(model_name: str = "openai/whisper-tiny", audio_path: Optional[str] = None, language: str = "en", task: str = "transcribe"):
    if audio_path is None:
        audio_path = Path(__file__).parent / "audio_samples" / "bush_speech.wav"
        print(f"Using default audio: {audio_path}")

    print("Loading HF Whisper...")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    # Force transcription in a target language for more deterministic results
    try:
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    except Exception:
        pass

    audio = load_audio_file(str(audio_path))
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

    print("Generating with HF model...")
    generated_ids = model.generate(inputs["input_features"])  # type: ignore
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("Transcription:")
    print(transcription)
    return transcription


if __name__ == "__main__":
    run_model_hf()
