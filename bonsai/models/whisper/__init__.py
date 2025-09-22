"""
JAX Whisper Model - Pure JAX implementation of OpenAI's Whisper speech recognition model.
"""

from .audio import load_audio, log_mel_spectrogram, pad_or_trim
from .modeling import ModelDimensions, Whisper, load_model, create_model, greedy_decode, greedy_decode_jit
from .tokenizer import get_tokenizer, Tokenizer
from .params import load_whisper_weights, convert_hf_whisper_to_nnx

# Model configurations
TINY_CONFIG = ModelDimensions(
    n_mels=80,
    n_audio_ctx=1500,
    n_audio_state=384,
    n_audio_head=6,
    n_audio_layer=4,
    n_vocab=51865,
    n_text_ctx=448,
    n_text_state=384,
    n_text_head=6,
    n_text_layer=4,
)

BASE_CONFIG = ModelDimensions(
    n_mels=80,
    n_audio_ctx=1500,
    n_audio_state=512,
    n_audio_head=8,
    n_audio_layer=6,
    n_vocab=51865,
    n_text_ctx=448,
    n_text_state=512,
    n_text_head=8,
    n_text_layer=6,
)

SMALL_CONFIG = ModelDimensions(
    n_mels=80,
    n_audio_ctx=1500,
    n_audio_state=768,
    n_audio_head=12,
    n_audio_layer=12,
    n_vocab=51865,
    n_text_ctx=448,
    n_text_state=768,
    n_text_head=12,
    n_text_layer=12,
)

MEDIUM_CONFIG = ModelDimensions(
    n_mels=80,
    n_audio_ctx=1500,
    n_audio_state=1024,
    n_audio_head=16,
    n_audio_layer=24,
    n_vocab=51865,
    n_text_ctx=448,
    n_text_state=1024,
    n_text_head=16,
    n_text_layer=24,
)

LARGE_CONFIG = ModelDimensions(
    n_mels=80,
    n_audio_ctx=1500,
    n_audio_state=1280,
    n_audio_head=20,
    n_audio_layer=32,
    n_vocab=51865,
    n_text_ctx=448,
    n_text_state=1280,
    n_text_head=20,
    n_text_layer=32,
)

# Available model configurations
AVAILABLE_MODELS = {
    "tiny": TINY_CONFIG,
    "tiny.en": TINY_CONFIG,
    "base": BASE_CONFIG,
    "base.en": BASE_CONFIG,
    "small": SMALL_CONFIG,
    "small.en": SMALL_CONFIG,
    "medium": MEDIUM_CONFIG,
    "medium.en": MEDIUM_CONFIG,
    "large": LARGE_CONFIG,
    "large-v1": LARGE_CONFIG,
    "large-v2": LARGE_CONFIG,
    "large-v3": LARGE_CONFIG,
    "large-v3-turbo": LARGE_CONFIG,
    "turbo": LARGE_CONFIG,
}

def available_models():
    """Returns the names of available model configurations."""
    return list(AVAILABLE_MODELS.keys())

def get_model_config(name: str) -> ModelDimensions:
    """Get model configuration by name."""
    if name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {name}. Available models: {available_models()}")
    return AVAILABLE_MODELS[name]

# Special tokens
START_OF_TRANSCRIPT = 50258
ENGLISH_TOKEN = 50259
TRANSCRIBE_TOKEN = 50359
NO_TIMESTAMPS_TOKEN = 50363
END_OF_TEXT = 50257

__version__ = "1.0.0"
__all__ = [
    # Core classes
    "ModelDimensions",
    "Whisper", 
    "Tokenizer",
    
    # Functions
    "load_model",
    "create_model",
    "greedy_decode",
    "greedy_decode_jit",
    "get_tokenizer",
    "load_whisper_weights",
    "convert_hf_whisper_to_nnx",
    "available_models",
    "get_model_config",
    
    # Audio functions
    "load_audio",
    "log_mel_spectrogram", 
    "pad_or_trim",
    
    # Configurations
    "TINY_CONFIG",
    "BASE_CONFIG", 
    "SMALL_CONFIG",
    "MEDIUM_CONFIG",
    "LARGE_CONFIG",
    "AVAILABLE_MODELS",
    
    # Special tokens
    "START_OF_TRANSCRIPT",
    "ENGLISH_TOKEN",
    "TRANSCRIBE_TOKEN", 
    "NO_TIMESTAMPS_TOKEN",
    "END_OF_TEXT",
]