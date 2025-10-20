from .audio import load_audio, log_mel_spectrogram, pad_or_trim
from .modeling import WhisperConfig, Whisper, load_model
from .params import convert_hf_whisper_to_nnx

# Backward compatibility
ModelDimensions = WhisperConfig