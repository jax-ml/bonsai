import os
from functools import lru_cache
from subprocess import CalledProcessError, run
from typing import Optional, Union

import numpy as np
import jax.numpy as jnp
import librosa

from .utils import exact_div

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token


def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    # Use librosa to load audio
    try:
        audio, _ = librosa.load(file, sr=sr, mono=True)
        return audio.astype(np.float32)
    except Exception as e:
        # Fallback to ffmpeg if librosa fails
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads", "0",
            "-i", file,
            "-f", "s16le",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-ar", str(sr),
            "-"
        ]
        try:
            out = run(cmd, capture_output=True, check=True).stdout
            return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
        except CalledProcessError as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if isinstance(array, jnp.ndarray):
        if array.shape[axis] > length:
            array = jnp.take(array, jnp.arange(length), axis=axis)
        
        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = jnp.pad(array, pad_widths)
    else:
        # Handle numpy arrays
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(n_mels: int) -> np.ndarray:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Uses librosa to generate mel filters on the fly.
    """
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"
    
    # Generate mel filters using librosa
    mel_filters = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=n_mels)
    return mel_filters.astype(np.float32)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, jnp.ndarray],
    n_mels: int = 80,
    padding: int = 0,
):
    """
    Compute the log-Mel spectrogram using librosa

    Parameters
    ----------
    audio: Union[str, np.ndarray, jnp.ndarray], shape = (*)
        The path to audio or either a NumPy array or JAX array containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 and 128 are supported

    padding: int
        Number of zero samples to pad to the right

    Returns
    -------
    jnp.ndarray, shape = (n_mels, n_frames)
        A JAX array that contains the Mel spectrogram
    """
    if isinstance(audio, str):
        audio = load_audio(audio)
    
    # Convert to numpy if it's a JAX array
    if isinstance(audio, jnp.ndarray):
        audio = np.array(audio)
    
    # Ensure it's a numpy array
    audio = np.array(audio)
    
    if padding > 0:
        audio = np.pad(audio, (0, padding))
    
    # Compute mel spectrogram using librosa
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=n_mels,
        fmin=0,
        fmax=None,
        htk=False,
        norm='slaney',
        dtype=np.float32
    )
    
    # Convert to log scale (same as original implementation)
    log_spec = np.log10(np.clip(mel_spec, a_min=1e-10, a_max=None))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    
    # Drop last frame to match PyTorch (librosa produces 3001 frames, PyTorch produces 3000)
    log_spec = log_spec[:, :-1]
    
    # Convert to JAX array with correct dtype
    return jnp.array(log_spec, dtype=jnp.float32)