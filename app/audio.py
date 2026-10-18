import io
from typing import Union, Tuple
import librosa
import numpy as np
import torch

from app.config import SAMPLE_RATE, MAX_SAMPLES


def preprocess_audio(
    audio_source: Union[str, bytes, io.BytesIO]
) -> Tuple[torch.Tensor, float]:
    """
    Load, resample, standardize audio to 16 kHz mono, 4 seconds (64,000 samples).
    
    Args:
        audio_source: File path (str) or raw audio bytes/BytesIO.
        
    Returns:
        tensor: Standardized PyTorch tensor of shape [1, 1, 64000]
        original_duration: Original duration in seconds before padding/truncating
    """
    # If raw bytes were passed, wrap in BytesIO
    if isinstance(audio_source, bytes):
        audio_source = io.BytesIO(audio_source)

    # librosa.load loads audio as mono by default (mono=True)
    # sr=None preserves original sampling rate to check if resampling is needed
    audio, sr = librosa.load(audio_source, sr=None, mono=True)
    
    original_duration = float(len(audio) / sr)

    # Resample to 16 kHz if necessary
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    # Pad or truncate to exactly MAX_SAMPLES (64,000 samples / 4.0 seconds)
    standardized = np.zeros(MAX_SAMPLES, dtype=np.float32)
    if len(audio) > MAX_SAMPLES:
        standardized[:] = audio[:MAX_SAMPLES]
    else:
        standardized[:len(audio)] = audio

    # Shape: [1, 1, 64000] -> [Batch=1, Channel=1, Samples=64000]
    tensor = torch.from_numpy(standardized).unsqueeze(0).unsqueeze(0)
    
    return tensor, original_duration
