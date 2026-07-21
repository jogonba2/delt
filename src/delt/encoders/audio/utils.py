"""Module with utils for audio processing."""

from io import BytesIO

import numpy as np
import soundfile as sf
import torch
from torchaudio import functional as F

from delt.types import Audio


def decode_audio(audio: Audio, target_sampling_rate: int) -> np.ndarray:
    """
    Decode audio to get the waveform and sampling rate and resample if needed.

    Args:
        audio (Audio): audio input.
        target_sampling_rate (int): *target* sampling rate to resample.

    Returns:
        np.ndarray: a waveform as numpy array.

    """
    if isinstance(audio, bytes):
        waveform, sr = sf.read(BytesIO(audio), always_2d=True)
        waveform = torch.from_numpy(waveform.mean(axis=1).astype(np.float32))
    else:
        decoded = audio.get_all_samples()
        waveform = decoded.data.float().mean(dim=0)
        sr = decoded.sample_rate

    if sr != target_sampling_rate:
        waveform = F.resample(waveform, sr, target_sampling_rate)

    return waveform.numpy()
