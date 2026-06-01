"""
Shared audio utilities.
"""
from __future__ import annotations

import tempfile

import torch
import torchaudio


def extract_chunk(
    audio_path: str,
    start: float,
    end: float,
    target_sr: int = 16_000,
) -> str:
    """
    Extract a time slice from audio_path, resample to target_sr,
    and write to a temporary WAV file.

    Returns:
        Path to the temp WAV file (caller is responsible for deletion).
    """
    wav, sr = torchaudio.load(audio_path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    s = int(start * target_sr)
    e = int(end * target_sr)
    chunk = wav[:, s:e]

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    torchaudio.save(tmp.name, chunk, target_sr)
    tmp.close()
    return tmp.name
