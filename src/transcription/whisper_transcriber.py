"""
Whisper ASR via faster-whisper. Supports all model sizes.
"""
from __future__ import annotations

import io

import numpy as np
import soundfile as sf
import torch
import torchaudio
from faster_whisper import WhisperModel

from src.transcription.base import BaseTranscriber, TranscribedSegment

MODEL_SR = 16_000

# (display_label, model_id)
WHISPER_MODELS: list[tuple[str, str]] = [
    ("tiny (39M)",          "tiny"),
    ("base (74M)",          "base"),
    ("small (244M)",        "small"),
    ("medium (769M)",       "medium"),
    ("large-v2 (1.5B)",     "large-v2"),
    ("large-v3 (1.5B)",     "large-v3"),
    ("large-v3-turbo (809M)", "large-v3-turbo"),
]


class WhisperTranscriber(BaseTranscriber):
    """Multilingual Whisper ASR; model_name selects the variant."""

    def __init__(self, model_name: str = "large-v3"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute = "float16" if device == "cuda" else "int8"

        print(f"Loading Whisper '{model_name}' on {device} ({compute})...")
        self.model = WhisperModel(model_name, device=device, compute_type=compute)
        print(f"WhisperTranscriber ({model_name}) ready.")

    def transcribe_chunk(
        self,
        audio_path: str,
        start: float,
        end: float,
        language: str | None = None,
    ) -> TranscribedSegment:
        buf = self._load_buffer(audio_path, start, end)
        if buf is None:
            return TranscribedSegment(start=start, end=end, text="", language="")

        segments, info = self.model.transcribe(
            buf,
            language=language,
            beam_size=5,
            vad_filter=False,
        )

        text = " ".join(s.text.strip() for s in segments)
        detected_lang = info.language or (language or "")
        return TranscribedSegment(start=start, end=end, text=text, language=detected_lang)

    def _load_buffer(self, audio_path: str, start: float, end: float) -> io.BytesIO | None:
        wav, sr = torchaudio.load(audio_path)
        if sr != MODEL_SR:
            wav = torchaudio.functional.resample(wav, sr, MODEL_SR)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        s = int(start * MODEL_SR)
        e = int(end * MODEL_SR)
        chunk = wav[0, s:e]
        if chunk.numel() == 0:
            return None

        buf = io.BytesIO()
        sf.write(buf, chunk.numpy().astype(np.float32), MODEL_SR, format="WAV")
        buf.seek(0)
        return buf
