"""
Fallback ASR via faster-whisper (Whisper large-v3).
Used for: mixed-language chunks, unsupported languages,
and low-confidence LID results.
Also exposes its WhisperModel for LID reuse.
"""
from __future__ import annotations

import io

import numpy as np
import soundfile as sf
import torch
import torchaudio
from faster_whisper import WhisperModel

from src.transcription.base import BaseTranscriber, TranscribedSegment

MODEL_NAME = "Systran/faster-whisper-large-v3"
MODEL_SR = 16_000


class FallbackTranscriber(BaseTranscriber):
    """Whisper large-v3 via faster-whisper. Language=None → auto-detect."""

    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute = "float16" if device == "cuda" else "int8"

        print(f"Loading Whisper fallback ({MODEL_NAME}) on {device} ({compute})...")
        self.model = WhisperModel(MODEL_NAME, device=device, compute_type=compute)
        print("FallbackTranscriber (Whisper) ready.")

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
            language=language,   # None → Whisper auto-detects per segment
            beam_size=5,
            vad_filter=False,    # VAD already done upstream
        )

        text = " ".join(s.text.strip() for s in segments)
        detected_lang = info.language or (language or "")

        return TranscribedSegment(
            start=start, end=end, text=text, language=detected_lang
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _load_buffer(
        self, audio_path: str, start: float, end: float
    ) -> io.BytesIO | None:
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
