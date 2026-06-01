"""
Language identification using faster-whisper's built-in detector.
Reuses the fallback Whisper model — no extra model to load.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torchaudio


SUPPORTED_LANGS = {"vi", "en", "zh"}
CONFIDENCE_THRESHOLD = 0.75
LID_WINDOW_SEC = 5.0   # use at most first N seconds for LID
LID_SR = 16_000        # faster-whisper expects 16 kHz float32


@dataclass
class LIDResult:
    language: str           # BCP-47 code detected by Whisper
    confidence: float       # probability of top language
    is_mixed: bool          # True → route to Whisper fallback with language=None


class LangDetector:
    """
    Thin wrapper around faster-whisper's detect_language.
    Call detect() per chunk to get language + confidence.
    """

    def __init__(self, whisper_model):
        """
        Args:
            whisper_model: a loaded faster_whisper.WhisperModel instance
                           (shared with FallbackTranscriber to avoid double load).
        """
        self.model = whisper_model

    def detect(self, audio_path: str, start: float, end: float) -> LIDResult:
        """
        Detect the dominant language of an audio chunk.

        Uses at most LID_WINDOW_SEC seconds from the start of the chunk
        so detection is fast even on long segments.

        Args:
            audio_path: path to the full audio file.
            start, end: chunk boundaries in seconds.

        Returns:
            LIDResult with language code, confidence, and mixed flag.
        """
        audio = self._load_window(audio_path, start, end)
        lang, prob, _ = self.model.detect_language(audio)

        is_mixed = (prob < CONFIDENCE_THRESHOLD) or (lang not in SUPPORTED_LANGS)
        return LIDResult(language=lang, confidence=prob, is_mixed=is_mixed)

    # ── internals ─────────────────────────────────────────────────────────────

    def _load_window(
        self, audio_path: str, start: float, end: float
    ) -> np.ndarray:
        """Load a short window as 16 kHz float32 numpy array for Whisper."""
        wav, sr = torchaudio.load(audio_path)

        if sr != LID_SR:
            wav = torchaudio.functional.resample(wav, sr, LID_SR)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        s = int(start * LID_SR)
        # cap at LID_WINDOW_SEC to keep detection fast
        window_end = min(start + LID_WINDOW_SEC, end)
        e = int(window_end * LID_SR)

        chunk = wav[0, s:e].numpy().astype(np.float32)
        return chunk
