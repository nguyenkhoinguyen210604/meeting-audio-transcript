"""
Chinese ASR via FunASR SenseVoice.
Model: iic/SenseVoiceSmall
"""
from __future__ import annotations

import os
import tempfile

import torch
import torchaudio

from src.transcription.base import BaseTranscriber, TranscribedSegment

MODEL_SR = 16_000
SENSEVOICE_MODEL = "iic/SenseVoiceSmall"


class ZhTranscriber(BaseTranscriber):
    """Chinese ASR via FunASR SenseVoice."""

    def __init__(self):
        from funasr import AutoModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SenseVoice ({SENSEVOICE_MODEL}) on {device}...")

        self.model = AutoModel(
            model=SENSEVOICE_MODEL,
            trust_remote_code=True,
            device=device,
        )
        print("ZhTranscriber (SenseVoice) ready.")

    def transcribe_chunk(
        self,
        audio_path: str,
        start: float,
        end: float,
        language: str | None = "zh",
    ) -> TranscribedSegment:
        tmp_path = self._extract_chunk(audio_path, start, end)
        try:
            res = self.model.generate(
                input=tmp_path,
                language=language or "auto",
                use_itn=True,
            )
            # FunASR returns a list of dicts: [{"text": "..."}]
            text = res[0]["text"].strip() if res else ""
        finally:
            os.unlink(tmp_path)

        return TranscribedSegment(
            start=start, end=end, text=text, language=language or "zh"
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _extract_chunk(
        self, audio_path: str, start: float, end: float
    ) -> str:
        wav, sr = torchaudio.load(audio_path)
        if sr != MODEL_SR:
            wav = torchaudio.functional.resample(wav, sr, MODEL_SR)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        s = int(start * MODEL_SR)
        e = int(end * MODEL_SR)
        chunk = wav[:, s:e]

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        torchaudio.save(tmp.name, chunk, MODEL_SR)
        tmp.close()
        return tmp.name
