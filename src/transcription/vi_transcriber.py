"""
Vietnamese ASR via Zipformer (sherpa-onnx).
Model: k2-fsa/sherpa-onnx-zipformer-vi-2024-06-11 (HuggingFace)
"""
from __future__ import annotations

import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download

from src.transcription.base import BaseTranscriber, TranscribedSegment

MODEL_SR = 16_000
HF_REPO = "csukuangfj/sherpa-onnx-zipformer-vi-int8-2025-04-20"
MODEL_FILES = {
    "encoder": "encoder-epoch-12-avg-8.int8.onnx",
    "decoder": "decoder-epoch-12-avg-8.onnx",
    "joiner":  "joiner-epoch-12-avg-8.int8.onnx",
    "tokens":  "tokens.txt",
}


class ViTranscriber(BaseTranscriber):
    """Vietnamese Zipformer transducer via sherpa-onnx."""

    def __init__(self):
        import sherpa_onnx

        print(f"Downloading Vietnamese Zipformer from {HF_REPO}...")
        paths = {
            key: hf_hub_download(HF_REPO, fname)
            for key, fname in MODEL_FILES.items()
        }

        provider = "cuda" if torch.cuda.is_available() else "cpu"

        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=paths["encoder"],
            decoder=paths["decoder"],
            joiner=paths["joiner"],
            tokens=paths["tokens"],
            num_threads=4,
            provider=provider,
        )
        print("ViTranscriber (Zipformer) ready.")

    def transcribe_chunk(
        self,
        audio_path: str,
        start: float,
        end: float,
        language: str | None = "vi",
    ) -> TranscribedSegment:
        import sherpa_onnx

        samples = self._load_samples(audio_path, start, end)
        if samples is None:
            return TranscribedSegment(start=start, end=end, text="", language="vi")

        stream = self.recognizer.create_stream()
        stream.accept_waveform(MODEL_SR, samples)
        self.recognizer.decode_stream(stream)
        text = stream.result.text.strip()

        return TranscribedSegment(start=start, end=end, text=text, language="vi")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _load_samples(
        self, audio_path: str, start: float, end: float
    ) -> np.ndarray | None:
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

        return chunk.numpy().astype(np.float32)
