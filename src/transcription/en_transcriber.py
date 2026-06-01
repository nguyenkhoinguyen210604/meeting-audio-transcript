"""
English ASR via Parakeet-TDT-CTC-110M (sherpa-onnx, no NeMo required).
Model: csukuangfj/sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000
Uses from_nemo_ctc (single model.onnx, no encoder/decoder/joiner split).
"""
from __future__ import annotations

import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download

from src.transcription.base import BaseTranscriber, TranscribedSegment

MODEL_SR = 16_000
HF_REPO = "csukuangfj/sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000"
MODEL_FILES = {
    "model":  "model.onnx",
    "tokens": "tokens.txt",
}


class EnTranscriber(BaseTranscriber):
    """English Parakeet-TDT-CTC via sherpa-onnx (no NeMo dependency)."""

    def __init__(self):
        import sherpa_onnx

        print(f"Downloading Parakeet from {HF_REPO}...")
        paths = {
            key: hf_hub_download(HF_REPO, fname)
            for key, fname in MODEL_FILES.items()
        }

        self.recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=paths["model"],
            tokens=paths["tokens"],
            num_threads=4,
            provider="cpu",
        )
        print("EnTranscriber (Parakeet CTC) ready.")

    def transcribe_chunk(
        self,
        audio_path: str,
        start: float,
        end: float,
        language: str | None = "en",
    ) -> TranscribedSegment:
        import sherpa_onnx

        samples = self._load_samples(audio_path, start, end)
        if samples is None:
            return TranscribedSegment(start=start, end=end, text="", language="en")

        stream = self.recognizer.create_stream()
        stream.accept_waveform(MODEL_SR, samples)
        self.recognizer.decode_stream(stream)
        text = stream.result.text.strip()

        return TranscribedSegment(start=start, end=end, text=text, language="en")

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
