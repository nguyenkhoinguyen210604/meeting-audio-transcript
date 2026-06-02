"""
Qwen3-ASR via transformers.
Models: Qwen/Qwen3-ASR-0.6B, Qwen/Qwen3-ASR-1.7B
"""
from __future__ import annotations

import numpy as np
import torch
import torchaudio

from src.transcription.base import BaseTranscriber, TranscribedSegment

MODEL_SR = 16_000

QWEN_MODELS: list[tuple[str, str]] = [
    ("Qwen3-ASR 0.6B", "Qwen/Qwen3-ASR-0.6B"),
    ("Qwen3-ASR 1.7B", "Qwen/Qwen3-ASR-1.7B"),
]


class QwenTranscriber(BaseTranscriber):
    """Qwen3-ASR multilingual speech-to-text via transformers."""

    def __init__(self, model_id: str = "Qwen/Qwen3-ASR-0.6B"):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        print(f"Loading {model_id} on {device}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto",
        )
        self.dtype = dtype
        print(f"QwenTranscriber ({model_id}) ready.")

    def transcribe_chunk(
        self,
        audio_path: str,
        start: float,
        end: float,
        language: str | None = None,
    ) -> TranscribedSegment:
        audio_np = self._load(audio_path, start, end)
        if audio_np is None:
            return TranscribedSegment(start=start, end=end, text="")

        inputs = self.processor(
            audio_np, sampling_rate=MODEL_SR,
            return_tensors="pt", return_attention_mask=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        if self.dtype == torch.float16 and "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].half()

        with torch.no_grad():
            ids = self.model.generate(**inputs, language=None, task="transcribe")
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        return TranscribedSegment(start=start, end=end, text=text)

    def _load(self, audio_path: str, start: float, end: float) -> np.ndarray | None:
        wav, sr = torchaudio.load(audio_path)
        if sr != MODEL_SR:
            wav = torchaudio.functional.resample(wav, sr, MODEL_SR)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        chunk = wav[0, int(start * MODEL_SR):int(end * MODEL_SR)]
        return chunk.numpy().astype(np.float32) if chunk.numel() else None
