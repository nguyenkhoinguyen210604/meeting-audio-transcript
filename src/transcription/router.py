"""
ASR Runner: dispatch to Whisper or Qwen3-ASR backend.
No LID — both backends are multilingual and auto-detect language.
"""
from __future__ import annotations

from src.transcription.base import BaseTranscriber, TranscribedSegment
from src.transcription.whisper_transcriber import WhisperTranscriber, WHISPER_MODELS
from src.transcription.qwen_transcriber import QwenTranscriber, QWEN_MODELS
from src.vad.segmenter import AudioChunk


# Expose model lists for UI selection
BACKEND_MODELS: dict[str, list[tuple[str, str]]] = {
    "whisper": WHISPER_MODELS,
    "qwen":    QWEN_MODELS,
}


class ASRRunner:
    """
    Runs ASR on each VAD chunk using the selected multilingual backend.

    Backends:
        whisper — faster-whisper (tiny/base/small/medium/large-v1/v2/v3)
        qwen    — Qwen3-ASR (0.6B / 1.7B)
    """

    def __init__(self, backend: str = "whisper", model: str = "large-v3"):
        backend = backend.lower()
        if backend == "whisper":
            self._transcriber: BaseTranscriber = WhisperTranscriber(model)
        elif backend == "qwen":
            self._transcriber = QwenTranscriber(model)
        else:
            raise ValueError(
                f"Unknown ASR backend: {backend!r}. Choose 'whisper' or 'qwen'."
            )

    def transcribe(
        self,
        audio_path: str,
        chunks: list[AudioChunk],
    ) -> list[TranscribedSegment]:
        results: list[TranscribedSegment] = []
        for i, chunk in enumerate(chunks):
            print(f"  [{i+1}/{len(chunks)}] {chunk.start:.1f}s–{chunk.end:.1f}s")
            seg = self._transcriber.transcribe_chunk(audio_path, chunk.start, chunk.end)
            if seg.text:
                results.append(seg)
        results.sort(key=lambda s: s.start)
        return results
