"""
ASR Router: LID → dispatch chunk to the correct language-specific model.
All models are lazy-loaded on first use (only models actually needed get loaded).
"""
from __future__ import annotations

from src.lid.detector import LangDetector, LIDResult
from src.transcription.base import BaseTranscriber, TranscribedSegment
from src.transcription.fallback_transcriber import FallbackTranscriber
from src.vad.segmenter import AudioChunk


class ASRRouter:
    """
    Runs LID on each chunk and dispatches to the appropriate ASR backend:

        vi  + high confidence  → ViTranscriber   (Zipformer, sherpa-onnx)
        en  + high confidence  → EnTranscriber   (Parakeet, NeMo)
        zh  + high confidence  → ZhTranscriber   (SenseVoice, FunASR)
        *   + low confidence   → FallbackTranscriber (Whisper large-v3, language=None)
        other language         → FallbackTranscriber (Whisper large-v3, language hint)
    """

    def __init__(self):
        self._fallback: FallbackTranscriber | None = None
        self._lid: LangDetector | None = None
        self._specialized: dict[str, BaseTranscriber] = {}

    # ── public ────────────────────────────────────────────────────────────────

    def transcribe(
        self,
        audio_path: str,
        chunks: list[AudioChunk],
    ) -> list[TranscribedSegment]:
        """
        Transcribe all chunks, routing each to the right model.

        Args:
            audio_path: path to the full (denoised) audio file.
            chunks:     list of AudioChunk from VADSegmenter.

        Returns:
            List of TranscribedSegment sorted by start time.
        """
        results: list[TranscribedSegment] = []

        for i, chunk in enumerate(chunks):
            lid = self._get_lid().detect(audio_path, chunk.start, chunk.end)

            print(
                f"  [{i+1}/{len(chunks)}] "
                f"{chunk.start:.1f}s–{chunk.end:.1f}s | "
                f"lang={lid.language} conf={lid.confidence:.2f}"
                + (" [mixed→fallback]" if lid.is_mixed else "")
            )

            seg = self._dispatch(audio_path, chunk, lid)
            if seg.text:
                results.append(seg)

        results.sort(key=lambda s: s.start)
        return results

    # ── internals ─────────────────────────────────────────────────────────────

    def _dispatch(
        self,
        audio_path: str,
        chunk: AudioChunk,
        lid: LIDResult,
    ) -> TranscribedSegment:
        if lid.is_mixed:
            # Low confidence or unsupported: Whisper with language=None
            return self._get_fallback().transcribe_chunk(
                audio_path, chunk.start, chunk.end, language=None
            )

        if lid.language == "vi":
            return self._get_specialized("vi").transcribe_chunk(
                audio_path, chunk.start, chunk.end, language="vi"
            )
        if lid.language == "en":
            return self._get_specialized("en").transcribe_chunk(
                audio_path, chunk.start, chunk.end, language="en"
            )
        if lid.language == "zh":
            return self._get_specialized("zh").transcribe_chunk(
                audio_path, chunk.start, chunk.end, language="zh"
            )

        # Supported by LID but not in our specialized set → Whisper with hint
        return self._get_fallback().transcribe_chunk(
            audio_path, chunk.start, chunk.end, language=lid.language
        )

    # ── lazy loaders ──────────────────────────────────────────────────────────

    def _get_fallback(self) -> FallbackTranscriber:
        if self._fallback is None:
            self._fallback = FallbackTranscriber()
        return self._fallback

    def _get_lid(self) -> LangDetector:
        if self._lid is None:
            self._lid = LangDetector(self._get_fallback().model)
        return self._lid

    def _get_specialized(self, lang: str) -> BaseTranscriber:
        if lang not in self._specialized:
            if lang == "vi":
                from src.transcription.vi_transcriber import ViTranscriber
                self._specialized["vi"] = ViTranscriber()
            elif lang == "en":
                from src.transcription.en_transcriber import EnTranscriber
                self._specialized["en"] = EnTranscriber()
            elif lang == "zh":
                from src.transcription.zh_transcriber import ZhTranscriber
                self._specialized["zh"] = ZhTranscriber()
        return self._specialized[lang]
