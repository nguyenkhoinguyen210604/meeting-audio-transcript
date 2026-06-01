"""
Shared dataclasses and base class for all ASR transcribers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class TranscribedSegment:
    start: float
    end: float
    text: str
    language: str = ""
    speaker: str = ""             # empty when diarization is not used
    words: list[dict] = field(default_factory=list)   # [{word, start, end}]

    def __repr__(self) -> str:
        tag = f" ({self.language})" if self.language else ""
        spk = f"{self.speaker}: " if self.speaker else ""
        return f"[{self.start:.2f}s–{self.end:.2f}s]{tag} {spk}{self.text}"


class BaseTranscriber(ABC):
    """Common interface for all language-specific ASR backends."""

    @abstractmethod
    def transcribe_chunk(
        self,
        audio_path: str,
        start: float,
        end: float,
        language: str | None = None,
    ) -> TranscribedSegment:
        """
        Transcribe a time slice of audio_path.

        Args:
            audio_path: full audio file (pipeline denoised WAV).
            start, end: boundaries in seconds.
            language:   BCP-47 hint; None means auto-detect inside the model.

        Returns:
            TranscribedSegment with text, start, end, language.
        """
