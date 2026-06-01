"""
Speaker diarization — assign speaker labels and timestamps to audio segments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from pyannote.audio import Pipeline


@dataclass
class Segment:
    speaker: str
    start: float   # seconds
    end: float     # seconds

    @property
    def duration(self) -> float:
        return self.end - self.start

    def __repr__(self) -> str:
        return f"[{self.start:.2f}s – {self.end:.2f}s] {self.speaker}"


class Diarizer:
    # Requires accepting pyannote terms of use at:
    # https://hf.co/pyannote/speaker-diarization-3.1
    # https://hf.co/pyannote/segmentation-3.0
    MODEL_NAME = "pyannote/speaker-diarization-3.1"

    def __init__(self, hf_token: str, num_speakers: int | None = None):
        """
        Args:
            hf_token:     Hugging Face access token (read permission).
            num_speakers: Fix number of speakers if known; None = auto-detect.
        """
        self.num_speakers = num_speakers

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading pyannote diarization pipeline on {device}...")

        self.pipeline = Pipeline.from_pretrained(
            self.MODEL_NAME,
            use_auth_token=hf_token,
        )
        self.pipeline.to(device)
        print("Diarization pipeline ready.")

    def process(self, input_wav: str) -> list[Segment]:
        """
        Run speaker diarization on a WAV file.

        Args:
            input_wav: path to mono WAV file (any sample rate).

        Returns:
            List of Segment(speaker, start, end), sorted by start time.
        """
        print(f"Diarizing: {input_wav}")

        kwargs: dict = {}
        if self.num_speakers is not None:
            kwargs["num_speakers"] = self.num_speakers

        diarization = self.pipeline(input_wav, **kwargs)

        segments = [
            Segment(
                speaker=str(label),
                start=round(turn.start, 3),
                end=round(turn.end, 3),
            )
            for turn, _, label in diarization.itertracks(yield_label=True)
        ]

        segments.sort(key=lambda s: s.start)

        print(f"Found {len(set(s.speaker for s in segments))} speaker(s), "
              f"{len(segments)} segment(s).")

        return segments

    def save_rttm(self, segments: list[Segment], output_path: str) -> None:
        """Save diarization result as RTTM file (standard format)."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            for seg in segments:
                f.write(
                    f"SPEAKER file 1 {seg.start:.3f} {seg.duration:.3f} "
                    f"<NA> <NA> {seg.speaker} <NA> <NA>\n"
                )
        print(f"RTTM saved → {output_path}")
