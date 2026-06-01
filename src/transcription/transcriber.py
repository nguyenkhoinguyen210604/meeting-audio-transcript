"""
Transcribe audio segments to text using PhoWhisper-large-ct2 + faster-whisper.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from faster_whisper import WhisperModel

from src.diarization.diarizer import Segment


@dataclass
class TranscribedSegment:
    speaker: str
    start: float
    end: float
    text: str
    words: list[dict] = field(default_factory=list)  # [{word, start, end}]

    def __repr__(self) -> str:
        return f"[{self.start:.2f}s – {self.end:.2f}s] {self.speaker}: {self.text}"


class Transcriber:
    MODEL_NAME = "vinai/PhoWhisper-large-ct2"

    def __init__(self, word_timestamps: bool = False):
        """
        Args:
            word_timestamps: include per-word timestamps in output.
        """
        self.word_timestamps = word_timestamps

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        print(f"Loading {self.MODEL_NAME} on {device} ({compute_type})...")

        self.model = WhisperModel(
            self.MODEL_NAME,
            device=device,
            compute_type=compute_type,
        )

        print("Transcriber ready.")

    def transcribe_file(self, audio_path: str) -> list[TranscribedSegment]:
        """
        Transcribe a full audio file without diarization.
        Useful for quick checks or single-speaker audio.
        """
        segments, info = self.model.transcribe(
            audio_path,
            language="vi",
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            word_timestamps=self.word_timestamps,
        )

        print(f"Detected language: {info.language} "
              f"(prob={info.language_probability:.2f})")

        results = []
        for seg in segments:
            words = []
            if self.word_timestamps and seg.words:
                words = [
                    {"word": w.word, "start": w.start, "end": w.end}
                    for w in seg.words
                ]
            results.append(TranscribedSegment(
                speaker="SPEAKER",
                start=round(seg.start, 3),
                end=round(seg.end, 3),
                text=seg.text.strip(),
                words=words,
            ))

        return results

    def transcribe_segments(
        self,
        audio_path: str,
        diarization: list[Segment],
    ) -> list[TranscribedSegment]:
        """
        Transcribe audio aligned with diarization segments.
        Each diarized segment is transcribed separately and tagged with
        its speaker label.

        Args:
            audio_path:   path to mono WAV file.
            diarization:  output from Diarizer.process().

        Returns:
            List of TranscribedSegment sorted by start time.
        """
        import torchaudio

        audio, sr = torchaudio.load(audio_path)
        results: list[TranscribedSegment] = []

        for i, seg in enumerate(diarization):
            start_sample = int(seg.start * sr)
            end_sample   = int(seg.end * sr)
            chunk = audio[:, start_sample:end_sample]

            if chunk.shape[1] == 0:
                continue

            # write chunk to a temp buffer faster-whisper can read
            import io
            import soundfile as sf
            buf = io.BytesIO()
            sf.write(buf, chunk.squeeze(0).numpy(), sr, format="WAV")
            buf.seek(0)

            whisper_segs, _ = self.model.transcribe(
                buf,
                language="vi",
                beam_size=5,
                word_timestamps=self.word_timestamps,
            )

            text = " ".join(s.text.strip() for s in whisper_segs)
            words = []

            if self.word_timestamps:
                for s in whisper_segs:
                    if s.words:
                        for w in s.words:
                            words.append({
                                "word": w.word,
                                # offset timestamps back to absolute position
                                "start": round(seg.start + w.start, 3),
                                "end":   round(seg.start + w.end,   3),
                            })

            if text:
                results.append(TranscribedSegment(
                    speaker=seg.speaker,
                    start=seg.start,
                    end=seg.end,
                    text=text,
                    words=words,
                ))

            print(f"  [{i+1}/{len(diarization)}] {seg.speaker} "
                  f"{seg.start:.1f}s–{seg.end:.1f}s: {text[:60]}")

        results.sort(key=lambda s: s.start)
        return results

    def save_txt(self, segments: list[TranscribedSegment], output_path: str) -> None:
        """Save transcript as plain text with speaker labels."""
        import os
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for seg in segments:
                f.write(f"[{seg.start:.2f}s – {seg.end:.2f}s] "
                        f"{seg.speaker}: {seg.text}\n")
        print(f"Transcript saved → {output_path}")
