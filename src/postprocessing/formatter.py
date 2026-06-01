"""
Format TranscribedSegment list into readable transcript outputs.
"""

from __future__ import annotations

import json
import os

from src.transcription.base import TranscribedSegment


def _srt_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


class Formatter:
    def __init__(self, segments: list[TranscribedSegment]):
        self.segments = sorted(segments, key=lambda s: s.start)

    def to_text(self) -> str:
        """Plain text with speaker labels and timestamps."""
        lines = []
        for seg in self.segments:
            lines.append(
                f"[{seg.start:.2f}s – {seg.end:.2f}s] {seg.speaker}: {seg.text}"
            )
        return "\n".join(lines)

    def to_plain_transcript(self) -> str:
        """
        Merged transcript grouping consecutive turns by the same speaker.
        Best format to pass to the summarizer.
        """
        if not self.segments:
            return ""

        lines = []
        current_speaker = self.segments[0].speaker
        current_texts = [self.segments[0].text]

        for seg in self.segments[1:]:
            if seg.speaker == current_speaker:
                current_texts.append(seg.text)
            else:
                lines.append(f"{current_speaker}: {' '.join(current_texts)}")
                current_speaker = seg.speaker
                current_texts = [seg.text]

        lines.append(f"{current_speaker}: {' '.join(current_texts)}")
        return "\n".join(lines)

    def to_srt(self) -> str:
        """SRT subtitle format."""
        blocks = []
        for i, seg in enumerate(self.segments, start=1):
            start = _srt_timestamp(seg.start)
            end   = _srt_timestamp(seg.end)
            blocks.append(f"{i}\n{start} --> {end}\n{seg.speaker}: {seg.text}")
        return "\n\n".join(blocks)

    def to_json(self) -> str:
        """JSON array of all segments."""
        data = [
            {
                "speaker": seg.speaker,
                "start":   seg.start,
                "end":     seg.end,
                "text":    seg.text,
                "words":   seg.words,
            }
            for seg in self.segments
        ]
        return json.dumps(data, ensure_ascii=False, indent=2)

    # ── save helpers ─────────────────────────────────────────────────────────

    def save(self, output_path: str, fmt: str = "txt") -> str:
        """
        Write transcript to file.

        Args:
            output_path: destination file path (extension can override fmt).
            fmt:         "txt" | "plain" | "srt" | "json"

        Returns:
            Absolute path of the written file.
        """
        ext = os.path.splitext(output_path)[1].lstrip(".")
        fmt = ext if ext in {"txt", "srt", "json"} else fmt

        renderers = {
            "txt":   self.to_text,
            "plain": self.to_plain_transcript,
            "srt":   self.to_srt,
            "json":  self.to_json,
        }

        if fmt not in renderers:
            raise ValueError(f"Unsupported format '{fmt}'. Choose: {list(renderers)}")

        content = renderers[fmt]()

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Transcript saved → {output_path}")
        return os.path.abspath(output_path)
