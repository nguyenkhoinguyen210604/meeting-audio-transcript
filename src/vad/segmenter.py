"""
VAD-based audio segmenter: detect speech regions, merge into ASR-ready chunks.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torchaudio


@dataclass
class AudioChunk:
    start: float   # seconds
    end: float     # seconds

    @property
    def duration(self) -> float:
        return self.end - self.start

    def __repr__(self) -> str:
        return f"AudioChunk({self.start:.2f}s–{self.end:.2f}s, {self.duration:.2f}s)"


class VADSegmenter:
    """
    Wraps Silero VAD to produce silence-aware ASR chunks.

    VAD finds raw speech segments (removes silence).
    The merge step groups adjacent segments into longer chunks so ASR models
    receive enough acoustic context to avoid hallucination.
    """

    TARGET_SR = 16_000  # Silero VAD requires 16 kHz

    def __init__(
        self,
        min_speech_ms: int = 300,
        min_silence_ms: int = 500,
        max_chunk_duration: float = 25.0,
        max_gap: float = 1.5,
        min_chunk_duration: float = 2.0,
    ):
        """
        Args:
            min_speech_ms:      minimum speech segment duration to keep.
            min_silence_ms:     minimum silence duration to split segments.
            max_chunk_duration: hard cap on merged chunk length (seconds).
            max_gap:            maximum silence gap (seconds) within which
                                adjacent segments are merged into one chunk.
        """
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.max_chunk_duration = max_chunk_duration
        self.max_gap = max_gap
        self.min_chunk_duration = min_chunk_duration

        print("Loading Silero VAD...")
        self.model, self._utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self._get_ts = self._utils[0]   # get_speech_timestamps
        self._read_audio = self._utils[2]  # read_audio
        print("Silero VAD ready.")

    # ── public ────────────────────────────────────────────────────────────────

    def get_chunks(self, audio_path: str) -> list[AudioChunk]:
        """
        Full pipeline: VAD on audio file → merge into ASR-ready chunks.

        Returns:
            List of AudioChunk sorted by start time.
        """
        raw_segments = self._run_vad(audio_path)
        chunks = self._merge(raw_segments)
        print(
            f"VAD: {len(raw_segments)} speech segment(s) "
            f"→ {len(chunks)} ASR chunk(s)."
        )
        return chunks

    # ── internals ─────────────────────────────────────────────────────────────

    def _run_vad(self, audio_path: str) -> list[AudioChunk]:
        wav, sr = torchaudio.load(audio_path)
        if sr != self.TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, self.TARGET_SR)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0)

        timestamps = self._get_ts(
            wav,
            self.model,
            sampling_rate=self.TARGET_SR,
            min_speech_duration_ms=self.min_speech_ms,
            min_silence_duration_ms=self.min_silence_ms,
            return_seconds=True,
        )
        return [AudioChunk(start=t["start"], end=t["end"]) for t in timestamps]

    def _merge(self, segments: list[AudioChunk]) -> list[AudioChunk]:
        """
        Merge adjacent segments into longer chunks, respecting two constraints:
        - gap between consecutive segments ≤ max_gap
        - resulting chunk duration ≤ max_chunk_duration

        A second pass absorbs any chunk shorter than min_chunk_duration into
        its nearest neighbour to avoid feeding micro-segments to ASR.
        """
        if not segments:
            return []

        chunks: list[AudioChunk] = []
        c_start = segments[0].start
        c_end = segments[0].end

        for seg in segments[1:]:
            gap = seg.start - c_end
            would_be_duration = seg.end - c_start

            if gap <= self.max_gap and would_be_duration <= self.max_chunk_duration:
                c_end = seg.end
            else:
                chunks.append(AudioChunk(start=c_start, end=c_end))
                c_start = seg.start
                c_end = seg.end

        chunks.append(AudioChunk(start=c_start, end=c_end))

        # Second pass: absorb chunks shorter than min_chunk_duration
        # into whichever neighbour has the smaller gap.
        merged: list[AudioChunk] = []
        i = 0
        while i < len(chunks):
            chunk = chunks[i]
            if chunk.duration < self.min_chunk_duration:
                gap_prev = (chunk.start - merged[-1].end) if merged else float("inf")
                gap_next = (chunks[i + 1].start - chunk.end) if i + 1 < len(chunks) else float("inf")

                if gap_prev <= gap_next and merged:
                    # absorb into previous chunk
                    merged[-1] = AudioChunk(start=merged[-1].start, end=chunk.end)
                elif i + 1 < len(chunks):
                    # absorb into next chunk
                    chunks[i + 1] = AudioChunk(start=chunk.start, end=chunks[i + 1].end)
                # else: first and only chunk — keep as-is
                else:
                    merged.append(chunk)
            else:
                merged.append(chunk)
            i += 1

        return merged
