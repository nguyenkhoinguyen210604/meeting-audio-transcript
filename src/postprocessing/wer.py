"""
WER (Word Error Rate) computation — load metadata, match ground truth,
compute per-file and aggregate WER stats.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import jiwer
import numpy as np


def load_metadata(path: str) -> list[dict]:
    """Load metadata from JSON or CSV. Returns list of dicts."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with open(path, encoding="utf-8") as f:
            return list(csv.DictReader(f))
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _filename_stems(entry: dict) -> list[str]:
    """Extract candidate filename stems from a metadata entry."""
    stems = []
    for key in ("file", "audio", "filename", "file_path", "audio_path"):
        val = entry.get(key)
        if val:
            stems.append(Path(str(val)).stem)
    return stems


def _detect_text_field(entry: dict) -> str:
    """Auto-detect the ground-truth text field name."""
    for key in ("text", "transcript", "answer", "sentence", "ground_truth"):
        if key in entry:
            return key
    for key in entry:
        if "text" in key.lower() or "transcript" in key.lower():
            return key
    raise ValueError(f"Cannot detect ground-truth field in keys: {list(entry.keys())}")


def build_ground_truth_map(metadata: list[dict]) -> dict[str, str]:
    """
    Build mapping from filename-stem → ground-truth text.
    Auto-detects file-key and text-key fields.
    """
    if not metadata:
        return {}
    text_field = _detect_text_field(metadata[0])
    mapping: dict[str, str] = {}
    for entry in metadata:
        stems = _filename_stems(entry)
        gt = (entry.get(text_field) or "").strip()
        for stem in stems:
            if stem:
                mapping[stem] = gt
    return mapping


def _match_ref(
    basename: str,
    gt_map: dict[str, str],
) -> str | None:
    """Try to find ground truth for an audio basename. Returns text or None."""
    if basename in gt_map:
        return gt_map[basename]
    # Try stripping leading zeros
    stripped = basename.lstrip("0")
    for stem, gt in gt_map.items():
        if stem.lstrip("0") == stripped:
            return gt
    # Try substring match
    for stem, gt in gt_map.items():
        if stem in basename or basename in stem:
            return gt
    return None


def compute_wer_score(reference: str, hypothesis: str) -> float:
    """Compute WER (0.0–1.0). Both strings are lowercased before comparison."""
    if not reference.strip():
        return 0.0
    return float(jiwer.wer(reference.lower(), hypothesis.lower()))


def compute_wer_stats(
    audio_paths: list[str],
    transcriptions: dict[str, str],
    ground_truth_map: dict[str, str],
) -> dict:
    """
    Compute per-file WER and aggregate statistics.

    Args:
        audio_paths: list of paths to audio files (basename used for matching).
        transcriptions: dict mapping basename → ASR transcript text.
        ground_truth_map: dict mapping basename → ground-truth text.

    Returns:
        {"per_file": [(basename, wer, ref_words, hyp_words), ...],
         "mean": float, "std": float, "min": float, "max": float, "count": int}
    """
    per_file: list[dict] = []
    scores: list[float] = []

    for ap in audio_paths:
        basename = Path(ap).stem
        hyp = transcriptions.get(basename, "")
        ref = _match_ref(basename, ground_truth_map)

        if ref is not None:
            wer_val = compute_wer_score(ref, hyp)
            scores.append(wer_val)
            per_file.append({
                "name": basename,
                "wer": wer_val,
                "ref_words": len(ref.split()),
                "hyp_words": len(hyp.split()),
            })
        else:
            per_file.append({
                "name": basename,
                "wer": None,
                "ref_words": 0,
                "hyp_words": len(hyp.split()) if hyp else 0,
            })

    if scores:
        arr = np.array(scores)
        return {
            "per_file": per_file,
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "count": len(scores),
        }
    return {
        "per_file": per_file,
        "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0,
    }
