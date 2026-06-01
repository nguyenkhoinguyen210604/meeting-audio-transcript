from datasets import load_dataset, Audio
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import subprocess
import json
import tempfile

OUTPUT_DIR = Path("./tedlium3_top500_longest_mp3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def ffprobe_duration(path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(json.loads(result.stdout)["format"]["duration"])

def materialize_audio(audio_obj, tmp_dir, idx):
    audio_path = audio_obj.get("path")
    audio_bytes = audio_obj.get("bytes")

    if audio_path is not None and Path(audio_path).exists():
        return Path(audio_path)

    if audio_bytes is not None:
        tmp_path = Path(tmp_dir) / f"{idx:06d}.audio"
        tmp_path.write_bytes(audio_bytes)
        return tmp_path

    raise RuntimeError(f"Cannot find audio data at idx={idx}")

def convert_to_mp3(src_path, out_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(src_path),
        "-codec:a", "libmp3lame",
        "-q:a", "2",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)

print("Loading dataset...")

ds = load_dataset(
    "AudioLLMs/tedlium3_test",
    split="test"
)

ds = ds.cast_column("context", Audio(decode=False))

print(ds)
print(ds.features)
print(ds.column_names)
print(f"Total samples: {len(ds)}")

print("Computing durations...")

records = []

with tempfile.TemporaryDirectory() as tmp_dir:
    for idx in tqdm(range(len(ds))):
        sample = ds[idx]

        audio_obj = sample["context"]
        src_path = materialize_audio(audio_obj, tmp_dir, idx)

        duration = ffprobe_duration(src_path)

        records.append({
            "idx": idx,
            "duration": duration,
        })

    duration_df = pd.DataFrame(records)

    top500 = (
        duration_df
        .sort_values("duration", ascending=False)
        .head(500)
        .reset_index(drop=True)
    )

    print(f"Selected {len(top500)} longest samples")

    metadata = []

    for rank, row in enumerate(
        tqdm(top500.itertuples(index=False), total=len(top500))
    ):
        sample = ds[int(row.idx)]

        audio_obj = sample["context"]
        src_path = materialize_audio(audio_obj, tmp_dir, int(row.idx))

        mp3_path = OUTPUT_DIR / f"{rank:04d}.mp3"
        convert_to_mp3(src_path, mp3_path)

        metadata.append({
            "rank": rank,
            "dataset_idx": int(row.idx),
            "file": mp3_path.name,
            "duration_sec": float(row.duration),
            "instruction": sample["instruction"],
            "transcript": sample["answer"],
        })

pd.DataFrame(metadata).to_csv(
    OUTPUT_DIR / "metadata.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Done.")
print("Saved to:", OUTPUT_DIR.resolve())