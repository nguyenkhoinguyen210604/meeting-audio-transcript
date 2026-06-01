from datasets import load_dataset, Audio, concatenate_datasets
from pathlib import Path
from tqdm import tqdm
import subprocess
import json
import tempfile

# =========================
# Config
# =========================

DATASET_ID = "thivux/phoaudiobook"
SPLITS = ["validation", "test"]

AUDIO_COL = "audio"
TEXT_COL = "text"
SPEAKER_COL = "speaker"

OUTPUT_DIR = Path("./phoaudiobook_val_test_top500_longest_mp3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METADATA_PATH = OUTPUT_DIR / "metadata.json"

# Nếu chạy Kaggle, nên dùng Kaggle Secret tên HF_TOKEN
# from kaggle_secrets import UserSecretsClient
# HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")

HF_TOKEN = ""  # sửa token của bạn ở đây

# =========================
# Helpers
# =========================

def ffprobe_duration(path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def materialize_audio(audio_obj, tmp_dir, uid):
    audio_path = audio_obj.get("path")
    audio_bytes = audio_obj.get("bytes")

    if audio_path is not None and Path(audio_path).exists():
        return Path(audio_path)

    if audio_bytes is not None:
        tmp_path = Path(tmp_dir) / f"{uid}.audio"
        tmp_path.write_bytes(audio_bytes)
        return tmp_path

    raise RuntimeError(f"Cannot find audio data for uid={uid}")


def convert_to_mp3(src_path, out_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(src_path),

        # Chỉ encode sang MP3
        # Không -ac, không -ar, không filter audio
        "-codec:a", "libmp3lame",
        "-q:a", "2",

        str(out_path),
    ]
    subprocess.run(cmd, check=True)


# =========================
# Load validation + test
# =========================

print("Loading dataset splits...")

datasets = []

for split_name in SPLITS:
    print(f"Loading split: {split_name}")

    ds_split = load_dataset(
        DATASET_ID,
        split=split_name,
        token=HF_TOKEN,
    )

    ds_split = ds_split.cast_column(AUDIO_COL, Audio(decode=False))

    # thêm cột source_split để biết sample đến từ validation hay test
    ds_split = ds_split.add_column(
        "source_split",
        [split_name] * len(ds_split)
    )

    ds_split = ds_split.add_column(
        "source_index",
        list(range(len(ds_split)))
    )

    datasets.append(ds_split)

ds = concatenate_datasets(datasets)

print(ds)
print(ds.features)
print(ds.column_names)
print(f"Total validation + test samples: {len(ds)}")

if len(ds) < 500:
    raise ValueError(f"validation + test chỉ có {len(ds)} samples, không đủ 500.")

# =========================
# Compute durations
# =========================

print("Computing durations...")

records = []

with tempfile.TemporaryDirectory() as tmp_dir:
    for idx in tqdm(range(len(ds))):
        sample = ds[idx]

        audio_obj = sample[AUDIO_COL]
        src_path = materialize_audio(audio_obj, tmp_dir, uid=f"dur_{idx:06d}")

        duration = ffprobe_duration(src_path)

        records.append({
            "combined_idx": idx,
            "source_split": sample["source_split"],
            "source_index": int(sample["source_index"]),
            "duration_sec": float(duration),
        })

    # =========================
    # Top 500 longest
    # =========================

    top500 = sorted(
        records,
        key=lambda x: x["duration_sec"],
        reverse=True
    )[:500]

    print(f"Selected {len(top500)} longest samples")

    # =========================
    # Save MP3 + metadata JSON
    # =========================

    metadata = []

    for rank, item in enumerate(tqdm(top500)):
        sample = ds[int(item["combined_idx"])]

        audio_obj = sample[AUDIO_COL]
        src_path = materialize_audio(
            audio_obj,
            tmp_dir,
            uid=f"save_{rank:04d}_{item['combined_idx']:06d}"
        )

        mp3_path = OUTPUT_DIR / f"{rank:04d}.mp3"
        convert_to_mp3(src_path, mp3_path)

        metadata.append({
            "rank": rank,
            "file": mp3_path.name,
            "dataset": DATASET_ID,
            "source_split": item["source_split"],
            "source_index": item["source_index"],
            "combined_idx": int(item["combined_idx"]),
            "duration_sec": item["duration_sec"],
            "speaker": sample[SPEAKER_COL],
            "text": sample[TEXT_COL],
        })

# =========================
# Save metadata.json
# =========================

with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("Done.")
print("Saved audio to:", OUTPUT_DIR.resolve())
print("Saved metadata to:", METADATA_PATH.resolve())