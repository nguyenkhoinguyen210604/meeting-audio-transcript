"""
Streamlit UI for the Meeting Record Transcript pipeline.
Upload → preprocess (normalize + denoise + VAD) → pick ASR model → transcribe → WER.
Switching models only re-runs ASR, not the full pipeline.
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Meeting Transcript Pipeline", page_icon="🎙️", layout="wide")
st.title("🎙️ Meeting Transcript Pipeline")
st.caption("Normalize → Denoise → VAD → ASR → WER")

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".opus", ".wma"}


# ── Cached resources ─────────────────────────────────────────────────────
@st.cache_resource
def get_pipeline():
    from src.pipeline import AudioPipeline
    return AudioPipeline(chunk_seconds=30, max_chunk_duration=25.0, max_gap=1.5)


@st.cache_data
def load_metadata_cached(path: str) -> dict[str, str]:
    from src.postprocessing.wer import load_metadata, build_ground_truth_map
    return build_ground_truth_map(load_metadata(path))


def run_wer(audio_paths: list[str], transcriptions: dict[str, str], gt_map: dict[str, str]) -> dict:
    from src.postprocessing.wer import compute_wer_stats
    return compute_wer_stats(audio_paths, transcriptions, gt_map)


# ── Session state ─────────────────────────────────────────────────────────
if "preprocessed" not in st.session_state:
    st.session_state.preprocessed = {}
if "asr_results" not in st.session_state:
    st.session_state.asr_results = {}
if "gt_map" not in st.session_state:
    st.session_state.gt_map = {}
if "file_labels" not in st.session_state:
    st.session_state.file_labels = {}


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ASR Model")
    asr_backend = st.selectbox(
        "Backend",
        options=["whisper", "qwen"],
        format_func=lambda x: "Whisper (faster-whisper)" if x == "whisper" else "Qwen3-ASR (transformers)",
    )

    if asr_backend == "whisper":
        _w = [
            ("tiny (39M)", "tiny"), ("base (74M)", "base"),
            ("small (244M)", "small"), ("medium (769M)", "medium"),
            ("large-v1 (1.5B)", "large-v1"), ("large-v2 (1.5B)", "large-v2"),
            ("large-v3 (1.5B)", "large-v3"),
        ]
        asr_model = st.selectbox(
            "Kích cỡ",
            options=[m for _, m in _w],
            format_func=lambda m: next(l for l, mid in _w if mid == m),
            index=6,
        )
    else:
        _q = [
            ("Qwen3-ASR 0.6B", "Qwen/Qwen3-ASR-0.6B"),
            ("Qwen3-ASR 1.7B", "Qwen/Qwen3-ASR-1.7B"),
        ]
        asr_model = st.selectbox(
            "Phiên bản",
            options=[m for _, m in _q],
            format_func=lambda m: next(l for l, mid in _q if mid == m),
        )

    st.divider()
    st.header("WER Evaluation")
    meta_mode = st.radio(
        "Metadata",
        ["Không dùng", "Tải file lên", "Đường dẫn file"],
        horizontal=False,
    )

    if meta_mode == "Tải file lên":
        meta_upload = st.file_uploader(
            "Chọn metadata (.json / .csv)",
            type=["json", "csv"],
            key="meta_uploader",
        )
        if meta_upload:
            _md = os.path.join(tempfile.gettempdir(), "mrt_metadata")
            os.makedirs(_md, exist_ok=True)
            _mp = os.path.join(_md, meta_upload.name)
            with open(_mp, "wb") as f:
                f.write(meta_upload.getvalue())
            try:
                st.session_state.gt_map = load_metadata_cached(_mp)
                st.success(f"Đã load {len(st.session_state.gt_map)} ground-truth mẫu.")
            except Exception as e:
                st.error(f"Lỗi load metadata: {e}")

    elif meta_mode == "Đường dẫn file":
        meta_path = st.text_input("Đường dẫn file metadata", placeholder="/path/to/metadata.json")
        if meta_path and os.path.isfile(meta_path):
            try:
                st.session_state.gt_map = load_metadata_cached(meta_path)
                st.success(f"Đã load {len(st.session_state.gt_map)} ground-truth mẫu.")
            except Exception as e:
                st.error(f"Lỗi load metadata: {e}")


# ── Helpers ───────────────────────────────────────────────────────────────

def _do_preprocess(pipeline, file_path: str, output_dir: str) -> dict | None:
    try:
        return pipeline.preprocess(file_path, output_dir)
    except Exception as e:
        st.error(f"Tiền xử lý lỗi: {os.path.basename(file_path)} — {e}")
        return None


def _do_transcribe(pipeline, pre: dict, output_dir: str, backend: str, model: str) -> dict | None:
    try:
        return pipeline.transcribe(
            denoised_wav=pre["denoised_wav"],
            chunks=pre["chunks"],
            output_dir=output_dir,
            asr_backend=backend,
            asr_model=model,
            basename=pre["basename"],
        )
    except Exception as e:
        st.error(f"ASR lỗi: {pre['basename']} — {e}")
        return None


# ── Main UI ───────────────────────────────────────────────────────────────

input_mode = st.radio(
    "Nguồn dữ liệu:",
    ["Tải file lên", "Đường dẫn thư mục cục bộ"],
    horizontal=True,
)

files_to_preprocess: list[str] = []
new_labels: dict[str, str] = {}

if input_mode == "Tải file lên":
    uploaded_files = st.file_uploader(
        "Chọn một hoặc nhiều file audio",
        type=[e.lstrip(".") for e in AUDIO_EXTS],
        accept_multiple_files=True,
        key="file_uploader",
    )
    if uploaded_files:
        upload_dir = os.path.join(tempfile.gettempdir(), "mrt_uploads")
        os.makedirs(upload_dir, exist_ok=True)
        for uf in uploaded_files:
            dst = os.path.join(upload_dir, uf.name)
            if not os.path.exists(dst):
                with open(dst, "wb") as f:
                    f.write(uf.getvalue())
            if dst not in st.session_state.preprocessed:
                files_to_preprocess.append(dst)
            new_labels[dst] = uf.name
else:
    dir_path = st.text_input(
        "Đường dẫn thư mục chứa file audio",
        placeholder="/kaggle/input/.../audio_folder",
    )
    if dir_path and os.path.isdir(dir_path):
        found = sorted(
            p for p in Path(dir_path).iterdir()
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS
        )
        if found:
            for fp in found:
                fp_s = str(fp)
                if fp_s not in st.session_state.preprocessed:
                    files_to_preprocess.append(fp_s)
                new_labels[fp_s] = fp.name
            if files_to_preprocess:
                st.info(f"Tìm thấy {len(found)} file. {len(files_to_preprocess)} file mới cần tiền xử lý.")
            elif found:
                st.success(f"{len(found)} file đã được tiền xử lý — sẵn sàng chạy ASR.")
        else:
            st.warning("Không tìm thấy file audio nào trong thư mục.")
    elif dir_path:
        st.warning("Đường dẫn không tồn tại.")

st.session_state.file_labels.update(new_labels)

# ── Preprocess ────────────────────────────────────────────────────────────
if files_to_preprocess:
    st.divider()
    n_new = len(files_to_preprocess)
    st.caption(f"{n_new} file mới cần tiền xử lý (normalize → denoise → VAD).")

    if st.button(f"⚡ Tiền xử lý {n_new} file", type="primary", use_container_width=True):
        pipeline = get_pipeline()
        out_base = os.path.join("output", "preprocessed")
        os.makedirs(out_base, exist_ok=True)

        prog = st.progress(0, text="Đang tiền xử lý...")
        for i, fp in enumerate(files_to_preprocess):
            prog.progress((i + 1) / len(files_to_preprocess),
                          text=f"Tiền xử lý [{i+1}/{len(files_to_preprocess)}]")
            pre = _do_preprocess(pipeline, fp, out_base)
            if pre:
                st.session_state.preprocessed[fp] = pre
        prog.empty()
        st.rerun()

# ── ASR ───────────────────────────────────────────────────────────────────
pre_files = list(st.session_state.preprocessed.keys())
if pre_files:
    st.divider()
    st.caption(
        f"{len(pre_files)} file đã sẵn sàng. "
        f"Mô hình: **{asr_backend}** / **{asr_model}**. "
        "Đổi model ở sidebar rồi chạy lại ASR — không cần tiền xử lý lại."
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Chạy ASR", type="primary", use_container_width=True):
            pipeline = get_pipeline()
            out_base = os.path.join("output", "transcripts", time.strftime("%Y%m%d_%H%M%S"))
            os.makedirs(out_base, exist_ok=True)

            st.session_state.asr_results = {}
            prog = st.progress(0, text="Đang chạy ASR...")
            valid = [(fp, pre) for fp in pre_files
                     if (pre := st.session_state.preprocessed.get(fp)) and pre["chunks"]]

            for i, (fp, pre) in enumerate(valid):
                prog.progress((i + 1) / max(len(valid), 1),
                              text=f"ASR [{i+1}/{len(valid)}]")
                res = _do_transcribe(pipeline, pre, out_base, asr_backend, asr_model)
                if res:
                    res["_audio_path"] = fp
                    st.session_state.asr_results[pre["basename"]] = res
            prog.empty()
            st.rerun()

    with col2:
        if st.button("🗑 Xóa cache", use_container_width=True):
            st.session_state.preprocessed = {}
            st.session_state.asr_results = {}
            st.session_state.file_labels = {}
            st.rerun()

# ── Results ───────────────────────────────────────────────────────────────
asr_results = st.session_state.asr_results
if asr_results:
    st.divider()
    names = sorted(asr_results.keys())
    gt_map = st.session_state.gt_map

    # ── WER ────────────────────────────────────────────────────────────────
    wer_stats = None
    if gt_map and names:
        audio_paths = [asr_results[n].get("_audio_path", "") for n in names]
        transcriptions = {Path(ap).stem: asr_results[n].get("transcript_text", "")
                          for n, ap in zip(names, audio_paths) if ap}
        if audio_paths and transcriptions:
            wer_stats = run_wer(audio_paths, transcriptions, gt_map)

    if wer_stats and wer_stats["count"] > 0:
        st.subheader("📊 Word Error Rate (WER)")

        _m, _s, _mn, _mx = wer_stats["mean"], wer_stats["std"], wer_stats["min"], wer_stats["max"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean WER", f"{_m*100:.2f}%")
        c2.metric("Std", f"±{_s*100:.2f}%")
        c3.metric("Min WER", f"{_mn*100:.2f}%")
        c4.metric("Max WER", f"{_mx*100:.2f}%")
        st.caption(f"Thống kê trên {wer_stats['count']} file có ground truth khớp.")

        df_wer = pd.DataFrame([
            {"File": pf["name"],
             "WER": f"{pf['wer']*100:.2f}%" if pf["wer"] is not None else "N/A",
             "Ref words": pf["ref_words"], "Hyp words": pf["hyp_words"]}
            for pf in wer_stats["per_file"]
        ])
        st.dataframe(df_wer, use_container_width=True, hide_index=True)

    # ── Transcript ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader(f"📄 Transcripts ({len(names)} file)")

    if len(names) == 1:
        _tabs = st.tabs(["📄 Transcript", "⬇ Tải xuống"])
        res = asr_results[names[0]]
        with _tabs[0]:
            st.text_area("Transcript", res.get("transcript_text", ""), height=400)
        with _tabs[1]:
            path = res.get("transcript_txt")
            if path and os.path.exists(path):
                with open(path, "rb") as fh:
                    st.download_button(
                        label="📄 Transcript (.txt)", data=fh.read(),
                        file_name=os.path.basename(path),
                        mime="text/plain", use_container_width=True,
                    )
    else:
        for name in names:
            res = asr_results[name]
            with st.expander(f"📄 {name}", expanded=False):
                st.text_area(
                    f"Transcript: {name}",
                    res.get("transcript_text", ""), height=250,
                    label_visibility="collapsed",
                )

        with st.expander("⬇ Tải tất cả", expanded=False):
            import zipfile, io
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for name in names:
                    path = asr_results[name].get("transcript_txt")
                    if path and os.path.exists(path):
                        zf.write(path, os.path.basename(path))
            buf.seek(0)
            st.download_button(
                "⬇ Tải tất cả (.zip)", data=buf,
                file_name="transcripts.zip", mime="application/zip",
                use_container_width=True,
            )
