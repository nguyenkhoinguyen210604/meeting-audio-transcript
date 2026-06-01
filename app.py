"""
Streamlit UI for the Meeting Record Transcript pipeline.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Meeting Transcript Pipeline",
    page_icon="🎙️",
    layout="wide",
)

st.title("🎙️ Meeting Transcript Pipeline")
st.caption("Normalize → Denoise → VAD → ASR → Summarize")

# ── Sidebar — pipeline settings ──────────────────────────────────────────────
openai_api_key = os.environ.get("OPENAI_KEY") or os.environ.get("OPENAI_API_KEY") or ""
summarize_model = "gpt-4o-mini"
chunk_seconds = 30
separate_speakers = False
max_chunk_duration = 25.0
max_gap = 1.5

# ── File upload ───────────────────────────────────────────────────────────────
st.subheader("1. Tải lên file audio")
uploaded_file = st.file_uploader(
    "Chọn file audio (wav, mp3, m4a, ogg, flac, ...)",
    type=["wav", "mp3", "m4a", "ogg", "flac", "aac", "opus", "wma"],
    help="Hỗ trợ mọi định dạng được ffmpeg decode.",
)

if uploaded_file is not None:
    st.audio(uploaded_file, format=uploaded_file.type)
    st.caption(f"File: **{uploaded_file.name}** — {uploaded_file.size / 1024:.1f} KB")

    run_btn = st.button("▶ Chạy Pipeline", type="primary", use_container_width=True)

    if run_btn:
        from src.pipeline import AudioPipeline  # lazy import — heavy models

        progress = st.progress(0, text="Đang khởi tạo...")
        log_box = st.empty()
        logs: list[str] = []

        def log(msg: str) -> None:
            logs.append(msg)
            log_box.code("\n".join(logs), language=None)

        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join("output", "transcripts", timestamp)
            os.makedirs(output_dir, exist_ok=True)

            with tempfile.TemporaryDirectory() as work_dir:
                # Save upload to disk
                input_path = os.path.join(work_dir, uploaded_file.name)
                with open(input_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                log(f"[INPUT] {uploaded_file.name}")
                progress.progress(5, text="Khởi tạo pipeline...")

                pipeline = AudioPipeline(
                    chunk_seconds=chunk_seconds,
                    separate=separate_speakers,
                    openai_api_key=openai_api_key or None,
                    summarize_model=summarize_model,
                    max_chunk_duration=max_chunk_duration,
                    max_gap=max_gap,
                )

                progress.progress(10, text="[1/5] Normalizing audio...")
                log("[1/5] Normalizing audio...")

                # Monkey-patch print so pipeline logs appear in UI
                import builtins
                _orig_print = builtins.print

                def _ui_print(*args, **kwargs):
                    msg = " ".join(str(a) for a in args)
                    _orig_print(*args, **kwargs)
                    log(msg)

                builtins.print = _ui_print

                # Progress hook via patched print —
                # update bar based on keywords in pipeline output
                _step_map = {
                    "[2]": (25, "[2/5] Reducing noise..."),
                    "[3]": (50, "[3/5] VAD segmentation..."),
                    "[4]": (65, "[4/5] Transcribing..."),
                    "[5]": (85, "[5/5] Summarizing..."),
                }

                orig_ui_print = _ui_print

                def _tracked_print(*args, **kwargs):
                    msg = " ".join(str(a) for a in args)
                    for key, (pct, label) in _step_map.items():
                        if key in msg:
                            progress.progress(pct, text=label)
                            break
                    orig_ui_print(*args, **kwargs)

                builtins.print = _tracked_print

                try:
                    results = pipeline.run(input_path, output_dir)
                finally:
                    builtins.print = _orig_print

                progress.progress(100, text="Hoàn tất!")
                log("Pipeline hoàn tất.")

                # ── Display results ───────────────────────────────────────
                st.divider()
                st.subheader("2. Kết quả")

                tab_transcript, tab_summary, tab_download = st.tabs(
                    ["📄 Transcript", "✨ Tóm tắt AI", "⬇ Tải xuống"]
                )

                # ── Transcript tab ────────────────────────────────────────
                with tab_transcript:
                    txt_path = results.get("transcript_txt")
                    if txt_path and os.path.exists(txt_path):
                        transcript_text = Path(txt_path).read_text(encoding="utf-8")
                        st.text_area(
                            "Transcript (với timestamp & speaker)",
                            transcript_text,
                            height=400,
                        )
                    else:
                        st.warning("Không tìm thấy file transcript.")

                # ── Summary tab ───────────────────────────────────────────
                with tab_summary:
                    summary_path = results.get("summary_path")
                    if summary_path and os.path.exists(summary_path):
                        summary_text = Path(summary_path).read_text(encoding="utf-8")
                        st.markdown(summary_text)
                    elif not openai_api_key:
                        st.info(
                            "Tóm tắt bị bỏ qua vì không có OpenAI API key. "
                            "Nhập key ở thanh bên trái để bật tính năng này."
                        )
                    else:
                        st.warning("Không tìm thấy file tóm tắt.")

                # ── Download tab ──────────────────────────────────────────
                with tab_download:
                    dl_cols = st.columns(3)

                    file_defs = [
                        ("transcript_txt",  "📄 Transcript (.txt)", "text/plain"),
                        ("transcript_json", "🗂 Transcript (.json)", "application/json"),
                        ("transcript_srt",  "🎬 Subtitle (.srt)",   "text/plain"),
                        ("summary_path",    "✨ Tóm tắt (.md)",     "text/markdown"),
                        ("denoised_wav",    "🔊 Audio đã lọc (.wav)", "audio/wav"),
                    ]

                    col_idx = 0
                    for key, label, mime in file_defs:
                        path = results.get(key)
                        if path and os.path.exists(path):
                            with dl_cols[col_idx % 3]:
                                with open(path, "rb") as fh:
                                    st.download_button(
                                        label=label,
                                        data=fh.read(),
                                        file_name=os.path.basename(path),
                                        mime=mime,
                                        use_container_width=True,
                                    )
                            col_idx += 1

        except Exception as exc:
            builtins.print = _orig_print  # ensure always restored
            progress.empty()
            st.error(f"Pipeline lỗi: {exc}")
            with st.expander("Chi tiết lỗi"):
                import traceback
                st.code(traceback.format_exc())

else:
    st.info("Tải lên file audio để bắt đầu.")
