import argparse
import os

from src.pipeline import AudioPipeline


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Meeting audio pipeline: "
            "normalize → denoise → [separate] → VAD → LID → ASR → summarize"
        )
    )

    parser.add_argument(
        "input",
        help="Input audio file (any format supported by ffmpeg)"
    )

    parser.add_argument(
        "output_dir",
        help="Directory where all output files are written"
    )

    # ── audio enhancement ────────────────────────────────────────────────────
    parser.add_argument(
        "--chunk_seconds",
        type=int,
        default=30,
        help="Chunk size (seconds) for noise reduction (default: 30)"
    )

    parser.add_argument(
        "--separate",
        action="store_true",
        help=(
            "Run speaker separation after denoising (MossFormer2_SS_16K). "
            "Tracks saved to <output_dir>/separated/"
        )
    )

    # ── VAD / chunking ────────────────────────────────────────────────────────
    parser.add_argument(
        "--max_chunk_duration",
        type=float,
        default=25.0,
        help="Maximum ASR chunk duration in seconds (default: 25)"
    )

    parser.add_argument(
        "--max_gap",
        type=float,
        default=1.5,
        help=(
            "Maximum silence gap (seconds) within which adjacent VAD segments "
            "are merged into one ASR chunk (default: 1.5)"
        )
    )

    # ── summarization ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--openai_api_key",
        default=os.environ.get("OPENAI_API_KEY"),
        help=(
            "OpenAI API key for summarization. "
            "Falls back to OPENAI_API_KEY env var. Summarization is skipped if absent."
        )
    )

    parser.add_argument(
        "--summarize_model",
        default="gpt-4o-mini",
        help="OpenAI model for summarization (default: gpt-4o-mini)"
    )

    args = parser.parse_args()

    pipeline = AudioPipeline(
        chunk_seconds=args.chunk_seconds,
        separate=args.separate,
        openai_api_key=args.openai_api_key,
        summarize_model=args.summarize_model,
        max_chunk_duration=args.max_chunk_duration,
        max_gap=args.max_gap,
    )

    pipeline.run(args.input, args.output_dir)
