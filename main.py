import argparse

from src.pipeline import AudioPipeline


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Meeting audio pipeline: normalize → denoise → [separate] → VAD → ASR"
    )

    parser.add_argument("input",      help="Input audio file (any format supported by ffmpeg)")
    parser.add_argument("output_dir", help="Directory where all output files are written")

    parser.add_argument(
        "--asr_backend",
        default="whisper",
        choices=["whisper", "qwen"],
        help="ASR backend: 'whisper' (faster-whisper) or 'qwen' (Qwen3-ASR). Default: whisper",
    )
    parser.add_argument(
        "--asr_model",
        default="large-v3",
        help="Model variant. whisper: tiny|base|small|medium|large-v1|large-v2|large-v3. "
             "qwen: Qwen/Qwen3-ASR-0.6B | Qwen/Qwen3-ASR-1.7B",
    )
    parser.add_argument("--chunk_seconds", type=int, default=30)
    parser.add_argument("--separate", action="store_true")
    parser.add_argument("--max_chunk_duration", type=float, default=25.0)
    parser.add_argument("--max_gap", type=float, default=1.5)

    args = parser.parse_args()

    pipeline = AudioPipeline(
        chunk_seconds=args.chunk_seconds,
        separate=args.separate,
        max_chunk_duration=args.max_chunk_duration,
        max_gap=args.max_gap,
    )

    pipeline.run(
        args.input,
        args.output_dir,
        asr_backend=args.asr_backend,
        asr_model=args.asr_model,
    )
