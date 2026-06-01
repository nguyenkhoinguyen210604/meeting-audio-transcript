import os
import shutil
import tempfile

from src.preprocessing.normalizer import normalize
from src.enhancement.noise_reducer import NoiseReducer
from src.enhancement.speech_separator import SpeechSeparator
from src.vad.segmenter import VADSegmenter
from src.transcription.router import ASRRouter
from src.postprocessing.formatter import Formatter
from src.postprocessing.summarizer import Summarizer


class AudioPipeline:
    """
    Meeting audio pipeline:
    normalize → denoise → [separate] → VAD chunk → LID → ASR → summarize
    """

    def __init__(
        self,
        chunk_seconds: int = 30,
        separate: bool = False,
        openai_api_key: str | None = None,
        summarize_model: str = "gpt-4o-mini",
        summarize: bool = False,
        max_chunk_duration: float = 25.0,
        max_gap: float = 1.5,
    ):
        self.noise_reducer = NoiseReducer(chunk_seconds=chunk_seconds)
        self.separator = SpeechSeparator() if separate else None
        self.segmenter = VADSegmenter(
            max_chunk_duration=max_chunk_duration,
            max_gap=max_gap,
        )
        self.router = ASRRouter()
        self.summarizer = (
            Summarizer(api_key=openai_api_key, model=summarize_model)
            if (summarize and openai_api_key)
            else None
        )

    def run(self, input_file: str, output_dir: str) -> dict:
        """
        Full pipeline: input → normalize → denoise → [separate]
                       → VAD chunk → LID → ASR route → summarize → output

        Returns:
            dict with paths to all produced files.
        """
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(input_file))[0]
        results: dict = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            normalized_wav = os.path.join(tmpdir, "normalized.wav")
            denoised_wav   = os.path.join(tmpdir, "denoised.wav")

            print("[1] Normalizing audio...")
            normalize(input_file, normalized_wav)

            print("[2] Reducing noise...")
            self.noise_reducer.process(normalized_wav, denoised_wav)

            audio_dir = os.path.join(output_dir, "audio")
            os.makedirs(audio_dir, exist_ok=True)
            denoised_out = os.path.join(audio_dir, f"{basename}_denoised.wav")
            shutil.copy(denoised_wav, denoised_out)
            results["denoised_wav"] = denoised_out
            print(f"  Denoised audio → {denoised_out}")

            asr_input = denoised_wav

            if self.separator:
                print("[3] Separating speakers...")
                separated_dir = os.path.join(output_dir, "separated")
                tracks = self.separator.process(denoised_wav, separated_dir)
                results["separated_tracks"] = tracks
                print(f"  Speaker tracks → {separated_dir}")
                for t in tracks:
                    print(f"    - {t}")
                if tracks:
                    asr_input = tracks[0]

            print("[3] Running VAD segmentation...")
            chunks = self.segmenter.get_chunks(asr_input)
            if not chunks:
                print("No speech detected — stopping pipeline.")
                return results

            print(f"[4] Transcribing {len(chunks)} chunk(s) (LID + ASR routing)...")
            transcribed = self.router.transcribe(asr_input, chunks)

            formatter = Formatter(transcribed)

            transcript_txt  = os.path.join(output_dir, f"{basename}_transcript.txt")
            transcript_json = os.path.join(output_dir, f"{basename}_transcript.json")
            transcript_srt  = os.path.join(output_dir, f"{basename}_transcript.srt")
            formatter.save(transcript_txt,  fmt="txt")
            formatter.save(transcript_json, fmt="json")
            formatter.save(transcript_srt,  fmt="srt")
            results["transcript_txt"]  = transcript_txt
            results["transcript_json"] = transcript_json
            results["transcript_srt"]  = transcript_srt

            if self.summarizer is None:
                print("\nNo OpenAI API key supplied — skipping summarization.")
                return results

            print("[5] Summarizing...")
            plain = formatter.to_plain_transcript()
            summary = self.summarizer.summarize(plain)
            summary_path = os.path.join(output_dir, f"{basename}_summary.md")
            self.summarizer.save(summary, summary_path)
            results["summary_path"] = summary_path

        print("\nPipeline complete. Outputs:")
        for key, val in results.items():
            if isinstance(val, list):
                for v in val:
                    print(f"  [{key}] {v}")
            else:
                print(f"  [{key}] {val}")

        return results
