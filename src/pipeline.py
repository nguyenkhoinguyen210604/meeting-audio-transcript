import os
import shutil
import tempfile

from src.preprocessing.normalizer import normalize
from src.enhancement.noise_reducer import NoiseReducer
from src.enhancement.speech_separator import SpeechSeparator
from src.vad.segmenter import VADSegmenter
from src.transcription.router import ASRRunner
from src.postprocessing.formatter import Formatter


class AudioPipeline:
    """
    Meeting audio pipeline, split into two stages:
    Stage 1 — preprocess() : normalize → denoise → [separate] → VAD
    Stage 2 — transcribe()  : ASR → format
    """

    def __init__(
        self,
        chunk_seconds: int = 30,
        separate: bool = False,
        max_chunk_duration: float = 25.0,
        max_gap: float = 1.5,
    ):
        self.noise_reducer = NoiseReducer(chunk_seconds=chunk_seconds)
        self.separator = SpeechSeparator() if separate else None
        self.segmenter = VADSegmenter(
            max_chunk_duration=max_chunk_duration,
            max_gap=max_gap,
        )
        self._asr_runner = None
        self._asr_key: tuple | None = None

    # ── Stage 1: pre-ASR (cacheable) ──────────────────────────────────────

    def preprocess(self, input_file: str, output_dir: str) -> dict:
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(input_file))[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            normalized_wav = os.path.join(tmpdir, "normalized.wav")
            denoised_wav = os.path.join(tmpdir, "denoised.wav")

            print(f"[pre] Normalizing: {basename}")
            normalize(input_file, normalized_wav)

            print(f"[pre] Denoising: {basename}")
            self.noise_reducer.process(normalized_wav, denoised_wav)

            audio_dir = os.path.join(output_dir, "audio")
            os.makedirs(audio_dir, exist_ok=True)
            denoised_out = os.path.join(audio_dir, f"{basename}_denoised.wav")
            shutil.copy(denoised_wav, denoised_out)

            asr_input = denoised_wav

            if self.separator:
                print(f"[pre] Separating speakers: {basename}")
                separated_dir = os.path.join(output_dir, "separated")
                tracks = self.separator.process(denoised_wav, separated_dir)
                if tracks:
                    asr_input = tracks[0]

            print(f"[pre] VAD: {basename}")
            chunks = self.segmenter.get_chunks(asr_input)

        return {
            "denoised_wav": denoised_out,
            "chunks": [{"start": c.start, "end": c.end} for c in chunks],
            "basename": basename,
        }

    # ── Stage 2: ASR only ─────────────────────────────────────────────────

    def transcribe(
        self,
        denoised_wav: str,
        chunks: list[dict],
        output_dir: str,
        asr_backend: str = "whisper",
        asr_model: str = "large-v3",
        basename: str = "audio",
    ) -> dict:
        os.makedirs(output_dir, exist_ok=True)

        from src.vad.segmenter import AudioChunk
        chunk_objs = [AudioChunk(c["start"], c["end"]) for c in chunks]

        if not chunk_objs:
            print(f"[ASR] No speech chunks — skipping: {basename}")
            return {}

        asr_key = (asr_backend, asr_model)
        if self._asr_key != asr_key:
            print(f"[ASR] Loading model: {asr_backend}/{asr_model}")
            self._asr_runner = ASRRunner(backend=asr_backend, model=asr_model)
            self._asr_key = asr_key

        runner = self._asr_runner

        print(f"[ASR] Transcribing {len(chunk_objs)} chunk(s): {basename}")
        transcribed = runner.transcribe(denoised_wav, chunk_objs)

        formatter = Formatter(transcribed)

        results: dict = {}
        path = os.path.join(output_dir, f"{basename}_transcript.txt")
        formatter.save(path, fmt="txt")
        results["transcript_txt"] = path

        results["transcript_text"] = formatter.to_plain_transcript()
        return results

    # ── Full pipeline (CLI convenience) ────────────────────────────────────

    def run(
        self,
        input_file: str,
        output_dir: str,
        asr_backend: str = "whisper",
        asr_model: str = "large-v3",
    ) -> dict:
        pre = self.preprocess(input_file, output_dir)
        asr = self.transcribe(
            denoised_wav=pre["denoised_wav"],
            chunks=pre["chunks"],
            output_dir=output_dir,
            asr_backend=asr_backend,
            asr_model=asr_model,
            basename=pre["basename"],
        )
        return {"denoised_wav": pre["denoised_wav"], **asr}
