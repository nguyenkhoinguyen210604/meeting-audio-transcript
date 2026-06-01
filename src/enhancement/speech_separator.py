import os

import torch
import torchaudio
from clearvoice import ClearVoice


class SpeechSeparator:
    # MossFormer2_SS_16K requires 16kHz mono audio
    TARGET_SR = 16000
    MODEL_NAME = "MossFormer2_SS_16K"

    def __init__(self):
        if not torch.cuda.is_available():
            n = os.cpu_count() or 1
            torch.set_num_threads(n)
            torch.set_num_interop_threads(n)
            print(f"CPU mode: using {n} threads")

        print("Loading MossFormer2_SS_16K model via ClearVoice...")

        self.cv = ClearVoice(
            task='speech_separation',
            model_names=[self.MODEL_NAME]
        )

        print("MossFormer2_SS_16K ready.")

    def process(self, input_wav, output_dir):
        """
        Separate mixed speech into individual speaker tracks.

        Args:
            input_wav: path to mono wav file (will be resampled to 16kHz if needed)
            output_dir: directory to write separated speaker tracks

        Returns:
            list of output file paths, one per speaker
        """

        os.makedirs(output_dir, exist_ok=True)

        # resample to 16kHz if needed
        audio, sr = torchaudio.load(input_wav)
        if sr != self.TARGET_SR:
            resampler = torchaudio.transforms.Resample(sr, self.TARGET_SR)
            audio = resampler(audio)
            resampled_path = os.path.join(output_dir, "_input_16k.wav")
            torchaudio.save(resampled_path, audio, self.TARGET_SR)
            input_wav = resampled_path

        self.cv(input_path=input_wav, online_write=True, output_path=output_dir)

        # ClearVoice writes to output_dir/<MODEL_NAME>/
        model_out_dir = os.path.join(output_dir, self.MODEL_NAME)
        search_dir = model_out_dir if os.path.isdir(model_out_dir) else output_dir

        output_files = sorted([
            os.path.join(search_dir, f)
            for f in os.listdir(search_dir)
            if f.endswith(".wav") and not f.startswith("_input")
        ])

        # clean up temp resample file if created
        tmp = os.path.join(output_dir, "_input_16k.wav")
        if os.path.exists(tmp):
            os.remove(tmp)

        print(f"Separated into {len(output_files)} speaker track(s).")

        return output_files
