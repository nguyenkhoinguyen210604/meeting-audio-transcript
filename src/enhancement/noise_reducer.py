import math
import os

import torch
import torchaudio
from df.enhance import enhance, init_df


class NoiseReducer:
    def __init__(self, chunk_seconds=30):
        print("Loading DeepFilterNet model...")

        self.model, self.df_state, _ = init_df()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)
        self.chunk_seconds = chunk_seconds

        print(f"Using device: {self.device}")
        print(f"Chunk size: {self.chunk_seconds} seconds")

    def process(self, input_wav, output_wav):
        """
        Apply DeepFilterNet noise reduction in chunks, then save result.
        """

        print("Loading preprocessed audio...")

        audio, sr = torchaudio.load(input_wav)

        print(f"Sample rate: {sr}")
        print(f"Audio shape: {audio.shape}")

        chunk_samples = sr * self.chunk_seconds
        total_samples = audio.shape[1]
        num_chunks = math.ceil(total_samples / chunk_samples)

        print(f"Total chunks: {num_chunks}")

        enhanced_chunks = []

        for i in range(num_chunks):

            start = i * chunk_samples
            end = min((i + 1) * chunk_samples, total_samples)

            print(
                f"Processing chunk "
                f"{i + 1}/{num_chunks} "
                f"({start}:{end})"
            )

            chunk = audio[:, start:end].to(self.device)

            with torch.no_grad():
                enhanced_chunk = enhance(
                    self.model,
                    self.df_state,
                    chunk
                )

            enhanced_chunks.append(enhanced_chunk.cpu())

            del chunk, enhanced_chunk

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("Concatenating chunks...")

        enhanced_audio = torch.cat(enhanced_chunks, dim=1)

        output_dir = os.path.dirname(output_wav)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        print("Saving denoised audio...")

        torchaudio.save(output_wav, enhanced_audio, sr)
