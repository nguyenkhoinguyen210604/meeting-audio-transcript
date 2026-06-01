import subprocess


def run_cmd(cmd):
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed:\n{' '.join(cmd)}\n"
            f"{result.stderr.decode()}"
        )

    return result


def normalize(input_file, output_wav):
    """
    Convert input audio to mono, 48kHz, normalized wav via ffmpeg.
    """

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_file,
        "-ac", "1",
        "-ar", "48000",
        "-af", "loudnorm",
        "-c:a", "pcm_s16le",
        output_wav
    ]

    run_cmd(cmd)
