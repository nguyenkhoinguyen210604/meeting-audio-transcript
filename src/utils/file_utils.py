"""
Shared file/path utilities (ensure dir exists, temp paths, cleanup, etc.).
"""

import os


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def temp_path(directory, filename):
    ensure_dir(directory)
    return os.path.join(directory, filename)
