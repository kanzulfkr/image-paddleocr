import os
import re
from pathlib import Path
from datetime import datetime

import cv2
import pandas as pd
import numpy as np

from .constants import SUPPORTED_EXTS  # Import eksplisit

def get_timestamp():
    """Dapatkan timestamp untuk nama file"""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def list_images(input_dir: Path):
    """Kumpulkan semua gambar yang didukung dari folder input."""
    files = [p for p in sorted(input_dir.iterdir()) 
             if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    return files

def ensure_directories(*dirs):
    """Pastikan direktori-direktori ada"""
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)