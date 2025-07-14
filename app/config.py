# app/config.py
from pathlib import Path

class Config:
    MODEL_DIR = Path("models")
    PREDOWNLOAD_LANGUAGES = ["eng", "khm", "mya"]  # Pre-download these
    DEFAULT_LANGUAGE = "eng"
    
config = Config()