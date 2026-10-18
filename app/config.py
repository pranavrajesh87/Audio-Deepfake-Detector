import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent

# Audio Specifications
SAMPLE_RATE = 16000
DURATION_SECONDS = 4
MAX_SAMPLES = SAMPLE_RATE * DURATION_SECONDS  # 64,000 samples

# Model Configuration
# Can be overridden by an environment variable MODEL_PATH, otherwise defaults to local path
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "best_model.pt"))
MODEL_VERSION = "v1"

# Decision Threshold (ASVspoof calibrated)
# Scores >= THRESHOLD are classified as FAKE, < THRESHOLD as REAL
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.90"))

# Allowed Audio File Extensions for validation
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit
