import os
from pathlib import Path
import pytest
import torch

from app.config import MODEL_PATH
from app.model import Detector


@pytest.fixture(scope="session", autouse=True)
def ensure_model_checkpoint_exists():
    """
    Session fixture: Ensures a model checkpoint file exists before any tests run.
    On local dev, uses your real trained weights.
    In CI/CD (GitHub Actions), automatically initializes a dummy checkpoint so
    all API endpoint and inference tests pass without needing 400MB weights in git.
    """
    checkpoint_path = Path(MODEL_PATH)
    created_dummy = False

    if not checkpoint_path.exists():
        print(f"\n[CI Notice] Checkpoint {checkpoint_path} not found. Creating test dummy checkpoint...")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        # Initialize architecture and save initialized weights
        dummy_detector = Detector()
        torch.save(dummy_detector.state_dict(), checkpoint_path)
        created_dummy = True

    yield

    # Clean up dummy checkpoint after tests finish if it was created specifically for this run
    if created_dummy and checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
        except OSError:
            pass
