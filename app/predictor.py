import os
from pathlib import Path
from typing import Union, Dict, Any
import io
import urllib.request
import torch

from app.config import MODEL_PATH, MODEL_DOWNLOAD_URL, DECISION_THRESHOLD, MODEL_VERSION
from app.model import Detector
from app.audio import preprocess_audio


def download_weights(destination: str, url: str) -> None:
    """
    Downloads model weights from a remote URL to destination with progress display.
    Streams to a temporary file first to guarantee atomic write and prevent corrupted checkpoints.
    """
    dest_path = Path(destination)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")

    print(f"\n[Model Checkpoint] Checkpoint not found at: {destination}")
    print(f"[Model Checkpoint] Auto-downloading weights from release: {url}")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "AudioDeepfakeDetector/1.0"}
    )

    try:
        with urllib.request.urlopen(req) as response, open(temp_path, "wb") as out_file:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB chunk

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(
                        f"\r[Model Checkpoint] Downloading: {percent:5.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)",
                        end="",
                        flush=True,
                    )
                else:
                    mb_downloaded = downloaded / (1024 * 1024)
                    print(f"\r[Model Checkpoint] Downloading: {mb_downloaded:.1f} MB", end="", flush=True)

        print("\n[Model Checkpoint] Download finished successfully! Finalizing checkpoint...")
        temp_path.replace(dest_path)
        print(f"[Model Checkpoint] Saved checkpoint to {dest_path}")
    except Exception as e:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise RuntimeError(
            f"Failed to auto-download model checkpoint from {url}: {e}\n"
            f"Please download 'best_model.pt' manually and place it at '{destination}'."
        ) from e


class AudioPredictor:
    """
    Manages model loading and runs inference on audio inputs.
    """
    def __init__(self, model_path: str = MODEL_PATH, device: str = None):
        # Choose GPU if available, otherwise CPU
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_path = model_path
        self.threshold = DECISION_THRESHOLD
        self.model_version = MODEL_VERSION

        print(f"Loading deepfake detector from {self.model_path} onto {self.device}...")
        self.model = self._load_model()
        print("Model loaded and ready for inference!")

    def _load_model(self) -> Detector:
        """Instantiates the architecture and loads trained checkpoint weights."""
        if not os.path.exists(self.model_path):
            download_weights(self.model_path, MODEL_DOWNLOAD_URL)

        model = Detector()
        state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()  # Set to evaluation mode (turns off Dropout)
        return model

    def predict(self, audio_source: Union[str, bytes, io.BytesIO]) -> Dict[str, Any]:
        """
        Runs end-to-end preprocessing and prediction on audio.
        
        Returns:
            Dictionary with prediction ('real' or 'fake'), score, threshold, and metadata.
        """
        # 1. Preprocess audio into standardized tensor [1, 1, 64000]
        tensor, original_duration = preprocess_audio(audio_source)
        tensor = tensor.to(self.device)

        # 2. Run forward pass with gradients disabled (saves memory and CPU/GPU cycles)
        with torch.no_grad():
            logit = self.model(tensor)
            # Sigmoid turns the logit into a probability score between 0.0 and 1.0
            score = torch.sigmoid(logit).item()

        # 3. Apply decision threshold
        is_fake = score >= self.threshold
        prediction = "fake" if is_fake else "real"

        return {
            "prediction": prediction,
            "score": round(score, 4),
            "threshold": self.threshold,
            "model_version": self.model_version,
            "audio_duration_seconds": round(original_duration, 2),
        }


# Global singleton instance (initialized when imported or on service startup)
_predictor_instance = None

def get_predictor() -> AudioPredictor:
    """Returns the singleton predictor instance, initializing it if necessary."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = AudioPredictor()
    return _predictor_instance
