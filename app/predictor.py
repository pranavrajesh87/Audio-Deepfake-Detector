import os
from typing import Union, Dict, Any
import io
import torch

from app.config import MODEL_PATH, DECISION_THRESHOLD, MODEL_VERSION
from app.model import Detector
from app.audio import preprocess_audio


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
            raise FileNotFoundError(
                f"Model checkpoint not found at: {self.model_path}. "
                f"Ensure 'models/best_model.pt' exists or set the MODEL_PATH environment variable."
            )

        model = Detector()
        state_dict = torch.load(self.model_path, map_location=self.device)
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
