import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class Detector(nn.Module):
    """
    Wav2Vec2-based Binary Audio Classifier for Deepfake Voice Detection.
    """
    def __init__(self, pretrained_model_name: str = "facebook/wav2vec2-base"):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model_name)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw audio waveform tensor of shape [Batch, 1, 64000] or [Batch, 64000]
        Returns:
            Logit tensor of shape [Batch, 1]
        """
        # If input has channel dimension [B, 1, T], remove it -> [B, T]
        if x.dim() == 3:
            x = x.squeeze(1)
            
        # Extract features from Wav2Vec2: [Batch, Time_Frames, 768]
        h = self.wav2vec2(x).last_hidden_state
        
        # Mean pooling across the time dimension -> [Batch, 768]
        pooled = h.mean(dim=1)
        
        # Binary classification head -> [Batch, 1]
        return self.classifier(pooled)
