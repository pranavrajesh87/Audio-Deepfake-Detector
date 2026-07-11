# app/demo.py — Gradio Demo for Audio Deepfake Detector

import os
import torch
import torch.nn as nn
import numpy as np
import librosa
import gradio as gr
from transformers import Wav2Vec2Model

# ========================
# CONFIGURATION
# ========================
DEVICE = torch.device("cpu")
SAMPLE_RATE = 16000
MAX_SAMPLES = 64000  # 4 seconds at 16kHz

# Path to your trained model (update if your path is different)
MODEL_PATH = r"C:\Users\prana\OneDrive\Desktop\AI PROJECTS\DEEP FAKE AUDIO DETECTOR\models\best_model.pt"

# ========================
# MODEL
# ========================
class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        x = x.squeeze(1)  # Remove channel dim: [B, 1, 64000] -> [B, 64000]
        h = self.wav2vec2(x).last_hidden_state  # [B, T, 768]
        pooled = h.mean(dim=1)  # [B, 768]
        return self.classifier(pooled)  # [B, 1]

print("Loading model...")
model = Detector()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("Model ready!")

# ========================
# AUDIO PREPROCESSING
# ========================
def load_audio(audio_path):
    """Load audio file and standardize to 4 seconds, 16kHz mono"""
    audio, sr = librosa.load(audio_path, sr=None)
    
    # Resample if needed
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    
    # Pad or truncate to exactly 4 seconds (64,000 samples)
    standardized = np.zeros(MAX_SAMPLES, dtype=np.float32)
    if len(audio) > MAX_SAMPLES:
        standardized[:] = audio[:MAX_SAMPLES]
    else:
        standardized[:len(audio)] = audio
    
    # Add batch and channel dimensions: [1, 1, 64000]
    return torch.from_numpy(standardized).unsqueeze(0).unsqueeze(0)

# ========================
# PREDICTION FUNCTION
# ========================
def predict(audio_file):
    """
    Takes an audio file path from Gradio, returns formatted prediction.
    """
    try:
        # Load and preprocess
        tensor = load_audio(audio_file).to(DEVICE)
        
        # Run inference
        with torch.no_grad():
            prob = torch.sigmoid(model(tensor)).item()
        
        # Format output
        confidence = prob * 100 if prob > 0.5 else (1 - prob) * 100
        label = "AI-GENERATED / SYNTHETIC" if prob > 0.5 else "REAL HUMAN VOICE"
        
        result = f"## {label}\n\n**Confidence:** {confidence:.1f}%"
        
        # Add domain shift warning for "fake" predictions
        if prob > 0.5:
            result += (
                "\n\n⚠️ **Domain Shift Warning:** "
                "This model was trained on studio-quality ASVspoof recordings. "
                "It may misclassify real-world audio (phone recordings, laptop mics, "
                "compressed voice notes) as synthetic due to acoustic mismatch. "
                "See project README for the full stress test analysis."
            )
        else:
            result += (
                "\n\n✅ Audio characteristics match the training distribution "
                "(studio-quality speech)."
            )
        
        return result
        
    except Exception as e:
        return f"**Error processing audio:** {str(e)}"

# ========================
# GRADIO INTERFACE
# ========================
demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(
        type="filepath",
        label="Upload Audio File (.wav, .mp3, .flac, .m4a)"
    ),
    outputs=gr.Markdown(label="Detection Result"),
    title="🔊 Audio Deepfake Detector",
    description=(
        "Detect AI-generated synthetic speech vs. real human voice using a "
        "fine-tuned Wav2Vec2 model.\n\n"
        "**Note:** This model achieves EER 4.26% on ASVspoof 2019 benchmark, "
        "but exhibits known domain shift on real-world recordings. "
        "Upload a clean studio recording for best results."
    ),
    allow_flagging="never",
    examples=[]  # You can add example file paths here if desired
)

# ========================
# LAUNCH
# ========================
if __name__ == "__main__":
    demo.launch()