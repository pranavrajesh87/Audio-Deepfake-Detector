# test_clean_only.py — Test clean laptop recordings ONLY

import os
import torch
import torch.nn as nn
import numpy as np
import librosa
from transformers import Wav2Vec2Model

DEVICE = torch.device("cpu")
SAMPLE_RATE = 16000
MAX_SAMPLES = 64000

MODEL_PATH = r"C:\Users\prana\OneDrive\Desktop\AI PROJECTS\DEEP FAKE AUDIO DETECTOR\models\best_model.pt"
CLEAN_FOLDER = r"C:\Users\prana\OneDrive\Desktop\AI PROJECTS\DEEP FAKE AUDIO DETECTOR\results\clean_recordings"

class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 1)
        )
    def forward(self, x):
        x = x.squeeze(1)
        h = self.wav2vec2(x).last_hidden_state
        pooled = h.mean(dim=1)
        return self.classifier(pooled)

print("Loading model...")
model = Detector()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("Model loaded!\n")

def load_audio(path):
    audio, sr = librosa.load(path, sr=None)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    out = np.zeros(MAX_SAMPLES, dtype=np.float32)
    if len(audio) > MAX_SAMPLES:
        out[:] = audio[:MAX_SAMPLES]
    else:
        out[:len(audio)] = audio
    return torch.from_numpy(out).unsqueeze(0).unsqueeze(0)

def predict(audio_path):
    tensor = load_audio(audio_path).to(DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).item()
    return prob

print("=" * 60)
print("CLEAN LAPTOP RECORDING TEST")
print("=" * 60)
print("Threshold: 0.50")
print("Below 0.50 = REAL (human voice)")
print("Above 0.50 = FAKE (AI/synthetic)")
print("Close to 0.50 = borderline/uncertain")
print("=" * 60 + "\n")

files = sorted([f for f in os.listdir(CLEAN_FOLDER) if f.endswith((".m4a", ".wav", ".mp3"))])
real_count = 0
total = 0

for fname in files:
    path = os.path.join(CLEAN_FOLDER, fname)
    try:
        prob = predict(path)
        label = "REAL" if prob < 0.5 else "FAKE"
        distance = abs(prob - 0.5)
        
        # Highlight borderline cases
        if distance < 0.1:
            marker = "⚠️ BORDERLINE"
        elif prob < 0.5:
            marker = "✓ CORRECT"
        else:
            marker = "✗ WRONG"
        
        print(f"{fname}")
        print(f"  Probability: {prob:.4f}  →  {label}  {marker}")
        print()
        
        if prob < 0.5:
            real_count += 1
        total += 1
        
    except Exception as e:
        print(f"{fname}: ERROR - {e}")

print("=" * 60)
print(f"SUMMARY: {real_count}/{total} detected as REAL ({real_count/total*100:.1f}%)")
print("=" * 60)

if real_count > 0:
    print("\n✓ SUCCESS! Model CAN detect clean human voice as real.")
    print("  This proves the WhatsApp failure was compression artifacts.")
else:
    print("\n✗ All clean recordings still detected as fake.")
    print("  Possible reasons:")
    print("    1. Model is over-rejecting (learned to say 'fake' too often)")
    print("    2. Windows Voice Recorder .m4a still has mild compression")
    print("    3. Background noise or room echo in your recordings")
    print("    4. Need to try even closer to microphone, quieter room")