# test_asvspoof_bonafide.py — Test model on ASVspoof's own real voices

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
EVAL_AUDIO = r"C:\Users\prana\OneDrive\Desktop\AI PROJECTS\DEEP FAKE AUDIO DETECTOR\data\LA\ASVspoof2019_LA_eval\flac"
EVAL_PROTOCOL = r"C:\Users\prana\OneDrive\Desktop\AI PROJECTS\DEEP FAKE AUDIO DETECTOR\data\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.eval.trl.txt"

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

def predict(path):
    tensor = load_audio(path).to(DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).item()
    return prob

# Parse eval protocol and find bonafide clips
bonafide_files = []
with open(EVAL_PROTOCOL, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if parts[4] == "bonafide":
            bonafide_files.append(parts[1])

print(f"Found {len(bonafide_files)} bonafide clips in eval set")
print("Testing 10 random ASVspoof real voices...\n")

import random
random.seed(42)
test_files = random.sample(bonafide_files, 10)

real_detected = 0
for fname in test_files:
    path = os.path.join(EVAL_AUDIO, f"{fname}.flac")
    prob = predict(path)
    label = "REAL" if prob < 0.5 else "FAKE"
    marker = "✓ CORRECT" if prob < 0.5 else "✗ WRONG"
    print(f"{fname}: {prob:.4f} → {label} {marker}")
    if prob < 0.5:
        real_detected += 1

print(f"\n{'='*50}")
print(f"ASVspoof real voices detected as REAL: {real_detected}/10")
print(f"{'='*50}")

if real_detected >= 8:
    print("\n✓ MODEL IS NOT BROKEN.")
    print("  It correctly identifies ASVspoof studio real voices.")
    print("  It fails on YOUR voice because of domain shift:")
    print("    - Different microphone (laptop vs studio mic)")
    print("    - Different room acoustics")
    print("    - Different speaking style")
    print("  This is the REAL finding: the model learned 'ASVspoof studio', not 'human'.")
else:
    print("\n✗ MODEL IS BROKEN — even ASVspoof real voices are called fake.")
    print("  Possible causes: label flip during training, corrupted weights, or bug.")