# phase3_stress_test.py — Test the model on modern fakes and real voice

import os
import torch
import torch.nn as nn
import numpy as np
import librosa
from transformers import Wav2Vec2Model

# ========================
# CONFIGURATION
# ========================
DEVICE = torch.device("cpu")
SAMPLE_RATE = 16000
MAX_SAMPLES = 64000

MODEL_PATH = r"C:\Users\prana\OneDrive\Desktop\AI PROJECTS\DEEP FAKE AUDIO DETECTOR\models\best_model.pt"
MODERN_FOLDER = r"C:\Users\prana\OneDrive\Desktop\AI PROJECTS\DEEP FAKE AUDIO DETECTOR\results\edge_tts_fakes"
REAL_FOLDER = r"C:\Users\prana\OneDrive\Desktop\AI PROJECTS\DEEP FAKE AUDIO DETECTOR\results\my_real_voice"

# ========================
# MODEL
# ========================
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

# ========================
# AUDIO LOADER
# ========================
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
    label = 1 if prob > 0.5 else 0
    return label, prob

# ========================
# TEST MODERN FAKES (Edge TTS)
# ========================
print("=== Edge TTS (Modern Neural Fakes) ===")
modern_files = sorted([f for f in os.listdir(MODERN_FOLDER) if f.endswith(".wav")])
modern_correct = 0

for fname in modern_files:
    path = os.path.join(MODERN_FOLDER, fname)
    label, prob = predict(path)
    status = "FAKE" if label == 1 else "REAL (WRONG!)"
    print(f"  {fname}: {status} ({prob:.4f})")
    if label == 1:
        modern_correct += 1

modern_rate = modern_correct / len(modern_files) * 100
print(f"\nModern Fake Detection Rate: {modern_rate:.1f}% ({modern_correct}/{len(modern_files)} detected as fake)\n")

# ========================
# TEST REAL VOICE
# ========================
print("=== Your Real Voice ===")
# FIXED: Now accepts .aac, .wav, .m4a, .mp3
real_files = sorted([f for f in os.listdir(REAL_FOLDER) if f.endswith((".wav", ".aac", ".m4a", ".mp3"))])
real_correct = 0

for fname in real_files:
    path = os.path.join(REAL_FOLDER, fname)
    label, prob = predict(path)
    status = "REAL" if label == 0 else "FAKE (WRONG!)"
    print(f"  {fname}: {status} ({prob:.4f})")
    if label == 0:
        real_correct += 1

real_rate = real_correct / len(real_files) * 100
print(f"\nReal Voice Accuracy: {real_rate:.1f}% ({real_correct}/{len(real_files)} correctly identified as real)\n")

# ========================
# THE TABLE
# ========================
print("=" * 65)
print("STRESS TEST RESULTS TABLE")
print("=" * 65)
print(f"{'Test Group':<40} | {'Result'}")
print("-" * 65)
print(f"{'ASVspoof 2019 Eval (Baseline)':<40} | EER: 4.26%, Acc: 84.2%")
print(f"{'Edge TTS Neural (Modern Fakes)':<40} | Detected as fake: {modern_rate:.1f}%")
print(f"{'Your Real Voice (Out-of-Distribution)':<40} | Correctly real: {real_rate:.1f}%")
print("=" * 65)

# Save to file
with open(r"C:\Users\prana\OneDrive\Desktop\AI PROJECTS\DEEP FAKE AUDIO DETECTOR\results\stress_test_results.txt", "w") as f:
    f.write("Audio Deepfake Detector - Stress Test Results\n")
    f.write("=" * 55 + "\n\n")
    f.write("ASVspoof 2019 Eval (Baseline):\n")
    f.write("  EER: 4.26%\n")
    f.write("  Accuracy: 84.2%\n")
    f.write("  F1: 0.9046\n\n")
    f.write(f"Edge TTS Neural (Modern Synthetic Voices):\n")
    f.write(f"  Total clips: {len(modern_files)}\n")
    f.write(f"  Detected as fake: {modern_correct}/{len(modern_files)}\n")
    f.write(f"  Detection rate: {modern_rate:.1f}%\n\n")
    f.write(f"Self-Recorded Real Voice:\n")
    f.write(f"  Total clips: {len(real_files)}\n")
    f.write(f"  Correctly identified as real: {real_correct}/{len(real_files)}\n")
    f.write(f"  Accuracy: {real_rate:.1f}%\n")

print("\nResults saved to: results/stress_test_results.txt")