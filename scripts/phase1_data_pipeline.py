# phase1_data.py - COMPLETE Phase 1: Data Pipeline

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np

# ========================
# 1. CONFIGURATION
# ========================

# Paths to your data
BASE_PATH = r"C:\Users\prana\OneDrive\Desktop\AI PROJECTS\DEEP FAKE AUDIO DETECTOR\data\LA"

# Protocol files (the label lists)
TRAIN_PROTOCOL = os.path.join(BASE_PATH, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.train.trn.txt")
DEV_PROTOCOL   = os.path.join(BASE_PATH, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.dev.trl.txt")
EVAL_PROTOCOL  = os.path.join(BASE_PATH, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.eval.trl.txt")

# Audio folders
TRAIN_AUDIO = os.path.join(BASE_PATH, "ASVspoof2019_LA_train", "flac")
DEV_AUDIO   = os.path.join(BASE_PATH, "ASVspoof2019_LA_dev", "flac")
EVAL_AUDIO  = os.path.join(BASE_PATH, "ASVspoof2019_LA_eval", "flac")

# Audio settings
SAMPLE_RATE = 16000      # Wav2Vec2 expects 16,000 samples per second
DURATION = 4             # We want exactly 4 seconds per clip
MAX_SAMPLES = SAMPLE_RATE * DURATION  # 16,000 * 4 = 64,000 numbers per clip


# ========================
# 2. PARSE PROTOCOL FILES
# ========================

def parse_protocol(protocol_path):
    """
    Reads a protocol file and returns a list of dictionaries.
    Each dictionary = one audio clip with its label and info.
    """
    data = []
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            # Format: SPEAKER_ID  FILE_NAME  -  SYSTEM_ID  LABEL
            data.append({
                "speaker_id": parts[0],
                "file_name": parts[1],
                "system_id": parts[3],   # A01-A19 for fake, '-' for real
                "label": parts[4],       # 'bonafide' or 'spoof'
                "audio_path": None       # We will fill this in later
            })
    return data

# Parse all three sets
train_data = parse_protocol(TRAIN_PROTOCOL)
dev_data = parse_protocol(DEV_PROTOCOL)
eval_data = parse_protocol(EVAL_PROTOCOL)

print(f"Train set: {len(train_data)} clips")
print(f"Dev set:   {len(dev_data)} clips")
print(f"Eval set:  {len(eval_data)} clips")


# ========================
# 3. BALANCE THE TRAINING SET
# ========================

# Separate real and fake in training data
train_real = [item for item in train_data if item["label"] == "bonafide"]
train_fake = [item for item in train_data if item["label"] == "spoof"]

print(f"\nBefore balancing:")
print(f"  Real:  {len(train_real)}")
print(f"  Fake:  {len(train_fake)}")

# Take ALL real clips, and randomly pick the same number of fake clips
random.seed(42)  # For reproducibility (same random picks every time)
train_fake_subset = random.sample(train_fake, len(train_real))

# Combine them back together
train_data_balanced = train_real + train_fake_subset
random.shuffle(train_data_balanced)  # Mix them up so real and fake are not grouped

print(f"\nAfter balancing:")
print(f"  Total train clips: {len(train_data_balanced)}")
print(f"  Real: {sum(1 for x in train_data_balanced if x['label'] == 'bonafide')}")
print(f"  Fake: {sum(1 for x in train_data_balanced if x['label'] == 'spoof')}")


# ========================
# 4. ADD AUDIO PATHS
# ========================

# Tell each item where its actual .flac file lives
for item in train_data_balanced:
    item["audio_path"] = os.path.join(TRAIN_AUDIO, item["file_name"] + ".flac")

for item in dev_data:
    item["audio_path"] = os.path.join(DEV_AUDIO, item["file_name"] + ".flac")

for item in eval_data:
    item["audio_path"] = os.path.join(EVAL_AUDIO, item["file_name"] + ".flac")


# ========================
# 5. LOAD AND STANDARDIZE AUDIO
# ========================

def load_audio(audio_path):
    """
    Loads a .flac file and forces it to be exactly MAX_SAMPLES long.
    - If too short: pad with zeros (silence) at the end
    - If too long: cut off the end
    Returns a PyTorch tensor.
    """
    # Load the audio file (sr=None keeps original sample rate, which is already 16kHz)
    audio, sr = librosa.load(audio_path, sr=None)
    
    # Safety check: if sample rate is wrong, resample it
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    
    # Create an empty array of the correct size, filled with zeros (silence)
    standardized = np.zeros(MAX_SAMPLES, dtype=np.float32)
    
    # Copy the actual audio into it
    if len(audio) > MAX_SAMPLES:
        # Too long: keep only the first MAX_SAMPLES numbers
        standardized[:] = audio[:MAX_SAMPLES]
    else:
        # Too short: copy what we have, rest stays zero (silence)
        standardized[:len(audio)] = audio
    
    # Convert to PyTorch tensor and add a "channel" dimension
    # Shape becomes: (1, 64000) — the 1 is because audio is mono (not stereo)
    tensor = torch.from_numpy(standardized).unsqueeze(0)
    
    return tensor


# ========================
# 6. PYTORCH DATASET CLASS
# ========================

class AudioDeepfakeDataset(Dataset):
    """
    A PyTorch Dataset knows how to:
    1. Say "I have N items"
    2. Return item i as (audio_tensor, label)
    """
    
    def __init__(self, data_list):
        """
        data_list: list of dictionaries from parse_protocol()
        """
        self.data = data_list
        # Convert text labels to numbers: bonafide=0, spoof=1
        self.label_map = {"bonafide": 0, "spoof": 1}
    
    def __len__(self):
        """How many clips do we have?"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns one training example:
        - audio_tensor: shape (1, 64000)
        - label: 0 (real) or 1 (fake)
        """
        item = self.data[idx]
        
        # Load and standardize the audio
        audio_tensor = load_audio(item["audio_path"])
        
        # Convert label text to number
        label = self.label_map[item["label"]]
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return audio_tensor, label_tensor


# ========================
# 7. CREATE DATASETS AND DATALOADERS
# ========================

# Create Dataset objects
train_dataset = AudioDeepfakeDataset(train_data_balanced)
dev_dataset = AudioDeepfakeDataset(dev_data)
eval_dataset = AudioDeepfakeDataset(eval_data)

print(f"\nDataset sizes:")
print(f"  Train: {len(train_dataset)}")
print(f"  Dev:   {len(dev_dataset)}")
print(f"  Eval:  {len(eval_dataset)}")

# Create DataLoaders (they batch multiple clips together)
# batch_size=4 means we process 4 clips at a time (small, fits in 4GB VRAM)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=4, shuffle=False)

# ========================
# 8. TEST: Grab one batch and inspect
# ========================

print(f"\n=== Testing one batch from train_loader ===")
for batch_audio, batch_labels in train_loader:
    print(f"Batch audio shape: {batch_audio.shape}")
    print(f"  -> This means: {batch_audio.shape[0]} clips, {batch_audio.shape[1]} channel, {batch_audio.shape[2]} samples")
    print(f"Batch labels: {batch_labels}")
    print(f"  -> 0 = real, 1 = fake")
    print(f"Label counts in this batch: {batch_labels.sum()} fake, {len(batch_labels) - batch_labels.sum()} real")
    break  # Only show the first batch

print("\n=== Phase 1 Complete ===")
print("Data is ready. We can now move to Phase 2: Training the model.")