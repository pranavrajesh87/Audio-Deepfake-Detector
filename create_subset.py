import os
import shutil
import random

# Paths
BASE = r"C:\Users\prana\OneDrive\Desktop\AI PROJECTS\DEEP FAKE AUDIO DETECTOR\data\LA"
SUBSET = r"C:\Users\prana\OneDrive\Desktop\AI PROJECTS\DEEP FAKE AUDIO DETECTOR\data_subset\LA"

# Folders to create
folders = {
    "train_audio": os.path.join(SUBSET, "ASVspoof2019_LA_train", "flac"),
    "dev_audio":   os.path.join(SUBSET, "ASVspoof2019_LA_dev", "flac"),
    "eval_audio":  os.path.join(SUBSET, "ASVspoof2019_LA_eval", "flac"),
    "protocols":   os.path.join(SUBSET, "ASVspoof2019_LA_cm_protocols"),
}

for f in folders.values():
    os.makedirs(f, exist_ok=True)

# Helper: parse protocol and write subset
def make_subset(protocol_path, audio_src, audio_dst, max_total, output_protocol_name):
    # Parse
    rows = []
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            rows.append(parts)
    
    # For train: balance real/fake. For dev/eval: just random sample.
    if "train" in protocol_path:
        real = [r for r in rows if r[4] == "bonafide"]
        fake = [r for r in rows if r[4] == "spoof"]
        # Take 1000 real + 1000 fake (or all real if less than 1000)
        n = min(1000, len(real))
        subset_rows = random.sample(real, n) + random.sample(fake, n)
    else:
        # Dev/Eval: random sample of max_total
        subset_rows = random.sample(rows, min(max_total, len(rows)))
    
    random.shuffle(subset_rows)
    
    # Write new protocol file
    out_protocol = os.path.join(folders["protocols"], output_protocol_name)
    with open(out_protocol, 'w') as f:
        for parts in subset_rows:
            f.write(" ".join(parts) + "\n")
    
    # Copy audio files
    copied = 0
    for parts in subset_rows:
        fname = parts[1] + ".flac"
        src = os.path.join(audio_src, fname)
        dst = os.path.join(audio_dst, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
        else:
            print(f"Missing: {fname}")
    
    print(f"{output_protocol_name}: {len(subset_rows)} rows, {copied} audio files copied.")
    return len(subset_rows)

# 1. Train subset: 1000 real + 1000 fake = 2000 clips
make_subset(
    os.path.join(BASE, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.train.trn.txt"),
    os.path.join(BASE, "ASVspoof2019_LA_train", "flac"),
    folders["train_audio"],
    2000,
    "ASVspoof2019.LA.cm.train.trn.txt"
)

# 2. Dev subset: 1000 random clips
make_subset(
    os.path.join(BASE, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.dev.trl.txt"),
    os.path.join(BASE, "ASVspoof2019_LA_dev", "flac"),
    folders["dev_audio"],
    1000,
    "ASVspoof2019.LA.cm.dev.trl.txt"
)

# 3. Eval subset: 1000 random clips (for final testing)
make_subset(
    os.path.join(BASE, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.eval.trl.txt"),
    os.path.join(BASE, "ASVspoof2019_LA_eval", "flac"),
    folders["eval_audio"],
    1000,
    "ASVspoof2019.LA.cm.eval.trl.txt"
)

print("\nDone! Now zip the 'data_subset' folder and upload to Google Drive.")