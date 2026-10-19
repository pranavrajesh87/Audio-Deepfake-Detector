import os
import subprocess
from imageio_ffmpeg import get_ffmpeg_exe

# Folder with your .m4a files
input_folder = r"C:\Users\prana\OneDrive\Desktop\AI PROJECTS\DEEP FAKE AUDIO DETECTOR\results\clean_recordings"

# Get the bundled ffmpeg path (no system install needed)
ffmpeg_path = get_ffmpeg_exe()
print(f"Using ffmpeg: {ffmpeg_path}")

# Find all .m4a files
m4a_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".m4a")]

if not m4a_files:
    print("No .m4a files found. Check the folder path.")
else:
    print(f"Found {len(m4a_files)} files to convert...\n")
    
    for fname in m4a_files:
        input_path = os.path.join(input_folder, fname)
        output_path = os.path.join(input_folder, fname.replace(".m4a", ".wav"))
        
        # Run ffmpeg: convert m4a to wav, 16kHz, mono
        cmd = [
            ffmpeg_path,
            "-i", input_path,      # input
            "-ar", "16000",         # sample rate 16kHz
            "-ac", "1",             # mono
            "-y",                   # overwrite if exists
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Converted: {fname} → {os.path.basename(output_path)}")
        else:
            print(f"✗ Failed: {fname}")
            print(result.stderr[:200])  # Show first 200 chars of error

print("\nDone! Check your clean_recordings folder for .wav files.")