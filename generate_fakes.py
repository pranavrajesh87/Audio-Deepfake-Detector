# generate_fakes.py — Create modern synthetic voices using Edge TTS (neural voices)

import edge_tts
import asyncio
import os

output_dir = r"C:\Users\prana\OneDrive\Desktop\AI PROJECTS\DEEP FAKE AUDIO DETECTOR\results\edge_tts_fakes"
os.makedirs(output_dir, exist_ok=True)

# Modern neural voices from Microsoft Edge
# "en-US-AriaNeural" is a high-quality female neural voice
# "en-US-GuyNeural" is a high-quality male neural voice
VOICES = ["en-US-AriaNeural", "en-US-GuyNeural"]

sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming how we detect synthetic media.",
    "Please verify your identity by repeating the following phrase.",
    "Artificial intelligence can now generate realistic human voices.",
    "The weather today is sunny with a slight chance of rain.",
    "I need to schedule a meeting for tomorrow afternoon.",
    "Security systems must adapt to new threats continuously.",
    "This is a test of the emergency broadcast system.",
    "Voice cloning technology has improved dramatically in recent years.",
    "Can you hear me clearly through this connection?",
    "The project deadline is approaching faster than expected.",
    "Deep learning models require large amounts of training data.",
    "Please confirm your appointment by pressing one.",
    "The meeting will reconvene after a short break.",
    "Authentication systems rely on biometric voice patterns.",
    "I am calling regarding your recent application.",
    "The package has been delivered to your front door.",
    "Thank you for choosing our service today.",
    "Your verification code is four two eight one.",
    "This call may be recorded for quality assurance purposes."
]

async def generate():
    print(f"Generating {len(sentences)} modern neural TTS clips...")
    
    for i, sentence in enumerate(sentences):
        voice = VOICES[i % 2]  # Alternate between male and female voices
        output_path = os.path.join(output_dir, f"edge_tts_fake_{i+1:02d}.wav")
        
        communicate = edge_tts.Communicate(sentence, voice)
        await communicate.save(output_path)
        
        print(f"  Done: edge_tts_fake_{i+1:02d}.wav (voice: {voice})")
    
    print(f"\nAll clips saved to: {output_dir}")

# Run the async function
asyncio.run(generate())