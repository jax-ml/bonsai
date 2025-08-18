#!/usr/bin/env python3
"""Generate speech audio for Bush's speech text."""

import os
from pathlib import Path

# Text from the first paragraph of Bush's speech
BUSH_SPEECH_TEXT = """
Ladies and gentlemen, this is a difficult moment for America. 
Unfortunately, we'll be going back to Washington after my remarks. 
Secretary of Ride Pays and Lieutenant Governor will take the podium and discuss education. 
I do want to thank the folks here at the Bucket Elementary School for their hospitality. 
Today, we've had an national tragedy. Two airplanes have crashed into the world trade center.
"""

def generate_speech_audio():
    """Generate speech audio using text-to-speech."""
    output_dir = Path("bonsai/models/whisper/tests/audio_samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "generated_bush_speech.wav"
    
    print("=== Generating Speech Audio ===")
    print(f"Text to synthesize: {BUSH_SPEECH_TEXT.strip()}")
    print(f"Output file: {output_file}")
    
    try:
        # Try using gTTS (Google Text-to-Speech)
        from gtts import gTTS
        
        print("Using Google Text-to-Speech...")
        tts = gTTS(text=BUSH_SPEECH_TEXT, lang='en', slow=False)
        tts.save(str(output_file))
        print(f"✅ Audio saved to {output_file}")
        
    except ImportError:
        print("gTTS not available, trying alternative methods...")
        
        try:
            # Try using pyttsx3 (offline TTS)
            import pyttsx3
            
            print("Using pyttsx3...")
            engine = pyttsx3.init()
            engine.save_to_file(BUSH_SPEECH_TEXT, str(output_file))
            engine.runAndWait()
            print(f"✅ Audio saved to {output_file}")
            
        except ImportError:
            print("pyttsx3 not available, trying espeak...")
            
            try:
                # Try using espeak command line
                import subprocess
                
                print("Using espeak...")
                cmd = [
                    "espeak", 
                    "-w", str(output_file),
                    "-s", "150",  # Speed
                    "-p", "50",   # Pitch
                    "-a", "100",  # Amplitude
                    BUSH_SPEECH_TEXT
                ]
                subprocess.run(cmd, check=True)
                print(f"✅ Audio saved to {output_file}")
                
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("❌ No text-to-speech method available")
                print("Please install one of: gtts, pyttsx3, or espeak")
                return None
    
    return output_file


if __name__ == "__main__":
    audio_file = generate_speech_audio()
    if audio_file and audio_file.exists():
        print(f"\n✅ Successfully generated speech audio: {audio_file}")
        print("You can now test the Whisper model on this generated audio.")
    else:
        print("\n❌ Failed to generate speech audio")
