#!/usr/bin/env python3
"""Download CC0 speech samples for testing."""

import os
import urllib.request
from pathlib import Path

# CC0 speech samples from Wikimedia Commons
AUDIO_SAMPLES = {
    "bush_speech": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/8/8c/George_W._Bush_Radio_Address_%28January_27%2C_2001%29.ogg",
        "description": "George W. Bush Radio Address (January 27, 2001) - CC0 licensed"
    },
    "bush_short": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/1/1f/George_W._Bush_Speech_-_September_11%2C_2001.ogg",
        "description": "George W. Bush Speech - September 11, 2001 - CC0 licensed"
    }
}

def download_audio_samples():
    """Download CC0 audio samples."""
    audio_dir = Path("bonsai/models/whisper/tests/audio_samples")
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading CC0 speech samples from Wikimedia Commons...")
    
    for name, info in AUDIO_SAMPLES.items():
        output_path = audio_dir / f"{name}.ogg"
        
        if output_path.exists():
            print(f"  {name}.ogg already exists, skipping...")
            continue
            
        print(f"  Downloading {name}.ogg...")
        print(f"    Source: {info['description']}")
        try:
            urllib.request.urlretrieve(info["url"], output_path)
            print(f"    ✓ Downloaded {name}.ogg")
        except Exception as e:
            print(f"    ✗ Failed to download {name}.ogg: {e}")
    
    print("\nSpeech samples ready for testing!")

if __name__ == "__main__":
    download_audio_samples()
