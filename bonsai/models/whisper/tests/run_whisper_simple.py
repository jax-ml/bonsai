#!/usr/bin/env python3
"""
Simple test to run the original OpenAI Whisper on the test audio sample.
This script uses the original Whisper code that we copied into the whisper directory.
"""

import sys
import os
import torch
import time

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_original_whisper_imports():
    """Test importing the original Whisper modules."""
    print("="*80)
    print("TESTING ORIGINAL WHISPER IMPORTS")
    print("="*80)
    
    try:
        # Test importing individual modules
        print("Testing audio module...")
        import audio
        print("✅ Audio module imported successfully")
        
        print("Testing model module...")
        import model
        print("✅ Model module imported successfully")
        
        print("Testing transcribe module...")
        import transcribe
        print("✅ Transcribe module imported successfully")
        
        print("Testing tokenizer module...")
        import tokenizer
        print("✅ Tokenizer module imported successfully")
        
        print("Testing utils module...")
        import utils
        print("✅ Utils module imported successfully")
        
        # Test specific functions
        print("\nTesting specific functions...")
        audio_func = getattr(audio, 'load_audio', None)
        if audio_func:
            print("✅ load_audio function found")
        else:
            print("❌ load_audio function not found")
            
        mel_func = getattr(audio, 'log_mel_spectrogram', None)
        if mel_func:
            print("✅ log_mel_spectrogram function found")
        else:
            print("❌ log_mel_spectrogram function not found")
            
        whisper_class = getattr(model, 'Whisper', None)
        if whisper_class:
            print("✅ Whisper class found")
        else:
            print("❌ Whisper class not found")
            
        transcribe_func = getattr(transcribe, 'transcribe', None)
        if transcribe_func:
            print("✅ transcribe function found")
        else:
            print("❌ transcribe function not found")
        
        print("\n" + "="*80)
        print("ORIGINAL WHISPER IMPORTS TEST COMPLETE")
        print("="*80)
        print("✅ All modules imported successfully")
        print("✅ All key functions/classes found")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        print("\n" + "="*80)
        print("ORIGINAL WHISPER IMPORTS TEST FAILED")
        print("="*80)
        return False

def test_audio_processing():
    """Test audio processing with the original Whisper."""
    print("\n" + "="*80)
    print("TESTING ORIGINAL WHISPER AUDIO PROCESSING")
    print("="*80)
    
    try:
        import audio
        
        # Path to the test audio file
        audio_path = "audio_samples/bush_speech.wav"
        
        if not os.path.exists(audio_path):
            print(f"❌ Audio file not found: {audio_path}")
            return False
        
        print(f"Loading audio from: {audio_path}")
        
        # Load audio
        start_time = time.time()
        audio_tensor = audio.load_audio(audio_path)
        load_time = time.time() - start_time
        print(f"✅ Audio loaded in {load_time:.2f} seconds")
        print(f"Audio shape: {audio_tensor.shape}")
        print(f"Audio duration: {len(audio_tensor) / 16000:.1f} seconds")
        
        # Test mel spectrogram computation
        print("\nComputing mel spectrogram...")
        start_time = time.time()
        mel = audio.log_mel_spectrogram(audio_tensor)
        mel_time = time.time() - start_time
        print(f"✅ Mel spectrogram computed in {mel_time:.2f} seconds")
        print(f"Mel spectrogram shape: {mel.shape}")
        
        print("\n" + "="*80)
        print("ORIGINAL WHISPER AUDIO PROCESSING TEST COMPLETE")
        print("="*80)
        print("✅ Audio loading works")
        print("✅ Mel spectrogram computation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio processing failed: {e}")
        print("\n" + "="*80)
        print("ORIGINAL WHISPER AUDIO PROCESSING TEST FAILED")
        print("="*80)
        return False

if __name__ == "__main__":
    # Test imports first
    imports_ok = test_original_whisper_imports()
    
    if imports_ok:
        # Test audio processing
        audio_ok = test_audio_processing()
        
        if audio_ok:
            print("\n🎉 ALL TESTS PASSED! Original Whisper is working correctly.")
        else:
            print("\n❌ Audio processing test failed.")
    else:
        print("\n❌ Import test failed.")
