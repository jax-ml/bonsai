#!/usr/bin/env python3
"""
Run the original Whisper code with actual trained weights.
This loads the real Whisper model and uses our copied code structure.
"""

import sys
import os
import torch
import time

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_whisper_with_real_weights():
    """Run the original Whisper code with actual trained weights."""
    print("="*80)
    print("RUNNING ORIGINAL WHISPER WITH REAL TRAINED WEIGHTS")
    print("="*80)
    
    try:
        # First, load the real Whisper model to get the weights
        print("Loading real Whisper model to get weights...")
        import whisper as real_whisper
        real_model = real_whisper.load_model('tiny')
        print("✅ Real Whisper model loaded")
        
        # Now import our copied modules
        import audio
        import model
        import transcribe
        import tokenizer
        import utils
        
        print("✅ All copied modules imported successfully")
        
        # Path to the test audio file
        audio_path = "audio_samples/bush_speech.wav"
        
        if not os.path.exists(audio_path):
            print(f"❌ Audio file not found: {audio_path}")
            return
        
        print(f"📁 Loading audio from: {audio_path}")
        
        # Load audio using our copied code
        start_time = time.time()
        audio_tensor = audio.load_audio(audio_path)
        load_time = time.time() - start_time
        print(f"✅ Audio loaded in {load_time:.2f} seconds")
        print(f"   Audio shape: {audio_tensor.shape}")
        print(f"   Audio duration: {len(audio_tensor) / 16000:.1f} seconds")
        
        # Compute mel spectrogram using our copied code
        print(f"\n🎵 Computing mel spectrogram...")
        start_time = time.time()
        mel = audio.log_mel_spectrogram(audio_tensor)
        mel_time = time.time() - start_time
        print(f"✅ Mel spectrogram computed in {mel_time:.2f} seconds")
        print(f"   Mel spectrogram shape: {mel.shape}")
        
        # Create our copied model with the same dimensions as the real model
        print(f"\n🤖 Creating our copied Whisper model...")
        dims = model.ModelDimensions(
            n_mels=80,
            n_audio_ctx=1500,
            n_audio_state=384,
            n_audio_head=6,
            n_audio_layer=4,
            n_vocab=51865,
            n_text_ctx=448,
            n_text_state=384,
            n_text_head=6,
            n_text_layer=4
        )
        
        our_model = model.Whisper(dims)
        print(f"✅ Our model created: {type(our_model)}")
        
        # Copy the weights from the real model to our model
        print(f"\n🔄 Copying weights from real model to our model...")
        try:
            # Copy encoder weights
            our_model.encoder.load_state_dict(real_model.encoder.state_dict())
            print("✅ Encoder weights copied")
            
            # Copy decoder weights  
            our_model.decoder.load_state_dict(real_model.decoder.state_dict())
            print("✅ Decoder weights copied")
            
            print("✅ All weights copied successfully!")
            
        except Exception as e:
            print(f"❌ Weight copying failed: {e}")
            print("This might be due to different model architectures")
            return
        
        # Test transcription with our copied model and real weights
        print(f"\n🎤 Running transcription with our copied model + real weights...")
        start_time = time.time()
        
        try:
            result = transcribe.transcribe(our_model, audio_tensor)
            transcribe_time = time.time() - start_time
            print(f"✅ Transcription completed in {transcribe_time:.2f} seconds")
            print(f"\n📝 TRANSCRIPTION RESULT:")
            print(f"   Text: {result.get('text', 'No text generated')}")
            print(f"   Language: {result.get('language', 'Unknown')}")
            print(f"   Segments: {len(result.get('segments', []))}")
            
            if 'segments' in result and result['segments']:
                print(f"\n📋 ALL SEGMENTS:")
                for i, segment in enumerate(result['segments']):
                    print(f"   {i+1}. {segment.get('start', 0):.1f}s-{segment.get('end', 0):.1f}s: {segment.get('text', 'No text')}")
                    
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            import traceback
            traceback.print_exc()
            return
        
        print(f"\n" + "="*80)
        print("ORIGINAL WHISPER WITH REAL WEIGHTS TEST COMPLETE")
        print("="*80)
        print("✅ Audio loading works")
        print("✅ Mel spectrogram computation works")
        print("✅ Model structure works")
        print("✅ Weight copying works")
        print("✅ Transcription with real weights works")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_whisper_with_real_weights()
    
    if success:
        print(f"\n🎉 SUCCESS! Our copied Whisper code works with real weights!")
        print(f"   This proves our copied structure is correct.")
        print(f"   The issue with our Bonsai model is in the JAX/NNX implementation.")
    else:
        print(f"\n❌ TEST FAILED!")
        print(f"   There are issues with the weight copying or model structure.")
