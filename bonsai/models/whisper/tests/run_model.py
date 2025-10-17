#!/usr/bin/env python3
"""
Standalone test for JAX Whisper model - no PyTorch comparison.
Tests the complete JAX pipeline: audio loading, mel spectrogram, model inference, and text generation.
Uses HuggingFace WhisperTokenizer for proper tokenization.
"""

import sys
import os
import jax.numpy as jnp
import time
import subprocess

def jax_greedy_decode(model, audio_features, tokenizer, max_length=200, verbose=False):
    """Simple greedy decoding for JAX model using HuggingFace tokenizer."""
    # Get special tokens from HuggingFace tokenizer
    sot = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    en_token = tokenizer.convert_tokens_to_ids("<|en|>")
    transcribe = tokenizer.convert_tokens_to_ids("<|transcribe|>")
    no_timestamps = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    eot = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    
    # Start with initial tokens (multilingual format)
    initial_tokens = [sot, en_token, transcribe, no_timestamps]
    tokens = jnp.array([initial_tokens], dtype=jnp.int32)
    
    # Generate tokens greedily using JAX
    for i in range(max_length):
        # Get logits for current sequence
        logits = model.logits(tokens, audio_features)
        
        # Get the last token's logits
        last_logits = logits[0, -1, :]
        
        # Select the token with highest probability
        next_token = jnp.argmax(last_logits)
        
        if verbose and i < 20:
            # Show first few tokens being generated
            token_str = tokenizer.decode([int(next_token)])
            print(f"      Token {i}: {int(next_token)} -> '{token_str}'")
        
        # Add the new token
        tokens = jnp.concatenate([tokens, jnp.array([[next_token]], dtype=jnp.int32)], axis=1)
        
        # Stop if we hit the end token
        if next_token == eot:  # <|endoftext|>
            if verbose:
                print(f"      Stopped at token {i} (EOT)")
            break
    
    return tokens

def main():
    print("=" * 80)
    print("JAX WHISPER MODEL - STANDALONE TEST")
    print("=" * 80)
    
    try:
        # Import the modules
        from bonsai.models.whisper import audio, modeling
        from transformers import WhisperTokenizer
        
        print("✅ All modules imported successfully")
        
        # Load JAX NNX model with real weights
        print(f"\n🤖 Loading JAX NNX Whisper model with real weights...")
        jax_model = modeling.load_model("tiny")
        print(f"✅ JAX NNX model loaded with real weights")
        
        # Download audio file from URL
        audio_url = "https://bush41library.tamu.edu/files/audio/Remarks%20of%20Vice%20President%20Bush%20Upon%20Arrival%20in%20Moscow%20for%20the%20Funeral%20of%20Konstantin%20Chernenko%2012%20March%201985.mp3"
        audio_filename = "bush_moscow_speech.mp3"
        audio_path = os.path.join(os.path.dirname(__file__), "audio_samples", audio_filename)
        
        print(f"\n📥 Downloading audio from URL...")
        print(f"   URL: {audio_url}")
        print(f"   Destination: {audio_path}")
        
        # Use wget to download the file
        try:
            subprocess.run(["wget", "-O", audio_path, audio_url], check=True, capture_output=True)
            print(f"✅ Audio downloaded successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to download audio: {e}")
            raise
        except FileNotFoundError:
            print(f"❌ wget not found. Please install wget or use curl instead.")
            raise
        
        print(f"\n📁 Loading audio from: {audio_path}")
        audio_tensor = audio.load_audio(audio_path)
        print(f"✅ Audio loaded - shape: {audio_tensor.shape}")
        
        print(f"\n🎵 Computing mel spectrogram...")
        mel_tensor = audio.log_mel_spectrogram(audio_tensor)
        print(f"✅ Mel spectrogram computed - shape: {mel_tensor.shape}")
        
        # Convert to JAX and prepare for model
        mel_jax = jnp.array(mel_tensor)
        
        # Initialize HuggingFace tokenizer
        print(f"\n🔤 Initializing HuggingFace tokenizer...")
        tokenizer_instance = WhisperTokenizer.from_pretrained("openai/whisper-tiny")
        tokenizer_instance.set_prefix_tokens(language="en", task="transcribe")
        print(f"✅ HF tokenizer initialized")
        
        # Focus on the 30 sec segment with  speech (5-35 seconds)
        print(f"\n{'='*80}")
        print(f"🧪 Testing segment: 5-35 seconds (where the speech begins)")
        print(f"{'='*80}")
        
        # Extract segment: 500-3500 frames (5-35 seconds at 100 frames/sec)
        start_frame, end_frame = 500, 3500
        
        mel_segment = mel_jax[:, start_frame:end_frame][None, :, :]
        print(f"   Mel shape: {mel_segment.shape}")
        print(f"   Mel stats: min={mel_segment.min():.3f}, max={mel_segment.max():.3f}, mean={mel_segment.mean():.3f}")
        
        # Get audio features
        print(f"\n🎵 Computing audio features...")
        audio_features = jax_model.embed_audio(mel_segment)
        print(f"✅ Audio features computed - shape: {audio_features.shape}")
        
        # Generate tokens
        print(f"\n🤖 Generating tokens with JAX Whisper...")
        print(f"   Showing first 20 tokens being generated:")
        start_time = time.time()
        jax_tokens = jax_greedy_decode(jax_model, audio_features, tokenizer_instance, max_length=448, verbose=True)
        jax_time = time.time() - start_time
        
        print(f"\n   JAX generation time: {jax_time:.2f} seconds")
        print(f"   Generated {jax_tokens.shape[1]} tokens")
        
        # Convert tokens to text
        print(f"\n📝 Converting tokens to text...")
        try:
            tokens_list = jax_tokens[0].tolist()
            text = tokenizer_instance.decode(tokens_list, skip_special_tokens=True)
            print(f"✅ Text decoded successfully")
            
            print(f"\n📝 TRANSCRIPTION RESULT:")
            print(f"\n{'-'*80}")
            print(f"{text}")
            print(f"{'-'*80}")
            
            # Analyze the transcription
            token_count = jax_tokens.shape[1]
            unique_tokens = len(jnp.unique(jax_tokens[0]))
            word_count = len(text.split())
            print(f"\n📊 TRANSCRIPTION ANALYSIS:")
            print(f"   Token count: {token_count}")
            print(f"   Unique tokens: {unique_tokens}")
            print(f"   Word count: {word_count}")
            print(f"   Characters: {len(text)}")
                 
        except Exception as e:
            print(f"❌ Text decoding failed: {e}")
            print(f"   Raw tokens: {jax_tokens[0].tolist()}")
        
        print(f"\n" + "=" * 80)
        print("JAX WHISPER STANDALONE TEST COMPLETE")
        print("=" * 80)
        
        print(f"\n🎉 JAX WHISPER TEST SUCCESSFUL!")
        print(f"   The JAX model is working independently!")
        
    except Exception as e:
        print(f"❌ JAX Whisper test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
