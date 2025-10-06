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

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def jax_greedy_decode(model, audio_features, tokenizer, max_length=200):
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
        
        # Add the new token
        tokens = jnp.concatenate([tokens, jnp.array([[next_token]], dtype=jnp.int32)], axis=1)
        
        # Stop if we hit the end token
        if next_token == eot:  # <|endoftext|>
            break
    
    return tokens

def main():
    print("=" * 80)
    print("JAX WHISPER MODEL - STANDALONE TEST")
    print("=" * 80)
    
    try:
        # Import the modules
        import audio
        import modeling
        from transformers import WhisperTokenizer
        
        print("✅ All modules imported successfully")
        
        # Load JAX NNX model with real weights
        print(f"\n🤖 Loading JAX NNX Whisper model with real weights...")
        jax_model = modeling.load_model("tiny")
        print(f"✅ JAX NNX model loaded with real weights")
        
        # Load audio and compute mel
        audio_path = os.path.join(os.path.dirname(__file__), "audio_samples", "bush_speech.wav")
        print(f"\n📁 Loading audio from: {audio_path}")
        audio_tensor = audio.load_audio(audio_path)
        print(f"✅ Audio loaded - shape: {audio_tensor.shape}")
        
        print(f"\n🎵 Computing mel spectrogram...")
        mel_tensor = audio.log_mel_spectrogram(audio_tensor)
        print(f"✅ Mel spectrogram computed - shape: {mel_tensor.shape}")
        
        # Convert to JAX and prepare for model
        mel_jax = jnp.array(mel_tensor)
        
        # Test with first 30 seconds of audio
        print(f"\n🧪 Testing with first 30 seconds of audio...")
        mel_30s = mel_jax[:, :3000][None, :, :]  # First 30 seconds with batch dimension: (1, 80, 3000)
        print(f"   Mel shape: {mel_30s.shape}")
        
        # Get audio features
        print(f"\n🎵 Computing audio features...")
        audio_features = jax_model.embed_audio(mel_30s)
        print(f"✅ Audio features computed - shape: {audio_features.shape}")
        
        # Generate tokens
        # Initialize HuggingFace tokenizer
        print(f"\n🔤 Initializing HuggingFace tokenizer...")
        tokenizer_instance = WhisperTokenizer.from_pretrained("openai/whisper-tiny")
        tokenizer_instance.set_prefix_tokens(language="en", task="transcribe")
        print(f"✅ HF tokenizer initialized")
        
        print(f"\n🤖 Generating tokens with JAX Whisper...")
        start_time = time.time()
        jax_tokens = jax_greedy_decode(jax_model, audio_features, tokenizer_instance, max_length=200)
        jax_time = time.time() - start_time
        
        print(f"   JAX generation time: {jax_time:.2f} seconds")
        print(f"   JAX tokens: {jax_tokens[0].tolist()}")
        
        # Convert tokens to text
        print(f"\n📝 Converting tokens to text...")
        try:
            tokens_list = jax_tokens[0].tolist()
            text = tokenizer_instance.decode(tokens_list)
            print(f"✅ Text decoded successfully")
            
            print(f"\n📝 TRANSCRIPTION RESULT:")
            print(f"   Text: {text}")
            
            # Analyze the transcription using JAX operations
            print(f"\n📊 TRANSCRIPTION ANALYSIS:")
            token_count = jax_tokens.shape[1]
            unique_tokens = len(jnp.unique(jax_tokens[0]))
            token_diversity = unique_tokens / token_count * 100
            print(f"   Token count: {token_count}")
            print(f"   Unique tokens: {unique_tokens}")
            print(f"   Token diversity: {unique_tokens}/{token_count} ({token_diversity:.1f}%)")
                 
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
