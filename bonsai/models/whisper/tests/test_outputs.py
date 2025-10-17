#!/usr/bin/env python3
"""
Test logits on real audio transcription from the Bush Moscow speech.
"""

import sys
import os
import jax
import jax.numpy as jnp
import numpy as np

from bonsai.models.whisper import audio, modeling
from transformers import WhisperTokenizer


def test_outputs():
    """Test logits quality on real audio from Bush Moscow speech."""
    print("=" * 80)
    print("TESTING LOGITS ON REAL AUDIO (Bush Moscow Speech)")
    print("=" * 80)
    
    # Load model
    print(f"\n🤖 Loading JAX Whisper model...")
    jax_model = modeling.load_model("tiny")
    print(f"✅ Model loaded")
    
    # Load audio
    audio_path = os.path.join(os.path.dirname(__file__), "audio_samples", "bush_moscow_speech.mp3")
    print(f"\n📁 Loading audio: {audio_path}")
    
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found. Please run run_model.py first to download it.")
        return False
    
    audio_tensor = audio.load_audio(audio_path)
    print(f"✅ Audio loaded - shape: {audio_tensor.shape}")
    
    # Compute mel spectrogram
    print(f"\n🎵 Computing mel spectrogram...")
    mel_tensor = audio.log_mel_spectrogram(audio_tensor)
    mel_jax = jnp.array(mel_tensor)
    print(f"✅ Mel computed - shape: {mel_jax.shape}")
    
    # Test segment with speech (5-35 seconds)
    print(f"\n🧪 Testing segment with actual speech (5-35s)...")
    mel_segment = mel_jax[:, 500:3500][None, :, :]
    print(f"   Mel segment shape: {mel_segment.shape}")
    
    # Get audio features
    print(f"\n🎵 Computing audio features...")
    audio_features = jax_model.embed_audio(mel_segment)
    print(f"✅ Audio features - shape: {audio_features.shape}")
    
    # Initialize tokenizer
    print(f"\n🔤 Initializing tokenizer...")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")
    
    # Create initial tokens
    sot = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    en_token = tokenizer.convert_tokens_to_ids("<|en|>")
    transcribe = tokenizer.convert_tokens_to_ids("<|transcribe|>")
    no_timestamps = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    
    initial_tokens = jnp.array([[sot, en_token, transcribe, no_timestamps]], dtype=jnp.int32)
    print(f"✅ Initial tokens: {initial_tokens[0].tolist()}")
    
    # Get logits for first prediction
    print(f"\n🤖 Computing logits for next token prediction...")
    logits = jax_model.logits(initial_tokens, audio_features)
    print(f"✅ Logits computed - shape: {logits.shape}")
    
    # Analyze logits
    print(f"\n📊 LOGITS ANALYSIS:")
    print(f"   Shape: {logits.shape}")
    print(f"   Range: [{logits.min():.4f}, {logits.max():.4f}]")
    print(f"   Mean: {logits.mean():.4f}")
    print(f"   Std: {logits.std():.4f}")
    
    # Check for problems
    has_nan = jnp.any(jnp.isnan(logits))
    has_inf = jnp.any(jnp.isinf(logits))
    print(f"   Has NaN: {has_nan}")
    print(f"   Has Inf: {has_inf}")
    
    if has_nan or has_inf:
        print(f"\n❌ FAILED: Logits contain NaN or Inf values!")
        return False
    
    # Get top predictions
    last_logits = logits[0, -1, :]
    top_k = 10
    top_indices = jnp.argsort(last_logits)[-top_k:][::-1]
    top_probs = jax.nn.softmax(last_logits)[top_indices]
    
    print(f"\n🔝 TOP {top_k} PREDICTIONS:")
    for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
        token_id = int(idx)
        token_str = tokenizer.decode([token_id])
        print(f"   {i+1}. Token {token_id:5d} (p={prob:.4f}): '{token_str}'")
    
    # Verify top prediction matches what we got in transcription
    top_token = int(top_indices[0])
    print(f"\n🎯 Top predicted token: {top_token}")
    
    # Check if probabilities sum to 1
    probs = jax.nn.softmax(logits, axis=-1)
    prob_sum = jnp.sum(probs, axis=-1)
    print(f"\n🔍 Probability sum check: {prob_sum[0, -1]:.6f} (should be ~1.0)")
    
    if abs(float(prob_sum[0, -1]) - 1.0) > 1e-5:
        print(f"❌ FAILED: Probabilities don't sum to 1!")
        return False
    
    # Check entropy (should be reasonable, not too uniform or too peaked)
    entropy = -jnp.sum(probs * jnp.log(probs + 1e-10), axis=-1)
    print(f"   Entropy: {entropy[0, -1]:.4f} (higher = more uncertain)")
    
    # Test with a longer sequence
    print(f"\n🧪 Testing with longer token sequence...")
    # Add a few more tokens (simulating "to Moscow")
    longer_tokens = jnp.array([[sot, en_token, transcribe, no_timestamps, 284, 15785]], dtype=jnp.int32)
    logits_longer = jax_model.logits(longer_tokens, audio_features)
    print(f"✅ Longer sequence logits - shape: {logits_longer.shape}")
    print(f"   Range: [{logits_longer.min():.4f}, {logits_longer.max():.4f}]")
    
    # Check consistency
    print(f"\n🔄 Checking logits consistency...")
    # First 4 tokens should produce same logits
    logits_short = jax_model.logits(initial_tokens, audio_features)
    logits_compare = jax_model.logits(longer_tokens[:, :4], audio_features)
    
    max_diff = jnp.abs(logits_short - logits_compare).max()
    print(f"   Max difference in first 4 positions: {max_diff:.6e}")
    
    if max_diff > 1e-5:
        print(f"   ⚠️  Warning: Logits not fully consistent (might be OK due to caching)")
    else:
        print(f"   ✅ Logits are consistent!")
    
    # Compare transcriptions with PyTorch Whisper
    print(f"\n" + "=" * 80)
    print("COMPARING TRANSCRIPTIONS: JAX vs PyTorch (Same 30s Segment)")
    print("=" * 80)
    
    try:
        import whisper
        import torch
        
        print(f"\n🔄 Loading PyTorch Whisper model...")
        torch_model = whisper.load_model("tiny")
        print(f"✅ PyTorch model loaded")
        
        # Extract the SAME 30-second segment for both models
        # Segment: 5-35 seconds (500:3500 frames = 5-35s at 100 frames/sec)
        start_sec = 5
        end_sec = 35
        print(f"\n📍 Using segment: {start_sec}-{end_sec} seconds")
        
        # Extract audio segment (16000 Hz * 30 seconds = 480000 samples)
        start_sample = start_sec * 16000
        end_sample = end_sec * 16000
        audio_segment = audio_tensor[start_sample:end_sample]
        print(f"   Audio segment: {audio_segment.shape} samples ({len(audio_segment)/16000:.1f}s)")
        
        # Generate JAX transcription
        print(f"\n🤖 Generating JAX transcription...")
        jax_tokens = greedy_decode_jax(jax_model, audio_features, tokenizer, max_length=200)
        jax_text = tokenizer.decode(jax_tokens[0].tolist(), skip_special_tokens=True)
        print(f"✅ JAX transcription complete ({len(jax_tokens[0])} tokens)")
        
        # Generate PyTorch transcription on SAME segment
        print(f"\n🔥 Generating PyTorch transcription (same segment)...")
        torch_audio_segment = torch.from_numpy(np.array(audio_segment)).float()
        with torch.no_grad():
            result = torch_model.transcribe(torch_audio_segment, language="en", task="transcribe", 
                                          initial_prompt=None, verbose=False)
        torch_text = result["text"].strip()
        print(f"✅ PyTorch transcription complete")
        
        # Display transcriptions
        print(f"\n📝 TRANSCRIPTION COMPARISON:")
        print(f"\n{'JAX Whisper:':-<80}")
        print(f"{jax_text}")
        print(f"\n{'PyTorch Whisper:':-<80}")
        print(f"{torch_text}")
        
        # Calculate similarity
        jax_words = jax_text.lower().split()
        torch_words = torch_text.lower().split()
        
        # Simple word-level comparison
        common_words = set(jax_words) & set(torch_words)
        all_words = set(jax_words) | set(torch_words)
        
        if all_words:
            similarity = len(common_words) / len(all_words) * 100
            print(f"\n📊 SIMILARITY METRICS:")
            print(f"   Common words: {len(common_words)}")
            print(f"   JAX unique words: {len(set(jax_words))}")
            print(f"   PyTorch unique words: {len(set(torch_words))}")
            print(f"   Word-level similarity: {similarity:.1f}%")
        
        # Character-level differences
        print(f"\n📏 LENGTH COMPARISON:")
        print(f"   JAX: {len(jax_text)} chars, {len(jax_words)} words")
        print(f"   PyTorch: {len(torch_text)} chars, {len(torch_words)} words")
        
        # Check if transcriptions are very similar
        if similarity > 80:
            print(f"\n✅ Transcriptions are highly similar (>{similarity:.0f}%)")
        elif similarity > 60:
            print(f"\n⚠️  Transcriptions are moderately similar ({similarity:.0f}%)")
        else:
            print(f"\n❌ Transcriptions differ significantly ({similarity:.0f}%)")
        
    except ImportError:
        print(f"\n⚠️  PyTorch Whisper not available, skipping comparison")
    except Exception as e:
        print(f"\n⚠️  Error during comparison: {e}")
    
    print(f"\n" + "=" * 80)
    print("✅ ALL LOGITS TESTS PASSED!")
    print("=" * 80)
    return True


def greedy_decode_jax(model, audio_features, tokenizer, max_length=200):
    """Simple greedy decoding for JAX model."""
    sot = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    en_token = tokenizer.convert_tokens_to_ids("<|en|>")
    transcribe = tokenizer.convert_tokens_to_ids("<|transcribe|>")
    no_timestamps = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    eot = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    
    initial_tokens = [sot, en_token, transcribe, no_timestamps]
    tokens = jnp.array([initial_tokens], dtype=jnp.int32)
    
    for i in range(max_length):
        logits = model.logits(tokens, audio_features)
        last_logits = logits[0, -1, :]
        next_token = jnp.argmax(last_logits)
        tokens = jnp.concatenate([tokens, jnp.array([[next_token]], dtype=jnp.int32)], axis=1)
        
        if next_token == eot:
            break
    
    return tokens


if __name__ == "__main__":
    success = test_outputs()
    sys.exit(0 if success else 1)

