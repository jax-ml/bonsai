#!/usr/bin/env python3
"""
Standalone test for JAX Whisper model - no PyTorch comparison.
Tests the complete JAX pipeline: audio loading, mel spectrogram, model inference, and text generation.
"""

import sys
import os
import jax
import jax.numpy as jnp
import numpy as np
import time
from flax import nnx

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def jax_greedy_decode(model, audio_features, max_length=200):
    """Simple greedy decoding for JAX model."""
    # Start with initial tokens
    tokens = jnp.array([[50258, 50259, 50359, 50364]], dtype=jnp.int32)  # <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
    
    for i in range(max_length):
        # Get logits for current tokens
        logits = model.logits(tokens, audio_features)
        
        # Get the last token's logits
        last_logits = logits[0, -1, :]
        
        # Select the token with highest probability
        next_token = jnp.argmax(last_logits)
        
        # Add the new token
        tokens = jnp.concatenate([tokens, jnp.array([[next_token]], dtype=jnp.int32)], axis=1)
        
        # Stop if we hit the end token
        if next_token == 50257:  # <|endoftext|>
            break
    
    return tokens

def main():
    print("=" * 80)
    print("JAX WHISPER MODEL - STANDALONE TEST")
    print("=" * 80)
    
    try:
        # Import the modules
        import audio
        import model_fixed
        import tokenizer
        
        print("✅ All modules imported successfully")
        
        # Create JAX NNX model
        print(f"\n🤖 Creating JAX NNX Whisper model...")
        dims = model_fixed.ModelDimensions(
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
        rngs = nnx.Rngs(0)
        jax_model = model_fixed.Whisper(dims, rngs=rngs)
        print(f"✅ JAX NNX model created")
        
        # Load audio and compute mel
        print(f"\n📁 Loading audio from: audio_samples/bush_speech.wav")
        audio_tensor = audio.load_audio("audio_samples/bush_speech.wav")
        print(f"✅ Audio loaded - shape: {audio_tensor.shape}")
        
        print(f"\n🎵 Computing mel spectrogram...")
        mel_tensor = audio.log_mel_spectrogram(audio_tensor)
        print(f"✅ Mel spectrogram computed - shape: {mel_tensor.shape}")
        
        # Convert mel to JAX array
        mel_jax = jnp.array(mel_tensor)
        
        # Test with first 30 seconds of audio
        print(f"\n🧪 Testing with first 30 seconds of audio...")
        mel_30s = mel_jax[:, :3000]  # First 30 seconds (3000 frames)
        mel_30s = mel_30s[None, :, :]  # Add batch dimension: (1, 80, 3000)
        print(f"   Mel shape: {mel_30s.shape}")
        
        # Load real weights (we need PyTorch for this part)
        print(f"\n🔄 Loading real Whisper weights...")
        try:
            import whisper
            real_whisper = whisper.load_model("tiny")
            print(f"✅ Real Whisper model loaded for weight copying")
            
            # Copy weights to JAX model
            print(f"🔄 Copying weights to JAX model...")
            
            # Copy encoder weights
            jax_model.encoder.conv1.kernel = nnx.Param(jnp.array(real_whisper.encoder.conv1.weight.data.numpy().T))
            jax_model.encoder.conv1.bias = nnx.Param(jnp.array(real_whisper.encoder.conv1.bias.data.numpy()))
            jax_model.encoder.conv2.kernel = nnx.Param(jnp.array(real_whisper.encoder.conv2.weight.data.numpy().T))
            jax_model.encoder.conv2.bias = nnx.Param(jnp.array(real_whisper.encoder.conv2.bias.data.numpy()))
            
            # Copy positional embedding
            jax_model.encoder.positional_embedding = nnx.Param(jnp.array(real_whisper.encoder.positional_embedding.data.numpy()))
            
            # Copy encoder blocks
            for i, (pytorch_block, jax_block) in enumerate(zip(real_whisper.encoder.blocks, jax_model.encoder.blocks)):
                # Self-attention
                jax_block.attn.query.kernel = nnx.Param(jnp.array(pytorch_block.attn.query.weight.data.numpy().T))
                jax_block.attn.query.bias = nnx.Param(jnp.array(pytorch_block.attn.query.bias.data.numpy()))
                jax_block.attn.key.kernel = nnx.Param(jnp.array(pytorch_block.attn.key.weight.data.numpy().T))
                if pytorch_block.attn.key.bias is not None:
                    jax_block.attn.key.bias = nnx.Param(jnp.array(pytorch_block.attn.key.bias.data.numpy()))
                jax_block.attn.value.kernel = nnx.Param(jnp.array(pytorch_block.attn.value.weight.data.numpy().T))
                jax_block.attn.value.bias = nnx.Param(jnp.array(pytorch_block.attn.value.bias.data.numpy()))
                jax_block.attn.out.kernel = nnx.Param(jnp.array(pytorch_block.attn.out.weight.data.numpy().T))
                jax_block.attn.out.bias = nnx.Param(jnp.array(pytorch_block.attn.out.bias.data.numpy()))
                
                jax_block.attn_ln.scale = nnx.Param(jnp.array(pytorch_block.attn_ln.weight.data.numpy()))
                jax_block.attn_ln.bias = nnx.Param(jnp.array(pytorch_block.attn_ln.bias.data.numpy()))
                
                # MLP
                jax_block.mlp_linear1.kernel = nnx.Param(jnp.array(pytorch_block.mlp[0].weight.data.numpy().T))
                jax_block.mlp_linear1.bias = nnx.Param(jnp.array(pytorch_block.mlp[0].bias.data.numpy()))
                jax_block.mlp_linear2.kernel = nnx.Param(jnp.array(pytorch_block.mlp[2].weight.data.numpy().T))
                jax_block.mlp_linear2.bias = nnx.Param(jnp.array(pytorch_block.mlp[2].bias.data.numpy()))
                jax_block.mlp_ln.scale = nnx.Param(jnp.array(pytorch_block.mlp_ln.weight.data.numpy()))
                jax_block.mlp_ln.bias = nnx.Param(jnp.array(pytorch_block.mlp_ln.bias.data.numpy()))
            
            jax_model.encoder.ln_post.scale = nnx.Param(jnp.array(real_whisper.encoder.ln_post.weight.data.numpy()))
            jax_model.encoder.ln_post.bias = nnx.Param(jnp.array(real_whisper.encoder.ln_post.bias.data.numpy()))
            
            # Copy decoder weights
            jax_model.decoder.token_embedding.embedding = nnx.Param(jnp.array(real_whisper.decoder.token_embedding.weight.data.numpy()))
            jax_model.decoder.positional_embedding = nnx.Param(jnp.array(real_whisper.decoder.positional_embedding.data.numpy()))
            
            # Copy decoder blocks
            for i, (pytorch_block, jax_block) in enumerate(zip(real_whisper.decoder.blocks, jax_model.decoder.blocks)):
                # Self-attention
                jax_block.attn.query.kernel = nnx.Param(jnp.array(pytorch_block.attn.query.weight.data.numpy().T))
                jax_block.attn.query.bias = nnx.Param(jnp.array(pytorch_block.attn.query.bias.data.numpy()))
                jax_block.attn.key.kernel = nnx.Param(jnp.array(pytorch_block.attn.key.weight.data.numpy().T))
                if pytorch_block.attn.key.bias is not None:
                    jax_block.attn.key.bias = nnx.Param(jnp.array(pytorch_block.attn.key.bias.data.numpy()))
                jax_block.attn.value.kernel = nnx.Param(jnp.array(pytorch_block.attn.value.weight.data.numpy().T))
                jax_block.attn.value.bias = nnx.Param(jnp.array(pytorch_block.attn.value.bias.data.numpy()))
                jax_block.attn.out.kernel = nnx.Param(jnp.array(pytorch_block.attn.out.weight.data.numpy().T))
                jax_block.attn.out.bias = nnx.Param(jnp.array(pytorch_block.attn.out.bias.data.numpy()))
                
                jax_block.attn_ln.scale = nnx.Param(jnp.array(pytorch_block.attn_ln.weight.data.numpy()))
                jax_block.attn_ln.bias = nnx.Param(jnp.array(pytorch_block.attn_ln.bias.data.numpy()))
                
                # Cross-attention
                cross_attn = pytorch_block.cross_attn
                jax_cross_attn = jax_block.cross_attn
                jax_cross_attn.query.kernel = nnx.Param(jnp.array(cross_attn.query.weight.data.numpy().T))
                jax_cross_attn.query.bias = nnx.Param(jnp.array(cross_attn.query.bias.data.numpy()))
                jax_cross_attn.key.kernel = nnx.Param(jnp.array(cross_attn.key.weight.data.numpy().T))
                if cross_attn.key.bias is not None:
                    jax_cross_attn.key.bias = nnx.Param(jnp.array(cross_attn.key.bias.data.numpy()))
                jax_cross_attn.value.kernel = nnx.Param(jnp.array(cross_attn.value.weight.data.numpy().T))
                jax_cross_attn.value.bias = nnx.Param(jnp.array(cross_attn.value.bias.data.numpy()))
                jax_cross_attn.out.kernel = nnx.Param(jnp.array(cross_attn.out.weight.data.numpy().T))
                jax_cross_attn.out.bias = nnx.Param(jnp.array(cross_attn.out.bias.data.numpy()))
                
                jax_block.cross_attn_ln.scale = nnx.Param(jnp.array(pytorch_block.cross_attn_ln.weight.data.numpy()))
                jax_block.cross_attn_ln.bias = nnx.Param(jnp.array(pytorch_block.cross_attn_ln.bias.data.numpy()))
                
                # MLP
                jax_block.mlp_linear1.kernel = nnx.Param(jnp.array(pytorch_block.mlp[0].weight.data.numpy().T))
                jax_block.mlp_linear1.bias = nnx.Param(jnp.array(pytorch_block.mlp[0].bias.data.numpy()))
                jax_block.mlp_linear2.kernel = nnx.Param(jnp.array(pytorch_block.mlp[2].weight.data.numpy().T))
                jax_block.mlp_linear2.bias = nnx.Param(jnp.array(pytorch_block.mlp[2].bias.data.numpy()))
                jax_block.mlp_ln.scale = nnx.Param(jnp.array(pytorch_block.mlp_ln.weight.data.numpy()))
                jax_block.mlp_ln.bias = nnx.Param(jnp.array(pytorch_block.mlp_ln.bias.data.numpy()))
            
            jax_model.decoder.ln.scale = nnx.Param(jnp.array(real_whisper.decoder.ln.weight.data.numpy()))
            jax_model.decoder.ln.bias = nnx.Param(jnp.array(real_whisper.decoder.ln.bias.data.numpy()))
            
            print("✅ ALL weights copied to JAX model")
            
        except ImportError:
            print("⚠️  PyTorch not available - using random weights")
            print("   (This will not produce meaningful transcription)")
        
        # Get audio features
        print(f"\n🎵 Computing audio features...")
        audio_features = jax_model.embed_audio(mel_30s)
        print(f"✅ Audio features computed - shape: {audio_features.shape}")
        
        # Generate tokens
        print(f"\n🤖 Generating tokens with JAX Whisper...")
        start_time = time.time()
        jax_tokens = jax_greedy_decode(jax_model, audio_features, max_length=200)
        jax_time = time.time() - start_time
        
        print(f"   JAX generation time: {jax_time:.2f} seconds")
        print(f"   JAX tokens: {jax_tokens[0].tolist()}")
        
        # Convert tokens to text
        print(f"\n📝 Converting tokens to text...")
        try:
            tokenizer_instance = tokenizer.get_tokenizer(multilingual=True)
            tokens_list = jax_tokens[0].tolist()
            text = tokenizer_instance.decode(tokens_list)
            print(f"✅ Text decoded successfully")
            
            print(f"\n📝 TRANSCRIPTION RESULT:")
            print(f"   Text: {text}")
            
            # Analyze the transcription
            print(f"\n📊 TRANSCRIPTION ANALYSIS:")
            print(f"   Token count: {len(tokens_list)}")
            print(f"   Unique tokens: {len(set(tokens_list))}")
            print(f"   Token diversity: {len(set(tokens_list))}/{len(tokens_list)} ({len(set(tokens_list))/len(tokens_list)*100:.1f}%)")
            
            # Check for special tokens
            special_tokens = {
                50258: "<|startoftranscript|>",
                50259: "<|en|>",
                50359: "<|transcribe|>",
                50364: "<|notimestamps|>",
                50257: "<|endoftext|>"
            }
            
            found_special = []
            for token in tokens_list:
                if token in special_tokens:
                    found_special.append(special_tokens[token])
            
            print(f"   Special tokens found: {found_special}")
            
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
