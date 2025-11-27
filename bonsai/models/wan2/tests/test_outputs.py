# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test output correctness by comparing JAX implementation with HuggingFace reference.

This script loads weights from the HuggingFace checkpoint and compares outputs
between the JAX implementation and PyTorch reference to ensure correctness.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from huggingface_hub import snapshot_download

import os
from bonsai.models.wan2 import modeling, params, vae
from wan.configs import WAN_CONFIGS
from wan.modules.t5 import T5EncoderModel
import torch
from transformers import AutoTokenizer

def check_weight_loading(jax_model, torch_model):
    """比较 JAX (safetensor) 和 PyTorch (pth) 加载的权重"""

    # 1. Embedding weights
    # torch_model is WanT5EncoderModel, torch_model.model is T5Encoder
    torch_emb = torch_model.model.token_embedding.weight.float().detach().cpu().numpy()
    jax_emb = np.array(jax_model.encoder.token_embedding.embedding.value)

    print("Embedding weights:")
    print(f"  Shapes: torch={torch_emb.shape}, jax={jax_emb.shape}")
    print(f"  Max diff: {np.abs(torch_emb - jax_emb).max():.2e}")
    print(f"  Mean: torch={torch_emb.mean():.4f}, jax={jax_emb.mean():.4f}")

    # 2. 第一个 block 的 query weight
    # PyTorch: encoder.block[0].layer[0].SelfAttention.q.weight
    torch_q = torch_model.model.blocks[0].attn.q.weight.float().detach().cpu().numpy()

    # JAX: encoder.blocks[0] 的对应参数
    # 需要知道你的参数结构，可能是：
    jax_q = np.array(jax_model.encoder.blocks[0].attn.q.kernel.value)

    print("\nFirst block query weight:")
    print(f"  Shapes: torch={torch_q.shape}, jax={jax_q.shape}")
    print(f"  Max diff: {np.abs(torch_q.T - jax_q).max():.2e}")  # 注意可能需要转置

    # 3. Layer norm (检查 gamma/beta)
    torch_ln_weight = torch_model.model.norm.weight.float().detach().cpu().numpy()
    jax_ln_weight = np.array(jax_model.encoder.norm.weight.value)

    print("\nFinal LayerNorm weight:")
    print(f"  Shapes: torch={torch_ln_weight.shape}, jax={jax_ln_weight.shape}")
    print(f"  Max diff: {np.abs(torch_ln_weight - jax_ln_weight).max():.2e}")


def compare_intermediate_outputs(jax_model, torch_model, input_ids, attention_mask):
    """Compare intermediate layer outputs between JAX and PyTorch models.

    Args:
        jax_model: JAX T5Encoder model
        torch_model: WanT5EncoderModel instance
        input_ids: Tokenized input IDs [batch, seq_len]
        attention_mask: Attention mask [batch, seq_len]
    """
    print("\n" + "=" * 80)
    print("COMPARING INTERMEDIATE LAYER OUTPUTS")
    print("=" * 80)

    # Convert inputs to appropriate formats
    input_ids_jax = jnp.array(input_ids)
    attention_mask_jax = jnp.array(attention_mask)

    input_ids_torch = torch.from_numpy(np.array(input_ids)).long()
    attention_mask_torch = torch.from_numpy(np.array(attention_mask)).long()

    # Get PyTorch encoder
    torch_encoder = torch_model.model

    # ========================================
    # 1. Embedding Layer
    # ========================================
    print("\n[Layer 0] Token Embeddings")

    # JAX
    jax_emb = jax_model.encoder.token_embedding(input_ids_jax)

    # PyTorch
    with torch.no_grad():
        torch_emb = torch_encoder.token_embedding(input_ids_torch)

    compare_outputs(jax_emb, torch_emb, "Token Embeddings", rtol=1e-3, atol=1e-4)

    # ========================================
    # 2. After dropout (before blocks)
    # ========================================
    print("\n[Layer 0.5] After Dropout")

    # JAX - manually apply dropout with deterministic=True (no dropout)
    jax_x = jax_emb

    # PyTorch
    with torch.no_grad():
        torch_x = torch_encoder.dropout(torch_emb)

    compare_outputs(jax_x, torch_x, "After Initial Dropout", rtol=1e-3, atol=1e-4)

    # ========================================
    # 3. Each Transformer Block
    # ========================================
    num_layers = len(torch_encoder.blocks)
    print(f"\n[Blocks] Processing {num_layers} transformer layers...")

    # Initialize with embedding output
    jax_hidden = jax_x
    torch_hidden = torch_x

    for layer_idx in range(num_layers):
        print(f"\n[Block {layer_idx}] Transformer Layer {layer_idx}")

        # JAX block forward
        jax_block = jax_model.encoder.blocks[layer_idx]
        jax_pos_bias = jax_model.encoder.pos_embedding(
            jax_hidden.shape[1],
            jax_hidden.shape[1],
        )
        jax_hidden = jax_block(jax_hidden, mask=attention_mask_jax, pos_bias=jax_pos_bias)

        # PyTorch block forward
        with torch.no_grad():
            torch_block = torch_encoder.blocks[layer_idx]
            torch_pos_bias = torch_encoder.pos_embedding(
                torch_hidden.shape[1],
                torch_hidden.shape[1],
            )
            torch_hidden = torch_block(torch_hidden, mask=attention_mask_torch, pos_bias=torch_pos_bias)

        # Compare
        compare_outputs(
            jax_hidden,
            torch_hidden,
            f"Block {layer_idx} Output",
            rtol=1e-3,
            atol=1e-4
        )

        # Optional: Compare sub-components for first few layers
        if layer_idx < 3:
            print(f"  [Block {layer_idx}] Sub-component details:")

            # Norm1 output
            with torch.no_grad():
                jax_norm1_out = jax_block.norm1(jax_hidden)
                torch_norm1_out = torch_block.norm1(torch_hidden)

            diff = np.abs(np.array(jax_norm1_out) - np.array(torch_norm1_out.float()))
            print(f"    Norm1: max_diff={diff.max():.2e}, mean_diff={diff.mean():.2e}")

    # ========================================
    # 5. Final Layer Norm
    # ========================================
    print("\n[Final] Layer Normalization")

    # JAX
    jax_output = jax_model.encoder.norm(jax_hidden)

    # PyTorch
    with torch.no_grad():
        torch_output = torch_encoder.norm(torch_hidden)

    compare_outputs(jax_output, torch_output, "Final LayerNorm Output", rtol=1e-3, atol=1e-4)

    # ========================================
    # 6. Final Dropout
    # ========================================
    print("\n[Final] After Final Dropout")

    # JAX - no dropout in eval mode
    jax_final = jax_output

    # PyTorch
    with torch.no_grad():
        torch_final = torch_encoder.dropout(torch_output)

    compare_outputs(jax_final, torch_final, "Final Output", rtol=1e-3, atol=1e-4)

    print("\n" + "=" * 80)
    print("INTERMEDIATE COMPARISON COMPLETE")
    print("=" * 80)

def compare_outputs(jax_output: jax.Array, torch_output, name: str, rtol: float = 1e-3, atol: float = 1e-5):
    """Compare JAX and PyTorch outputs and report differences.

    Args:
        jax_output: Output from JAX model
        torch_output: Output from PyTorch model (torch.Tensor)
        name: Name of the output being compared
        rtol: Relative tolerance
        atol: Absolute tolerance
    """
    import torch
    if torch_output.dtype == torch.bfloat16:
        torch_output = torch_output.float()

    # Convert PyTorch to numpy
    if isinstance(torch_output, torch.Tensor):
        torch_np = torch_output.detach().cpu().numpy()
    else:
        torch_np = np.array(torch_output)

    # Convert JAX to numpy
    jax_np = np.array(jax_output)

    print(f"\n{'=' * 80}")
    print(f"Comparing: {name}")
    print(f"{'=' * 80}")
    print(f"JAX shape:   {jax_np.shape}")
    print(f"Torch shape: {torch_np.shape}")
    print(f"JAX dtype:   {jax_np.dtype}")
    print(f"Torch dtype: {torch_np.dtype}")

    # Check shapes match
    if jax_np.shape != torch_np.shape:
        print("❌ Shape mismatch!")
        return False

    # Compute differences
    abs_diff = np.abs(jax_np - torch_np)
    rel_diff = abs_diff / (np.abs(torch_np) + 1e-10)

    max_abs_diff = np.max(abs_diff)
    max_rel_diff = np.max(rel_diff)
    mean_abs_diff = np.mean(abs_diff)
    mean_rel_diff = np.mean(rel_diff)

    print("\nStatistics:")
    print(f"  Max absolute difference: {max_abs_diff:.2e}")
    print(f"  Max relative difference: {max_rel_diff:.2e}")
    print(f"  Mean absolute difference: {mean_abs_diff:.2e}")
    print(f"  Mean relative difference: {mean_rel_diff:.2e}")

    print(f"\nJAX output range:   [{np.min(jax_np):.4f}, {np.max(jax_np):.4f}]")
    print(f"Torch output range: [{np.min(torch_np):.4f}, {np.max(torch_np):.4f}]")

    # Check if within tolerance
    close = np.allclose(jax_np, torch_np, rtol=rtol, atol=atol)

    if close:
        print(f"\n✅ Outputs match within tolerance (rtol={rtol}, atol={atol})")
    else:
        print(f"\n❌ Outputs do NOT match (rtol={rtol}, atol={atol})")
        # Show some mismatched locations
        mismatch_mask = ~np.isclose(jax_np, torch_np, rtol=rtol, atol=atol)
        n_mismatches = np.sum(mismatch_mask)
        print(f"  Number of mismatches: {n_mismatches} / {jax_np.size} ({100 * n_mismatches / jax_np.size:.2f}%)")

    return close


def test_t5_encoder():
    """Test T5 encoder output against Wan T5 reference implementation."""
    print("\n" + "=" * 80)
    print("TEST 1: T5 Encoder (UMT5-XXL)")
    print("=" * 80)
    # Download checkpoint
    model_ckpt_path = snapshot_download("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")

    # Test prompt
    prompt = "A beautiful sunset over the ocean with waves crashing on the shore"
    max_length = 512

    print(f"\nTest prompt: {prompt}")
    print(f"Max length: {max_length}")

    # Tokenize using transformers tokenizer for JAX model
    print("\nTokenizing for JAX model...")
    tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
    inputs = tokenizer(prompt, max_length=max_length, padding="max_length", truncation=True, return_tensors="np")

    # JAX model
    print("\n[1/2] Loading JAX T5 encoder...")
    jax_model = params.create_t5_encoder_from_safe_tensors(model_ckpt_path, mesh=None)

    # Run JAX
    # print("\nRunning JAX model...")
    # input_ids_jax = jnp.array(inputs["input_ids"])
    # attention_mask_jax = jnp.array(inputs["attention_mask"])
    # jax_output = jax_model(input_ids_jax, attention_mask_jax, deterministic=True)
    # print(f"✓ JAX output shape: {jax_output.shape}")

    # ========================================
    # Configuration
    # ========================================
    ckpt_dir = "./Wan2.1-T2V-1.3B"  # Change this to your checkpoint directory
    device = torch.device("cpu")  # Force CPU

    print("=" * 60)
    print("T5 Encoder Unit Test")
    print("=" * 60)
    print(f"Checkpoint directory: {ckpt_dir}")
    print(f"Input prompt: {prompt}")
    print(f"Device: {device}")
    print()

    # ========================================
    # Load Configuration
    # ========================================
    config = WAN_CONFIGS["t2v-1.3B"]
    print(f"Model config: {config.__name__}")
    print(f"T5 checkpoint: {config.t5_checkpoint}")
    print(f"T5 tokenizer: {config.t5_tokenizer}")
    print(f"Text length: {config.text_len}")
    print(f"T5 dtype: {config.t5_dtype}")
    print()

    # ========================================
    # Initialize T5 Encoder
    # ========================================
    print("Initializing T5 Encoder...")
    text_encoder = T5EncoderModel(
        text_len=config.text_len,
        dtype=config.t5_dtype,
        device=device,  # Use CPU
        checkpoint_path=os.path.join(ckpt_dir, config.t5_checkpoint),
        tokenizer_path=os.path.join(ckpt_dir, config.t5_tokenizer),
        shard_fn=None,  # No FSDP on CPU
    )
    print("T5 Encoder loaded successfully!")
    print()
    # 运行检查
    check_weight_loading(jax_model, text_encoder)

    # ========================================
    # Compare Intermediate Outputs
    # ========================================
    compare_intermediate_outputs(jax_model, text_encoder, inputs["input_ids"], inputs["attention_mask"])

    # ========================================
    # Encode Prompt
    # ========================================
    print("Encoding prompt...")
    context = text_encoder([prompt], device)
    print("Encoding complete!")
    print()

    torch_embeddings = context[0]  # Get first (and only) result
    actual_seq_len = torch_embeddings.shape[0]

    print(f"✓ PyTorch output shape: {torch_embeddings.shape} (actual length: {actual_seq_len})")
    print(f"  JAX output shape: {jax_output.shape}")

    # Extract only the valid (non-padded) portion from JAX output for comparison
    # JAX output: [1, max_length, hidden_dim], we want [1, actual_seq_len, hidden_dim]
    jax_output_trimmed = jax_output[:, :actual_seq_len, :]

    # Add batch dimension to PyTorch output to match JAX format
    torch_embeddings = torch_embeddings.unsqueeze(0)

    print(f"\nComparing only valid tokens (first {actual_seq_len} tokens):")
    print(f"  JAX (trimmed): {jax_output_trimmed.shape}")
    print(f"  PyTorch: {torch_embeddings.shape}")

    # Compare only the valid portion (ignore padding)
    return compare_outputs(jax_output_trimmed, torch_embeddings, "T5 Encoder Output", rtol=1e-3, atol=1e-4)


def test_dit_transformer():
    """Test DiT transformer output against HuggingFace reference."""
    print("\n" + "=" * 80)
    print("TEST 2: DiT Transformer (Wan2.1-T2V-1.3B)")
    print("=" * 80)

    try:
        import torch
        from diffusers import WanDiT2DModel
    except ImportError:
        print("❌ PyTorch or diffusers not installed. Skipping DiT test.")
        return False

    # Download checkpoint
    model_ckpt_path = snapshot_download("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")

    # Config
    config = modeling.ModelConfig(num_layers=24)  # Use default full config

    # Create dummy inputs (matching expected shapes)
    batch_size = 1
    num_frames = config.num_frames  # 21
    latent_h, latent_w = config.latent_size  # (104, 60)
    latent_channels = config.input_dim  # 16
    text_seq_len = config.max_text_len  # 512
    text_embed_dim = config.text_embed_dim  # 4096

    latents = jnp.zeros((batch_size, num_frames, latent_h, latent_w, latent_channels))
    text_embeds = jnp.zeros((batch_size, text_seq_len, text_embed_dim))
    timestep = jnp.array([25])  # Middle of diffusion

    print("\nInput shapes:")
    print(f"  Latents: {latents.shape}")
    print(f"  Text embeds: {text_embeds.shape}")
    print(f"  Timestep: {timestep}")

    # JAX model
    print("\n[1/2] Loading JAX DiT model...")
    jax_model = params.create_model_from_safe_tensors(model_ckpt_path, config, mesh=None, load_transformer_only=True)

    # PyTorch reference
    print("[2/2] Loading PyTorch DiT reference...")
    torch_model = WanDiT2DModel.from_pretrained(f"{model_ckpt_path}/transformer")
    torch_model.eval()

    # Run JAX
    print("\nRunning JAX model...")
    jax_output = jax_model.forward(latents, text_embeds, timestep, deterministic=True)

    # Run PyTorch (need to convert to channel-first format)
    print("Running PyTorch model...")
    with torch.no_grad():
        # PyTorch expects [B, C, T, H, W] (channel-first)
        latents_torch = torch.from_numpy(np.array(latents))
        latents_torch = latents_torch.permute(0, 4, 1, 2, 3)  # [B, T, H, W, C] -> [B, C, T, H, W]

        text_embeds_torch = torch.from_numpy(np.array(text_embeds))
        timestep_torch = torch.tensor([25])

        torch_output = torch_model(
            hidden_states=latents_torch,
            encoder_hidden_states=text_embeds_torch,
            timestep=timestep_torch,
            return_dict=False,
        )[0]

        # Convert back to channel-last for comparison
        torch_output = torch_output.permute(0, 2, 3, 4, 1)  # [B, C, T, H, W] -> [B, T, H, W, C]

    # Compare
    return compare_outputs(jax_output, torch_output, "DiT Transformer Output", rtol=1e-3, atol=1e-4)


def test_vae_decoder():
    """Test VAE decoder output against HuggingFace reference."""
    print("\n" + "=" * 80)
    print("TEST 3: VAE Decoder (Wan-VAE)")
    print("=" * 80)

    try:
        import torch
        from diffusers import AutoencoderKLWan
    except ImportError:
        print("❌ PyTorch or diffusers not installed. Skipping VAE test.")
        return False

    # Download checkpoint
    model_ckpt_path = snapshot_download("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")

    # Create dummy latent input
    # Expected: [B, T, H, W, C] = [1, 21, 104, 60, 16]
    batch_size = 1
    num_frames = 21
    latent_h, latent_w = 104, 60
    latent_channels = 16

    latents_jax = jnp.zeros((batch_size, num_frames, latent_h, latent_w, latent_channels))

    print(f"\nInput latents shape: {latents_jax.shape}")

    # JAX model
    print("\n[1/2] Loading JAX VAE decoder...")
    jax_vae = params.create_vae_decoder_from_safe_tensors(model_ckpt_path, mesh=None)

    # PyTorch reference
    print("[2/2] Loading PyTorch VAE reference...")
    torch_vae = AutoencoderKLWan.from_pretrained(f"{model_ckpt_path}/vae")
    torch_vae.eval()

    # Run JAX
    print("\nRunning JAX VAE decoder...")
    jax_output = jax_vae.decode(latents_jax)

    # Run PyTorch (need to convert to channel-first format)
    print("Running PyTorch VAE decoder...")
    with torch.no_grad():
        # PyTorch expects [B, C, T, H, W] (channel-first)
        latents_torch = torch.from_numpy(np.array(latents_jax))
        latents_torch = latents_torch.permute(0, 4, 1, 2, 3)  # [B, T, H, W, C] -> [B, C, T, H, W]

        torch_output = torch_vae.decode(latents_torch).sample

        # Convert back to channel-last for comparison: [B, C, T, H, W] -> [B, T, H, W, C]
        torch_output = torch_output.permute(0, 2, 3, 4, 1)

    # Compare
    # VAE outputs are typically less precise due to stochastic operations
    return compare_outputs(jax_output, torch_output, "VAE Decoder Output", rtol=5e-3, atol=1e-3)






def run_all_tests():
    """Run all output comparison tests."""
    print("\n" + "=" * 80)
    print("RUNNING ALL OUTPUT COMPARISON TESTS")
    print("=" * 80)
    print("\nThis will compare JAX implementations with HuggingFace reference models")
    print("to ensure correctness of the ported models.\n")

    results = {}

    # Test T5
    try:
        results["T5 Encoder"] = test_t5_encoder()
    except Exception as e:
        print(f"\n❌ T5 test failed with error: {e}")
        import traceback

        traceback.print_exc()
        results["T5 Encoder"] = False

    # Test DiT
    try:
        results["DiT Transformer"] = test_dit_transformer()
    except Exception as e:
        print(f"\n❌ DiT test failed with error: {e}")
        import traceback

        traceback.print_exc()
        results["DiT Transformer"] = False

    # Test VAE
    try:
        results["VAE Decoder"] = test_vae_decoder()
    except Exception as e:
        print(f"\n❌ VAE test failed with error: {e}")
        import traceback

        traceback.print_exc()
        results["VAE Decoder"] = False

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:20s} {status}")

    all_passed = all(results.values())
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 80 + "\n")

    return all_passed


if __name__ == "__main__":
    test_t5_encoder()
