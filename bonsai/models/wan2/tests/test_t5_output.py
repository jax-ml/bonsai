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
from huggingface_hub import snapshot_download
from bonsai.models.wan2 import params
from wan.configs import WAN_CONFIGS
import torch
from transformers import AutoTokenizer, UMT5EncoderModel

def check_weight_loading(jax_model, torch_model):
    torch_emb = torch_model.shared.weight.detach().cpu().numpy()
    jax_emb = np.array(jax_model.encoder.token_embedding.embedding.value)

    print("Embedding weights:")
    print(f"  Shapes: torch={torch_emb.shape}, jax={jax_emb.shape}")
    print(f"  Max diff: {np.abs(torch_emb - jax_emb).max():.2e}")
    print(f"  Mean diff: {np.abs(torch_emb - jax_emb).mean():.2e}")
    torch_q = torch_model.encoder.block[0].layer[0].SelfAttention.q.weight.detach().cpu().numpy()
    jax_q = np.array(jax_model.encoder.blocks[0].attn.q.kernel.value)

    print("\nFirst block query weight:")
    print(f"  Shapes: torch={torch_q.shape}, jax={jax_q.shape}")
    print(f"  Max diff: {np.abs(torch_q.T - jax_q).max():.2e}")
    print(f"  Mean diff: {np.abs(torch_q.T - jax_q).mean():.2e}")

    torch_ln_weight = torch_model.encoder.final_layer_norm.weight.detach().cpu().numpy()
    jax_ln_weight = np.array(jax_model.encoder.norm.weight.value)

    print("\nFinal LayerNorm weight:")
    print(f"  Shapes: torch={torch_ln_weight.shape}, jax={jax_ln_weight.shape}")
    print(f"  Max diff: {np.abs(torch_ln_weight - jax_ln_weight).max():.2e}")
    print(f"  Mean diff: {np.abs(torch_ln_weight - jax_ln_weight).mean():.2e}")
    


def compare_intermediate_outputs(jax_model, torch_model, input_ids, attention_mask):
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
        jax_pos_bias = jax_block.pos_embedding(
            jax_hidden.shape[1],
            jax_hidden.shape[1],
        )
        jax_hidden = jax_block(jax_hidden, mask=attention_mask_jax, pos_bias=jax_pos_bias,deterministic=True)

        # PyTorch block forward
        with torch.no_grad():
            torch_block = torch_encoder.blocks[layer_idx]
            torch_pos_bias = torch_block.pos_embedding(
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

            diff = np.abs(np.array(jax_norm1_out) - np.array(torch_norm1_out))
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
        print("Shape mismatch!")
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

    print("\nTokenizing...")
    tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
    inputs_j = tokenizer(prompt, return_tensors="np")
    inputs_p = tokenizer(prompt, return_tensors="pt")

    print("\n[1/2] Loading T5 encoder...")
    jax_t5 = params.create_t5_encoder_from_safe_tensors(model_ckpt_path, mesh=None)
    hf_t5 = UMT5EncoderModel.from_pretrained(model_ckpt_path, subfolder="text_encoder", torch_dtype=torch.float32)

    check_weight_loading(jax_t5, hf_t5)

    print("\nRunning model...")
    input_ids_jax = jnp.array(inputs_j.input_ids)
    jax_output = jax_t5(input_ids_jax, deterministic=True)

    pytorch_output = hf_t5(inputs_p.input_ids)
    torch_embeddings = pytorch_output.last_hidden_state

    # Compare only the valid portion (ignore padding)
    return compare_outputs(jax_output, torch_embeddings, "T5 Encoder Output", rtol=1e-3, atol=1e-4)

if __name__ == "__main__":
    test_t5_encoder()
