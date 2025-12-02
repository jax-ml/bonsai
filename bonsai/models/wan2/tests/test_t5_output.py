"""Test output correctness by comparing JAX implementation with HuggingFace reference."""

import jax
import jax.numpy as jnp
import numpy as np
from huggingface_hub import snapshot_download
from bonsai.models.wan2 import params, t5
import torch
from transformers import AutoTokenizer, UMT5EncoderModel, UMT5ForConditionalGeneration
import os

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
    
def compare_outputs(jax_output: jax.Array, torch_output, name: str, rtol: float = 1e-2, atol: float = 1e-4):
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


def test_t5_intermediate():
    """Compare intermediate layer outputs between JAX and PyTorch T5 encoder."""
    print("\n" + "=" * 80)
    print("TEST 2: T5 Encoder Intermediate Outputs")
    print("=" * 80)

    # Download checkpoint
    model_ckpt_path = snapshot_download("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")

    # Test prompt
    prompt = "A beautiful sunset over the ocean with waves crashing on the shore"

    print(f"\nTest prompt: {prompt}")

    print("\nTokenizing...")
    tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
    inputs_j = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=512, truncation=True)
    inputs_p = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=512, truncation=True)

    print("\n[1/3] Loading models...")
    jax_t5 = params.create_t5_encoder_from_safe_tensors(model_ckpt_path, mesh=None)
    pytorch_t5 = UMT5EncoderModel.from_pretrained(
        model_ckpt_path,
        subfolder="text_encoder",
        torch_dtype=torch.float32
    )
    pytorch_t5.eval()

    # Register hooks to capture PyTorch intermediate outputs
    print("\n[2/3] Capturing PyTorch intermediate outputs...")
    pytorch_intermediates = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                pytorch_intermediates[name] = output[0].detach().cpu()
            else:
                pytorch_intermediates[name] = output.detach().cpu()
        return hook

    # Hook embedding
    pytorch_t5.encoder.embed_tokens.register_forward_hook(make_hook("embeddings"))

    # Hook each block with detailed attention captures
    for i, block in enumerate(pytorch_t5.encoder.block):
        block.register_forward_hook(make_hook(f"block_{i}_output"))
        block.layer[0].register_forward_hook(make_hook(f"block_{i}_attn_output"))

        # Hook attention Q, K, V projections
        block.layer[0].layer_norm.register_forward_hook(make_hook(f"block_{i}_attn_norm"))
        attn = block.layer[0].SelfAttention
        attn.q.register_forward_hook(make_hook(f"block_{i}_q_proj"))
        attn.k.register_forward_hook(make_hook(f"block_{i}_k_proj"))
        attn.v.register_forward_hook(make_hook(f"block_{i}_v_proj"))

        # # Hook to capture position bias (computed inside attention)
        # def make_pos_bias_hook(block_idx):
        #     def pos_bias_hook(module, input, output):
        #         # T5 attention returns (output, position_bias, ...)
        #         if isinstance(output, tuple) and len(output) > 1:
        #             pos_bias = output[1]  # position_bias is second return value
        #             if pos_bias is not None:
        #                 pytorch_intermediates[f"block_{block_idx}_position_bias"] = pos_bias.detach().cpu()
        #     return pos_bias_hook

        # attn.register_forward_hook(make_pos_bias_hook(i))

        if len(block.layer) > 1:
            block.layer[1].register_forward_hook(make_hook(f"block_{i}_ffn_output"))

    # Hook final norm
    pytorch_t5.encoder.final_layer_norm.register_forward_hook(make_hook("final_norm"))

    # Run PyTorch forward
    with torch.no_grad():
        pytorch_output = pytorch_t5(inputs_p.input_ids)

    print("\n[3/3] Comparing intermediate outputs...")

    # Convert inputs to JAX
    input_ids_jax = jnp.array(inputs_j.input_ids)

    # Manual forward pass for JAX to get intermediates
    # 1. Embeddings
    x_jax = jax_t5.encoder.token_embedding(input_ids_jax)
    embeddings_torch = pytorch_intermediates["embeddings"]
    compare_outputs(x_jax, embeddings_torch, "Token Embeddings", rtol=1e-4, atol=1e-6)

    # 2. Dropout (not comparing, just applying)
    x_jax = jax_t5.encoder.dropout(x_jax, deterministic=True)

    # 3. Position bias setup (only once)
    batch_size, seq_len = input_ids_jax.shape
    position_bias_jax = None

    # 4. Process through blocks
    num_layers = len(jax_t5.encoder.blocks)
    print(f"\nComparing {num_layers} transformer blocks...")

    for i in range(min(3, num_layers)):  # Compare first 3 blocks
        print(f"\n{'='*80}")
        print(f"BLOCK {i}")
        print(f"{'='*80}")

        block = jax_t5.encoder.blocks[i]

        # Manual attention computation to capture Q, K, V
        print(f"\n--- Attention Details ---")

        # Norm
        normed_x_jax = block.norm1(x_jax)

        compare_outputs(normed_x_jax, pytorch_intermediates[f"block_{i}_attn_norm"], f"Block {i} Attention Norm", rtol=1e-5, atol=1e-6)

        # Q, K, V projections
        q_jax = block.attn.q(normed_x_jax)
        k_jax = block.attn.k(normed_x_jax)
        v_jax = block.attn.v(normed_x_jax)

        # Compare with PyTorch
        q_torch = pytorch_intermediates[f"block_{i}_q_proj"]
        k_torch = pytorch_intermediates[f"block_{i}_k_proj"]
        v_torch = pytorch_intermediates[f"block_{i}_v_proj"]

        compare_outputs(q_jax, q_torch, f"Block {i} Q after Linear", rtol=1e-5, atol=1e-6)
        compare_outputs(k_jax, k_torch, f"Block {i} K after Linear", rtol=1e-5, atol=1e-6)
        compare_outputs(v_jax, v_torch, f"Block {i} V after Linear", rtol=1e-5, atol=1e-6)

        # Now run full attention (for comparison)
        if position_bias_jax is None:
            attn_output = block.attn(
                x_jax,
                mask=None,
                pos_bias=None,
                deterministic=True
            )
            # Compare position bias (only computed in first block)
            if f"block_{i}_position_bias" in pytorch_intermediates:
                pos_bias_torch = pytorch_intermediates[f"block_{i}_position_bias"]
                compare_outputs(position_bias_jax, pos_bias_torch, f"Block {i} Position Bias", rtol=1e-5, atol=1e-6)
        else:
            attn_output = block.attn(
                x_jax,
                mask=None,
                pos_bias=position_bias_jax,
                deterministic=True
            )

        # PyTorch attention output (after layer[0] which includes norm + attn + dropout)
        attn_torch = pytorch_intermediates[f"block_{i}_attn_output"]
        compare_outputs(attn_output, attn_torch, f"Block {i} Attention Output", rtol=1e-3, atol=1e-5)

        x_jax = attn_output

        # FFN
        if hasattr(block, 'ffn'):
            ffn_output = block.ffn(x_jax, deterministic=True)
            x_jax = ffn_output

            if f"block_{i}_ffn_output" in pytorch_intermediates:
                ffn_torch = pytorch_intermediates[f"block_{i}_ffn_output"]
                compare_outputs(ffn_output, ffn_torch, f"Block {i} FFN Output", rtol=1e-3, atol=1e-5)

        # Final block output
        block_torch = pytorch_intermediates[f"block_{i}_output"]
        compare_outputs(x_jax, block_torch, f"Block {i} Final Output", rtol=1e-3, atol=1e-5)

    # 5. Final layer norm
    x_jax = jax_t5.encoder.norm(x_jax)
    final_norm_torch = pytorch_intermediates["final_norm"]
    compare_outputs(x_jax, final_norm_torch, "Final Layer Norm", rtol=1e-4, atol=1e-6)

    # 6. Final output
    torch_final = pytorch_output.last_hidden_state
    compare_outputs(x_jax, torch_final, "Final Output", rtol=1e-3, atol=1e-4)

    print("\n" + "="*80)
    print("Intermediate comparison complete!")
    print("="*80)

def test_t5_e2e():
    """Test JAX T5 encoder with PyTorch decoder on end-to-end generation task."""
    print("\n" + "=" * 80)
    print("TEST 2: T5 E2E (JAX Encoder + PyTorch Decoder)")
    print("=" * 80)
    # Test prompts
    test_prompts = [
        "translate English to Chinese: The house is wonderful.",
    ]

    print("\n[1/3] Loading models...")
    tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
    model_ckpt_path = snapshot_download("google/umt5-xxl")

    # Load JAX encoder
    jax_t5 = params.create_t5_encoder_from_safe_tensors(model_ckpt_path, mesh=None, is_sf=False,config=t5.T5Config.umt5_xxl())

    # Load full PyTorch model (encoder + decoder)
    pytorch_full_model = UMT5ForConditionalGeneration.from_pretrained(
        model_ckpt_path,
        torch_dtype=torch.float32
    )
    pytorch_full_model.eval()

    print("\n[2/3] Running generation tests...")

    for i, prompt in enumerate(test_prompts):
        print(f"\n{'='*80}")
        print(f"Test Case {i+1}: {prompt}")
        print(f"{'='*80}")

        # Tokenize
        inputs_jax = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=512, truncation=True)
        inputs_torch = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=512, truncation=True)

        # ============================================================
        # Baseline: Full PyTorch model
        # ============================================================
        print("\n[Baseline] Full PyTorch model:")
        with torch.no_grad():
            pytorch_outputs = pytorch_full_model.generate(
                input_ids=inputs_torch.input_ids,
                attention_mask=inputs_torch.attention_mask,
                max_length=50,
                num_beams=1,  # Greedy decoding
                do_sample=False,
            )

        pytorch_text = tokenizer.decode(pytorch_outputs[0], skip_special_tokens=True)
        print(f"  Output: {pytorch_text}")

        # ============================================================
        # Hybrid: JAX encoder + PyTorch decoder
        # ============================================================
        print("\n[Hybrid] JAX encoder + PyTorch decoder:")

        # Get encoder hidden states from JAX
        input_ids_jax = jnp.array(inputs_jax.input_ids)
        jax_encoder_output = jax_t5(input_ids_jax, deterministic=True)

        # Convert to PyTorch
        encoder_hidden_states = torch.from_numpy(np.array(jax_encoder_output))

        print(f"  JAX encoder output shape: {encoder_hidden_states.shape}")
        print(f"  JAX encoder output range: [{encoder_hidden_states.min():.4f}, {encoder_hidden_states.max():.4f}]")

        # Create encoder outputs object for decoder
        from transformers.modeling_outputs import BaseModelOutput
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_hidden_states
        )

        # Generate using decoder with JAX encoder outputs
        with torch.no_grad():
            hybrid_outputs = pytorch_full_model.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=inputs_torch.attention_mask,
                max_length=50,
                num_beams=1,
                do_sample=False,
            )

        hybrid_text = tokenizer.decode(hybrid_outputs[0], skip_special_tokens=True)
        print(f"  Output: {hybrid_text}")

    print("\n[3/3] Summary")
    print("=" * 80)
    print("E2E test complete. Check outputs above to verify encoder correctness.")


if __name__ == "__main__":
    # Uncomment the test you want to run:
    # test_t5_encoder()           # Test final outputs only
    # test_t5_intermediate()    # Test intermediate layer outputs (detailed)
    test_t5_e2e()              # End-to-end generation test
