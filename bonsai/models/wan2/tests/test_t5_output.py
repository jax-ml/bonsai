#!/usr/bin/env python3
"""
Unit test to extract T5 encoder output from Wan2.1-T2V-1.3B model on CPU.
"""

import os

import torch
from wan.configs import WAN_CONFIGS
from wan.modules.t5 import T5EncoderModel


def test_t5_encoder_output():
    """Test T5 encoder and get its output."""

    # ========================================
    # Configuration
    # ========================================
    ckpt_dir = "/home/gcpuser/sky_workdir/bonsai/Wan2.1-T2V-1.3B"  # Change this to your checkpoint directory
    prompt = "A beautiful sunset over the ocean with waves crashing on the shore"
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

    # ========================================
    # Encode Prompt
    # ========================================
    print("Encoding prompt...")
    context = text_encoder([prompt], device)
    print("Encoding complete!")
    print()

    # ========================================
    # Output Information
    # ========================================
    print("=" * 60)
    print("T5 Encoder Output")
    print("=" * 60)

    # Context is a list of tensors (one per prompt)
    print(f"Number of outputs: {len(context)}")
    print(f"Output shape: {context[0].shape}")
    print(f"Output dtype: {context[0].dtype}")
    print(f"Output device: {context[0].device}")
    print(f"Output range: [{context[0].min().item():.4f}, {context[0].max().item():.4f}]")
    print()

    # Detailed shape breakdown
    seq_len, hidden_dim = context[0].shape
    print("Shape breakdown:")
    print(f"  - Sequence length: {seq_len} (actual tokens, no padding)")
    print(f"  - Hidden dimension: {hidden_dim} (T5-XXL dimension)")
    print()

    # ========================================
    # Sample Output Values
    # ========================================
    print("Sample output values (first 10 tokens, first 5 dims):")
    print(context[0][:10, :5])
    print()

    # ========================================
    # Statistics
    # ========================================
    print("Statistics:")
    print(f"  - Mean: {context[0].mean().item():.6f}")
    print(f"  - Std:  {context[0].std().item():.6f}")
    print(f"  - Min:  {context[0].min().item():.6f}")
    print(f"  - Max:  {context[0].max().item():.6f}")
    print()

    # ========================================
    # Save Output (Optional)
    # ========================================
    output_file = "t5_encoder_output.pt"
    torch.save(
        {
            "prompt": prompt,
            "context": context[0],
            "config": {
                "text_len": config.text_len,
                "dtype": str(config.t5_dtype),
                "actual_length": seq_len,
                "hidden_dim": hidden_dim,
            },
        },
        output_file,
    )
    print(f"Output saved to: {output_file}")
    print()

    # ========================================
    # Test Multiple Prompts
    # ========================================
    print("=" * 60)
    print("Testing Multiple Prompts")
    print("=" * 60)

    prompts = ["A cat walking on the street", "A beautiful sunset over the ocean", "Two anthropomorphic cats boxing"]

    print(f"Number of prompts: {len(prompts)}")
    contexts = text_encoder(prompts, device)

    for i, (prompt, ctx) in enumerate(zip(prompts, contexts)):
        print(f"\nPrompt {i + 1}: {prompt}")
        print(f"  Shape: {ctx.shape}")
        print(f"  Length: {ctx.shape[0]} tokens")

    print()
    print("=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

    return context[0]


if __name__ == "__main__":
    # Run the test
    try:
        output = test_t5_encoder_output()
        print(f"\nFinal output shape: {output.shape}")
        print(f"Final output dtype: {output.dtype}")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback

        traceback.print_exc()
