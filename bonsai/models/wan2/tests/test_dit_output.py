"""Test output correctness by comparing JAX implementation with HuggingFace reference."""

import jax
import jax.numpy as jnp
import numpy as np
from huggingface_hub import snapshot_download
from bonsai.models.wan2 import params, modeling
import torch
from transformers import AutoTokenizer, UMT5EncoderModel, AutoModel

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

    if isinstance(torch_output, torch.Tensor):
        torch_np = torch_output.detach().cpu().numpy()
    else:
        torch_np = np.array(torch_output)

    jax_np = np.array(jax_output)

    print(f"\n{'=' * 80}")
    print(f"Comparing: {name}")
    print(f"{'=' * 80}")
    print(f"JAX shape:   {jax_np.shape}")
    print(f"Torch shape: {torch_np.shape}")
    print(f"JAX dtype:   {jax_np.dtype}")
    print(f"Torch dtype: {torch_np.dtype}")

    if jax_np.shape != torch_np.shape:
        print("Shape mismatch!")
        return False

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


def test_dit():
    print("\n" + "=" * 80)
    print("TEST 2: DiT")
    print("=" * 80)

    model_ckpt_path = snapshot_download("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    config = modeling.ModelConfig

    print("\n[1/2] Loading transformer")
    transformer = AutoModel.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="transformer", torch_dtype=torch.bfloat16)
    jax_dit = params.create_model_from_safe_tensors(model_ckpt_path, config,mesh=None)
    print("transformer loaded:", transformer, transformer.config)

    batch_size = 1
    num_channels = 16  # in_channels
    num_frames = 9 
    height = 60      
    width = 104      
    text_seq_len = 128  
    text_dim = 4096     # UMT5 hidden dimension

    # Create dummy inputs
    hidden_states = torch.randn(
        batch_size, num_channels, num_frames, height, width,
        dtype=torch.float32
    )
    hidden_states_jax = jnp.array(hidden_states.numpy())
    timestep = torch.randint(
        0, 1000,
        (batch_size,),
        dtype=torch.long
    )
    timestep_jax = jnp.array(timestep.numpy())
    encoder_hidden_states = torch.randn(
        batch_size, text_seq_len, text_dim,
        dtype=torch.float32
    )
    encoder_hidden_states_jax = jnp.array(encoder_hidden_states.numpy())

    # 3. Run forward pass
    with torch.no_grad():
        output = transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=None,  # Only for I2V models
            return_dict=True,
            attention_kwargs=None,
        )
    jax_dit_output = jax_dit.forward(
        hidden_states_jax,
        encoder_hidden_states_jax,
        timestep_jax,
        deterministic=True
    )

    # 4. Verify output shape
    # Output should have same shape as input
    expected_shape = (batch_size, num_channels, num_frames, height, width)
    assert output.sample.shape == expected_shape

    # Compare only the valid portion (ignore padding)
    return compare_outputs(jax_dit_output, output, "Dit", rtol=1e-3, atol=1e-4)

if __name__ == "__main__":
    test_dit()
