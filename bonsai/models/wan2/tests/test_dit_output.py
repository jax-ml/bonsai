"""Test output correctness by comparing JAX implementation with HuggingFace reference."""

import jax
import jax.numpy as jnp
import numpy as np
from huggingface_hub import snapshot_download
from bonsai.models.wan2 import params, modeling
import torch
from diffusers import AutoModel
from collections import OrderedDict

def check_weight_loading(jax_model, torch_model):
    torch_emb = torch_model.patch_embedding.weight.detach().cpu().numpy()
    jax_emb = np.array(jax_model.patch_embed.kernel.value).transpose(2, 3, 4, 1, 0)

    print("Embedding weights:")
    print(f"  Shapes: torch={torch_emb.shape}, jax={jax_emb.shape}")
    print(f"  Max diff: {np.abs(torch_emb - jax_emb).max():.2e}")
    print(f"  Mean diff: {np.abs(torch_emb - jax_emb).mean():.2e}")
    
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

def check_intermediate_weights(jax_model, torch_model):


def test_dit():
    print("\n" + "=" * 80)
    print("TEST 2: DiT")
    print("=" * 80)

    model_ckpt_path = snapshot_download("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    config = modeling.ModelConfig

    print("\n[1/2] Loading transformer")
    transformer = AutoModel.from_pretrained(model_ckpt_path, subfolder="transformer", torch_dtype=torch.float32)

    jax_dit = params.create_model_from_safe_tensors(model_ckpt_path, config,mesh=None)
    print("transformer loaded:", transformer, transformer.config)

    check_weight_loading(jax_dit, transformer)

    batch_size = 1
    num_channels = 16  # in_channels
    num_frames = 9 
    height = 60      
    width = 104      
    text_seq_len = 128  
    text_dim = 4096     # UMT5 hidden dimension

    debugger = WanTransformerDebugger(transformer)
    debugger.register_hooks()

    # Create dummy inputs
    hidden_states = torch.randn(
        batch_size, num_channels, num_frames, height, width,
        dtype=torch.float32
    )
    # jax channels last
    hidden_states_jax = jnp.array(np.transpose(hidden_states.numpy(), (0, 2, 3, 4, 1)))    
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

    print("\n[2/2] Running forward pass")
    with torch.no_grad():
        output = transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=None,  # Only for I2V models
            return_dict=True,
            attention_kwargs=None,
        )
        
    # 5. Get intermediate outputs
    intermediate_outputs = debugger.get_outputs()

    print("=" * 80)
    print("INTERMEDIATE OUTPUTS")
    print("=" * 80)

    # Print all intermediate output shapes
    for name, tensor in intermediate_outputs.items():
        if isinstance(tensor, torch.Tensor):
            print(f"{name:50s} : {tuple(tensor.shape)}")

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

    # change to channels last for comparison
    expected_output = np.transpose(output.sample.numpy(), (0, 2, 3, 4, 1))

    # Compare only the valid portion (ignore padding)
    return compare_outputs(jax_dit_output, expected_output, "Dit", rtol=1e-3, atol=1e-4)

class WanTransformerDebugger:
    """Helper class to extract intermediate outputs from WanTransformer3DModel"""

    def __init__(self, model):
        self.model = model
        self.intermediate_outputs = OrderedDict()
        self.hooks = []

    def register_hooks(self):
        """Register forward hooks to capture intermediate outputs"""

        # Hook for patch embedding
        def patch_embed_hook(module, input, output):
            self.intermediate_outputs['patch_embed_output'] = output.detach().cpu()

        # Hook for rotary embeddings
        def rope_hook(module, input, output):
            freqs_cos, freqs_sin = output
            self.intermediate_outputs['rope_freqs_cos'] = freqs_cos.detach().cpu()
            self.intermediate_outputs['rope_freqs_sin'] = freqs_sin.detach().cpu()

        # Hook for condition embedder
        def condition_embedder_hook(module, input, output):
            temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = output
            self.intermediate_outputs['condition_temb'] = temb.detach().cpu()
            self.intermediate_outputs['condition_timestep_proj'] = timestep_proj.detach().cpu()
            self.intermediate_outputs['condition_encoder_hidden_states'] = encoder_hidden_states.detach().cpu()
            if encoder_hidden_states_image is not None:
                self.intermediate_outputs['condition_encoder_hidden_states_image'] = encoder_hidden_states_image.detach().cpu()

        # Hook for each transformer block
        for i, block in enumerate(self.model.blocks):
            def make_block_hook(block_idx):
                def block_hook(module, input, output):
                    self.intermediate_outputs[f'block_{block_idx}_output'] = output.detach().cpu()
                return block_hook

            # Hook for block output
            handle = block.register_forward_hook(make_block_hook(i))
            self.hooks.append(handle)

            # Hook for self-attention in each block
            def make_attn1_hook(block_idx):
                def attn1_hook(module, input, output):
                    self.intermediate_outputs[f'block_{block_idx}_attn1_output'] = output.detach().cpu()
                return attn1_hook

            handle = block.attn1.register_forward_hook(make_attn1_hook(i))
            self.hooks.append(handle)

            # Hook for cross-attention in each block
            def make_attn2_hook(block_idx):
                def attn2_hook(module, input, output):
                    self.intermediate_outputs[f'block_{block_idx}_attn2_output'] = output.detach().cpu()
                return attn2_hook

            handle = block.attn2.register_forward_hook(make_attn2_hook(i))
            self.hooks.append(handle)

            # Hook for FFN in each block
            def make_ffn_hook(block_idx):
                def ffn_hook(module, input, output):
                    self.intermediate_outputs[f'block_{block_idx}_ffn_output'] = output.detach().cpu()
                return ffn_hook

            handle = block.ffn.register_forward_hook(make_ffn_hook(i))
            self.hooks.append(handle)

        # Hook for patch embedding
        handle = self.model.patch_embedding.register_forward_hook(patch_embed_hook)
        self.hooks.append(handle)

        # Hook for rope
        handle = self.model.rope.register_forward_hook(rope_hook)
        self.hooks.append(handle)

        # Hook for condition embedder
        handle = self.model.condition_embedder.register_forward_hook(condition_embedder_hook)
        self.hooks.append(handle)

        # Hook for final norm
        def norm_out_hook(module, input, output):
            self.intermediate_outputs['norm_out_output'] = output.detach().cpu()

        handle = self.model.norm_out.register_forward_hook(norm_out_hook)
        self.hooks.append(handle)

        # Hook for final projection
        def proj_out_hook(module, input, output):
            self.intermediate_outputs['proj_out_output'] = output.detach().cpu()

        handle = self.model.proj_out.register_forward_hook(proj_out_hook)
        self.hooks.append(handle)

    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_outputs(self):
        """Get all captured intermediate outputs"""
        return self.intermediate_outputs

    def clear_outputs(self):
        """Clear stored outputs"""
        self.intermediate_outputs = OrderedDict()


def test_wan_transformer_with_intermediate_outputs():
    """
    Test WanTransformer3DModel and extract all intermediate outputs
    """
    # 1. Create model with Wan2.1-T2V-1.3B config
    config = {
        "patch_size": (1, 2, 2),
        "num_attention_heads": 12,
        "attention_head_dim": 128,
        "in_channels": 16,
        "out_channels": 16,
        "text_dim": 4096,
        "freq_dim": 256,
        "ffn_dim": 8960,
        "num_layers": 30,
        "cross_attn_norm": True,
        "qk_norm": "rms_norm_across_heads",
        "eps": 1e-6,
        "added_kv_proj_dim": None,
        "rope_max_seq_len": 1024,
    }

    model = WanTransformer3DModel(**config)
    model.eval()

    # 2. Create debugger and register hooks


    # 3. Prepare inputs
    batch_size = 1
    num_channels = 16
    num_frames = 9
    height = 60
    width = 104
    text_seq_len = 512
    text_dim = 4096

    # Set seed for reproducibility
    torch.manual_seed(42)

    hidden_states = torch.randn(
        batch_size, num_channels, num_frames, height, width,
        dtype=torch.float32
    )

    timestep = torch.tensor([500], dtype=torch.long)

    encoder_hidden_states = torch.randn(
        batch_size, text_seq_len, text_dim,
        dtype=torch.float32
    )

    print("Input shapes:")
    print(f"  hidden_states: {hidden_states.shape}")
    print(f"  timestep: {timestep.shape}")
    print(f"  encoder_hidden_states: {encoder_hidden_states.shape}")
    print()

    # 4. Run forward pass
    with torch.no_grad():
        output = model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=None,
            return_dict=True,
            attention_kwargs=None,
        )

    # 5. Get intermediate outputs
    intermediate_outputs = debugger.get_outputs()

    print("=" * 80)
    print("INTERMEDIATE OUTPUTS")
    print("=" * 80)

    # Print all intermediate output shapes
    for name, tensor in intermediate_outputs.items():
        if isinstance(tensor, torch.Tensor):
            print(f"{name:50s} : {tuple(tensor.shape)}")

    print("=" * 80)
    print(f"Final output shape: {output.sample.shape}")
    print("=" * 80)

    # 6. Save outputs for comparison
    outputs_dict = {
        'inputs': {
            'hidden_states': hidden_states.cpu(),
            'timestep': timestep.cpu(),
            'encoder_hidden_states': encoder_hidden_states.cpu(),
        },
        'intermediate': intermediate_outputs,
        'output': output.sample.cpu(),
    }

    # Save to file
    torch.save(outputs_dict, 'wan_transformer_outputs.pt')
    print("\n✓ Saved outputs to 'wan_transformer_outputs.pt'")

    # 7. Clean up
    debugger.remove_hooks()

    return outputs_dict


def compare_specific_layers(outputs_dict_diffusers, outputs_dict_yours):
    """
    Compare specific layer outputs between two implementations
    
    Args:
        outputs_dict_diffusers: Output dict from WanTransformer3DModel
        outputs_dict_yours: Output dict from your implementation
    """
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    # Compare each intermediate output
    diffusers_intermediate = outputs_dict_diffusers['intermediate']
    yours_intermediate = outputs_dict_yours['intermediate']

    for name in diffusers_intermediate.keys():
        if name not in yours_intermediate:
            print(f"⚠ {name:50s} : MISSING in your implementation")
            continue

        diffusers_tensor = diffusers_intermediate[name]
        yours_tensor = yours_intermediate[name]

        # Check shape match
        if diffusers_tensor.shape != yours_tensor.shape:
            print(f"✗ {name:50s} : SHAPE MISMATCH")
            print(f"    Diffusers: {diffusers_tensor.shape}")
            print(f"    Yours:     {yours_tensor.shape}")
            continue

        # Calculate difference metrics
        abs_diff = torch.abs(diffusers_tensor - yours_tensor)
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        rel_diff = (abs_diff / (torch.abs(diffusers_tensor) + 1e-8)).mean().item()

        # Determine if match is good
        if max_diff < 1e-5:
            status = "✓ EXACT MATCH"
        elif max_diff < 1e-3:
            status = "✓ CLOSE MATCH"
        elif max_diff < 1e-1:
            status = "⚠ SLIGHT DIFF"
        else:
            status = "✗ LARGE DIFF"

        print(f"{status:15s} {name:50s} : max={max_diff:.2e}, mean={mean_diff:.2e}, rel={rel_diff:.2e}")

    # Compare final output
    diffusers_output = outputs_dict_diffusers['output']
    yours_output = outputs_dict_yours['output']

    abs_diff = torch.abs(diffusers_output - yours_output)
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    print("=" * 80)
    print(f"Final Output Difference: max={max_diff:.2e}, mean={mean_diff:.2e}")
    print("=" * 80)


if __name__ == "__main__":
    test_dit()


