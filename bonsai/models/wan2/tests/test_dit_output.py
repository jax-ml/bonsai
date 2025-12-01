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
    # torch :(out, in, t, h, w)
    torch_emb = torch_model.patch_embedding.weight.detach().cpu().numpy()
    # jax: (t, h, w, in, out)
    jax_emb = np.array(jax_model.patch_embed.kernel.value).transpose(4,3,0,1,2)

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
    height = 30      
    width = 30      
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

    # Manual forward pass with intermediate comparisons
    print("\n" + "=" * 80)
    print("STEP-BY-STEP FORWARD PASS WITH COMPARISONS")
    print("=" * 80)

    # 1. Text projection
    text_embeds_jax = jax_dit.text_proj(encoder_hidden_states_jax)
    text_embeds_torch = intermediate_outputs['condition_encoder_hidden_states']
    compare_outputs(text_embeds_jax, text_embeds_torch, "Text Projection", rtol=1e-3, atol=1e-4)

    # 2. Patch embedding
    x_jax = jax_dit.patch_embed(hidden_states_jax)
    # PyTorch is BCTHW, need to convert to BTHWC for comparison
    patch_embed_torch = intermediate_outputs['patch_embed_output']
    patch_embed_torch_channels_last = np.transpose(patch_embed_torch.numpy(), (0, 2, 3, 4, 1))
    compare_outputs(x_jax, patch_embed_torch_channels_last, "Patch Embedding", rtol=1e-3, atol=1e-4)

    # Reshape to sequence
    b, t_out, h_out, w_out, d = x_jax.shape
    x_jax = x_jax.reshape(b, t_out * h_out * w_out, d)
    grid_sizes = (t_out, h_out, w_out)

    # 3. RoPE frequencies
    max_seq = max(grid_sizes)
    rope_freqs = tuple(
        jax.lax.stop_gradient(arr)
        for arr in modeling.precompute_freqs_cis_3d(
            dim=jax_dit.cfg.head_dim,
            theta=jax_dit.rope_theta,
            max_seq_len=max_seq
        )
    )

    # Build full RoPE frequency grid for comparison
    freqs_t, freqs_h, freqs_w = rope_freqs
    f, h, w = grid_sizes
    head_dim = jax_dit.cfg.head_dim
    dim_base = head_dim // 6
    dim_t, dim_h, dim_w = head_dim - 4 * dim_base, 2 * dim_base, 2 * dim_base

    freqs_grid = jnp.concatenate([
        jnp.broadcast_to(freqs_t[:f, None, None, :, :], (f, h, w, dim_t // 2, 2)),
        jnp.broadcast_to(freqs_h[None, :h, None, :, :], (f, h, w, dim_h // 2, 2)),
        jnp.broadcast_to(freqs_w[None, None, :w, :, :], (f, h, w, dim_w // 2, 2)),
    ], axis=3).reshape(t_out * h_out * w_out, head_dim // 2, 2)

    rope_freqs_cos_jax = freqs_grid[..., 0]
    rope_freqs_cos_jax = jnp.stack([rope_freqs_cos_jax, rope_freqs_cos_jax], axis=-1).reshape(1, -1, 1, head_dim)

    # PyTorch RoPE freqs are in BCHW format, convert to sequence format
    rope_freqs_cos_torch = intermediate_outputs['rope_freqs_cos']
    compare_outputs(rope_freqs_cos_jax, rope_freqs_cos_torch, "RoPE Freqs Cos", rtol=1e-5, atol=1e-6)

    # 4. Time embeddings
    time_emb_jax, time_proj_jax = jax_dit.time_embed(timestep_jax)
    time_emb_torch = intermediate_outputs['condition_temb']
    time_proj_torch = intermediate_outputs['condition_timestep_proj']
    compare_outputs(time_emb_jax, time_emb_torch, "Time Embedding", rtol=1e-3, atol=1e-4)
    compare_outputs(time_proj_jax, time_proj_torch, "Time Projection", rtol=1e-3, atol=1e-4)

    # 5. Process through transformer blocks with detailed attention comparison
    for i, block in enumerate(jax_dit.blocks):
        print(f"\n{'='*80}")
        print(f"BLOCK {i} - DETAILED COMPARISON")
        print(f"{'='*80}")

        # Get modulation parameters
        b_size = time_proj_jax.shape[0]
        d = jax_dit.cfg.hidden_dim
        reshaped_time_emb = time_proj_jax.reshape(b_size, 6, d)
        modulation = jax.nn.silu(reshaped_time_emb + block.scale_shift_table.value)
        modulation = modulation.reshape(b_size, -1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(modulation, 6, axis=-1)

        # Self-attention with detailed steps
        print(f"\n--- Self-Attention ---")
        norm_x = block.norm1(x_jax)
        norm_x_modulated = norm_x * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]

        # Q, K, V projections
        num_heads = block.self_attn.num_heads
        head_dim = block.self_attn.head_dim
        b_size, n = norm_x_modulated.shape[:2]

        q = block.self_attn.q_norm(block.self_attn.q_proj(norm_x_modulated))
        k = block.self_attn.k_norm(block.self_attn.k_proj(norm_x_modulated))
        v = block.self_attn.v_proj(norm_x_modulated)

        print(f"Q after norm (before reshape): shape={q.shape}, range=[{q.min():.4f}, {q.max():.4f}]")
        print(f"K after norm (before reshape): shape={k.shape}, range=[{k.min():.4f}, {k.max():.4f}]")
        print(f"V (before reshape): shape={v.shape}, range=[{v.min():.4f}, {v.max():.4f}]")

        # Reshape to heads
        q = q.reshape(b_size, n, num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(b_size, n, num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(b_size, n, num_heads, head_dim).transpose(0, 2, 1, 3)

        print(f"Q after reshape: shape={q.shape}, range=[{q.min():.4f}, {q.max():.4f}]")
        print(f"K after reshape: shape={k.shape}, range=[{k.min():.4f}, {k.max():.4f}]")
        print(f"V after reshape: shape={v.shape}, range=[{v.min():.4f}, {v.max():.4f}]")

        # Apply RoPE
        q_before_rope = q.copy()
        k_before_rope = k.copy()

        q, k = jnp.transpose(q, (0, 2, 1, 3)), jnp.transpose(k, (0, 2, 1, 3))
        q = modeling.rope_apply(q, grid_sizes, rope_freqs)
        k = modeling.rope_apply(k, grid_sizes, rope_freqs)
        q, k = jnp.transpose(q, (0, 2, 1, 3)), jnp.transpose(k, (0, 2, 1, 3))

        print(f"Q after RoPE: shape={q.shape}, range=[{q.min():.4f}, {q.max():.4f}]")
        print(f"K after RoPE: shape={k.shape}, range=[{k.min():.4f}, {k.max():.4f}]")

        # Attention scores
        attn_scores = jnp.einsum("bhid,bhjd->bhij", q, k) / jnp.sqrt(head_dim)
        print(f"Attention scores (before softmax): shape={attn_scores.shape}, range=[{attn_scores.min():.4f}, {attn_scores.max():.4f}]")

        # Attention weights
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)
        print(f"Attention weights (after softmax): shape={attn_weights.shape}, range=[{attn_weights.min():.4f}, {attn_weights.max():.4f}]")

        # Attention output
        attn_out = jnp.einsum("bhij,bhjd->bhid", attn_weights, v)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(b_size, n, -1)
        print(f"Attention output (before proj): shape={attn_out.shape}, range=[{attn_out.min():.4f}, {attn_out.max():.4f}]")

        # Output projection
        attn_out = block.self_attn.out_proj(attn_out)
        print(f"Attention output (after proj): shape={attn_out.shape}, range=[{attn_out.min():.4f}, {attn_out.max():.4f}]")

        # Compare with PyTorch
        attn1_output_torch = intermediate_outputs[f'block_{i}_attn1_output']
        compare_outputs(attn_out, attn1_output_torch, f"Block {i} Attn1 Output", rtol=1e-2, atol=1e-3)

        # Apply gate and residual
        x_jax = x_jax + gate_msa[:, None, :] * attn_out

        # Cross-attention
        print(f"\n--- Cross-Attention ---")
        norm_x = block.norm2(x_jax)
        cross_out = block.cross_attn(norm_x, text_embeds_jax, deterministic=True)

        attn2_output_torch = intermediate_outputs[f'block_{i}_attn2_output']
        compare_outputs(cross_out, attn2_output_torch, f"Block {i} Attn2 Output", rtol=1e-2, atol=1e-3)

        x_jax = x_jax + cross_out

        # MLP
        print(f"\n--- MLP ---")
        norm_x = block.norm3(x_jax)
        norm_x_modulated = norm_x * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        mlp_out = block.mlp(norm_x_modulated)

        ffn_output_torch = intermediate_outputs[f'block_{i}_ffn_output']
        compare_outputs(mlp_out, ffn_output_torch, f"Block {i} FFN Output", rtol=1e-2, atol=1e-3)

        x_jax = x_jax + gate_mlp[:, None, :] * mlp_out

        # Compare final block output
        block_output_torch = intermediate_outputs[f'block_{i}_output']
        compare_outputs(x_jax, block_output_torch, f"Block {i} Final Output", rtol=1e-2, atol=1e-3)

        if i >= 0:  # Only compare first block in detail
            break

    # 6. Final layer
    jax_dit_output = jax_dit.final_layer(x_jax, time_emb_jax)

    # Compare with final norm output
    norm_out_torch = intermediate_outputs['norm_out_output']
    # Note: final_layer does norm + linear, so we can't directly compare with norm_out

    # Reshape to video format
    jax_dit_output = jax_dit.unpatchify(jax_dit_output, grid_sizes)

    # 4. Verify output shape
    expected_shape = (batch_size, num_channels, num_frames, height, width)
    assert output.sample.shape == expected_shape

    # change to channels last for comparison
    expected_output = np.transpose(output.sample.numpy(), (0, 2, 3, 4, 1))

    debugger.remove_hooks()
    # Compare final output
    return compare_outputs(jax_dit_output, expected_output, "Final DiT Output", rtol=1e-3, atol=1e-4)

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

if __name__ == "__main__":
    test_dit()


