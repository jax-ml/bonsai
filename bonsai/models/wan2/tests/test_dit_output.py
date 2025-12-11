"""Test output correctness by comparing JAX implementation with HuggingFace reference."""

import jax
import jax.numpy as jnp
import numpy as np
from huggingface_hub import snapshot_download
from bonsai.models.wan2 import params, modeling
import torch
from diffusers import AutoModel
from jax.lax import Precision
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

    # check fused kv projection weights in first block
    torch_k_weight = torch_model.blocks[0].attn2.to_k.weight.detach().cpu().numpy().T
    torch_v_weight = torch_model.blocks[0].attn2.to_v.weight.detach().cpu().numpy().T
    torch_kv = np.concatenate([torch_k_weight, torch_v_weight], axis=1)

    jax_kv_weight = np.array(jax_model.blocks[0].cross_attn.kv_proj.kernel.value)
    print("First block cross-attention KV projection weights:")
    print(f"  Shapes: torch={torch_kv.shape}, jax={jax_kv_weight.shape}")
    print(f"  Max diff: {np.abs(torch_kv - jax_kv_weight).max():.2e}")
    print(f"  Mean diff: {np.abs(torch_kv - jax_kv_weight).mean():.2e}")
    
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

def test_dit_output():
    print("\n" + "=" * 80)
    print("TEST 2: DiT")
    print("=" * 80)

    model_ckpt_path = snapshot_download("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    config = modeling.ModelConfig

    print("\n[1/2] Loading transformer")
    transformer = AutoModel.from_pretrained(model_ckpt_path, subfolder="transformer", torch_dtype=torch.float32)

    jax_dit = params.create_model_from_safe_tensors(model_ckpt_path, config,mesh=None)
    print("transformer loaded:", transformer, transformer.config)

    batch_size = 1
    num_channels = 16  # in_channels
    num_frames = 9 
    height = 30      
    width = 30      
    text_seq_len = 128  
    text_dim = 4096     # UMT5 hidden dimension

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
    pred_noise = jax_dit.forward(hidden_states_jax, encoder_hidden_states_jax, timestep_jax, deterministic=True)
    expected_output = np.transpose(output.sample.numpy(), (0, 2, 3, 4, 1))
    
    # Compare final output
    return compare_outputs(pred_noise, expected_output, "Final DiT Output", rtol=1e-3, atol=1e-4)
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
    debugger_attn = WanAttentionDebugger(transformer)
    debugger_attn.register_attention_hooks(block_indices=[0])

    # Create dummy inputs
    hidden_states = torch.randn(
        batch_size, num_channels, num_frames, height, width,
        dtype=torch.float32
    )
    # jax channels last
    hidden_states_jax = jnp.array(np.transpose(hidden_states.numpy(), (0, 2, 3, 4, 1))).astype(jnp.bfloat16)
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
    encoder_hidden_states_jax = jnp.array(encoder_hidden_states.numpy()).astype(jnp.bfloat16)

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
    states = debugger_attn.get_attention_states()

    print("=" * 80)
    print("INTERMEDIATE OUTPUTS")
    print("=" * 80)

    # # Print all intermediate output shapes
    # for name, tensor in intermediate_outputs.items():
    #     if isinstance(tensor, torch.Tensor):
    #         print(f"{name:50s} : {tuple(tensor.shape)}")

    for name, tensor in states.items():
        print(f"{name:50s}: {tuple(tensor.shape)}")

    # Restore original processors
    debugger_attn.restore_processors()

    # Manual forward pass with intermediate comparisons
    print("\n" + "=" * 80)
    print("STEP-BY-STEP FORWARD PASS WITH COMPARISONS")
    print("=" * 80)

    # 1. Text projection
    text_embeds_jax = jax_dit.text_proj(encoder_hidden_states_jax)
    text_embeds_torch = intermediate_outputs['condition_encoder_hidden_states']
    # compare_outputs(text_embeds_jax, text_embeds_torch, "Text Projection", rtol=1e-3, atol=1e-4)

    # 2. Patch embedding
    x_jax = jax_dit.patch_embed(hidden_states_jax)
    # PyTorch is BCTHW, need to convert to BTHWC for comparison
    patch_embed_torch = intermediate_outputs['patch_embed_output']
    patch_embed_torch_channels_last = np.transpose(patch_embed_torch.numpy(), (0, 2, 3, 4, 1))
    # compare_outputs(x_jax, patch_embed_torch_channels_last, "Patch Embedding", rtol=1e-3, atol=1e-4)

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
    # compare_outputs(rope_freqs_cos_jax, rope_freqs_cos_torch, "RoPE Freqs Cos", rtol=1e-5, atol=1e-6)

    # 4. Time embeddings
    time_emb_jax, time_proj_jax = jax_dit.time_embed(timestep_jax)
    time_emb_torch = intermediate_outputs['condition_temb']
    time_proj_torch = intermediate_outputs['condition_timestep_proj']
    # compare_outputs(time_emb_jax, time_emb_torch, "Time Embedding", rtol=1e-3, atol=1e-4)
    # compare_outputs(time_proj_jax, time_proj_torch, "Time Projection", rtol=1e-3, atol=1e-4)

    # 5. Process through transformer blocks with detailed attention comparison
    for i, block in enumerate(jax_dit.blocks):
        if i==0:
            print(f"\n{'='*80}")
            print(f"BLOCK {i} - DETAILED COMPARISON")
            print(f"{'='*80}")

            # Get modulation parameters
            b_size = time_proj_jax.shape[0]
            d = jax_dit.cfg.hidden_dim
            reshaped_time_emb = time_proj_jax.reshape(b_size, 6, d)
            modulation = reshaped_time_emb + block.scale_shift_table.value
            modulation = modulation.reshape(b_size, -1)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(modulation, 6, axis=-1)

            # Self-attention with detailed steps
            print(f"\n--- Self-Attention ---")
            norm_x = block.norm1(x_jax)
            norm_x_modulated = norm_x * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]

            attn_out = block.self_attn(norm_x_modulated, deterministic=True, rope_state=(rope_freqs, grid_sizes))

            # # Q, K, V projections
            num_heads = block.self_attn.num_heads
            head_dim = block.self_attn.head_dim
            b_size, n = norm_x_modulated.shape[:2]

            # Compare with PyTorch
            attn1_output_torch = intermediate_outputs[f'block_{i}_attn1_output']
            compare_outputs(attn_out, attn1_output_torch, f"Block {i} Attn1 Output", rtol=1e-2, atol=1e-3)

            # Apply gate and residual
            x_jax = x_jax + gate_msa[:, None, :] * attn_out

            # Cross-attention
            print(f"\n--- Cross-Attention ---")
            norm_x = block.norm2(x_jax)
            compare_outputs(norm_x, intermediate_outputs[f'block_{i}_norm2_output'], f"Block {i} Norm2 Output", rtol=1e-3, atol=1e-4)
            b, n, m = norm_x.shape[0], norm_x.shape[1], text_embeds_jax.shape[1]
            q_norm = block.cross_attn.q_norm(block.cross_attn.q_proj(norm_x))
            compare_outputs(q_norm, intermediate_outputs[f'block_{i}_attn2_query_normed'], f"Block {i} Attn2 Q after Norm", rtol=1e-5, atol=1e-6)
            kv = block.cross_attn.kv_proj(text_embeds_jax)
            k, v = jnp.split(kv, 2, axis=-1)
            k_norm = block.cross_attn.k_norm(k)
            compare_outputs(k_norm, intermediate_outputs[f'block_{i}_attn2_key_normed'], f"Block {i} Attn2 K after Norm", rtol=1e-5, atol=1e-6)

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

        if i > 0:  # Only compare first block in detail
            x_jax = block(x_jax, text_embeds_jax, time_proj_jax, deterministic=True, rope_state=(rope_freqs, grid_sizes))
            compare_outputs(x_jax, intermediate_outputs[f'block_{i}_output'], f"Block {i} Output", rtol=1e-2, atol=1e-3)

    # 6. Final layer
    jax_dit_output = jax_dit.final_layer(x_jax, time_emb_jax)

    compare_outputs(jax_dit_output, intermediate_outputs['proj_out_output'], "Final Projection Output", rtol=1e-3, atol=1e-4)

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
            
            def make_hook(name):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        self.intermediate_outputs[name] = output[0].detach().cpu()
                    else:
                        self.intermediate_outputs[name] = output.detach().cpu()
                return hook

            handle = block.attn2.register_forward_hook(make_attn2_hook(i))
            self.hooks.append(handle)
            handle = block.norm2.register_forward_hook(make_hook(f'block_{i}_norm2_output'))
            self.hooks.append(handle)

            attn = block.attn2
          # 1. Hook Q projection
            h = attn.to_q.register_forward_hook(
                make_hook(f'block_{i}_attn2_query')
            )
            self.hooks.append(h)

            # 2. Hook K projection
            h = attn.to_k.register_forward_hook(
                make_hook(f'block_{i}_attn2_key')
            )
            self.hooks.append(h)

            # 3. Hook V projection
            h = attn.to_v.register_forward_hook(
                make_hook(f'block_{i}_attn2_value')
            )
            self.hooks.append(h)

            # 4. Hook Q norm
            h = attn.norm_q.register_forward_hook(
                make_hook(f'block_{i}_attn2_query_normed')
            )
            self.hooks.append(h)

            # 5. Hook K norm
            h = attn.norm_k.register_forward_hook(
                make_hook(f'block_{i}_attn2_key_normed')
            )
            self.hooks.append(h)

            # 6. Hook output projection
            h = attn.to_out[0].register_forward_hook(
                make_hook(f'block_{i}_attn2_output')
            )
            self.hooks.append(h)
            h = attn.register_forward_hook(make_hook(f'block_{i}_attn2_attention'))
            self.hooks.append(h)


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

class WanAttentionDebugger:
    """Capture internal attention states (Q, K, V, attention scores, etc.)"""

    def __init__(self, model):
        self.model = model
        self.attention_states = OrderedDict()
        self.hooks = []
        self.original_processors = {}

    def register_attention_hooks(self, block_indices=None):
        """
        Register hooks to capture attention internal states.
        
        Args:
            block_indices: List of block indices to hook, or None for all blocks
        """
        if block_indices is None:
            block_indices = range(len(self.model.blocks))

        for i in block_indices:
            block = self.model.blocks[i]

            # Hook self-attention (attn1)
            self._hook_attention_module(block.attn1, f'block_{i}_attn1')

            # Hook cross-attention (attn2)
            self._hook_attention_module(block.attn2, f'block_{i}_attn2')

    def _hook_attention_module(self, attn_module, prefix):
        """Hook a single attention module to capture Q, K, V, and attention outputs"""

        # Save original processor
        self.original_processors[prefix] = attn_module.processor

        # Create custom processor that captures intermediates
        original_processor = attn_module.processor
        attention_states = self.attention_states

        class InstrumentedProcessor:
            """Wrapper processor that captures intermediate values"""

            def __call__(
                self,
                attn,
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=None,
                rotary_emb=None,
                **kwargs
            ):
                # Get encoder hidden states
                encoder_hidden_states_img = None
                if attn.add_k_proj is not None and encoder_hidden_states is not None:
                    image_context_length = encoder_hidden_states.shape[1] - 512
                    encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
                    encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

                # 1. Capture Q, K, V projections (before normalization)
                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states

                if attn.fused_projections:
                    if attn.cross_attention_dim_head is None:
                        query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
                    else:
                        query = attn.to_q(hidden_states)
                        key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
                else:
                    query = attn.to_q(hidden_states)
                    key = attn.to_k(encoder_hidden_states)
                    value = attn.to_v(encoder_hidden_states)

                attention_states[f'{prefix}_query_raw'] = query.detach().cpu()
                attention_states[f'{prefix}_key_raw'] = key.detach().cpu()
                attention_states[f'{prefix}_value_raw'] = value.detach().cpu()

                # 2. Capture after normalization
                query = attn.norm_q(query)
                key = attn.norm_k(key)

                attention_states[f'{prefix}_query_normed'] = query.detach().cpu()
                attention_states[f'{prefix}_key_normed'] = key.detach().cpu()

                # 3. Reshape to heads
                query = query.unflatten(2, (attn.heads, -1))
                key = key.unflatten(2, (attn.heads, -1))
                value = value.unflatten(2, (attn.heads, -1))

                attention_states[f'{prefix}_query_heads'] = query.detach().cpu()
                attention_states[f'{prefix}_key_heads'] = key.detach().cpu()
                attention_states[f'{prefix}_value_heads'] = value.detach().cpu()

                # 4. Capture after RoPE (if applied)
                if rotary_emb is not None:
                    def apply_rotary_emb(hidden_states, freqs_cos, freqs_sin):
                        x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                        cos = freqs_cos[..., 0::2]
                        sin = freqs_sin[..., 1::2]
                        out = torch.empty_like(hidden_states)
                        out[..., 0::2] = x1 * cos - x2 * sin
                        out[..., 1::2] = x1 * sin + x2 * cos
                        return out.type_as(hidden_states)

                    query = apply_rotary_emb(query, *rotary_emb)
                    key = apply_rotary_emb(key, *rotary_emb)

                    attention_states[f'{prefix}_query_rope'] = query.detach().cpu()
                    attention_states[f'{prefix}_key_rope'] = key.detach().cpu()

                # 5. Handle I2V additional K, V
                hidden_states_img = None
                if encoder_hidden_states_img is not None:
                    if attn.fused_projections:
                        key_img, value_img = attn.to_added_kv(encoder_hidden_states_img).chunk(2, dim=-1)
                    else:
                        key_img = attn.add_k_proj(encoder_hidden_states_img)
                        value_img = attn.add_v_proj(encoder_hidden_states_img)

                    key_img = attn.norm_added_k(key_img)
                    key_img = key_img.unflatten(2, (attn.heads, -1))
                    value_img = value_img.unflatten(2, (attn.heads, -1))

                    attention_states[f'{prefix}_key_img'] = key_img.detach().cpu()
                    attention_states[f'{prefix}_value_img'] = value_img.detach().cpu()

                    # Compute image attention (for I2V)
                    from diffusers.models.attention_dispatch import dispatch_attention_fn
                    hidden_states_img = dispatch_attention_fn(
                        query, key_img, value_img,
                        attn_mask=None, dropout_p=0.0, is_causal=False,
                        backend=original_processor._attention_backend,
                    )
                    hidden_states_img = hidden_states_img.flatten(2, 3)

                    attention_states[f'{prefix}_img_attention_output'] = hidden_states_img.detach().cpu()

                # 6. Compute main attention
                from diffusers.models.attention_dispatch import dispatch_attention_fn

                # Note: We can't easily capture attention weights with dispatch_attention_fn
                # because it uses optimized kernels (flash attention, etc.)
                # For debugging attention weights, we'd need to use manual computation

                hidden_states = dispatch_attention_fn(
                    query, key, value,
                    attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
                    backend=original_processor._attention_backend,
                )

                hidden_states = hidden_states.flatten(2, 3)
                hidden_states = hidden_states.type_as(query)

                attention_states[f'{prefix}_attention_output'] = hidden_states.detach().cpu()

                # 7. Combine with image attention if present
                if hidden_states_img is not None:
                    hidden_states = hidden_states + hidden_states_img
                    attention_states[f'{prefix}_combined_output'] = hidden_states.detach().cpu()

                # 8. Output projection
                hidden_states = attn.to_out[0](hidden_states)
                attention_states[f'{prefix}_output_proj'] = hidden_states.detach().cpu()

                hidden_states = attn.to_out[1](hidden_states)  # Dropout
                attention_states[f'{prefix}_final_output'] = hidden_states.detach().cpu()

                return hidden_states

        # Replace processor
        attn_module.set_processor(InstrumentedProcessor())

    def register_attention_weight_hooks(self, block_indices=None):
        """
        Capture actual attention weights (scores).
        Warning: This uses manual attention computation, not optimized kernels.
        """
        if block_indices is None:
            block_indices = range(len(self.model.blocks))

        for i in block_indices:
            block = self.model.blocks[i]
            self._hook_attention_with_weights(block.attn1, f'block_{i}_attn1')
            self._hook_attention_with_weights(block.attn2, f'block_{i}_attn2')

    def _hook_attention_with_weights(self, attn_module, prefix):
        """Hook that computes attention manually to capture weights"""

        attention_states = self.attention_states

        class WeightCapturingProcessor:
            def __call__(self, attn, hidden_states, encoder_hidden_states=None, 
                        attention_mask=None, rotary_emb=None, **kwargs):

                # [Same Q, K, V projection code as above...]
                # ... (omitted for brevity)

                # Manual attention computation
                import math

                # Scaled dot-product attention
                scale = 1.0 / math.sqrt(query.shape[-1])

                # (B, seq_q, heads, head_dim) @ (B, seq_k, heads, head_dim).T
                # -> (B, heads, seq_q, seq_k)
                attn_weights = torch.einsum('bqhd,bkhd->bhqk', query, key) * scale

                attention_states[f'{prefix}_attention_scores'] = attn_weights.detach().cpu()

                # Apply mask if present
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask

                # Softmax
                attn_weights = torch.softmax(attn_weights, dim=-1)

                attention_states[f'{prefix}_attention_weights'] = attn_weights.detach().cpu()

                # Apply attention to values
                # (B, heads, seq_q, seq_k) @ (B, seq_k, heads, head_dim)
                hidden_states = torch.einsum('bhqk,bkhd->bqhd', attn_weights, value)

                # Continue with output projection...
                hidden_states = hidden_states.flatten(2, 3)
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

                return hidden_states

        attn_module.set_processor(WeightCapturingProcessor())

    def restore_processors(self):
        """Restore original attention processors"""
        for prefix, original_processor in self.original_processors.items():
            # Parse prefix to get module
            parts = prefix.split('_')
            block_idx = int(parts[1])
            attn_name = parts[2]

            if attn_name == 'attn1':
                self.model.blocks[block_idx].attn1.set_processor(original_processor)
            elif attn_name == 'attn2':
                self.model.blocks[block_idx].attn2.set_processor(original_processor)

    def get_attention_states(self):
        """Get all captured attention states"""
        return self.attention_states

    def clear_states(self):
        """Clear captured states"""
        self.attention_states = OrderedDict()

if __name__ == "__main__":
    test_dit_output()
    # test_dit()
