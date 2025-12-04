import jax
import jax.numpy as jnp
import numpy as np
from modelscope import snapshot_download
from bonsai.models.wan2 import params
from bonsai.models.wan2 import vae as vae_lib
import torch
from diffusers import AutoencoderKLWan
from jax.lax import Precision
from collections import OrderedDict
from flax import nnx
import torch

class WanVAEDecoderHooks:
    """Extract intermediate outputs from Wan VAE Decoder using forward hooks"""

    def __init__(self, vae):
        self.vae = vae
        self.decoder = vae.decoder
        self.outputs = OrderedDict()
        self.hooks = []
        self.capture_enabled = False  # Control when to capture

    def register_decoder_hooks(self):
        """Register hooks on all decoder layers"""

        # 1. Hook post_quant_conv (before decoder)
        h = self.vae.post_quant_conv.register_forward_hook(
            self._make_conditional_hook('post_quant_conv')
        )
        self.hooks.append(h)

        # 2. Hook conv_in
        h = self.decoder.conv_in.register_forward_hook(
            self._make_conditional_hook('conv_in')
        )
        self.hooks.append(h)

        # 3. Hook mid_block
        h = self.decoder.mid_block.register_forward_hook(
            self._make_conditional_hook('mid_block')
        )
        self.hooks.append(h)

        # 4. Hook each mid_block residual block
        if hasattr(self.decoder.mid_block, 'resnets'):
            for i, res_block in enumerate(self.decoder.mid_block.resnets):
                h = res_block.register_forward_hook(
                    self._make_conditional_hook(f'mid_block_res_{i}')
                )
                self.hooks.append(h)
        if hasattr(self.decoder.mid_block, 'attentions'):
            for i, attn_block in enumerate(self.decoder.mid_block.attentions):
                h = attn_block.register_forward_hook(
                    self._make_conditional_hook(f'mid_block_attn_{i}')
                )
                self.hooks.append(h)

        # 5. Hook each up_block
        for i, up_block in enumerate(self.decoder.up_blocks):
            h = up_block.register_forward_hook(
                self._make_conditional_hook(f'up_block_{i}')
            )
            self.hooks.append(h)

            # Hook residual blocks within up_block
            if hasattr(up_block, 'resnets'):
                for j, res_block in enumerate(up_block.resnets):
                    h = res_block.register_forward_hook(
                        self._make_conditional_hook(f'up_block_{i}_res_{j}')
                    )
                    self.hooks.append(h)

            # Hook upsample layers
            if hasattr(up_block, 'upsamplers') and up_block.upsamplers is not None:
                h = up_block.upsamplers[0].register_forward_hook(
                    self._make_conditional_hook(f'up_block_{i}_upsample')
                )
                self.hooks.append(h)

        # 6. Hook norm_out
        h = self.decoder.norm_out.register_forward_hook(
            self._make_conditional_hook('norm_out')
        )
        self.hooks.append(h)

        # 7. Hook nonlinearity (after norm_out)
        # We need to hook this differently since it's a function, not a module
        # We'll hook conv_out input instead

        # 8. Hook conv_out (final output)
        h = self.decoder.conv_out.register_forward_hook(
            self._make_conditional_hook('conv_out')
        )
        self.hooks.append(h)

    def _make_conditional_hook(self, name):
        """Create hook that only captures when enabled"""
        def hook(module, input, output):
            if self.capture_enabled:
                self.outputs[name] = output.detach().cpu()
        return hook


    def decode_first_frame_only(self, latents):
        """Decode only first frame with hooks enabled"""

        # Extract first frame
        z_first = latents[:, :, 0:1, :, :]  # (B, C, 1, H, W)

        # Enable capture
        self.capture_enabled = True

        # Run through decoder directly (bypass frame loop)
        with torch.no_grad():
            x = self.vae.post_quant_conv(z_first)
            out = self.decoder(
                x,
                feat_cache=None,  # No cache for single frame
                feat_idx=[0],
                first_chunk=True
            )

        # Disable capture
        self.capture_enabled = False

        return out


    def _make_hook(self, name):
        """Create a hook function with closure over name"""
        def hook(module, input, output):
            self.outputs[name] = output.detach().cpu()
        return hook

    def remove_hooks(self):
        """Remove all registered hooks"""
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def clear_outputs(self):
        """Clear captured outputs"""
        self.outputs = OrderedDict()

    def get_outputs(self):
        """Get all captured outputs"""
        return self.outputs


def test_vae_decoder():
    # Load VAE model
    print("Loading AutoencoderKLWan...")
    model_ckpt_path = snapshot_download("Wan-AI/Wan2.1-T2V-1.3B-Diffusers",allow_patterns='vae/*',)

    vae_jax = params.create_vae_decoder_from_safe_tensors(model_ckpt_path, mesh=None)
    vae = AutoencoderKLWan.from_pretrained(
        model_ckpt_path,
        subfolder="vae",
        torch_dtype=torch.float32
    )
    vae.eval()

    # Register hooks
    hook_manager = WanVAEDecoderHooks(vae)
    hook_manager.register_decoder_hooks()

    # Create dummy latent input
    torch.manual_seed(42)
    batch_size = 1
    z_dim = 16
    num_frames = 9
    height = 30  # After spatial compression (8x)
    width = 52

    latents_mean = (
        torch.tensor(vae_lib.VAEConfig.latent_mean)
        .view(1, 16, 1, 1, 1)
        .to(dtype=torch.float32)
    )
    latents_std = 1.0 / torch.tensor(vae_lib.VAEConfig.latent_std).view(
        1, 16, 1, 1, 1
    ).to(dtype=torch.float32)

    latents = torch.randn(batch_size, z_dim, num_frames, height, width,dtype=torch.float32)
    latents = latents / latents_std + latents_mean
    latents_jax = jnp.array(latents.numpy().transpose(0,2,3,4,1))


    print(f"\nInput latents shape: {latents.shape}")
    print("Running decoder forward pass...\n")

    z, _ = vae_jax.conv2(latents_jax, None)
    t = z.shape[1]
    frames = []
    decoder = vae_jax.decoder
    output_jax = {}

    # Initialize cache list for feature caching
    cache_list = [None] * 50

    # Run decoder (through VAE decode)
    vae.clear_cache()
    x = vae.post_quant_conv(latents)
    with torch.no_grad():
        for i in range(num_frames):
            vae._conv_idx = [0]
            cache_idx = [0]
            frame_latent = z[:, i : i + 1, :, :, :]
            idx = cache_idx[0]
            # frame_out: [B, 4, H_out, W_out, 3] (4 frames per latent due to temporal upsampling)
            if i == 0:
                out = vae.decoder(
                    x[:, :, i : i + 1, :, :], feat_cache=vae._feat_map, feat_idx=vae._conv_idx, first_chunk=True
                )
                frame_out, cache_list = decoder(frame_latent, cache_list, cache_idx)
                compare_outputs(frame_out, out, f"frame_{i}_output", rtol=1e-2, atol=1e-4)
            elif i==1:
                hook_manager.clear_outputs()
                hook_manager.capture_enabled = True
                try:
                    out_ = vae.decoder(x[:, :, i : i + 1, :, :], feat_cache=vae._feat_map, feat_idx=vae._conv_idx)
                finally:
                    hook_manager.capture_enabled = False
                outputs = hook_manager.get_outputs()
                x, cache_list[idx] = decoder.conv_in(frame_latent, cache_list[idx])
                cache_idx[0] += 1
                compare_outputs(x, outputs['conv_in'], 'conv_in', rtol=1e-2, atol=1e-4)
                output_jax['conv_in'] = x

                # Mid block 1
                x, cache_list = decoder.mid_block1(x, cache_list, cache_idx)
                compare_outputs(x, outputs['mid_block_res_0'], 'mid_block_res_0', rtol=1e-2, atol=1e-4)
                output_jax['mid_block_res_0'] = x

                # Mid attention
                x = decoder.mid_attn(x)
                compare_outputs(x, outputs['mid_block_attn_0'], 'mid_block_attn_0', rtol=1e-2, atol=1e-4)
                output_jax['mid_block_attn_0'] = x

                # Mid block 2
                x, cache_list = decoder.mid_block2(x, cache_list, cache_idx)
                compare_outputs(x, outputs['mid_block_res_1'], 'mid_block_res_1', rtol=1e-2, atol=1e-4)
                output_jax['mid_block_res_1'] = x

                # Upsample stage 0
                for j, block in enumerate(decoder.up_blocks_0):
                    x, cache_list = block(x, cache_list, cache_idx)
                    compare_outputs(x, outputs[f'up_block_0_res_{j}'], f'up_block_0_res_{j}', rtol=1e-2, atol=1e-4)
                    output_jax[f'up_block_0_res_{j}'] = x
                x, cache_list = decoder.up_sample_0(x, cache_list, cache_idx)
                compare_outputs(x, outputs['up_block_0_upsample'], 'up_block_0_upsample', rtol=1e-2, atol=1e-4)
                output_jax['up_block_0_upsample'] = x

                # Upsample stage 1
                for j, block in enumerate(decoder.up_blocks_1):
                    x, cache_list = block(x, cache_list, cache_idx)
                    compare_outputs(x, outputs[f'up_block_1_res_{j}'], f'up_block_1_res_{j}', rtol=1e-2, atol=1e-4)
                    output_jax[f'up_block_1_res_{j}'] = x
                x, cache_list = decoder.up_sample_1(x, cache_list, cache_idx)
                compare_outputs(x, outputs['up_block_1_upsample'], 'up_block_1_upsample', rtol=1e-2, atol=1e-4)
                output_jax['up_block_1_upsample'] = x

                # Upsample stage 2 (spatial only, no cache)
                for j, block in enumerate(decoder.up_blocks_2):
                    x, cache_list = block(x, cache_list, cache_idx)
                    compare_outputs(x, outputs[f'up_block_2_res_{j}'], f'up_block_2_res_{j}', rtol=1e-2, atol=1e-4)
                    output_jax[f'up_block_2_res_{j}'] = x
                x = decoder.up_sample_2(x)  # Spatial-only, no cache
                compare_outputs(x, outputs['up_block_2_upsample'], 'up_block_2_upsample', rtol=1e-2, atol=1e-4)
                output_jax['up_block_2_upsample'] = x

                # Upsample stage 3 (no spatial upsample)
                for j, block in enumerate(decoder.up_blocks_3):
                    x, cache_list = block(x, cache_list, cache_idx)
                    compare_outputs(x, outputs[f'up_block_3_res_{j}'], f'up_block_3_res_{j}', rtol=1e-2, atol=1e-4)
                    output_jax[f'up_block_3_res_{j}'] = x

                x = decoder.norm_out(x)
                compare_outputs(x, outputs['norm_out'], 'norm_out', rtol=1e-2, atol=1e-4)
                output_jax['norm_out'] = x
                x = nnx.silu(x)

                idx = cache_idx[0]
                x, cache_list[idx] = decoder.conv_out(x, cache_list[idx])
                cache_idx[0] += 1
                compare_outputs(x, outputs['conv_out'], 'conv_out', rtol=1e-2, atol=1e-4)
                output_jax['conv_out'] = x
                frames.append(x)
                # compare_outputs(frame_out, out_, f"frame_{i}_output", rtol=1e-2, atol=1e-4)
                out = torch.cat([out, out_], 2)
            frames.append(frame_out)
        out = torch.clamp(out, min=-1.0, max=1.0)
        x = jnp.concatenate(frames, axis=1)  # [B, T_total, H_out, W_out, 3]
        x = jnp.clip(x, -1.0, 1.0)
        compare_outputs(x, out, "final_output", rtol=1e-2, atol=1e-4)
        # decoded = vae.decode(latents).sample

    # first_frame_output = hook_manager.decode_first_frame_only(latents)
    # Get captured outputs
    # outputs = hook_manager.get_outputs()

    print("=" * 80)
    print("CAPTURED VAE DECODER INTERMEDIATE OUTPUTS")
    print("=" * 80)

    # for name, tensor in outputs.items():
    #     print(f"{name:40s}: {tuple(tensor.shape)}")

    # output_jax = {}
    # z, _ = vae_jax.conv2(latents_jax, None)
    # compare_outputs(z, outputs['post_quant_conv'], 'post_quant_conv', rtol=1e-2, atol=1e-4)
    # output_jax['post_quant_conv'] = z

    # t = z.shape[1]
    # frames = []
    # decoder = vae_jax.decoder

    # # Initialize cache list for feature caching
    # cache_list = [None] * 50

    # for i in range(t):
    #     print(f"\n{'='*80}")
    #     print(f"Processing frame {i+1}/{t}")
    #     print(f"{'='*80}")

    #     # Reset cache index for each frame
    #     cache_idx = [0]
    #     frame_latent = z[:, i : i + 1, :, :, :]

    #     if i == 0:
    #         idx = cache_idx[0]
    #         x, cache_list[idx] = decoder.conv_in(frame_latent, cache_list[idx])
    #         cache_idx[0] += 1
    #         compare_outputs(x, outputs['conv_in'], 'conv_in', rtol=1e-2, atol=1e-4)
    #         output_jax['conv_in'] = x

    #         # Mid block 1
    #         x, cache_list = decoder.mid_block1(x, cache_list, cache_idx)
    #         compare_outputs(x, outputs['mid_block_res_0'], 'mid_block_res_0', rtol=1e-2, atol=1e-4)
    #         output_jax['mid_block_res_0'] = x

    #         # Mid attention
    #         x = decoder.mid_attn(x)
    #         compare_outputs(x, outputs['mid_block_attn_0'], 'mid_block_attn_0', rtol=1e-2, atol=1e-4)
    #         output_jax['mid_block_attn_0'] = x

    #         # Mid block 2
    #         x, cache_list = decoder.mid_block2(x, cache_list, cache_idx)
    #         compare_outputs(x, outputs['mid_block_res_1'], 'mid_block_res_1', rtol=1e-2, atol=1e-4)
    #         output_jax['mid_block_res_1'] = x

    #         # Upsample stage 0
    #         for j, block in enumerate(decoder.up_blocks_0):
    #             x, cache_list = block(x, cache_list, cache_idx)
    #             compare_outputs(x, outputs[f'up_block_0_res_{j}'], f'up_block_0_res_{j}', rtol=1e-2, atol=1e-4)
    #             output_jax[f'up_block_0_res_{j}'] = x
    #         x, cache_list = decoder.up_sample_0(x, cache_list, cache_idx)
    #         compare_outputs(x, outputs['up_block_0_upsample'], 'up_block_0_upsample', rtol=1e-2, atol=1e-4)
    #         output_jax['up_block_0_upsample'] = x

    #         # Upsample stage 1
    #         for j, block in enumerate(decoder.up_blocks_1):
    #             x, cache_list = block(x, cache_list, cache_idx)
    #             compare_outputs(x, outputs[f'up_block_1_res_{j}'], f'up_block_1_res_{j}', rtol=1e-2, atol=1e-4)
    #             output_jax[f'up_block_1_res_{j}'] = x
    #         x, cache_list = decoder.up_sample_1(x, cache_list, cache_idx)
    #         compare_outputs(x, outputs['up_block_1_upsample'], 'up_block_1_upsample', rtol=1e-2, atol=1e-4)
    #         output_jax['up_block_1_upsample'] = x

    #         # Upsample stage 2 (spatial only, no cache)
    #         for j, block in enumerate(decoder.up_blocks_2):
    #             x, cache_list = block(x, cache_list, cache_idx)
    #             compare_outputs(x, outputs[f'up_block_2_res_{j}'], f'up_block_2_res_{j}', rtol=1e-2, atol=1e-4)
    #             output_jax[f'up_block_2_res_{j}'] = x
    #         x = decoder.up_sample_2(x)  # Spatial-only, no cache
    #         compare_outputs(x, outputs['up_block_2_upsample'], 'up_block_2_upsample', rtol=1e-2, atol=1e-4)
    #         output_jax['up_block_2_upsample'] = x

    #         # Upsample stage 3 (no spatial upsample)
    #         for j, block in enumerate(decoder.up_blocks_3):
    #             x, cache_list = block(x, cache_list, cache_idx)
    #             compare_outputs(x, outputs[f'up_block_3_res_{j}'], f'up_block_3_res_{j}', rtol=1e-2, atol=1e-4)
    #             output_jax[f'up_block_3_res_{j}'] = x

    #         x = decoder.norm_out(x)
    #         compare_outputs(x, outputs['norm_out'], 'norm_out', rtol=1e-2, atol=1e-4)
    #         output_jax['norm_out'] = x
    #         x = nnx.silu(x)

    #         idx = cache_idx[0]
    #         x, cache_list[idx] = decoder.conv_out(x, cache_list[idx])
    #         cache_idx[0] += 1
    #         compare_outputs(x, outputs['conv_out'], 'conv_out', rtol=1e-2, atol=1e-4)
    #         output_jax['conv_out'] = x
    #         frames.append(x)
    #     else:
    #         # Subsequent frames: use cached features
    #         print(f"Subsequent frame: using feature cache")
    #         frame_out, cache_list = decoder(frame_latent, cache_list, cache_idx)
    #         frames.append(frame_out)

    # print("\n" + "=" * 80)
    # print(f"Final decoded output shape: {decoded.shape}")
    # print("=" * 80)

    # # Save outputs for comparison
    # outputs_dict = {
    #     'inputs': {
    #         'latents': latents.cpu(),
    #     },
    #     'intermediate': outputs,
    #     'output': decoded.cpu(),
    # }

    # outputs_dict_jax = {
    #     'intermediate': output_jax
    # }

    # compare_with_jax_decoder(outputs_dict, outputs_dict_jax)

    # torch.save(outputs_dict, 'wan_vae_decoder_outputs.pt')
    # print("\n✓ Saved outputs to 'wan_vae_decoder_outputs.pt'")

    # Clean up
    hook_manager.remove_hooks()

    # return outputs_dict

def compare_outputs(jax_output: jax.Array, torch_output, name: str, rtol: float = 1e-2, atol: float = 1e-4):
    if torch_output.dtype == torch.bfloat16:
        torch_output = torch_output.float()

    if isinstance(torch_output, torch.Tensor):
        torch_np = torch_output.detach().cpu().numpy()
    else:
        torch_np = np.array(torch_output)

    jax_np = np.array(jax_output).transpose(0,4,1,2,3)  # Convert JAX [B,T,H,W,C] to [B,C,T,H,W]

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

def compare_with_jax_decoder(outputs_dict_torch, outputs_dict_jax):
    """
    Compare PyTorch decoder outputs with JAX implementation
    
    Args:
        outputs_dict_torch: Output dict from PyTorch decoder
        outputs_dict_jax: Output dict from JAX decoder (convert to numpy/torch first)
    """
    print("\n" + "=" * 80)
    print("COMPARISON: PyTorch vs JAX")
    print("=" * 80)

    torch_intermediate = outputs_dict_torch['intermediate']
    jax_intermediate = outputs_dict_jax['intermediate']

    for name in torch_intermediate.keys():
        if name not in jax_intermediate:
            print(f"⚠  {name:40s}: MISSING in JAX implementation")
            continue

        torch_output = torch_intermediate[name]
        torch_np = torch_output.cpu().numpy()
        jax_output = jax_intermediate[name]
        jax_np = np.array(jax_output).transpose(0,4,1,2,3)  # Convert JAX [B,T,H,W,C] to [B,C,T,H,W]

        # Check shape match
        if torch_np.shape != jax_np.shape:
            print(f"✗  {name:40s}: SHAPE MISMATCH")
            print(f"     PyTorch: {torch_np.shape}")
            print(f"     JAX:     {jax_np.shape}")
            continue

        abs_diff = np.abs(jax_np - torch_np)
        rel_diff = abs_diff / (np.abs(torch_np) + 1e-10)

        max_abs_diff = np.max(abs_diff)
        max_rel_diff = np.max(rel_diff)
        mean_abs_diff = np.mean(abs_diff)
        mean_rel_diff = np.mean(rel_diff)

        rtol = 1e-2
        atol = 1e-4
        close = np.allclose(jax_np, torch_np, rtol=rtol, atol=atol)
        if close:
            print(f"\n✅ Outputs match within tolerance (rtol={rtol}, atol={atol})")
        else:
            print(f"\n❌ Outputs do NOT match (rtol={rtol}, atol={atol})")
            # Show some mismatched locations
            mismatch_mask = ~np.isclose(jax_np, torch_np, rtol=rtol, atol=atol)
            n_mismatches = np.sum(mismatch_mask)
            print(f"  Number of mismatches: {n_mismatches} / {jax_np.size} ({100 * n_mismatches / jax_np.size:.2f}%)")
        # # Status
        # if max_diff < 1e-5:
        #     status = "✓ EXACT"
        # elif max_diff < 1e-3:
        #     status = "✓ CLOSE"
        # elif max_diff < 1e-1:
        #     status = "⚠ DIFF"
        # else:
        #     status = "✗ LARGE DIFF"

        print(f"{name:40s}: max_diff={max_abs_diff:.2e}, mean_diff={mean_abs_diff:.2e}, mean_rel_diff={mean_rel_diff:.2e}")

    # # Compare final output
    # torch_output = outputs_dict_torch['output']
    # jax_output = outputs_dict_jax['output']

    # if not isinstance(jax_output, torch.Tensor):
    #     jax_output = torch.from_numpy(jax_output)

    # abs_diff = torch.abs(torch_output - jax_output)
    # max_diff = abs_diff.max().item()
    # mean_diff = abs_diff.mean().item()

    # print("=" * 80)
    # print(f"Final Output: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
    # print("=" * 80)


if __name__ == "__main__":
    test_vae_decoder()