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

"""Weight loading utilities for Wan2.1-T2V-1.3B model."""

import gc
import re
from enum import Enum

import jax
import safetensors
from etils import epath
from flax import nnx

from bonsai.models.wan2 import modeling as model_lib
from bonsai.models.wan2 import t5 as t5_lib
from bonsai.models.wan2 import vae as vae_lib


def _get_dit_mapping(cfg: model_lib.ModelConfig):
    class Transform(Enum):
        """Transformations for model parameters"""

        NONE = None
        TRANSPOSE = ((1, 0), None, False)  # For linear layers: (out, in) -> (in, out)
        TRANSPOSE_CONV = ((2, 3, 4, 1, 0), None, False)  # For 3D conv: (out, in, t, h, w) -> (t, h, w, in, out)

    mapping = {
        # Patch embedding (input projection)
        r"patch_embedding\.weight": ("patch_embed.kernel", Transform.TRANSPOSE_CONV),
        r"patch_embedding\.bias": ("patch_embed.bias", Transform.NONE),
        # Time embedder - Sequential uses integer indices (0, 1, 2), not layers_0
        r"condition_embedder\.time_embedder\.linear_1\.weight": (
            "time_embed.time_embedding.layers.0.kernel",
            Transform.TRANSPOSE,
        ),
        r"condition_embedder\.time_embedder\.linear_1\.bias": (
            "time_embed.time_embedding.layers.0.bias",
            Transform.NONE,
        ),
        r"condition_embedder\.time_embedder\.linear_2\.weight": (
            "time_embed.time_embedding.layers.2.kernel",
            Transform.TRANSPOSE,
        ),
        r"condition_embedder\.time_embedder\.linear_2\.bias": (
            "time_embed.time_embedding.layers.2.bias",
            Transform.NONE,
        ),
        r"condition_embedder\.time_proj\.weight": ("time_embed.time_projection.layers.1.kernel", Transform.TRANSPOSE),
        r"condition_embedder\.time_proj\.bias": ("time_embed.time_projection.layers.1.bias", Transform.NONE),
        # Text embedder (projects T5 embeddings to hidden dim)
        r"condition_embedder\.text_embedder\.linear_1\.weight": ("text_proj.layers.0.kernel", Transform.TRANSPOSE),
        r"condition_embedder\.text_embedder\.linear_1\.bias": ("text_proj.layers.0.bias", Transform.NONE),
        r"condition_embedder\.text_embedder\.linear_2\.weight": ("text_proj.layers.2.kernel", Transform.TRANSPOSE),
        r"condition_embedder\.text_embedder\.linear_2\.bias": ("text_proj.layers.2.bias", Transform.NONE),
        # Transformer blocks - Self attention (attn1)
        r"blocks\.([0-9]+)\.attn1\.norm_q\.weight": (r"blocks.\1.self_attn.q_norm.scale", Transform.NONE),
        r"blocks\.([0-9]+)\.attn1\.norm_k\.weight": (r"blocks.\1.self_attn.k_norm.scale", Transform.NONE),
        r"blocks\.([0-9]+)\.attn1\.to_q\.weight": (r"blocks.\1.self_attn.q_proj.kernel", Transform.TRANSPOSE),
        r"blocks\.([0-9]+)\.attn1\.to_q\.bias": (r"blocks.\1.self_attn.q_proj.bias", Transform.NONE),
        r"blocks\.([0-9]+)\.attn1\.to_k\.weight": (r"blocks.\1.self_attn.k_proj.kernel", Transform.TRANSPOSE),
        r"blocks\.([0-9]+)\.attn1\.to_k\.bias": (r"blocks.\1.self_attn.k_proj.bias", Transform.NONE),
        r"blocks\.([0-9]+)\.attn1\.to_v\.weight": (r"blocks.\1.self_attn.v_proj.kernel", Transform.TRANSPOSE),
        r"blocks\.([0-9]+)\.attn1\.to_v\.bias": (r"blocks.\1.self_attn.v_proj.bias", Transform.NONE),
        r"blocks\.([0-9]+)\.attn1\.to_out\.0\.weight": (r"blocks.\1.self_attn.out_proj.kernel", Transform.TRANSPOSE),
        r"blocks\.([0-9]+)\.attn1\.to_out\.0\.bias": (r"blocks.\1.self_attn.out_proj.bias", Transform.NONE),
        # Transformer blocks - Cross attention (attn2)
        # Note: CrossAttention only has q_norm, not k_norm; norm_k is skipped
        r"blocks\.([0-9]+)\.attn2\.norm_q\.weight": (r"blocks.\1.cross_attn.q_norm.scale", Transform.NONE),
        r"blocks\.([0-9]+)\.attn2\.to_q\.weight": (r"blocks.\1.cross_attn.q_proj.kernel", Transform.TRANSPOSE),
        r"blocks\.([0-9]+)\.attn2\.to_q\.bias": (r"blocks.\1.cross_attn.q_proj.bias", Transform.NONE),
        # Note: to_k and to_v need special handling - they're fused into kv_proj in JAX
        # See _load_fused_kv_weights() below
        r"blocks\.([0-9]+)\.attn2\.to_out\.0\.weight": (r"blocks.\1.cross_attn.out_proj.kernel", Transform.TRANSPOSE),
        r"blocks\.([0-9]+)\.attn2\.to_out\.0\.bias": (r"blocks.\1.cross_attn.out_proj.bias", Transform.NONE),
        # Transformer blocks - Feed forward (Sequential creates 'layers' dict with 0, 2 keys)
        r"blocks\.([0-9]+)\.ffn\.net\.0\.proj\.weight": (r"blocks.\1.mlp.layers.0.kernel", Transform.TRANSPOSE),
        r"blocks\.([0-9]+)\.ffn\.net\.0\.proj\.bias": (r"blocks.\1.mlp.layers.0.bias", Transform.NONE),
        r"blocks\.([0-9]+)\.ffn\.net\.2\.weight": (r"blocks.\1.mlp.layers.2.kernel", Transform.TRANSPOSE),
        r"blocks\.([0-9]+)\.ffn\.net\.2\.bias": (r"blocks.\1.mlp.layers.2.bias", Transform.NONE),
        # Transformer blocks - Norm and modulation
        r"blocks\.([0-9]+)\.norm2\.weight": (r"blocks.\1.norm2.scale", Transform.NONE),
        r"blocks\.([0-9]+)\.norm2\.bias": (r"blocks.\1.norm2.bias", Transform.NONE),
        r"blocks\.([0-9]+)\.scale_shift_table": (r"blocks.\1.scale_shift_table", Transform.NONE),
        # Output projection
        r"scale_shift_table": ("final_layer.scale_shift_table", Transform.NONE),
        r"proj_out\.weight": ("final_layer.linear.kernel", Transform.TRANSPOSE),
        r"proj_out\.bias": ("final_layer.linear.bias", Transform.NONE),
    }

    return mapping


def _get_vae_key_mapping():
    """Define mapping from PyTorch VAE keys to JAX VAE keys."""

    class Transform(Enum):
        """Transformations for VAE parameters"""

        NONE = None
        TRANSPOSE_2D_CONV = ((2, 3, 1, 0), None, False)  # For 2D conv: (out, in, h, w) -> (h, w, in, out)
        TRANSPOSE_3D = ((2, 3, 4, 1, 0), None, False)  # For 3D conv: (out, in, t, h, w) -> (t, h, w, in, out)
        SQUEEZE = (None, (-1,), False)  # Squeeze to 1D: (C, 1, 1, 1) -> (C,)

    # PyTorch format: (out_channels, in_channels, kernel_size...)
    # JAX format: (kernel_size..., in_channels, out_channels)
    mapping = {
        # Post-quantization conv: 1x1x1 conv
        r"post_quant_conv\.weight": ("conv2.conv.kernel", Transform.TRANSPOSE_3D),
        r"post_quant_conv\.bias": ("conv2.conv.bias", Transform.NONE),
        # Decoder input conv
        r"decoder\.conv_in\.weight": ("decoder.conv_in.conv.kernel", Transform.TRANSPOSE_3D),
        r"decoder\.conv_in\.bias": ("decoder.conv_in.conv.bias", Transform.NONE),
        # Mid block resnets
        r"decoder\.mid_block\.resnets\.0\.norm1\.gamma": ("decoder.mid_block1.norm1.scale", Transform.SQUEEZE),
        r"decoder\.mid_block\.resnets\.0\.conv1\.weight": (
            "decoder.mid_block1.conv1.conv.kernel",
            Transform.TRANSPOSE_3D,
        ),
        r"decoder\.mid_block\.resnets\.0\.conv1\.bias": ("decoder.mid_block1.conv1.conv.bias", Transform.NONE),
        r"decoder\.mid_block\.resnets\.0\.norm2\.gamma": ("decoder.mid_block1.norm2.scale", Transform.SQUEEZE),
        r"decoder\.mid_block\.resnets\.0\.conv2\.weight": (
            "decoder.mid_block1.conv2.conv.kernel",
            Transform.TRANSPOSE_3D,
        ),
        r"decoder\.mid_block\.resnets\.0\.conv2\.bias": ("decoder.mid_block1.conv2.conv.bias", Transform.NONE),
        r"decoder\.mid_block\.resnets\.1\.norm1\.gamma": ("decoder.mid_block2.norm1.scale", Transform.SQUEEZE),
        r"decoder\.mid_block\.resnets\.1\.conv1\.weight": (
            "decoder.mid_block2.conv1.conv.kernel",
            Transform.TRANSPOSE_3D,
        ),
        r"decoder\.mid_block\.resnets\.1\.conv1\.bias": ("decoder.mid_block2.conv1.conv.bias", Transform.NONE),
        r"decoder\.mid_block\.resnets\.1\.norm2\.gamma": ("decoder.mid_block2.norm2.scale", Transform.SQUEEZE),
        r"decoder\.mid_block\.resnets\.1\.conv2\.weight": (
            "decoder.mid_block2.conv2.conv.kernel",
            Transform.TRANSPOSE_3D,
        ),
        r"decoder\.mid_block\.resnets\.1\.conv2\.bias": ("decoder.mid_block2.conv2.conv.bias", Transform.NONE),
        # Mid attention block
        r"decoder\.mid_block\.attentions\.0\.norm\.gamma": ("decoder.mid_attn.norm.scale", Transform.SQUEEZE),
        r"decoder\.mid_block\.attentions\.0\.to_qkv\.weight": (
            "decoder.mid_attn.qkv.kernel",
            Transform.TRANSPOSE_2D_CONV,
        ),
        r"decoder\.mid_block\.attentions\.0\.to_qkv\.bias": ("decoder.mid_attn.qkv.bias", Transform.NONE),
        r"decoder\.mid_block\.attentions\.0\.proj\.weight": (
            "decoder.mid_attn.proj.kernel",
            Transform.TRANSPOSE_2D_CONV,
        ),
        r"decoder\.mid_block\.attentions\.0\.proj\.bias": ("decoder.mid_attn.proj.bias", Transform.NONE),
        # Up blocks - resnets (pattern for all 4 stages, 3 resnets each)
        r"decoder\.up_blocks\.([0-3])\.resnets\.([0-2])\.norm1\.gamma": (
            r"decoder.up_blocks_\1.\2.norm1.scale",
            Transform.SQUEEZE,
        ),
        r"decoder\.up_blocks\.([0-3])\.resnets\.([0-2])\.conv1\.weight": (
            r"decoder.up_blocks_\1.\2.conv1.conv.kernel",
            Transform.TRANSPOSE_3D,
        ),
        r"decoder\.up_blocks\.([0-3])\.resnets\.([0-2])\.conv1\.bias": (
            r"decoder.up_blocks_\1.\2.conv1.conv.bias",
            Transform.NONE,
        ),
        r"decoder\.up_blocks\.([0-3])\.resnets\.([0-2])\.norm2\.gamma": (
            r"decoder.up_blocks_\1.\2.norm2.scale",
            Transform.SQUEEZE,
        ),
        r"decoder\.up_blocks\.([0-3])\.resnets\.([0-2])\.conv2\.weight": (
            r"decoder.up_blocks_\1.\2.conv2.conv.kernel",
            Transform.TRANSPOSE_3D,
        ),
        r"decoder\.up_blocks\.([0-3])\.resnets\.([0-2])\.conv2\.bias": (
            r"decoder.up_blocks_\1.\2.conv2.conv.bias",
            Transform.NONE,
        ),
        # Skip connections (only in block 1, resnet 0)
        r"decoder\.up_blocks\.1\.resnets\.0\.conv_shortcut\.weight": (
            "decoder.up_blocks_1.0.skip_conv.conv.kernel",
            Transform.TRANSPOSE_3D,
        ),
        r"decoder\.up_blocks\.1\.resnets\.0\.conv_shortcut\.bias": (
            "decoder.up_blocks_1.0.skip_conv.conv.bias",
            Transform.NONE,
        ),
        # Upsamplers for blocks 0, 1, 2 (block 3 has no upsampler)
        # Block 0: Upsample3D (time_conv + spatial_conv)
        r"decoder\.up_blocks\.0\.upsamplers\.0\.time_conv\.weight": (
            "decoder.up_sample_0.time_conv.conv.kernel",
            Transform.TRANSPOSE_3D,
        ),
        r"decoder\.up_blocks\.0\.upsamplers\.0\.time_conv\.bias": (
            "decoder.up_sample_0.time_conv.conv.bias",
            Transform.NONE,
        ),
        r"decoder\.up_blocks\.0\.upsamplers\.0\.resample\.1\.weight": (
            "decoder.up_sample_0.spatial_conv.kernel",
            Transform.TRANSPOSE_2D_CONV,
        ),
        r"decoder\.up_blocks\.0\.upsamplers\.0\.resample\.1\.bias": (
            "decoder.up_sample_0.spatial_conv.bias",
            Transform.NONE,
        ),
        # Block 1: Upsample3D (time_conv + spatial_conv)
        r"decoder\.up_blocks\.1\.upsamplers\.0\.time_conv\.weight": (
            "decoder.up_sample_1.time_conv.conv.kernel",
            Transform.TRANSPOSE_3D,
        ),
        r"decoder\.up_blocks\.1\.upsamplers\.0\.time_conv\.bias": (
            "decoder.up_sample_1.time_conv.conv.bias",
            Transform.NONE,
        ),
        r"decoder\.up_blocks\.1\.upsamplers\.0\.resample\.1\.weight": (
            "decoder.up_sample_1.spatial_conv.kernel",
            Transform.TRANSPOSE_2D_CONV,
        ),
        r"decoder\.up_blocks\.1\.upsamplers\.0\.resample\.1\.bias": (
            "decoder.up_sample_1.spatial_conv.bias",
            Transform.NONE,
        ),
        # Block 2: Upsample2D (conv only, no time_conv)
        r"decoder\.up_blocks\.2\.upsamplers\.0\.resample\.1\.weight": (
            "decoder.up_sample_2.conv.kernel",
            Transform.TRANSPOSE_2D_CONV,
        ),
        r"decoder\.up_blocks\.2\.upsamplers\.0\.resample\.1\.bias": ("decoder.up_sample_2.conv.bias", Transform.NONE),
        # Output layers
        r"decoder\.norm_out\.gamma": ("decoder.norm_out.scale", Transform.SQUEEZE),
        r"decoder\.conv_out\.weight": ("decoder.conv_out.conv.kernel", Transform.TRANSPOSE_3D),
        r"decoder\.conv_out\.bias": ("decoder.conv_out.conv.bias", Transform.NONE),
    }

    return mapping


def _torch_key_to_jax_key(mapping, source_key):
    """Convert a PyTorch/Diffusers key to JAX key with transform info."""
    subs = [
        (re.sub(pat, repl, source_key), transform)
        for pat, (repl, transform) in mapping.items()
        if re.match(pat, source_key)
    ]
    if len(subs) == 0:
        # Key not found in mapping, might be OK (e.g., VAE weights)
        return None, None
    if len(subs) > 1:
        raise ValueError(f"Multiple patterns matched for key {source_key}: {subs}")
    return subs[0]


def _assign_weights(keys, tensor, state_dict, st_key, transform, sharding_dict=None):
    """Recursively descend into state_dict and assign the (possibly permuted/reshaped) tensor."""
    key, *rest = keys
    if not rest:
        if transform is not None and transform.value is not None:
            permute, reshape, reshape_first = transform.value
            if reshape_first and reshape is not None:
                tensor = tensor.reshape(reshape)
            if permute:
                tensor = tensor.transpose(permute)
            if not reshape_first and reshape is not None:
                tensor = tensor.reshape(reshape)

        if key not in state_dict:
            raise KeyError(f"Key {key} not found in state_dict. Available keys: {list(state_dict.keys())[:10]}...")

        if tensor.shape != state_dict[key].shape:
            raise ValueError(f"Shape mismatch for {st_key}: {tensor.shape} vs {state_dict[key].shape}")

        # Assign with or without sharding
        if sharding_dict is not None and key in sharding_dict:
            state_dict[key] = jax.device_put(tensor, sharding_dict[key])
        else:
            state_dict[key] = jax.device_put(tensor)
    else:
        next_sharding = sharding_dict[key] if sharding_dict is not None and key in sharding_dict else None
        _assign_weights(rest, tensor, state_dict[key], st_key, transform, next_sharding)


def _stoi(s):
    """Convert string to int if possible, otherwise return string."""
    try:
        return int(s)
    except ValueError:
        return s


def create_model_from_safe_tensors(
    file_dir: str,
    cfg: model_lib.ModelConfig,
    mesh: jax.sharding.Mesh | None = None,
) -> model_lib.Wan2DiT:
    """
    Load Wan2.1-T2V-1.3B DiT model from safetensors checkpoint.

    Args:
        file_dir: Directory containing .safetensors files or path to transformer directory
        cfg: Model configuration
        mesh: Optional JAX mesh for sharding
        load_transformer_only: If True, only load transformer weights (not VAE/text encoder)

    Returns:
        Wan2DiT model with loaded weights
    """
    # Check if file_dir is the model root or transformer subdirectory
    file_path = epath.Path(file_dir).expanduser()
    transformer_path = file_path / "transformer"

    if transformer_path.exists():
        # Look in transformer subdirectory
        files = sorted(list(transformer_path.glob("diffusion_pytorch_model-*.safetensors")))
    else:
        # Look in provided directory
        files = sorted(list(file_path.glob("diffusion_pytorch_model-*.safetensors")))
        if not files:
            files = sorted(list(file_path.glob("*.safetensors")))

    if not files:
        raise ValueError(f"No safetensors found in {file_dir} or {file_dir}/transformer")

    print(f"Found {len(files)} DiT transformer safetensors file(s)")

    # Create model structure
    wan2_dit = nnx.eval_shape(lambda: model_lib.Wan2DiT(cfg, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(wan2_dit)
    state_dict = abs_state.to_pure_dict()

    # Setup sharding if mesh provided
    sharding = nnx.get_named_sharding(abs_state, mesh).to_pure_dict() if mesh is not None else None

    key_mapping = _get_dit_mapping(cfg)
    conversion_errors = []
    loaded_keys = []
    skipped_keys = []

    # Collect K/V weights for fusion into kv_proj
    kv_weights = {}  # {block_idx: {'k_weight': ..., 'k_bias': ..., 'v_weight': ..., 'v_bias': ...}}

    for f in files:
        print(f"Loading weights from {f.name}...")
        with safetensors.safe_open(f, framework="numpy") as sf:
            for torch_key in sf.keys():
                tensor = sf.get_tensor(torch_key)

                # Special handling for cross-attention K/V fusion
                kv_match = re.match(r"blocks\.([0-9]+)\.attn2\.to_([kv])\.(weight|bias)", torch_key)
                if kv_match:
                    block_idx = int(kv_match.group(1))
                    kv_type = kv_match.group(2)  # 'k' or 'v'
                    param_type = kv_match.group(3)  # 'weight' or 'bias'

                    if block_idx not in kv_weights:
                        kv_weights[block_idx] = {}
                    kv_weights[block_idx][f"{kv_type}_{param_type}"] = tensor
                    loaded_keys.append(torch_key)
                    continue

                jax_key, transform = _torch_key_to_jax_key(key_mapping, torch_key)

                if jax_key is None:
                    # Skip keys not in our mapping (e.g., VAE, text encoder, attn2.norm_k)
                    skipped_keys.append(torch_key)
                    continue

                keys = [_stoi(k) for k in jax_key.split(".")]
                try:
                    _assign_weights(keys, tensor, state_dict, torch_key, transform, sharding)
                    loaded_keys.append(torch_key)
                except Exception as e:
                    full_jax_key = ".".join([str(k) for k in keys])
                    conversion_errors.append(
                        f"Failed to assign '{torch_key}' to '{full_jax_key}': {type(e).__name__}: {e}"
                    )
        gc.collect()

    # Fuse collected K/V weights into kv_proj
    import jax.numpy as jnp

    for block_idx, weights in kv_weights.items():
        if all(k in weights for k in ["k_weight", "k_bias", "v_weight", "v_bias"]):
            # Transpose and concatenate: (out, in) -> (in, out) then concat -> (in, 2*out)
            k_weight = weights["k_weight"].T  # (in, out)
            v_weight = weights["v_weight"].T  # (in, out)
            kv_kernel = jnp.concatenate([k_weight, v_weight], axis=1)  # (in, 2*out)

            kv_bias = jnp.concatenate([weights["k_bias"], weights["v_bias"]])  # (2*out,)

            # Assign to state dict
            state_dict["blocks"][block_idx]["cross_attn"]["kv_proj"]["kernel"] = jax.device_put(kv_kernel)
            state_dict["blocks"][block_idx]["cross_attn"]["kv_proj"]["bias"] = jax.device_put(kv_bias)

    print(f"Loaded {len(loaded_keys)} weight tensors")
    print(f"Skipped {len(skipped_keys)} weight tensors (VAE/text encoder/attn2.norm_k)")

    if conversion_errors:
        print(f"\n Warning: {len(conversion_errors)} conversion errors occurred:")
        for err in conversion_errors:  # Show first 5 errors
            print(f"  {err}")
        # if len(conversion_errors) > 5:
        #     print(f"  ... and {len(conversion_errors) - 5} more")

    gc.collect()
    return nnx.merge(graph_def, state_dict)


def create_vae_decoder_from_safe_tensors(
    file_dir: str,
    mesh: jax.sharding.Mesh | None = None,
) -> vae_lib.WanVAEDecoder:
    """
    Load Wan-VAE decoder from safetensors checkpoint.

    Args:
        file_dir: Directory containing .safetensors files or path to VAE directory
        mesh: Optional JAX mesh for sharding

    Returns:
        WanVAEDecoder with loaded weights
    """
    # Check if file_dir is the model root or VAE subdirectory
    file_path = epath.Path(file_dir).expanduser()
    vae_path = file_path / "vae"

    if vae_path.exists():
        # Look in vae subdirectory
        files = list(vae_path.glob("*.safetensors"))
    else:
        # Look in provided directory
        files = list(file_path.glob("*.safetensors"))

    if not files:
        raise ValueError(f"No safetensors found in {file_dir} or {file_dir}/vae")

    print(f"Found {len(files)} VAE safetensors file(s)")

    # Create VAE decoder structure
    vae_decoder = nnx.eval_shape(lambda: vae_lib.WanVAEDecoder(rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(vae_decoder)
    state_dict = abs_state.to_pure_dict()

    # Setup sharding if mesh provided
    sharding = nnx.get_named_sharding(abs_state, mesh).to_pure_dict() if mesh is not None else None

    key_mapping = _get_vae_key_mapping()
    conversion_errors = []
    loaded_keys = []
    skipped_keys = []

    for f in files:
        print(f"Loading VAE weights from {f.name}...")
        with safetensors.safe_open(f, framework="numpy") as sf:
            for torch_key in sf.keys():
                tensor = sf.get_tensor(torch_key)

                jax_key, transform = _torch_key_to_jax_key(key_mapping, torch_key)

                if jax_key is None:
                    skipped_keys.append(torch_key)
                    print(f"{torch_key} is not mapped")
                    continue

                keys = [_stoi(k) for k in jax_key.split(".")]
                try:
                    _assign_weights(keys, tensor, state_dict, torch_key, transform, sharding)
                    loaded_keys.append(torch_key)
                except Exception as e:
                    full_jax_key = ".".join([str(k) for k in keys])
                    conversion_errors.append(
                        f"Failed to assign '{torch_key}' to '{full_jax_key}': {type(e).__name__}: {e}"
                    )
        gc.collect()

    print(f"Loaded {len(loaded_keys)} VAE weight tensors")
    print(f"Skipped {len(skipped_keys)} weight tensors")

    if conversion_errors:
        print(f"\nWarning: {len(conversion_errors)} conversion errors occurred:")
        for err in conversion_errors:  # Show first 10 errors
            print(f"  {err}")
        # if len(conversion_errors) > 10:
        #     print(f"  ... and {len(conversion_errors) - 10} more")

    if len(loaded_keys) == 0:
        raise ValueError("No VAE weights were loaded! Check the checkpoint structure and key mapping.")

    gc.collect()
    return nnx.merge(graph_def, state_dict)


def _get_t5_key_mapping():
    """Define mapping from HuggingFace T5 keys to JAX T5 keys."""

    class Transform(Enum):
        """Transformations for T5 parameters"""

        NONE = None
        TRANSPOSE = ((1, 0), None, False)  # For linear layers: (out, in) -> (in, out)

    # T5/UMT5 uses standard HuggingFace naming
    mapping = {
        # Shared token embeddings
        r"shared\.weight": ("encoder.token_embedding.embedding", Transform.NONE),
        # Encoder blocks - Self attention
        r"encoder\.block\.([0-9]+)\.layer\.0\.SelfAttention\.q\.weight": (
            r"encoder.blocks.\1.attn.q.kernel",
            Transform.TRANSPOSE,
        ),
        r"encoder\.block\.([0-9]+)\.layer\.0\.SelfAttention\.k\.weight": (
            r"encoder.blocks.\1.attn.k.kernel",
            Transform.TRANSPOSE,
        ),
        r"encoder\.block\.([0-9]+)\.layer\.0\.SelfAttention\.v\.weight": (
            r"encoder.blocks.\1.attn.v.kernel",
            Transform.TRANSPOSE,
        ),
        r"encoder\.block\.([0-9]+)\.layer\.0\.SelfAttention\.o\.weight": (
            r"encoder.blocks.\1.attn.o.kernel",
            Transform.TRANSPOSE,
        ),
        r"encoder\.block\.([0-9]+)\.layer\.0\.SelfAttention\.relative_attention_bias\.weight": (
            r"encoder.blocks.\1.pos_embedding.embedding.embedding",
            Transform.NONE,
        ),
        r"encoder\.block\.([0-9]+)\.layer\.0\.layer_norm\.weight": (r"encoder.blocks.\1.norm1.weight", Transform.NONE),
        # Encoder blocks - Feed forward
        r"encoder\.block\.([0-9]+)\.layer\.1\.DenseReluDense\.wi_0\.weight": (
            r"encoder.blocks.\1.ffn.gate.kernel",
            Transform.TRANSPOSE,
        ),
        r"encoder\.block\.([0-9]+)\.layer\.1\.DenseReluDense\.wi_1\.weight": (
            r"encoder.blocks.\1.ffn.fc1.kernel",
            Transform.TRANSPOSE,
        ),
        r"encoder\.block\.([0-9]+)\.layer\.1\.DenseReluDense\.wo\.weight": (
            r"encoder.blocks.\1.ffn.fc2.kernel",
            Transform.TRANSPOSE,
        ),
        r"encoder\.block\.([0-9]+)\.layer\.1\.layer_norm\.weight": (r"encoder.blocks.\1.norm2.weight", Transform.NONE),
        # Final layer norm
        r"encoder\.final_layer_norm\.weight": ("encoder.norm.weight", Transform.NONE),
    }

    return mapping


def create_t5_encoder_from_safe_tensors(
    file_dir: str,
    mesh: jax.sharding.Mesh | None = None,
    is_sf: bool = True,
) -> t5_lib.T5EncoderModel:
    """
    Load T5 encoder from safetensors checkpoint.

    Args:
        file_dir: Directory containing .safetensors files or path to text_encoder directory
        mesh: Optional JAX mesh for sharding

    Returns:
        T5EncoderModel with loaded weights
    """
    from bonsai.models.wan2 import t5

    t5_encoder = nnx.eval_shape(lambda: t5.T5EncoderModel(rngs=nnx.Rngs(params=0, dropout=0)))
    graph_def, abs_state = nnx.split(t5_encoder)
    state_dict = abs_state.to_pure_dict()

    sharding = nnx.get_named_sharding(abs_state, mesh).to_pure_dict() if mesh is not None else None

    key_mapping = _get_t5_key_mapping()
    conversion_errors = []
    loaded_keys = []
    skipped_keys = []

    # Check if file_dir is the model root or text_encoder subdirectory
    file_path = epath.Path(file_dir).expanduser()
    text_encoder_path = file_path / "text_encoder"

    def load_pytorch_weights(file_dir):
        from transformers import UMT5ForConditionalGeneration

        model = UMT5ForConditionalGeneration.from_pretrained(file_dir)
        encoder_state = {k: v for k, v in model.state_dict().items() if k.startswith("encoder.")}
        return encoder_state

    if is_sf:
        if text_encoder_path.exists():
            files = sorted(list(text_encoder_path.glob("model-*.safetensors")))
        else:
            files = sorted(list(file_path.glob("*.safetensors")))
        if not files:
            raise ValueError(f"No safetensors found in {file_dir} or {file_dir}/text_encoder")
        print(f"Found {len(files)} T5 encoder safetensors file(s)")

        for f in files:
            print(f"Loading T5 weights from {f.name}...")
            with safetensors.safe_open(f, framework="numpy") as sf:
                for torch_key in sf.keys():
                    tensor = sf.get_tensor(torch_key)

                    jax_key, transform = _torch_key_to_jax_key(key_mapping, torch_key)

                    if jax_key is None:
                        # Skip keys not in our mapping
                        skipped_keys.append(torch_key)
                        print(f"{torch_key} is not mapped")
                        continue

                    keys = [_stoi(k) for k in jax_key.split(".")]
                    try:
                        _assign_weights(keys, tensor, state_dict, torch_key, transform, sharding)
                        loaded_keys.append(torch_key)
                    except Exception as e:
                        full_jax_key = ".".join([str(k) for k in keys])
                        conversion_errors.append(
                            f"Failed to assign '{torch_key}' to '{full_jax_key}': {type(e).__name__}: {e}"
                        )
            gc.collect()
    else:
        print(f"Loading T5 weights from PyTorch checkpoint in {file_dir}...")
        pt_state = load_pytorch_weights(file_dir)
        for torch_key, tensor in pt_state.items():
            jax_key, transform = _torch_key_to_jax_key(key_mapping, torch_key)

            if jax_key is None:
                # Skip keys not in our mapping
                skipped_keys.append(torch_key)
                print(f"{torch_key} is not mapped")
                continue

            keys = [_stoi(k) for k in jax_key.split(".")]
            try:
                _assign_weights(keys, tensor.numpy(), state_dict, torch_key, transform, sharding)
                loaded_keys.append(torch_key)
            except Exception as e:
                full_jax_key = ".".join([str(k) for k in keys])
                conversion_errors.append(f"Failed to assign '{torch_key}' to '{full_jax_key}': {type(e).__name__}: {e}")
        gc.collect()

    print(f"Loaded {len(loaded_keys)} T5 weight tensors")
    print(f"Skipped {len(skipped_keys)} weight tensors")

    if conversion_errors:
        print(f"\nWarning: {len(conversion_errors)} conversion errors occurred:")
        for err in conversion_errors:  # Show first 10 errors
            print(f"  {err}")
        # if len(conversion_errors) > 10:
        #     print(f"  ... and {len(conversion_errors) - 10} more")

    if len(loaded_keys) == 0:
        raise ValueError("No T5 weights were loaded! Check the checkpoint structure and key mapping.")

    gc.collect()
    return nnx.merge(graph_def, state_dict)


__all__ = [
    "create_model_from_safe_tensors",
    "create_t5_encoder_from_safe_tensors",
    "create_vae_decoder_from_safe_tensors",
]
