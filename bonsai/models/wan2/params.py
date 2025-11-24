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
from bonsai.models.wan2 import vae as vae_lib


def _get_key_and_transform_mapping(cfg: model_lib.ModelConfig):
    """Define mapping from HuggingFace diffusers keys to our JAX model keys."""

    class Transform(Enum):
        """Transformations for model parameters"""

        NONE = None
        TRANSPOSE = ((1, 0), None, False)  # Simple transpose for linear layers

    # Mapping of diffusers_keys -> (nnx_keys, Transform)
    # This is a simplified version - you'll need to adjust based on actual checkpoint structure
    mapping = {
        # Input projection
        r"transformer\.pos_embed\.proj\.weight": ("input_proj.kernel", Transform.TRANSPOSE),
        r"transformer\.pos_embed\.proj\.bias": ("input_proj.bias", Transform.NONE),
        # Time embedding
        r"transformer\.time_embed\.linear_1\.weight": ("time_embed.linear1.kernel", Transform.TRANSPOSE),
        r"transformer\.time_embed\.linear_1\.bias": ("time_embed.linear1.bias", Transform.NONE),
        r"transformer\.time_embed\.linear_2\.weight": ("time_embed.linear2.kernel", Transform.TRANSPOSE),
        r"transformer\.time_embed\.linear_2\.bias": ("time_embed.linear2.bias", Transform.NONE),
        # Positional embeddings
        r"transformer\.pos_embed\.pos_embed": ("pos_embed.pos_embed", Transform.NONE),
        # Transformer blocks
        r"transformer\.blocks\.([0-9]+)\.norm1\.weight": (r"blocks.\1.norm1.scale", Transform.NONE),
        r"transformer\.blocks\.([0-9]+)\.norm1\.bias": (r"blocks.\1.norm1.bias", Transform.NONE),
        r"transformer\.blocks\.([0-9]+)\.norm2\.weight": (r"blocks.\1.norm2.scale", Transform.NONE),
        r"transformer\.blocks\.([0-9]+)\.norm2\.bias": (r"blocks.\1.norm2.bias", Transform.NONE),
        r"transformer\.blocks\.([0-9]+)\.norm3\.weight": (r"blocks.\1.norm3.scale", Transform.NONE),
        r"transformer\.blocks\.([0-9]+)\.norm3\.bias": (r"blocks.\1.norm3.bias", Transform.NONE),
        # Self-attention
        r"transformer\.blocks\.([0-9]+)\.attn\.qkv\.weight": (r"blocks.\1.self_attn.qkv.kernel", Transform.TRANSPOSE),
        r"transformer\.blocks\.([0-9]+)\.attn\.qkv\.bias": (r"blocks.\1.self_attn.qkv.bias", Transform.NONE),
        r"transformer\.blocks\.([0-9]+)\.attn\.proj\.weight": (
            r"blocks.\1.self_attn.out_proj.kernel",
            Transform.TRANSPOSE,
        ),
        r"transformer\.blocks\.([0-9]+)\.attn\.proj\.bias": (r"blocks.\1.self_attn.out_proj.bias", Transform.NONE),
        # Cross-attention
        r"transformer\.blocks\.([0-9]+)\.cross_attn\.q\.weight": (
            r"blocks.\1.cross_attn.q_proj.kernel",
            Transform.TRANSPOSE,
        ),
        r"transformer\.blocks\.([0-9]+)\.cross_attn\.q\.bias": (r"blocks.\1.cross_attn.q_proj.bias", Transform.NONE),
        r"transformer\.blocks\.([0-9]+)\.cross_attn\.kv\.weight": (
            r"blocks.\1.cross_attn.kv_proj.kernel",
            Transform.TRANSPOSE,
        ),
        r"transformer\.blocks\.([0-9]+)\.cross_attn\.kv\.bias": (r"blocks.\1.cross_attn.kv_proj.bias", Transform.NONE),
        r"transformer\.blocks\.([0-9]+)\.cross_attn\.proj\.weight": (
            r"blocks.\1.cross_attn.out_proj.kernel",
            Transform.TRANSPOSE,
        ),
        r"transformer\.blocks\.([0-9]+)\.cross_attn\.proj\.bias": (
            r"blocks.\1.cross_attn.out_proj.bias",
            Transform.NONE,
        ),
        # MLP
        r"transformer\.blocks\.([0-9]+)\.mlp\.fc1\.weight": (r"blocks.\1.mlp.fc1.kernel", Transform.TRANSPOSE),
        r"transformer\.blocks\.([0-9]+)\.mlp\.fc1\.bias": (r"blocks.\1.mlp.fc1.bias", Transform.NONE),
        r"transformer\.blocks\.([0-9]+)\.mlp\.fc2\.weight": (r"blocks.\1.mlp.fc2.kernel", Transform.TRANSPOSE),
        r"transformer\.blocks\.([0-9]+)\.mlp\.fc2\.bias": (r"blocks.\1.mlp.fc2.bias", Transform.NONE),
        # AdaLN modulation
        r"transformer\.blocks\.([0-9]+)\.adaLN_modulation\.1\.weight": (
            r"blocks.\1.adaLN_modulation.layers_1.kernel",
            Transform.TRANSPOSE,
        ),
        r"transformer\.blocks\.([0-9]+)\.adaLN_modulation\.1\.bias": (
            r"blocks.\1.adaLN_modulation.layers_1.bias",
            Transform.NONE,
        ),
        # Final layer
        r"transformer\.final_layer\.norm\.weight": ("final_layer.norm.scale", Transform.NONE),
        r"transformer\.final_layer\.norm\.bias": ("final_layer.norm.bias", Transform.NONE),
        r"transformer\.final_layer\.linear\.weight": ("final_layer.linear.kernel", Transform.TRANSPOSE),
        r"transformer\.final_layer\.linear\.bias": ("final_layer.linear.bias", Transform.NONE),
        r"transformer\.final_layer\.adaLN_modulation\.1\.weight": (
            "final_layer.adaLN_modulation.layers_1.kernel",
            Transform.TRANSPOSE,
        ),
        r"transformer\.final_layer\.adaLN_modulation\.1\.bias": (
            "final_layer.adaLN_modulation.layers_1.bias",
            Transform.NONE,
        ),
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
    load_transformer_only: bool = True,
) -> model_lib.Wan2DiT:
    """
    Load Wan2.1-T2V-1.3B DiT model from safetensors checkpoint.

    Args:
        file_dir: Directory containing .safetensors files
        cfg: Model configuration
        mesh: Optional JAX mesh for sharding
        load_transformer_only: If True, only load transformer weights (not VAE/text encoder)

    Returns:
        Wan2DiT model with loaded weights
    """
    files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors found in {file_dir}")

    # Filter to transformer-only files if requested
    if load_transformer_only:
        files = [f for f in files if "transformer" in f.name.lower() or "dit" in f.name.lower()]
        if not files:
            # If no specific transformer files, use all files
            files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))

    # Create model structure
    wan2_dit = nnx.eval_shape(lambda: model_lib.Wan2DiT(cfg, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(wan2_dit)
    state_dict = abs_state.to_pure_dict()

    # Setup sharding if mesh provided
    sharding = nnx.get_named_sharding(abs_state, mesh).to_pure_dict() if mesh is not None else None

    key_mapping = _get_key_and_transform_mapping(cfg)
    conversion_errors = []
    loaded_keys = []
    skipped_keys = []

    for f in files:
        print(f"Loading weights from {f.name}...")
        with safetensors.safe_open(f, framework="numpy") as sf:
            for torch_key in sf.keys():
                tensor = sf.get_tensor(torch_key)

                jax_key, transform = _torch_key_to_jax_key(key_mapping, torch_key)

                if jax_key is None:
                    # Skip keys not in our mapping (e.g., VAE, text encoder)
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

    print(f"Loaded {len(loaded_keys)} weight tensors")
    print(f"Skipped {len(skipped_keys)} weight tensors (VAE/text encoder)")

    if conversion_errors:
        print(f"\n Warning: {len(conversion_errors)} conversion errors occurred:")
        for err in conversion_errors[:5]:  # Show first 5 errors
            print(f"  {err}")
        if len(conversion_errors) > 5:
            print(f"  ... and {len(conversion_errors) - 5} more")

    gc.collect()
    return nnx.merge(graph_def, state_dict)


def create_vae_decoder_from_safe_tensors(
    file_dir: str,
    mesh: jax.sharding.Mesh | None = None,
) -> vae_lib.WanVAEDecoder:
    """
    Load Wan-VAE decoder from safetensors checkpoint.

    Args:
        file_dir: Directory containing .safetensors files
        mesh: Optional JAX mesh for sharding

    Returns:
        WanVAEDecoder with loaded weights
    """
    files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors found in {file_dir}")

    # Filter to VAE-specific files
    vae_files = [f for f in files if "vae" in f.name.lower()]
    if not vae_files:
        print("Warning: No VAE-specific files found, trying all safetensors files")
        vae_files = files

    # Create VAE decoder structure
    vae_decoder = nnx.eval_shape(lambda: vae_lib.WanVAEDecoder(rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(vae_decoder)
    state_dict = abs_state.to_pure_dict()

    # Setup sharding if mesh provided
    _sharding = nnx.get_named_sharding(abs_state, mesh).to_pure_dict() if mesh is not None else None

    # TODO: Add proper VAE key mapping
    # For now, this is a placeholder that will need to be filled in
    # with the actual mapping from PyTorch VAE keys to JAX keys
    loaded_keys = []
    skipped_keys = []

    for f in vae_files:
        print(f"Loading VAE weights from {f.name}...")
        with safetensors.safe_open(f, framework="numpy") as sf:
            for torch_key in sf.keys():
                # Skip non-VAE keys
                if not any(prefix in torch_key for prefix in ["vae", "decoder", "post_quant"]):
                    skipped_keys.append(torch_key)
                    continue

                # TODO: Implement actual key mapping for VAE
                # This requires knowing the exact structure of the checkpoint
                skipped_keys.append(torch_key)

        gc.collect()

    print(f"Loaded {len(loaded_keys)} VAE weight tensors")
    print(f"Skipped {len(skipped_keys)} weight tensors")

    if len(loaded_keys) == 0:
        print("\nWarning: No VAE weights were loaded!")
        print("The VAE decoder has random weights and will not produce meaningful output.")
        print("You may need to:")
        print("  1. Check the checkpoint structure with: safetensors.safe_open(file, 'numpy').keys()")
        print("  2. Implement the VAE key mapping in params.py")

    gc.collect()
    return nnx.merge(graph_def, state_dict)


__all__ = ["create_model_from_safe_tensors", "create_vae_decoder_from_safe_tensors"]
