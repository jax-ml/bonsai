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

import re
from typing import Dict, Any, Tuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
import safetensors.flax as safetensors
from etils import epath
from flax import nnx
from transformers import WhisperConfig as HFWhisperConfig

from bonsai.models.whisper import modeling as model_lib


def _get_key_and_transform_mapping(config: model_lib.WhisperConfig) -> Dict[str, Tuple[str, Optional[Tuple]]]:
    """Get mapping from HuggingFace Whisper keys to NNX keys with transformations."""
    return {
        # Audio encoder convolutional layers
        r"model\.encoder\.conv1\.weight": ("encoder.conv1.kernel", ((2, 1, 0), None)),
        r"model\.encoder\.conv1\.bias": ("encoder.conv1.bias", None),
        r"model\.encoder\.conv2\.weight": ("encoder.conv2.kernel", ((2, 1, 0), None)),
        r"model\.encoder\.conv2\.bias": ("encoder.conv2.bias", None),
        
        # Audio encoder positional embedding
        r"model\.encoder\.embed_positions\.weight": ("encoder.positional_embedding", None),
        
        # Audio encoder transformer blocks
        r"model\.encoder\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (
            r"encoder.blocks.\1.attn.query.kernel", ((1, 0), None)
        ),
        r"model\.encoder\.layers\.([0-9]+)\.self_attn\.q_proj\.bias": (
            r"encoder.blocks.\1.attn.query.bias", None
        ),
        r"model\.encoder\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (
            r"encoder.blocks.\1.attn.key.kernel", ((1, 0), None)
        ),
        r"model\.encoder\.layers\.([0-9]+)\.self_attn\.k_proj\.bias": (
            r"encoder.blocks.\1.attn.key.bias", None
        ),
        r"model\.encoder\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (
            r"encoder.blocks.\1.attn.value.kernel", ((1, 0), None)
        ),
        r"model\.encoder\.layers\.([0-9]+)\.self_attn\.v_proj\.bias": (
            r"encoder.blocks.\1.attn.value.bias", None
        ),
        r"model\.encoder\.layers\.([0-9]+)\.self_attn\.out_proj\.weight": (
            r"encoder.blocks.\1.attn.out.kernel", ((1, 0), None)
        ),
        r"model\.encoder\.layers\.([0-9]+)\.self_attn\.out_proj\.bias": (
            r"encoder.blocks.\1.attn.out.bias", None
        ),
        r"model\.encoder\.layers\.([0-9]+)\.self_attn_layer_norm\.weight": (
            r"encoder.blocks.\1.attn_ln.scale", None
        ),
        r"model\.encoder\.layers\.([0-9]+)\.self_attn_layer_norm\.bias": (
            r"encoder.blocks.\1.attn_ln.bias", None
        ),
        r"model\.encoder\.layers\.([0-9]+)\.fc1\.weight": (
            r"encoder.blocks.\1.mlp_fc1.kernel", ((1, 0), None)
        ),
        r"model\.encoder\.layers\.([0-9]+)\.fc1\.bias": (
            r"encoder.blocks.\1.mlp_fc1.bias", None
        ),
        r"model\.encoder\.layers\.([0-9]+)\.fc2\.weight": (
            r"encoder.blocks.\1.mlp_fc2.kernel", ((1, 0), None)
        ),
        r"model\.encoder\.layers\.([0-9]+)\.fc2\.bias": (
            r"encoder.blocks.\1.mlp_fc2.bias", None
        ),
        r"model\.encoder\.layers\.([0-9]+)\.final_layer_norm\.weight": (
            r"encoder.blocks.\1.mlp_ln.scale", None
        ),
        r"model\.encoder\.layers\.([0-9]+)\.final_layer_norm\.bias": (
            r"encoder.blocks.\1.mlp_ln.bias", None
        ),
        
        # Audio encoder final layer norm
        r"model\.encoder\.layer_norm\.weight": ("encoder.ln_post.scale", None),
        r"model\.encoder\.layer_norm\.bias": ("encoder.ln_post.bias", None),
        
        # Text decoder
        r"model\.decoder\.embed_tokens\.weight": ("decoder.token_embedding.embedding", None),
        r"model\.decoder\.embed_positions\.weight": ("decoder.positional_embedding", None),
        
        # Text decoder transformer blocks
        r"model\.decoder\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (
            r"decoder.blocks.\1.attn.query.kernel", ((1, 0), None)
        ),
        r"model\.decoder\.layers\.([0-9]+)\.self_attn\.q_proj\.bias": (
            r"decoder.blocks.\1.attn.query.bias", None
        ),
        r"model\.decoder\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (
            r"decoder.blocks.\1.attn.key.kernel", ((1, 0), None)
        ),
        r"model\.decoder\.layers\.([0-9]+)\.self_attn\.k_proj\.bias": (
            r"decoder.blocks.\1.attn.key.bias", None
        ),
        r"model\.decoder\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (
            r"decoder.blocks.\1.attn.value.kernel", ((1, 0), None)
        ),
        r"model\.decoder\.layers\.([0-9]+)\.self_attn\.v_proj\.bias": (
            r"decoder.blocks.\1.attn.value.bias", None
        ),
        r"model\.decoder\.layers\.([0-9]+)\.self_attn\.out_proj\.weight": (
            r"decoder.blocks.\1.attn.out.kernel", ((1, 0), None)
        ),
        r"model\.decoder\.layers\.([0-9]+)\.self_attn\.out_proj\.bias": (
            r"decoder.blocks.\1.attn.out.bias", None
        ),
        r"model\.decoder\.layers\.([0-9]+)\.self_attn_layer_norm\.weight": (
            r"decoder.blocks.\1.attn_ln.scale", None
        ),
        r"model\.decoder\.layers\.([0-9]+)\.self_attn_layer_norm\.bias": (
            r"decoder.blocks.\1.attn_ln.bias", None
        ),
        r"model\.decoder\.layers\.([0-9]+)\.encoder_attn\.q_proj\.weight": (
            r"decoder.blocks.\1.cross_attn.query.kernel", ((1, 0), None)
        ),
        r"model\.decoder\.layers\.([0-9]+)\.encoder_attn\.q_proj\.bias": (
            r"decoder.blocks.\1.cross_attn.query.bias", None
        ),
        r"model\.decoder\.layers\.([0-9]+)\.encoder_attn\.k_proj\.weight": (
            r"decoder.blocks.\1.cross_attn.key.kernel", ((1, 0), None)
        ),
        r"model\.decoder\.layers\.([0-9]+)\.encoder_attn\.k_proj\.bias": (
            r"decoder.blocks.\1.cross_attn.key.bias", None
        ),
        r"model\.decoder\.layers\.([0-9]+)\.encoder_attn\.v_proj\.weight": (
            r"decoder.blocks.\1.cross_attn.value.kernel", ((1, 0), None)
        ),
        r"model\.decoder\.layers\.([0-9]+)\.encoder_attn\.v_proj\.bias": (
            r"decoder.blocks.\1.cross_attn.value.bias", None
        ),
        r"model\.decoder\.layers\.([0-9]+)\.encoder_attn\.out_proj\.weight": (
            r"decoder.blocks.\1.cross_attn.out.kernel", ((1, 0), None)
        ),
        r"model\.decoder\.layers\.([0-9]+)\.encoder_attn\.out_proj\.bias": (
            r"decoder.blocks.\1.cross_attn.out.bias", None
        ),
        r"model\.decoder\.layers\.([0-9]+)\.encoder_attn_layer_norm\.weight": (
            r"decoder.blocks.\1.cross_attn_ln.scale", None
        ),
        r"model\.decoder\.layers\.([0-9]+)\.encoder_attn_layer_norm\.bias": (
            r"decoder.blocks.\1.cross_attn_ln.bias", None
        ),
        r"model\.decoder\.layers\.([0-9]+)\.fc1\.weight": (
            r"decoder.blocks.\1.mlp_fc1.kernel", ((1, 0), None)
        ),
        r"model\.decoder\.layers\.([0-9]+)\.fc1\.bias": (
            r"decoder.blocks.\1.mlp_fc1.bias", None
        ),
        r"model\.decoder\.layers\.([0-9]+)\.fc2\.weight": (
            r"decoder.blocks.\1.mlp_fc2.kernel", ((1, 0), None)
        ),
        r"model\.decoder\.layers\.([0-9]+)\.fc2\.bias": (
            r"decoder.blocks.\1.mlp_fc2.bias", None
        ),
        r"model\.decoder\.layers\.([0-9]+)\.final_layer_norm\.weight": (
            r"decoder.blocks.\1.mlp_ln.scale", None
        ),
        r"model\.decoder\.layers\.([0-9]+)\.final_layer_norm\.bias": (
            r"decoder.blocks.\1.mlp_ln.bias", None
        ),
        
        # Text decoder final layer norm and output projection
        r"model\.decoder\.layer_norm\.weight": ("decoder.ln.scale", None),
        r"model\.decoder\.layer_norm\.bias": ("decoder.ln.bias", None),
        r"model\.decoder\.embed_tokens\.weight": ("decoder.output_projection.kernel", ((1, 0), None)),
    }


def _torch_key_to_jax_key(mapping: Dict[str, Tuple[str, Optional[Tuple]]], source_key: str) -> Tuple[str, Optional[Tuple]]:
    """Convert PyTorch key to JAX key with transformation."""
    for pattern, (replacement, transform) in mapping.items():
        if re.match(pattern, source_key):
            # Handle numbered groups in replacement
            if r"\1" in replacement:
                match = re.match(pattern, source_key)
                if match:
                    replacement = replacement.replace(r"\1", match.group(1))
            return replacement, transform
    raise ValueError(f"No mapping found for key: {source_key}")


def _stoi(s):
    """Convert string to int if possible, otherwise return string."""
    try:
        return int(s)
    except ValueError:
        return s


def _assign_weights(keys: list, tensor: np.ndarray, state_dict: Dict[str, Any], torch_key: str, transform: Optional[Tuple]) -> None:
    """Convert weights and assign to NNX state_dict."""
    key = keys[0]
    if len(keys) == 1:
        try:
            if transform is not None:
                permute, reshape = transform
                if permute:
                    tensor = tensor.transpose(permute)
                if reshape:
                    tensor = tensor.reshape(reshape)
            state_dict[key] = jnp.array(tensor)
        except Exception as e:
            print(f"Error processing key {torch_key}: {e}")
            raise
    else:
        # Handle nested keys
        if key not in state_dict:
            state_dict[key] = {}
        _assign_weights(keys[1:], tensor, state_dict[key], torch_key, transform)


def convert_hf_whisper_to_nnx(hf_model_path: str, config: model_lib.WhisperConfig) -> model_lib.WhisperModel:
    """Convert HuggingFace Whisper model to NNX format."""
    # Load HuggingFace config to get model dimensions
    hf_config = HFWhisperConfig.from_pretrained(hf_model_path)
    
    # Create NNX model with proper RNGs
    whisper_model = nnx.eval_shape(lambda: model_lib.WhisperModel(config, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(whisper_model)
    state_dict = abs_state.to_pure_dict()
    
    # Load weights from safetensors or pytorch format
    model_path = epath.Path(hf_model_path)
    
    # Try to load from safetensors first
    safetensors_files = list(model_path.glob("*.safetensors"))
    if safetensors_files:
        hf_state_dict = safetensors.load_file(str(safetensors_files[0]))
    else:
        # Fall back to pytorch format
        import torch
        checkpoint = torch.load(model_path / "pytorch_model.bin", map_location="cpu")
        hf_state_dict = {k: v.numpy() for k, v in checkpoint.items()}
    
    # Convert weights
    mapping = _get_key_and_transform_mapping(config)
    
    for torch_key, tensor in hf_state_dict.items():
        try:
            jax_key, transform = _torch_key_to_jax_key(mapping, torch_key)
            if jax_key is None:
                continue
            keys = [_stoi(k) for k in jax_key.split(".")]
            _assign_weights(keys, tensor, state_dict, torch_key, transform)
        except ValueError:
            print(f"Skipping unmapped key: {torch_key}")
            continue
    
    # Device placement
    state_dict = jax.device_put(state_dict, jax.devices()[0])
    
    # Merge and return
    return nnx.merge(graph_def, state_dict)


def create_model_from_safe_tensors(model_path: str, config: model_lib.WhisperConfig) -> model_lib.WhisperModel:
    """Create Whisper model from safetensors checkpoint."""
    return convert_hf_whisper_to_nnx(model_path, config)


def load_whisper_model(model_name: str = "openai/whisper-tiny", config: Optional[model_lib.WhisperConfig] = None) -> model_lib.WhisperModel:
    """Load Whisper model from HuggingFace hub."""
    if config is None:
        # Auto-detect config based on model name
        if "tiny" in model_name:
            config = model_lib.WhisperConfig.whisper_tiny()
        elif "base" in model_name:
            config = model_lib.WhisperConfig.whisper_base()
        elif "small" in model_name:
            config = model_lib.WhisperConfig.whisper_small()
        elif "medium" in model_name:
            config = model_lib.WhisperConfig.whisper_medium()
        elif "large" in model_name:
            config = model_lib.WhisperConfig.whisper_large()
        else:
            config = model_lib.WhisperConfig.whisper_tiny()  # Default
    
    return convert_hf_whisper_to_nnx(model_name, config)
