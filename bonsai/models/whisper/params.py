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


def _get_key_and_transform_mapping(cfg: model_lib.WhisperConfig):
    """Mapping of HuggingFace keys to NNX keys with transformations."""
    return {
        # Audio encoder
        r"encoder\.conv1\.weight": ("encoder.conv1.kernel", ((2, 3, 0, 1), None)),
        r"encoder\.conv1\.bias": ("encoder.conv1.bias", None),
        r"encoder\.conv2\.weight": ("encoder.conv2.kernel", ((2, 3, 0, 1), None)),
        r"encoder\.conv2\.bias": ("encoder.conv2.bias", None),
        r"encoder\.conv3\.weight": ("encoder.conv3.kernel", ((2, 3, 0, 1), None)),
        r"encoder\.conv3\.bias": ("encoder.conv3.bias", None),
        r"encoder\.conv4\.weight": ("encoder.conv4.kernel", ((2, 3, 0, 1), None)),
        r"encoder\.conv4\.bias": ("encoder.conv4.bias", None),
        r"encoder\.conv5\.weight": ("encoder.conv5.kernel", ((2, 3, 0, 1), None)),
        r"encoder\.conv5\.bias": ("encoder.conv5.bias", None),
        r"encoder\.conv6\.weight": ("encoder.conv6.kernel", ((2, 3, 0, 1), None)),
        r"encoder\.conv6\.bias": ("encoder.conv6.bias", None),
        r"encoder\.conv7\.weight": ("encoder.conv7.kernel", ((2, 3, 0, 1), None)),
        r"encoder\.conv7\.bias": ("encoder.conv7.bias", None),
        r"encoder\.conv8\.weight": ("encoder.conv8.kernel", ((2, 3, 0, 1), None)),
        r"encoder\.conv8\.bias": ("encoder.conv8.bias", None),
        r"encoder\.conv9\.weight": ("encoder.conv9.kernel", ((2, 3, 0, 1), None)),
        r"encoder\.conv9\.bias": ("encoder.conv9.bias", None),
        r"encoder\.conv10\.weight": ("encoder.conv10.kernel", ((2, 3, 0, 1), None)),
        r"encoder\.conv10\.bias": ("encoder.conv10.bias", None),
        r"encoder\.conv11\.weight": ("encoder.conv11.kernel", ((2, 3, 0, 1), None)),
        r"encoder\.conv11\.bias": ("encoder.conv11.bias", None),
        r"encoder\.conv12\.weight": ("encoder.conv12.kernel", ((2, 3, 0, 1), None)),
        r"encoder\.conv12\.bias": ("encoder.conv12.bias", None),
        
        # Audio encoder projection
        r"encoder\.proj\.weight": ("encoder.projection.kernel", ((1, 0), None)),
        r"encoder\.proj\.bias": ("encoder.projection.bias", None),
        
        # Audio encoder positional embedding
        r"encoder\.positional_embedding": ("encoder.positional_embedding", None),
        
        # Audio encoder transformer blocks
        r"encoder\.blocks\.([0-9]+)\.attn\.query\.weight": (
            r"encoder.blocks.\1.attn.query.kernel", ((1, 0), None)
        ),
        r"encoder\.blocks\.([0-9]+)\.attn\.query\.bias": (
            r"encoder.blocks.\1.attn.query.bias", None
        ),
        r"encoder\.blocks\.([0-9]+)\.attn\.key\.weight": (
            r"encoder.blocks.\1.attn.key.kernel", ((1, 0), None)
        ),
        r"encoder\.blocks\.([0-9]+)\.attn\.key\.bias": (
            r"encoder.blocks.\1.attn.key.bias", None
        ),
        r"encoder\.blocks\.([0-9]+)\.attn\.value\.weight": (
            r"encoder.blocks.\1.attn.value.kernel", ((1, 0), None)
        ),
        r"encoder\.blocks\.([0-9]+)\.attn\.value\.bias": (
            r"encoder.blocks.\1.attn.value.bias", None
        ),
        r"encoder\.blocks\.([0-9]+)\.attn\.out\.weight": (
            r"encoder.blocks.\1.attn.out.kernel", ((1, 0), None)
        ),
        r"encoder\.blocks\.([0-9]+)\.attn\.out\.bias": (
            r"encoder.blocks.\1.attn.out.bias", None
        ),
        r"encoder\.blocks\.([0-9]+)\.attn_ln\.weight": (
            r"encoder.blocks.\1.attn_ln.weight", None
        ),
        r"encoder\.blocks\.([0-9]+)\.attn_ln\.bias": (
            r"encoder.blocks.\1.attn_ln.bias", None
        ),
        r"encoder\.blocks\.([0-9]+)\.mlp\.0\.weight": (
            r"encoder.blocks.\1.mlp.0.kernel", ((1, 0), None)
        ),
        r"encoder\.blocks\.([0-9]+)\.mlp\.0\.bias": (
            r"encoder.blocks.\1.mlp.0.bias", None
        ),
        r"encoder\.blocks\.([0-9]+)\.mlp\.2\.weight": (
            r"encoder.blocks.\1.mlp.2.kernel", ((1, 0), None)
        ),
        r"encoder\.blocks\.([0-9]+)\.mlp\.2\.bias": (
            r"encoder.blocks.\1.mlp.2.bias", None
        ),
        r"encoder\.blocks\.([0-9]+)\.mlp_ln\.weight": (
            r"encoder.blocks.\1.mlp_ln.weight", None
        ),
        r"encoder\.blocks\.([0-9]+)\.mlp_ln\.bias": (
            r"encoder.blocks.\1.mlp_ln.bias", None
        ),
        
        # Audio encoder final layer norm
        r"encoder\.ln_post\.weight": ("encoder.ln_post.weight", None),
        r"encoder\.ln_post\.bias": ("encoder.ln_post.bias", None),
        
        # Text decoder
        r"decoder\.token_embedding\.weight": ("decoder.token_embedding.embedding", None),
        r"decoder\.positional_embedding": ("decoder.positional_embedding", None),
        
        # Text decoder transformer blocks
        r"decoder\.blocks\.([0-9]+)\.attn\.query\.weight": (
            r"decoder.blocks.\1.attn.query.kernel", ((1, 0), None)
        ),
        r"decoder\.blocks\.([0-9]+)\.attn\.query\.bias": (
            r"decoder.blocks.\1.attn.query.bias", None
        ),
        r"decoder\.blocks\.([0-9]+)\.attn\.key\.weight": (
            r"decoder.blocks.\1.attn.key.kernel", ((1, 0), None)
        ),
        r"decoder\.blocks\.([0-9]+)\.attn\.key\.bias": (
            r"decoder.blocks.\1.attn.key.bias", None
        ),
        r"decoder\.blocks\.([0-9]+)\.attn\.value\.weight": (
            r"decoder.blocks.\1.attn.value.kernel", ((1, 0), None)
        ),
        r"decoder\.blocks\.([0-9]+)\.attn\.value\.bias": (
            r"decoder.blocks.\1.attn.value.bias", None
        ),
        r"decoder\.blocks\.([0-9]+)\.attn\.out\.weight": (
            r"decoder.blocks.\1.attn.out.kernel", ((1, 0), None)
        ),
        r"decoder\.blocks\.([0-9]+)\.attn\.out\.bias": (
            r"decoder.blocks.\1.attn.out.bias", None
        ),
        r"decoder\.blocks\.([0-9]+)\.attn_ln\.weight": (
            r"decoder.blocks.\1.attn_ln.weight", None
        ),
        r"decoder\.blocks\.([0-9]+)\.attn_ln\.bias": (
            r"decoder.blocks.\1.attn_ln.bias", None
        ),
        r"decoder\.blocks\.([0-9]+)\.cross_attn\.query\.weight": (
            r"decoder.blocks.\1.cross_attn.query.kernel", ((1, 0), None)
        ),
        r"decoder\.blocks\.([0-9]+)\.cross_attn\.query\.bias": (
            r"decoder.blocks.\1.cross_attn.query.bias", None
        ),
        r"decoder\.blocks\.([0-9]+)\.cross_attn\.key\.weight": (
            r"decoder.blocks.\1.cross_attn.key.kernel", ((1, 0), None)
        ),
        r"decoder\.blocks\.([0-9]+)\.cross_attn\.key\.bias": (
            r"decoder.blocks.\1.cross_attn.key.bias", None
        ),
        r"decoder\.blocks\.([0-9]+)\.cross_attn\.value\.weight": (
            r"decoder.blocks.\1.cross_attn.value.kernel", ((1, 0), None)
        ),
        r"decoder\.blocks\.([0-9]+)\.cross_attn\.value\.bias": (
            r"decoder.blocks.\1.cross_attn.value.bias", None
        ),
        r"decoder\.blocks\.([0-9]+)\.cross_attn\.out\.weight": (
            r"decoder.blocks.\1.cross_attn.out.kernel", ((1, 0), None)
        ),
        r"decoder\.blocks\.([0-9]+)\.cross_attn\.out\.bias": (
            r"decoder.blocks.\1.cross_attn.out.bias", None
        ),
        r"decoder\.blocks\.([0-9]+)\.cross_attn_ln\.weight": (
            r"decoder.blocks.\1.cross_attn_ln.weight", None
        ),
        r"decoder\.blocks\.([0-9]+)\.cross_attn_ln\.bias": (
            r"decoder.blocks.\1.cross_attn_ln.bias", None
        ),
        r"decoder\.blocks\.([0-9]+)\.mlp\.0\.weight": (
            r"decoder.blocks.\1.mlp.0.kernel", ((1, 0), None)
        ),
        r"decoder\.blocks\.([0-9]+)\.mlp\.0\.bias": (
            r"decoder.blocks.\1.mlp.0.bias", None
        ),
        r"decoder\.blocks\.([0-9]+)\.mlp\.2\.weight": (
            r"decoder.blocks.\1.mlp.2.kernel", ((1, 0), None)
        ),
        r"decoder\.blocks\.([0-9]+)\.mlp\.2\.bias": (
            r"decoder.blocks.\1.mlp.2.bias", None
        ),
        r"decoder\.blocks\.([0-9]+)\.mlp_ln\.weight": (
            r"decoder.blocks.\1.mlp_ln.weight", None
        ),
        r"decoder\.blocks\.([0-9]+)\.mlp_ln\.bias": (
            r"decoder.blocks.\1.mlp_ln.bias", None
        ),
        
        # Text decoder final layer norm and output projection
        r"decoder\.ln\.weight": ("decoder.ln.weight", None),
        r"decoder\.ln\.bias": ("decoder.ln.bias", None),
        r"decoder\.output_projection\.weight": ("decoder.output_projection.kernel", ((1, 0), None)),
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
    
    # Create NNX model
    model = model_lib.WhisperModel(config)
    
    # Load weights from safetensors or pytorch format
    model_path = epath.Path(hf_model_path)
    
    # Try to load from safetensors first
    safetensors_files = list(model_path.glob("*.safetensors"))
    if safetensors_files:
        state_dict = safetensors.load_file(str(safetensors_files[0]))
    else:
        # Fall back to pytorch format
        import torch
        checkpoint = torch.load(model_path / "pytorch_model.bin", map_location="cpu")
        state_dict = {k: v.numpy() for k, v in checkpoint.items()}
    
    # Convert weights
    nnx_state_dict = {}
    mapping = _get_key_and_transform_mapping(config)
    
    for torch_key, tensor in state_dict.items():
        try:
            jax_key, transform = _torch_key_to_jax_key(mapping, torch_key)
            keys = jax_key.split(".")
            _assign_weights(keys, tensor, nnx_state_dict, torch_key, transform)
        except ValueError:
            print(f"Skipping unmapped key: {torch_key}")
            continue
    
    # Load weights into model
    nnx.load_state_dict(model, nnx_state_dict)
    
    return model


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
