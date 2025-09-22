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
        r"model\.encoder\.embed_positions\.weight": ("encoder.embed_positions", None),
        
        # Audio encoder transformer blocks
        r"model\.encoder\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (
            r"encoder.layers.\1.self_attn.q_proj.kernel", ((1, 0), None)
        ),
        r"model\.encoder\.layers\.([0-9]+)\.self_attn\.q_proj\.bias": (
            r"encoder.layers.\1.self_attn.q_proj.bias", None
        ),
        r"model\.encoder\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (
            r"encoder.layers.\1.self_attn.k_proj.kernel", ((1, 0), None)
        ),
        r"model\.encoder\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (
            r"encoder.layers.\1.self_attn.v_proj.kernel", ((1, 0), None)
        ),
        r"model\.encoder\.layers\.([0-9]+)\.self_attn\.v_proj\.bias": (
            r"encoder.layers.\1.self_attn.v_proj.bias", None
        ),
        r"model\.encoder\.layers\.([0-9]+)\.self_attn\.out_proj\.weight": (
            r"encoder.layers.\1.self_attn.out_proj.kernel", ((1, 0), None)
        ),
        r"model\.encoder\.layers\.([0-9]+)\.self_attn\.out_proj\.bias": (
            r"encoder.layers.\1.self_attn.out_proj.bias", None
        ),
        r"model\.encoder\.layers\.([0-9]+)\.self_attn_layer_norm\.weight": (
            r"encoder.layers.\1.self_attn_layer_norm.scale", None
        ),
        r"model\.encoder\.layers\.([0-9]+)\.self_attn_layer_norm\.bias": (
            r"encoder.layers.\1.self_attn_layer_norm.bias", None
        ),
        r"model\.encoder\.layers\.([0-9]+)\.fc1\.weight": (
            r"encoder.layers.\1.fc1.kernel", ((1, 0), None)
        ),
        r"model\.encoder\.layers\.([0-9]+)\.fc1\.bias": (
            r"encoder.layers.\1.fc1.bias", None
        ),
        r"model\.encoder\.layers\.([0-9]+)\.fc2\.weight": (
            r"encoder.layers.\1.fc2.kernel", ((1, 0), None)
        ),
        r"model\.encoder\.layers\.([0-9]+)\.fc2\.bias": (
            r"encoder.layers.\1.fc2.bias", None
        ),
        r"model\.encoder\.layers\.([0-9]+)\.final_layer_norm\.weight": (
            r"encoder.layers.\1.final_layer_norm.scale", None
        ),
        r"model\.encoder\.layers\.([0-9]+)\.final_layer_norm\.bias": (
            r"encoder.layers.\1.final_layer_norm.bias", None
        ),
        
        # Audio encoder final layer norm
        r"model\.encoder\.layer_norm\.weight": ("encoder.layer_norm.scale", None),
        r"model\.encoder\.layer_norm\.bias": ("encoder.layer_norm.bias", None),
        
        # Text decoder
        r"model\.decoder\.embed_tokens\.weight": ("decoder.embed_tokens.embedding", None),
        r"model\.decoder\.embed_positions\.weight": ("decoder.embed_positions", None),
        
        # Text decoder transformer blocks
        r"model\.decoder\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (
            r"decoder.layers.\1.self_attn.q_proj.kernel", ((1, 0), None)
        ),
        r"model\.decoder\.layers\.([0-9]+)\.self_attn\.q_proj\.bias": (
            r"decoder.layers.\1.self_attn.q_proj.bias", None
        ),
        r"model\.decoder\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (
            r"decoder.layers.\1.self_attn.k_proj.kernel", ((1, 0), None)
        ),
        r"model\.decoder\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (
            r"decoder.layers.\1.self_attn.v_proj.kernel", ((1, 0), None)
        ),
        r"model\.decoder\.layers\.([0-9]+)\.self_attn\.v_proj\.bias": (
            r"decoder.layers.\1.self_attn.v_proj.bias", None
        ),
        r"model\.decoder\.layers\.([0-9]+)\.self_attn\.out_proj\.weight": (
            r"decoder.layers.\1.self_attn.out_proj.kernel", ((1, 0), None)
        ),
        r"model\.decoder\.layers\.([0-9]+)\.self_attn\.out_proj\.bias": (
            r"decoder.layers.\1.self_attn.out_proj.bias", None
        ),
        r"model\.decoder\.layers\.([0-9]+)\.self_attn_layer_norm\.weight": (
            r"decoder.layers.\1.self_attn_layer_norm.scale", None
        ),
        r"model\.decoder\.layers\.([0-9]+)\.self_attn_layer_norm\.bias": (
            r"decoder.layers.\1.self_attn_layer_norm.bias", None
        ),
        
        # Cross attention
        r"model\.decoder\.layers\.([0-9]+)\.encoder_attn\.q_proj\.weight": (
            r"decoder.layers.\1.encoder_attn.q_proj.kernel", ((1, 0), None)
        ),
        r"model\.decoder\.layers\.([0-9]+)\.encoder_attn\.q_proj\.bias": (
            r"decoder.layers.\1.encoder_attn.q_proj.bias", None
        ),
        r"model\.decoder\.layers\.([0-9]+)\.encoder_attn\.k_proj\.weight": (
            r"decoder.layers.\1.encoder_attn.k_proj.kernel", ((1, 0), None)
        ),
        r"model\.decoder\.layers\.([0-9]+)\.encoder_attn\.v_proj\.weight": (
            r"decoder.layers.\1.encoder_attn.v_proj.kernel", ((1, 0), None)
        ),
        r"model\.decoder\.layers\.([0-9]+)\.encoder_attn\.v_proj\.bias": (
            r"decoder.layers.\1.encoder_attn.v_proj.bias", None
        ),
        r"model\.decoder\.layers\.([0-9]+)\.encoder_attn\.out_proj\.weight": (
            r"decoder.layers.\1.encoder_attn.out_proj.kernel", ((1, 0), None)
        ),
        r"model\.decoder\.layers\.([0-9]+)\.encoder_attn\.out_proj\.bias": (
            r"decoder.layers.\1.encoder_attn.out_proj.bias", None
        ),
        r"model\.decoder\.layers\.([0-9]+)\.encoder_attn_layer_norm\.weight": (
            r"decoder.layers.\1.encoder_attn_layer_norm.scale", None
        ),
        r"model\.decoder\.layers\.([0-9]+)\.encoder_attn_layer_norm\.bias": (
            r"decoder.layers.\1.encoder_attn_layer_norm.bias", None
        ),
        
        # MLP
        r"model\.decoder\.layers\.([0-9]+)\.fc1\.weight": (
            r"decoder.layers.\1.fc1.kernel", ((1, 0), None)
        ),
        r"model\.decoder\.layers\.([0-9]+)\.fc1\.bias": (
            r"decoder.layers.\1.fc1.bias", None
        ),
        r"model\.decoder\.layers\.([0-9]+)\.fc2\.weight": (
            r"decoder.layers.\1.fc2.kernel", ((1, 0), None)
        ),
        r"model\.decoder\.layers\.([0-9]+)\.fc2\.bias": (
            r"decoder.layers.\1.fc2.bias", None
        ),
        r"model\.decoder\.layers\.([0-9]+)\.final_layer_norm\.weight": (
            r"decoder.layers.\1.final_layer_norm.scale", None
        ),
        r"model\.decoder\.layers\.([0-9]+)\.final_layer_norm\.bias": (
            r"decoder.layers.\1.final_layer_norm.bias", None
        ),
        
        # Decoder final layer norm
        r"model\.decoder\.layer_norm\.weight": ("decoder.layer_norm.scale", None),
        r"model\.decoder\.layer_norm\.bias": ("decoder.layer_norm.bias", None),
    }


def _torch_key_to_jax_key(torch_key: str) -> str:
    """Convert PyTorch-style key to JAX-style key."""
    # Replace .weight with .kernel for linear layers
    if torch_key.endswith('.weight') and 'embed' not in torch_key and 'norm' not in torch_key:
        return torch_key.replace('.weight', '.kernel')
    # Replace .weight with .scale for layer norms
    elif torch_key.endswith('.weight') and 'norm' in torch_key:
        return torch_key.replace('.weight', '.scale')
    return torch_key


def _assign_weights(hf_weights: Dict[str, np.ndarray], nnx_state: Dict[str, Any], 
                   key_mapping: Dict[str, Tuple[str, Optional[Tuple]]]) -> Dict[str, Any]:
    """Assign HuggingFace weights to NNX state."""
    assigned_count = 0
    skipped_count = 0
    
    for hf_key, hf_array in hf_weights.items():
        matched = False
        
        for pattern, (nnx_key, transform) in key_mapping.items():
            match = re.match(pattern, hf_key)
            if match:
                # Handle regex groups
                if '\\1' in nnx_key:
                    nnx_key = match.expand(nnx_key)
                
                # Apply transformation if specified
                if transform is not None:
                    perm, _ = transform
                    if perm is not None:
                        hf_array = np.transpose(hf_array, perm)
                
                # Convert to JAX array
                if hasattr(hf_array, 'numpy'):
                    hf_array = hf_array.numpy()
                jax_array = jax.device_put(hf_array.astype(np.float32))
                
                # Navigate to the correct location in the state dict
                keys = nnx_key.split('.')
                current = nnx_state
                for key in keys[:-1]:
                    # Convert layer indices to integers
                    if key.isdigit():
                        key = int(key)
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                # Keep the final key as string
                current[keys[-1]] = jax_array
                assigned_count += 1
                matched = True
                break
        
        if not matched:
            skipped_count += 1
    
    print(f"Assigned {assigned_count} tensors, skipped {skipped_count} tensors")
    return nnx_state


def convert_hf_whisper_to_nnx(model_dir: str, config: model_lib.WhisperConfig) -> Tuple[nnx.GraphDef, Dict[str, Any]]:
    """Convert HuggingFace Whisper model to NNX format."""
    # Load HuggingFace weights
    model_path = epath.Path(model_dir)
    safetensors_file = model_path / "model.safetensors"
    
    if not safetensors_file.exists():
        raise FileNotFoundError(f"No model.safetensors found in {model_dir}")
    
    print(f"Loading weights from {safetensors_file}")
    with safetensors.safe_open(str(safetensors_file), framework="pt") as f:
        hf_weights = {key: f.get_tensor(key) for key in f.keys()}
    
    print(f"Loaded {len(hf_weights)} tensors from HuggingFace model")
    
    # Create NNX model structure
    model = nnx.eval_shape(lambda: model_lib.WhisperModel(config, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(model)
    state_dict = abs_state.to_pure_dict()
    
    print(f"Created NNX model with {len(state_dict)} top-level keys")
    
    key_mapping = _get_key_and_transform_mapping(config)
    
    state_dict = _assign_weights(hf_weights, state_dict, key_mapping)
    
    return graph_def, state_dict


def create_model_from_safe_tensors(model_dir: str, config: model_lib.WhisperConfig) -> model_lib.WhisperModel:
    """Create NNX Whisper model from HuggingFace safetensors."""
    graph_def, state_dict = convert_hf_whisper_to_nnx(model_dir, config)
    
    model = nnx.merge(graph_def, state_dict)
    return model


def load_whisper_model(model_name: str = "openai/whisper-tiny", 
                      cache_dir: str = "/tmp/models-bonsai") -> model_lib.WhisperModel:
    """Load Whisper model from HuggingFace hub."""
    from transformers import WhisperConfig
    
    # Download model if not cached
    model_dir = f"{cache_dir}/{model_name.split('/')[-1]}"
    
    if not epath.Path(model_dir).exists():
        print(f"Downloading {model_name} to {model_dir}")
        from transformers import WhisperProcessor
        processor = WhisperProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        # The processor download will also download the model files
    
    # Load config and create model
    hf_config = WhisperConfig.from_pretrained(model_dir)
    config = model_lib.WhisperConfig(
        vocab_size=hf_config.vocab_size,
        n_mels=hf_config.num_mel_bins,
        n_audio_ctx=hf_config.max_source_positions,
        n_audio_state=hf_config.d_model,
        n_audio_head=hf_config.encoder_attention_heads,
        n_audio_layer=hf_config.encoder_layers,
        n_text_ctx=hf_config.max_target_positions,
        n_text_state=hf_config.d_model,
        n_text_head=hf_config.decoder_attention_heads,
        n_text_layer=hf_config.decoder_layers,
        n_vocab=hf_config.vocab_size,
        n_langs=hf_config.num_languages,
        dtype=jnp.float32
    )
    
    return create_model_from_safe_tensors(model_dir, config)
