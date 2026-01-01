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

"""Parameter loading utilities for Qwen3-VL Flax NNX model."""

import gc
import re
from enum import Enum
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

try:
    import safetensors

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

from bonsai.models.qwen3vl import modeling as model_lib


def _get_key_and_transform_mapping(cfg: model_lib.Qwen3VLConfig):
    """Build key mapping and transformation rules for PyTorch to Flax conversion."""

    class Transform(Enum):
        NONE = None
        LINEAR = ((1, 0), None, False)
        EMBED = None
        CONV3D = ((2, 3, 4, 1, 0), None, False)
        SCALE = None

    mapping = {
        # Vision Model
        r"visual\.patch_embed\.proj\.weight": ("visual.patch_embed.proj.kernel", Transform.CONV3D),
        r"visual\.patch_embed\.proj\.bias": ("visual.patch_embed.proj.bias", Transform.NONE),
        r"visual\.pos_embed\.weight": ("visual.pos_embed.embedding", Transform.EMBED),
        r"visual\.blocks\.(\d+)\.norm1\.weight": (r"visual.blocks.\1.norm1.scale", Transform.SCALE),
        r"visual\.blocks\.(\d+)\.norm1\.bias": (r"visual.blocks.\1.norm1.bias", Transform.NONE),
        r"visual\.blocks\.(\d+)\.norm2\.weight": (r"visual.blocks.\1.norm2.scale", Transform.SCALE),
        r"visual\.blocks\.(\d+)\.norm2\.bias": (r"visual.blocks.\1.norm2.bias", Transform.NONE),
        r"visual\.blocks\.(\d+)\.attn\.qkv\.weight": (r"visual.blocks.\1.attn.qkv.kernel", Transform.LINEAR),
        r"visual\.blocks\.(\d+)\.attn\.qkv\.bias": (r"visual.blocks.\1.attn.qkv.bias", Transform.NONE),
        r"visual\.blocks\.(\d+)\.attn\.proj\.weight": (r"visual.blocks.\1.attn.proj.kernel", Transform.LINEAR),
        r"visual\.blocks\.(\d+)\.attn\.proj\.bias": (r"visual.blocks.\1.attn.proj.bias", Transform.NONE),
        r"visual\.blocks\.(\d+)\.mlp\.fc1\.weight": (r"visual.blocks.\1.mlp.fc1.kernel", Transform.LINEAR),
        r"visual\.blocks\.(\d+)\.mlp\.fc1\.bias": (r"visual.blocks.\1.mlp.fc1.bias", Transform.NONE),
        r"visual\.blocks\.(\d+)\.mlp\.fc2\.weight": (r"visual.blocks.\1.mlp.fc2.kernel", Transform.LINEAR),
        r"visual\.blocks\.(\d+)\.mlp\.fc2\.bias": (r"visual.blocks.\1.mlp.fc2.bias", Transform.NONE),
        r"visual\.merger\.norm\.weight": ("visual.merger.norm.scale", Transform.SCALE),
        r"visual\.merger\.norm\.bias": ("visual.merger.norm.bias", Transform.NONE),
        r"visual\.merger\.linear_fc1\.weight": ("visual.merger.linear_fc1.kernel", Transform.LINEAR),
        r"visual\.merger\.linear_fc1\.bias": ("visual.merger.linear_fc1.bias", Transform.NONE),
        r"visual\.merger\.linear_fc2\.weight": ("visual.merger.linear_fc2.kernel", Transform.LINEAR),
        r"visual\.merger\.linear_fc2\.bias": ("visual.merger.linear_fc2.bias", Transform.NONE),
        r"visual\.deepstack_merger_list\.(\d+)\.norm\.weight": (
            r"visual.deepstack_merger_list.\1.norm.scale",
            Transform.SCALE,
        ),
        r"visual\.deepstack_merger_list\.(\d+)\.norm\.bias": (
            r"visual.deepstack_merger_list.\1.norm.bias",
            Transform.NONE,
        ),
        r"visual\.deepstack_merger_list\.(\d+)\.linear_fc1\.weight": (
            r"visual.deepstack_merger_list.\1.linear_fc1.kernel",
            Transform.LINEAR,
        ),
        r"visual\.deepstack_merger_list\.(\d+)\.linear_fc1\.bias": (
            r"visual.deepstack_merger_list.\1.linear_fc1.bias",
            Transform.NONE,
        ),
        r"visual\.deepstack_merger_list\.(\d+)\.linear_fc2\.weight": (
            r"visual.deepstack_merger_list.\1.linear_fc2.kernel",
            Transform.LINEAR,
        ),
        r"visual\.deepstack_merger_list\.(\d+)\.linear_fc2\.bias": (
            r"visual.deepstack_merger_list.\1.linear_fc2.bias",
            Transform.NONE,
        ),
        # Text Model
        r"language_model\.embed_tokens\.weight": ("embed_tokens.embedding", Transform.EMBED),
        r"embed_tokens\.weight": ("embed_tokens.embedding", Transform.EMBED),
        r"language_model\.layers\.(\d+)\.self_attn\.q_proj\.weight": (
            r"layers.\1.self_attn.q_proj.kernel",
            Transform.LINEAR,
        ),
        r"language_model\.layers\.(\d+)\.self_attn\.k_proj\.weight": (
            r"layers.\1.self_attn.k_proj.kernel",
            Transform.LINEAR,
        ),
        r"language_model\.layers\.(\d+)\.self_attn\.v_proj\.weight": (
            r"layers.\1.self_attn.v_proj.kernel",
            Transform.LINEAR,
        ),
        r"language_model\.layers\.(\d+)\.self_attn\.o_proj\.weight": (
            r"layers.\1.self_attn.o_proj.kernel",
            Transform.LINEAR,
        ),
        r"language_model\.layers\.(\d+)\.self_attn\.q_norm\.weight": (
            r"layers.\1.self_attn.q_norm.weight",
            Transform.SCALE,
        ),
        r"language_model\.layers\.(\d+)\.self_attn\.k_norm\.weight": (
            r"layers.\1.self_attn.k_norm.weight",
            Transform.SCALE,
        ),
        r"language_model\.layers\.(\d+)\.mlp\.gate_proj\.weight": (
            r"layers.\1.mlp.gate_proj.kernel",
            Transform.LINEAR,
        ),
        r"language_model\.layers\.(\d+)\.mlp\.up_proj\.weight": (r"layers.\1.mlp.up_proj.kernel", Transform.LINEAR),
        r"language_model\.layers\.(\d+)\.mlp\.down_proj\.weight": (
            r"layers.\1.mlp.down_proj.kernel",
            Transform.LINEAR,
        ),
        r"language_model\.layers\.(\d+)\.input_layernorm\.weight": (
            r"layers.\1.input_layernorm.weight",
            Transform.SCALE,
        ),
        r"language_model\.layers\.(\d+)\.post_attention_layernorm\.weight": (
            r"layers.\1.post_attention_layernorm.weight",
            Transform.SCALE,
        ),
        r"language_model\.norm\.weight": ("norm.weight", Transform.SCALE),
        r"model\.norm\.weight": ("norm.weight", Transform.SCALE),
    }
    return mapping


def _torch_key_to_jax_key(mapping, source_key) -> Tuple[Optional[str], Any]:
    """Map a PyTorch key to Flax key using regex patterns."""
    for pat, (repl, transform) in mapping.items():
        if re.match(pat, source_key):
            return re.sub(pat, repl, source_key), transform
    return None, None


def _stoi(s: str):
    """Convert string to int if possible."""
    try:
        return int(s)
    except ValueError:
        return s


def _assign_weights(keys, tensor, state_dict, st_key, transform, sharding_dict):
    """Recursively descend into state_dict and assign the transformed tensor."""
    key, *rest = keys
    if not rest:
        if transform is not None:
            permute, reshape, reshape_first = transform
            if reshape_first and reshape is not None:
                tensor = tensor.reshape(reshape)
            if permute:
                tensor = tensor.transpose(permute)
            if not reshape_first and reshape is not None:
                tensor = tensor.reshape(reshape)
        if tensor.shape != state_dict[key].shape:
            raise ValueError(f"Shape mismatch for {st_key}: {tensor.shape} vs {state_dict[key].shape}")
        if sharding_dict is not None:
            state_dict[key] = jax.device_put(tensor, sharding_dict[key])
        else:
            state_dict[key] = jax.device_put(tensor)
    else:
        next_sharding = sharding_dict[key] if sharding_dict is not None else None
        _assign_weights(rest, tensor, state_dict[key], st_key, transform, next_sharding)


def create_model_from_safe_tensors(
    file_dir: str,
    cfg: model_lib.Qwen3VLConfig,
    mesh: Optional[jax.sharding.Mesh] = None,
) -> model_lib.Qwen3VLModel:
    """Load tensors from safetensors file(s) and create a Qwen3VL model.

    Args:
        file_dir: Directory containing .safetensors file(s)
        cfg: Model configuration
        mesh: Optional JAX mesh for sharding

    Returns:
        Initialized Qwen3VLModel with loaded weights
    """
    if not HAS_SAFETENSORS:
        raise ImportError("safetensors is required. Install with: pip install safetensors")

    from etils import epath

    files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors found in {file_dir}")

    model = nnx.eval_shape(lambda: model_lib.Qwen3VLModel(cfg, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(model)
    state_dict = abs_state.to_pure_dict()
    sharding = nnx.get_named_sharding(abs_state, mesh).to_pure_dict() if mesh is not None else None

    key_mapping = _get_key_and_transform_mapping(cfg)
    conversion_errors = []

    for f in files:
        with safetensors.safe_open(str(f), framework="numpy") as sf:
            for torch_key in sf.keys():
                tensor = sf.get_tensor(torch_key)
                jax_key, transform = _torch_key_to_jax_key(key_mapping, torch_key)
                if jax_key is None:
                    continue
                keys = [_stoi(k) for k in jax_key.split(".")]
                try:
                    transform_value = transform.value if transform else None
                    _assign_weights(keys, tensor, state_dict, torch_key, transform_value, sharding)
                except Exception as e:
                    conversion_errors.append(f"Failed '{torch_key}' -> '{jax_key}': {e}")
        gc.collect()

    if conversion_errors:
        raise RuntimeError(f"{len(conversion_errors)} weight conversion errors:\n" + "\n".join(conversion_errors[:10]))

    gc.collect()
    return nnx.merge(graph_def, state_dict)
