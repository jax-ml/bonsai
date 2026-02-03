# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Parameter helpers for bonsai.models.gemma3.

Add functions to load or convert pretrained checkpoints and to return
default configuration values used by the model implementation.
"""

import logging
import re
from enum import Enum

import jax
import safetensors.flax as safetensors
from etils import epath
from flax import nnx

from bonsai.models.gemma3 import modeling as model_lib


class Transform(Enum):
    """
    Specifies default transformation types for model parameter names.
    """

    DEFAULT = None
    BIAS = None
    LINEAR = ((1, 0), None)
    CONV2D = ((2, 3, 1, 0), None)
    EMBED = None


def _get_key_and_transform_mapping():
    # Mapping st_keys -> (nnx_keys, (permute_rule, reshape_rule)).
    return {
        r"^language_model\.model\.embed_tokens\.weight$": (
            r"embed_tokens\.weight\.embedding",
            Transform.EMBED,
        ),
        r"^language_model\.model\.layers\.(\d+)\.input_layernorm\.weight$": (
            r"language_model\.layers\.\1\.input_layernorm\.scale",
            Transform.DEFAULT,
        ),
        r"^language_model\.model\.layers\.(\d+)\.mlp\.down_proj\.weight$": (
            r"language_model\.layers\.\1\.mlp\.down_proj\.kernel",
            Transform.LINEAR,
        ),
        r"^language_model\.model\.layers\.(\d+)\.mlp\.gate_proj\.weight$": (
            r"language_model\.layers\.\1\.mlp\.gate_proj\.kernel",
            Transform.LINEAR,
        ),
        r"^language_model\.model\.layers\.(\d+)\.mlp\.up_proj\.weight$": (
            r"language_model\.layers\.\1\.mlp\.up_proj\.kernel",
            Transform.LINEAR,
        ),
        r"^language_model\.model\.layers\.(\d+)\.post_attention_layernorm\.weight$": (
            r"language_model\.layers\.\1\.post_attention_layernorm\.scale",
            Transform.DEFAULT,
        ),
        r"^language_model\.model\.layers\.(\d+)\.post_feedforward_layernorm\.weight$": (
            r"language_model\.layers\.\1\.post_feedforward_layernorm\.scale",
            Transform.DEFAULT,
        ),
        r"^language_model\.model\.layers\.(\d+)\.pre_feedforward_layernorm\.weight$": (
            r"language_model\.layers\.\1\.pre_feedforward_layernorm\.scale",
            Transform.DEFAULT,
        ),
        r"^language_model\.model\.layers\.(\d+)\.self_attn\.k_norm\.weight$": (
            r"language_model\.layers\.\1\.self_attn\.k_norm\.scale",
            Transform.DEFAULT,
        ),
        r"^language_model\.model\.layers\.(\d+)\.self_attn\.k_proj\.weight$": (
            r"language_model\.layers\.\1\.self_attn\.k_proj\.kernel",
            Transform.LINEAR,
        ),
        r"^language_model\.model\.layers\.(\d+)\.self_attn\.o_proj\.weight$": (
            r"language_model\.layers\.\1\.self_attn\.o_proj\.kernel",
            Transform.LINEAR,
        ),
        r"^language_model\.model\.layers\.(\d+)\.self_attn\.q_norm\.weight$": (
            r"language_model\.layers\.\1\.self_attn\.q_norm\.scale",
            Transform.DEFAULT,
        ),
        r"^language_model\.model\.layers\.(\d+)\.self_attn\.q_proj\.weight$": (
            r"language_model\.layers\.\1\.self_attn\.q_proj\.kernel",
            Transform.LINEAR,
        ),
        r"^language_model\.model\.layers\.(\d+)\.self_attn\.v_proj\.weight$": (
            r"language_model\.layers\.\1\.self_attn\.v_proj\.kernel",
            Transform.LINEAR,
        ),
        r"^language_model\.model\.norm\.weight$": (r"language_model\.norm\.scale", Transform.DEFAULT),
        r"^multi_modal_projector\.mm_input_projection_weight$": (
            r"multi_modal_projector\.mm_input_projection_weight",
            Transform.DEFAULT,
        ),
        r"^multi_modal_projector\.mm_soft_emb_norm\.weight$": (
            r"multi_modal_projector\.mm_soft_emb_norm\.scale",
            Transform.DEFAULT,
        ),
        r"^vision_tower\.vision_model\.embeddings\.patch_embedding\.bias$": (
            r"vision_tower\.embeddings\.patch_embedding\.bias",
            Transform.BIAS,
        ),
        r"^vision_tower\.vision_model\.embeddings\.patch_embedding\.weight$": (
            r"vision_tower\.embeddings\.patch_embedding\.kernel",
            Transform.CONV2D,
        ),
        r"^vision_tower\.vision_model\.embeddings\.position_embedding\.weight$": (
            r"vision_tower\.embeddings\.position_embedding\.embedding",
            Transform.EMBED,
        ),
        r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.layer_norm(\d+)\.bias$": (
            r"vision_tower\.encoder\.layers\.\1\.layer_norm\2\.bias",
            Transform.BIAS,
        ),
        r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.layer_norm(\d+)\.weight$": (
            r"vision_tower\.encoder\.layers\.\1\.layer_norm\2\.scale",
            Transform.DEFAULT,
        ),
        r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.mlp\.fc(\d+)\.bias$": (
            r"vision_tower\.encoder\.layers\.\1\.mlp\.fc\2\.bias",
            Transform.BIAS,
        ),
        r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.mlp\.fc(\d+)\.weight$": (
            r"vision_tower\.encoder\.layers\.\1\.mlp\.fc\2\.kernel",
            Transform.LINEAR,
        ),
        r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.k_proj\.bias$": (
            r"vision_tower\.encoder\.layers\.\1\.self_attn\.k_proj\.bias",
            Transform.BIAS,
        ),
        r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.k_proj\.weight$": (
            r"vision_tower\.encoder\.layers\.\1\.self_attn\.k_proj\.kernel",
            Transform.LINEAR,
        ),
        r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.out_proj\.bias$": (
            r"vision_tower\.encoder\.layers\.\1\.self_attn\.out_proj\.bias",
            Transform.BIAS,
        ),
        r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.out_proj\.weight$": (
            r"vision_tower\.encoder\.layers\.\1\.self_attn\.out_proj\.kernel",
            Transform.LINEAR,
        ),
        r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.q_proj\.bias$": (
            r"vision_tower\.encoder\.layers\.\1\.self_attn\.q_proj\.bias",
            Transform.BIAS,
        ),
        r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.q_proj\.weight$": (
            r"vision_tower\.encoder\.layers\.\1\.self_attn\.q_proj\.kernel",
            Transform.LINEAR,
        ),
        r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.v_proj\.bias$": (
            r"vision_tower\.encoder\.layers\.\1\.self_attn\.v_proj\.bias",
            Transform.BIAS,
        ),
        r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.v_proj\.weight$": (
            r"vision_tower\.encoder\.layers\.\1\.self_attn\.v_proj\.kernel",
            Transform.LINEAR,
        ),
        r"^vision_tower\.vision_model\.post_layernorm\.bias$": (
            r"vision_tower\.post_layernorm\.bias",
            Transform.BIAS,
        ),
        r"^vision_tower\.vision_model\.post_layernorm\.weight$": (
            r"vision_tower\.post_layernorm\.scale",
            Transform.DEFAULT,
        ),
    }


def _st_key_to_jax_key(mapping, source_key):
    """Map a safetensors key to exactly one JAX key & transform, else warn/error."""
    subs = [
        (re.sub(pat, repl, source_key), transform)
        for pat, (repl, transform) in mapping.items()
        if re.match(pat, source_key)
    ]
    if not subs:
        logging.warning(f"No mapping found for key: {source_key!r}")
        return None, None
    if len(subs) > 1:
        keys = [s for s, _ in subs]
        raise ValueError(f"Multiple mappings found for {source_key!r}: {keys}")
    return subs[0]


def _assign_weights(keys, tensor, state_dict, st_key, transform, sharding_dict):
    """Recursively descend into state_dict and assign the (possibly permuted/reshaped) tensor."""
    key, *rest = keys
    if not rest:
        if transform is not None:
            permute, reshape = transform
            if permute:
                tensor = tensor.transpose(permute)
            if reshape:
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


def _stoi(s):
    try:
        return int(s)
    except ValueError:
        return s


# TODO: Update to optimize parameter loading for larger models
def create_gemma3_from_pretrained(file_dir: str, cfg: model_lib.ModelConfig, *, mesh: jax.sharding.Mesh | None = None):
    """
    Load safetensor weights from a file, then convert & merge into a flax.nnx ViT model.

    Returns:
      A flax.nnx.Model instance with loaded parameters.
    """
    files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors found in {file_dir}")

    tensor_dict = {}
    for f in files:
        tensor_dict |= safetensors.load_file(f)

    gemma3 = model_lib.Gemma3Model(cfg, rngs=nnx.Rngs(0))
    graph_def, abs_state = nnx.split(gemma3)
    jax_state = nnx.to_pure_dict(abs_state)
    sharding = nnx.to_pure_dict(nnx.get_named_sharding(abs_state, mesh)) if mesh is not None else None

    mapping = _get_key_and_transform_mapping()
    for st_key, tensor in tensor_dict.items():
        jax_key, transform = _st_key_to_jax_key(mapping, st_key)
        if jax_key is None:
            continue
        keys = [_stoi(k) for k in jax_key.split(r"\.")]
        try:
            _assign_weights(keys, tensor, jax_state, st_key, transform.value, sharding)
        except KeyError as e:
            print(f"Key error: {keys} at {e}")
        except ValueError as e:
            print(e)
        except Exception as e:
            print(keys)
            raise e

    return nnx.merge(graph_def, jax_state)
