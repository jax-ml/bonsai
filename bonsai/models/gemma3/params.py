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

from enum import Enum

import safetensors.flax as safetensors
from etils import epath
from flax import nnx

from bonsai.models.gemma3 import modeling as model_lib
from bonsai.utils.params import stoi, safetensors_key_to_bonsai_key, assign_weights_from_eval_shape


def _get_key_and_transform_mapping():
    class Transform(Enum):
        """
        Specifies default transformation types for model parameter names.
        """

        DEFAULT = None
        BIAS = None
        LINEAR = ((1, 0), None, False)
        CONV2D = ((2, 3, 1, 0), None, False)
        EMBED = None

    # Mapping st_keys -> (nnx_keys, (permute_rule, reshape_rule, reshape_first)).
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


# TODO: Update to optimize parameter loading for larger models
def create_gemma3_from_pretrained(file_dir: str, cfg: model_lib.ModelConfig):
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

    gemma3 = nnx.eval_shape(lambda: model_lib.Gemma3Model(cfg, rngs=nnx.Rngs(0)))
    graph_def, abs_state = nnx.split(gemma3)
    jax_state = nnx.to_pure_dict(abs_state)

    mapping = _get_key_and_transform_mapping()
    for st_key, tensor in tensor_dict.items():
        jax_key, transform = safetensors_key_to_bonsai_key(mapping, st_key)
        if jax_key is None:
            continue
        keys = [stoi(k) for k in jax_key.split(r"\.")]
        try:
            assign_weights_from_eval_shape(keys, tensor, jax_state, st_key, transform.value)
        except KeyError as e:
            print(f"Key error: {keys} at {e}")
        except ValueError as e:
            print(e)
        except Exception as e:
            print(keys)
            raise e

    return nnx.merge(graph_def, jax_state)
