import re
# Copyright 2026 The JAX Authors.
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
Parameter helpers for bonsai.models.gemma4.

Provides parameter matching and checkpoint utilities.
"""

from enum import Enum

import jax
import jax.numpy as jnp
import safetensors.flax as safetensors
from etils import epath
from flax import nnx

from bonsai.models.gemma4 import modeling as model_lib
from bonsai.utils.params import stoi, map_to_bonsai_key, assign_weights_from_eval_shape


def _get_key_and_transform_mapping():
    """Returns the mapping from safetensors keys to bonsai model keys and their transforms.

    Returns:
        dict: A dictionary mapping safetensors regex patterns to a tuple containing the
        corresponding bonsai model key pattern and the required Transform enum.
    """

    class Transform(Enum):
        """
        Specifies default transformation types for model parameter names.
        """

        DEFAULT = None
        BIAS = None
        LINEAR = ((1, 0), None, False)
        CONV2D = ((2, 3, 1, 0), None, False)
        EMBED = None
        LINEAR_3D = ((0, 2, 1), None, False)

    # Mapping st_keys -> (nnx_keys, (permute_rule, reshape_rule, reshape_first)).
    return {
        r"^model\.embed_tokens\.weight$": (r"model\.embed_tokens\.embedding", Transform.EMBED),
        r"^model\.embed_tokens_per_layer\.weight$": (r"model\.embed_tokens_per_layer\.embedding", Transform.EMBED),
        r"^model\.per_layer_model_projection\.weight$": (
            r"model\.per_layer_model_projection\.kernel",
            Transform.LINEAR,
        ),
        r"^model\.per_layer_projection_norm\.weight$": (r"model\.per_layer_projection_norm\.scale", Transform.DEFAULT),
        r"^model\.layers\.(\d+)\.per_layer_input_gate\.weight$": (
            r"model\.layers\.\1\.per_layer_input_gate\.kernel",
            Transform.LINEAR,
        ),
        r"^model\.layers\.(\d+)\.per_layer_projection\.weight$": (
            r"model\.layers\.\1\.per_layer_projection\.kernel",
            Transform.LINEAR,
        ),
        r"^model\.layers\.(\d+)\.post_per_layer_input_norm\.weight$": (
            r"model\.layers\.\1\.post_per_layer_input_norm\.scale",
            Transform.DEFAULT,
        ),
        r"^model\.layers\.(\d+)\.input_layernorm\.weight$": (
            r"model\.layers\.\1\.pre_self_attention_norm\.scale",
            Transform.DEFAULT,
        ),
        r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$": (
            r"model\.layers\.\1\.post_self_attention_norm\.scale",
            Transform.DEFAULT,
        ),
        r"^model\.layers\.(\d+)\.pre_feedforward_layernorm\.weight$": (
            r"model\.layers\.\1\.pre_ffw_norm\.scale",
            Transform.DEFAULT,
        ),
        r"^model\.layers\.(\d+)\.post_feedforward_layernorm\.weight$": (
            r"model\.layers\.\1\.post_ffw_norm\.scale",
            Transform.DEFAULT,
        ),
        r"^model\.layers\.(\d+)\.self_attn\.q_norm\.weight$": (
            r"model\.layers\.\1\.self_attention\.q_norm\.scale",
            Transform.DEFAULT,
        ),
        r"^model\.layers\.(\d+)\.self_attn\.k_norm\.weight$": (
            r"model\.layers\.\1\.self_attention\.k_norm\.scale",
            Transform.DEFAULT,
        ),
        r"^model\.layers\.(\d+)\.self_attn\.v_norm\.weight$": (
            r"model\.layers\.\1\.self_attention\.v_norm\.scale",
            Transform.DEFAULT,
        ),
        r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$": (
            r"model\.layers\.\1\.self_attention\.q_proj\.kernel",
            Transform.LINEAR,
        ),
        r"^model\.layers\.(\d+)\.self_attn\.k_proj\.weight$": (
            r"model\.layers\.\1\.self_attention\.k_proj\.kernel",
            Transform.LINEAR,
        ),
        r"^model\.layers\.(\d+)\.self_attn\.v_proj\.weight$": (
            r"model\.layers\.\1\.self_attention\.v_proj\.kernel",
            Transform.LINEAR,
        ),
        r"^model\.layers\.(\d+)\.self_attn\.o_proj\.weight$": (
            r"model\.layers\.\1\.self_attention\.o_proj\.kernel",
            Transform.LINEAR,
        ),
        r"^model\.layers\.(\d+)\.mlp\.gate_proj\.weight$": (
            r"model\.layers\.\1\.mlp\.gate_proj\.kernel",
            Transform.LINEAR,
        ),
        r"^model\.layers\.(\d+)\.mlp\.up_proj\.weight$": (r"model\.layers\.\1\.mlp\.up_proj\.kernel", Transform.LINEAR),
        r"^model\.layers\.(\d+)\.mlp\.down_proj\.weight$": (
            r"model\.layers\.\1\.mlp\.down_proj\.kernel",
            Transform.LINEAR,
        ),
        r"^model\.layers\.(\d+)\.block_sparse_moe\.gate\.weight$": (
            r"model\.layers\.\1\.mlp\.gate\.kernel",
            Transform.LINEAR,
        ),
        r"^model\.layers\.(\d+)\.mlp\.routed_experts\.gate_proj\.weight$": (
            r"model\.layers\.\1\.mlp\.routed_experts\.gate_proj_kernel",
            Transform.LINEAR_3D,
        ),
        r"^model\.layers\.(\d+)\.mlp\.routed_experts\.up_proj\.weight$": (
            r"model\.layers\.\1\.mlp\.routed_experts\.up_proj_kernel",
            Transform.LINEAR_3D,
        ),
        r"^model\.layers\.(\d+)\.mlp\.routed_experts\.down_proj\.weight$": (
            r"model\.layers\.\1\.mlp\.routed_experts\.down_proj_kernel",
            Transform.LINEAR_3D,
        ),
        r"^model\.layers\.(\d+)\.shared_expert\.gate_proj\.weight$": (
            r"model\.layers\.\1\.mlp\.shared_experts\.gate_proj\.kernel",
            Transform.LINEAR,
        ),
        r"^model\.layers\.(\d+)\.shared_expert\.up_proj\.weight$": (
            r"model\.layers\.\1\.mlp\.shared_experts\.up_proj\.kernel",
            Transform.LINEAR,
        ),
        r"^model\.layers\.(\d+)\.shared_expert\.down_proj\.weight$": (
            r"model\.layers\.\1\.mlp\.shared_experts\.down_proj\.kernel",
            Transform.LINEAR,
        ),
        r"^model\.layers\.(\d+)\.block_sparse_moe\.pre_forward_scale_2\.weight$": (
            r"model\.layers\.\1\.mlp\.pre_forward_scale_2",
            Transform.DEFAULT,
        ),
        r"^model\.layers\.(\d+)\.block_sparse_moe\.pre_feedforward_layernorm_2\.weight$": (
            r"model\.layers\.\1\.mlp\.pre_feedforward_layernorm_2\.scale",
            Transform.DEFAULT,
        ),
        r"^model\.layers\.(\d+)\.block_sparse_moe\.post_feedforward_layernorm_1\.weight$": (
            r"model\.layers\.\1\.mlp\.post_feedforward_layernorm_1\.scale",
            Transform.DEFAULT,
        ),
        r"^model\.layers\.(\d+)\.block_sparse_moe\.post_feedforward_layernorm_2\.weight$": (
            r"model\.layers\.\1\.mlp\.post_feedforward_layernorm_2\.scale",
            Transform.DEFAULT,
        ),
        r"^model\.layers\.(\d+)\.block_sparse_moe\.per_expert_scale$": (
            r"model\.layers\.\1\.mlp\.per_expert_scale",
            Transform.DEFAULT,
        ),
        r"^model\.layers\.(\d+)\.layer_scalar\.weight$": (r"model\.layers\.\1\.layer_scalar", Transform.DEFAULT),
        r"^model\.norm\.weight$": (r"model\.norm\.scale", Transform.DEFAULT),
        r"^lm_head\.weight$": (r"lm_head\.kernel", Transform.LINEAR),
        # Multimodal Text-Projectors
        r"^embed_audio\.embedding_projection\.weight$": (
            r"embed_audio\.embedding_projection\.kernel",
            Transform.LINEAR,
        ),
        r"^multi_modal_projector\.mm_input_projection_weight$": (
            r"multi_modal_projector\.mm_input_projection_weight",
            Transform.DEFAULT,
        ),
        r"^multi_modal_projector\.mm_soft_emb_norm\.weight$": (
            r"multi_modal_projector\.mm_soft_emb_norm\.scale",
            Transform.DEFAULT,
        ),
        # Audio Tower
        r"^audio_tower\.subsample_conv_projection\.layer(\d+)\.conv\.weight$": (
            r"audio_tower\.subsample_conv_projection\.layer\1\.conv\.kernel",
            Transform.CONV2D,
        ),
        r"^audio_tower\.subsample_conv_projection\.layer(\d+)\.norm\.weight$": (
            r"audio_tower\.subsample_conv_projection\.layer\1\.norm\.scale",
            Transform.DEFAULT,
        ),
        r"^audio_tower\.subsample_conv_projection\.layer(\d+)\.norm\.bias$": (
            r"audio_tower\.subsample_conv_projection\.layer\1\.norm\.bias",
            Transform.BIAS,
        ),
        r"^audio_tower\.subsample_conv_projection\.input_proj_linear\.weight$": (
            r"audio_tower\.subsample_conv_projection\.input_proj_linear\.kernel",
            Transform.LINEAR,
        ),
        r"^audio_tower\.layers\.(\d+)\.feed_forward(\d+)\.ffw_layer_(\d+)\.linear\.weight$": (
            r"audio_tower\.layers\.\1\.feed_forward\2\.ffw_layer_\3\.linear\.kernel",
            Transform.LINEAR,
        ),
        r"^audio_tower\.layers\.(\d+)\.feed_forward(\d+)\.(pre|post)_layer_norm\.weight$": (
            r"audio_tower\.layers\.\1\.feed_forward\2\.\3_layer_norm\.scale",
            Transform.DEFAULT,
        ),
        r"^audio_tower\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|post)\.linear\.weight$": (
            r"audio_tower\.layers\.\1\.self_attn\.\2\.linear\.kernel",
            Transform.LINEAR,
        ),
        r"^audio_tower\.layers\.(\d+)\.self_attn\.relative_k_proj\.weight$": (
            r"audio_tower\.layers\.\1\.self_attn\.relative_k_proj\.kernel",
            Transform.LINEAR,
        ),
        r"^audio_tower\.layers\.(\d+)\.self_attn\.per_dim_scale$": (
            r"audio_tower\.layers\.\1\.self_attn\.per_dim_scale",
            Transform.DEFAULT,
        ),
        r"^audio_tower\.layers\.(\d+)\.lconv1d\.(linear_start|linear_end)\.linear\.weight$": (
            r"audio_tower\.layers\.\1\.lconv1d\.\2\.linear\.kernel",
            Transform.LINEAR,
        ),
        r"^audio_tower\.layers\.(\d+)\.lconv1d\.depthwise_conv1d\.weight$": (
            r"audio_tower\.layers\.\1\.lconv1d\.depthwise_conv1d\.conv\.kernel",
            ((2, 1, 0), None, False),  # Conv1d PyTorch (out_c, in_c/group, K) -> Flax (K, in_c, out_c/group)
        ),
        r"^audio_tower\.layers\.(\d+)\.lconv1d\.(pre_layer_norm|conv_norm)\.weight$": (
            r"audio_tower\.layers\.\1\.lconv1d\.\2\.scale",
            Transform.DEFAULT,
        ),
        r"^audio_tower\.layers\.(\d+)\.norm_(pre_attn|post_attn|out)\.weight$": (
            r"audio_tower\.layers\.\1\.norm_\2\.scale",
            Transform.DEFAULT,
        ),
        r"^audio_tower\.output_proj\.weight$": (
            r"audio_tower\.output_proj\.kernel",
            Transform.LINEAR,
        ),
        r"^audio_tower\.output_proj\.bias$": (
            r"audio_tower\.output_proj\.bias",
            Transform.BIAS,
        ),
        # Vision Tower
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
        r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.layer_norm(\d+)\.weight$": (
            r"vision_tower\.layers\.\1\.layer_norm\2\.scale",
            Transform.DEFAULT,
        ),
        r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.mlp\.fc(\d+)\.bias$": (
            r"vision_tower\.layers\.\1\.mlp\.fc\2\.bias",
            Transform.BIAS,
        ),
        r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.mlp\.fc(\d+)\.weight$": (
            r"vision_tower\.layers\.\1\.mlp\.fc\2\.kernel",
            Transform.LINEAR,
        ),
        r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.k_proj\.bias$": (
            r"vision_tower\.layers\.\1\.self_attn\.k_proj\.bias",
            Transform.BIAS,
        ),
        r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.k_proj\.weight$": (
            r"vision_tower\.layers\.\1\.self_attn\.k_proj\.kernel",
            Transform.LINEAR,
        ),
        r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.out_proj\.bias$": (
            r"vision_tower\.layers\.\1\.self_attn\.out_proj\.bias",
            Transform.BIAS,
        ),
        r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.out_proj\.weight$": (
            r"vision_tower\.layers\.\1\.self_attn\.out_proj\.kernel",
            Transform.LINEAR,
        ),
        r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.q_proj\.bias$": (
            r"vision_tower\.layers\.\1\.self_attn\.q_proj\.bias",
            Transform.BIAS,
        ),
        r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.q_proj\.weight$": (
            r"vision_tower\.layers\.\1\.self_attn\.q_proj\.kernel",
            Transform.LINEAR,
        ),
        r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.v_proj\.bias$": (
            r"vision_tower\.layers\.\1\.self_attn\.v_proj\.bias",
            Transform.BIAS,
        ),
        r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.v_proj\.weight$": (
            r"vision_tower\.layers\.\1\.self_attn\.v_proj\.kernel",
            Transform.LINEAR,
        ),
        r"^vision_tower\.vision_model\.post_layernorm\.weight$": (
            r"vision_tower\.post_layernorm\.scale",
            Transform.DEFAULT,
        ),
    }


def create_gemma4_from_pretrained(file_dir: str, cfg: model_lib.ModelConfig):
    """
    Load safetensor weights from a file, then convert & merge into a flax.nnx model.

    Returns:
      A flax.nnx.Model instance with loaded parameters.
    """
    import gc

    files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors found in {file_dir}")

    gemma4 = nnx.eval_shape(lambda: model_lib.Gemma4ForCausalLM(cfg, rngs=nnx.Rngs(0)))
    graph_def, abs_state = nnx.split(gemma4)
    jax_state = nnx.to_pure_dict(abs_state)

    mapping = _get_key_and_transform_mapping()

    moe_pattern = re.compile(
        r"^model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
    )
    expert_tensors = {}

    for f in files:
        with safetensors.safe_open(f, framework="numpy") as sf:
            for torch_key in sf.keys():
                match = moe_pattern.match(torch_key)
                if match:
                    l_idx, e_idx, proj_type = match.groups()
                    l_idx, e_idx = int(l_idx), int(e_idx)
                    if l_idx not in expert_tensors:
                        expert_tensors[l_idx] = {}
                    if proj_type not in expert_tensors[l_idx]:
                        expert_tensors[l_idx][proj_type] = {}
                    expert_tensors[l_idx][proj_type][e_idx] = jnp.array(sf.get_tensor(torch_key))
                    continue

                tensor = jnp.array(sf.get_tensor(torch_key))
                jax_key, transform = map_to_bonsai_key(mapping, torch_key)
                if jax_key is None:
                    continue
                keys = [stoi(k) for k in jax_key.split(r"\.")]
                try:
                    assign_weights_from_eval_shape(keys, tensor, jax_state, torch_key, transform.value)
                except KeyError as e:
                    print(f"Key error: {keys} at {e}")
                except ValueError as e:
                    print(e)
                except Exception as e:
                    print(keys)
                    raise e
        gc.collect()

    for l_idx, projs in expert_tensors.items():
        for proj_type, e_dict in projs.items():
            tensors = [e_dict[i] for i in sorted(e_dict.keys())]
            stacked = jnp.stack(tensors, axis=0)
            st_key = f"model.layers.{l_idx}.mlp.routed_experts.{proj_type}.weight"
            jax_key, transform = map_to_bonsai_key(mapping, st_key)
            if jax_key is not None:
                keys = [stoi(k) for k in jax_key.split(r"\.")]
                assign_weights_from_eval_shape(keys, stacked, jax_state, st_key, transform.value)

    # Convert remaining ShapeDtypeStruct into arrays
    if isinstance(jax_state["model"]["embed_scale"], jax.ShapeDtypeStruct):
        jax_state["model"]["embed_scale"] = jnp.array(cfg.hidden_size**0.5, dtype=jnp.bfloat16).astype(jnp.float32)

    if cfg.vision_config:
        if isinstance(jax_state["vision_tower"]["embeddings"]["position_ids"], jax.ShapeDtypeStruct):
            jax_state["vision_tower"]["embeddings"]["position_ids"] = jnp.expand_dims(
                jnp.arange(gemma4.vision_tower.embeddings.num_patches), 0
            )

    return nnx.merge(graph_def, jax_state)
