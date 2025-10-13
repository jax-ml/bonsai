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

import jax
import safetensors.flax as safetensors
from etils import epath
from flax import nnx

from bonsai.models.qwen3 import modeling as model_lib


def _get_key_and_transform_mapping(cfg: model_lib.ModelCfg):
    # Mapping of torch_keys -> (nnx_keys, (permute_rule, reshape_rule)).
    return {
        r"model\.embed_tokens\.weight": ("embedder.input_emb", None),
        r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (
            r"layers.\1.attn.q_proj.w",
            ((1, 0), (cfg.emb_dim, cfg.num_heads, cfg.head_dim)),
        ),
        r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (
            r"layers.\1.attn.k_proj.w",
            ((1, 0), (cfg.emb_dim, cfg.num_kv_heads, cfg.head_dim)),
        ),
        r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (
            r"layers.\1.attn.v_proj.w",
            ((1, 0), (cfg.emb_dim, cfg.num_kv_heads, cfg.head_dim)),
        ),
        r"model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": (
            r"layers.\1.attn.o_proj.w",
            ((1, 0), (cfg.num_heads, cfg.head_dim, cfg.emb_dim)),
        ),
        # mlp
        r"model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": (
            r"layers.\1.mlp.gate_proj.kernel",
            ((1, 0), None),
        ),
        r"model\.layers\.([0-9]+)\.mlp\.up_proj\.weight": (
            r"layers.\1.mlp.up_proj.kernel",
            ((1, 0), None),
        ),
        r"model\.layers\.([0-9]+)\.mlp\.down_proj\.weight": (
            r"layers.\1.mlp.down_proj.kernel",
            ((1, 0), None),
        ),
        r"model\.norm\.weight": ("final_norm.w", None),
        # norms
        r"model\.layers\.([0-9]+)\.self_attn\.q_norm\.weight": (
            r"layers.\1.attn.q_norm.w",
            None,
        ),
        r"model\.layers\.([0-9]+)\.self_attn\.k_norm\.weight": (
            r"layers.\1.attn.k_norm.w",
            None,
        ),
        # layer norms (pre/post attention)
        r"model\.layers\.([0-9]+)\.input_layernorm\.weight": (
            r"layers.\1.input_layernorm.w",
            None,
        ),
        r"model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": (
            r"layers.\1.post_attention_layernorm.w",
            None,
        ),
        r"lm_head\.weight": ("lm_head.w", ((1, 0), None)),
    }


def _torch_key_to_jax_key(mapping, source_key):
    subs = [
        (re.sub(pat, repl, source_key), reshape)
        for pat, (repl, reshape) in mapping.items()
        if re.match(pat, source_key)
    ]
    if len(subs) != 1:
        raise ValueError(f"Only one key should be found: {subs[0]}")
    return subs[0]


def _assign_weights(keys, tensor, state_dict, torch_key, transform):
    """Convert weights and assign to nnx state_dict."""
    key = keys[0]
    if len(keys) == 1:
        try:
            if transform is not None:
                permute, reshape = transform
                tensor = tensor.transpose(permute) if permute else tensor
                tensor = tensor.reshape(reshape) if reshape else tensor
        except Exception as e:
            raise RuntimeError(f"Failed to transform tensor {torch_key} with shape {tensor.shape}: {e}") from e

        if tensor.shape != state_dict[key].shape:
            raise ValueError(f"shape must match for {torch_key}, got {tensor.shape} vs {state_dict[key].shape}")
        state_dict[key] = tensor
        return state_dict
    else:
        if key not in state_dict:
            raise ValueError(f"Unfound key {key} in {state_dict}")
        _assign_weights(keys[1:], tensor, state_dict[key], torch_key, transform)
        return state_dict


def _stoi(s):
    try:
        return int(s)
    except ValueError:
        return s


def create_model_from_safe_tensors(
    file_dir: str, cfg: model_lib.ModelCfg, mesh: jax.sharding.Mesh
) -> model_lib.Qwen3:
    """Load tensors from the safetensors file and create a Qwen3 model."""
    files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors found in {file_dir}")

    tensor_dict = {}
    for f in files:
        tensor_dict |= safetensors.load_file(f)
    qwen3 = nnx.eval_shape(lambda: model_lib.Qwen3(cfg, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(qwen3)
    state_dict = abs_state.to_pure_dict()

    for k, v in tensor_dict.items():
        jax_key, transform = _torch_key_to_jax_key(_get_key_and_transform_mapping(cfg), k)
        jax_keys = [_stoi(s) for s in jax_key.split(".")]
        _assign_weights(jax_keys, v, state_dict, k, transform)
    if cfg.tie_word_embeddings:
        state_dict["lm_head"]["w"] = state_dict["embedder"]["input_emb"].T
    sharding = nnx.get_named_sharding(abs_state, mesh).to_pure_dict()
    state_dict = jax.device_put(state_dict, sharding)
    return nnx.merge(graph_def, state_dict)
