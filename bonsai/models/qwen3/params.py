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

import gc
from enum import Enum

import safetensors
from etils import epath
from flax import nnx

from bonsai.models.qwen3 import modeling as model_lib
from bonsai.utils.params import stoi, map_to_bonsai_key, assign_weights_from_eval_shape


def _get_key_and_transform_mapping(cfg: model_lib.ModelConfig):
    class Transform(Enum):
        BIAS = None
        LINEAR = ((1, 0), None, False)
        EMBED = None
        ATTN_Q = ((2, 0, 1), (cfg.num_heads, cfg.head_dim, cfg.emb_dim), True)
        ATTN_KV = ((2, 0, 1), (cfg.num_kv_heads, cfg.head_dim, cfg.emb_dim), True)
        ATTN_OUT = ((1, 0), (cfg.num_heads, cfg.head_dim, cfg.emb_dim), False)
        SCALE = None

    # Mapping of torch_keys -> (nnx_keys, (permute_rule, reshape_rule)).
    return {
        r"model\.embed_tokens\.weight": ("embedder.embedding", Transform.EMBED),
        r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (r"layers.\1.attn.q_proj.w", Transform.ATTN_Q),
        r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (r"layers.\1.attn.k_proj.w", Transform.ATTN_KV),
        r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (r"layers.\1.attn.v_proj.w", Transform.ATTN_KV),
        r"model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": (r"layers.\1.attn.o_proj.w", Transform.ATTN_OUT),
        # mlp
        r"model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": (r"layers.\1.mlp.gate_proj.kernel", Transform.LINEAR),
        r"model\.layers\.([0-9]+)\.mlp\.up_proj\.weight": (r"layers.\1.mlp.up_proj.kernel", Transform.LINEAR),
        r"model\.layers\.([0-9]+)\.mlp\.down_proj\.weight": (r"layers.\1.mlp.down_proj.kernel", Transform.LINEAR),
        r"model\.norm\.weight": ("final_norm.scale", Transform.SCALE),
        # norms
        r"model\.layers\.([0-9]+)\.self_attn\.q_norm\.weight": (r"layers.\1.attn.q_norm.scale", Transform.SCALE),
        r"model\.layers\.([0-9]+)\.self_attn\.k_norm\.weight": (r"layers.\1.attn.k_norm.scale", Transform.SCALE),
        # layer norms (pre/post attention)
        r"model\.layers\.([0-9]+)\.input_layernorm\.weight": (r"layers.\1.input_layernorm.scale", Transform.SCALE),
        r"model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": (
            r"layers.\1.post_attention_layernorm.scale",
            Transform.SCALE,
        ),
        r"lm_head\.weight": ("lm_head.w", Transform.LINEAR),
    }


def create_model_from_safe_tensors(file_dir: str, cfg: model_lib.ModelConfig) -> model_lib.Qwen3:
    """Load tensors from the safetensors file and create a Qwen3 model (memory-optimized)."""
    files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors found in {file_dir}")

    qwen3 = nnx.eval_shape(lambda: model_lib.Qwen3(cfg, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(qwen3)
    state_dict = nnx.to_pure_dict(abs_state)

    key_mapping = _get_key_and_transform_mapping(cfg)
    conversion_errors = []
    for f in files:
        with safetensors.safe_open(f, framework="numpy") as sf:
            for torch_key in sf.keys():
                tensor = sf.get_tensor(torch_key)

                jax_key, transform = map_to_bonsai_key(key_mapping, torch_key)
                if jax_key is None:
                    continue
                keys = [stoi(k) for k in jax_key.split(".")]
                try:
                    assign_weights_from_eval_shape(keys, tensor, state_dict, torch_key, transform.value)
                except Exception as e:
                    full_jax_key = ".".join([str(k) for k in keys])
                    conversion_errors.append(
                        f"Failed to assign '{torch_key}' to '{full_jax_key}': {type(e).__name__}: {e}"
                    )
        gc.collect()

    if conversion_errors:
        full_error_log = "\n".join(conversion_errors)
        raise RuntimeError(f"Encountered {len(conversion_errors)} weight conversion errors. Log:\n{full_error_log}")

    if cfg.tie_word_embeddings:
        state_dict["lm_head"]["w"] = state_dict["embedder"]["embedding"].T
    gc.collect()
    return nnx.merge(graph_def, state_dict)
