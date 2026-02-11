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

from enum import Enum

import safetensors.flax as safetensors
from etils import epath
from flax import nnx

from bonsai.models.vit import modeling as model_lib
from bonsai.utils.params import stoi, map_to_bonsai_key, assign_weights_from_eval_shape


def _get_key_and_transform_mapping(cfg: model_lib.ModelConfig):
    head_dim = cfg.hidden_dim // cfg.num_heads

    class Transform(Enum):
        DEFAULT = None
        BIAS = None
        LINEAR = ((1, 0), None, False)
        SCALE = None
        ATTN_KQV_BIAS = (None, (cfg.num_heads, head_dim), False)
        ATTN_KQV_KERNEL = ((1, 0), (cfg.hidden_dim, cfg.num_heads, head_dim), False)
        ATTN_OUT = ((1, 0), (cfg.num_heads, head_dim, cfg.hidden_dim), False)
        EMBED = None
        CONV2D = ((2, 3, 1, 0), None, False)

    # Mapping st_keys -> (nnx_keys, (permute_rule, reshape_rule, reshape_first)).
    return {
        r"^classifier.bias$": (r"classifier.bias", Transform.BIAS),
        r"^classifier.weight$": (r"classifier.kernel", Transform.LINEAR),
        r"^vit.embeddings.cls_token$": (r"pos_embeddings.cls_token", Transform.DEFAULT),
        r"^vit.embeddings.patch_embeddings.projection.bias$": (r"pos_embeddings.projection.bias", Transform.BIAS),
        r"^vit.embeddings.patch_embeddings.projection.weight$": (r"pos_embeddings.projection.kernel", Transform.CONV2D),
        r"^vit.embeddings.position_embeddings$": (r"pos_embeddings.pos_embeddings", Transform.EMBED),
        r"^vit.encoder.layer.([0-9]+).attention.attention.key.bias$": (
            r"layers.\1.attention.key.bias",
            Transform.ATTN_KQV_BIAS,
        ),
        r"^vit.encoder.layer.([0-9]+).attention.attention.key.weight$": (
            r"layers.\1.attention.key.kernel",
            Transform.ATTN_KQV_KERNEL,
        ),
        r"^vit.encoder.layer.([0-9]+).attention.attention.query.bias$": (
            r"layers.\1.attention.query.bias",
            Transform.ATTN_KQV_BIAS,
        ),
        r"^vit.encoder.layer.([0-9]+).attention.attention.query.weight$": (
            r"layers.\1.attention.query.kernel",
            Transform.ATTN_KQV_KERNEL,
        ),
        r"^vit.encoder.layer.([0-9]+).attention.attention.value.bias$": (
            r"layers.\1.attention.value.bias",
            Transform.ATTN_KQV_BIAS,
        ),
        r"^vit.encoder.layer.([0-9]+).attention.attention.value.weight$": (
            r"layers.\1.attention.value.kernel",
            Transform.ATTN_KQV_KERNEL,
        ),
        r"^vit.encoder.layer.([0-9]+).attention.output.dense.bias$": (r"layers.\1.attention.out.bias", Transform.BIAS),
        r"^vit.encoder.layer.([0-9]+).attention.output.dense.weight$": (
            r"layers.\1.attention.out.kernel",
            Transform.ATTN_OUT,
        ),
        r"^vit.encoder.layer.([0-9]+).intermediate.dense.bias$": (r"layers.\1.linear1.bias", Transform.BIAS),
        r"^vit.encoder.layer.([0-9]+).intermediate.dense.weight$": (r"layers.\1.linear1.kernel", Transform.LINEAR),
        r"^vit.encoder.layer.([0-9]+).layernorm_after.bias$": (r"layers.\1.layernorm_after.bias", Transform.BIAS),
        r"^vit.encoder.layer.([0-9]+).layernorm_after.weight$": (r"layers.\1.layernorm_after.scale", Transform.SCALE),
        r"^vit.encoder.layer.([0-9]+).layernorm_before.bias$": (r"layers.\1.layernorm_before.bias", Transform.BIAS),
        r"^vit.encoder.layer.([0-9]+).layernorm_before.weight$": (r"layers.\1.layernorm_before.scale", Transform.SCALE),
        r"^vit.encoder.layer.([0-9]+).output.dense.bias$": (r"layers.\1.linear2.bias", Transform.BIAS),
        r"^vit.encoder.layer.([0-9]+).output.dense.weight$": (r"layers.\1.linear2.kernel", Transform.LINEAR),
        r"^vit.layernorm.bias$": (r"ln.bias", Transform.BIAS),
        r"^vit.layernorm.weight$": (r"ln.scale", Transform.SCALE),
    }


def create_vit_from_pretrained(file_dir: str, config: model_lib.ModelConfig):
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

    vit = nnx.eval_shape(lambda: model_lib.ViTClassificationModel(config, rngs=nnx.Rngs(0)))
    graph_def, abs_state = nnx.split(vit)
    jax_state = nnx.to_pure_dict(abs_state)

    mapping = _get_key_and_transform_mapping(config)
    conversion_errors = []
    for st_key, tensor in tensor_dict.items():
        jax_key, transform = map_to_bonsai_key(mapping, st_key)
        if jax_key is None:
            continue
        keys = [stoi(k) for k in jax_key.split(".")]
        try:
            assign_weights_from_eval_shape(keys, tensor, jax_state, st_key, transform.value)
        except Exception as e:
            full_jax_key = ".".join([str(k) for k in keys])
            conversion_errors.append(f"Failed to assign '{st_key}' to '{full_jax_key}': {type(e).__name__}: {e}")

    if conversion_errors:
        full_error_log = "\n".join(conversion_errors)
        raise RuntimeError(f"Encountered {len(conversion_errors)} weight conversion errors. Log:\n{full_error_log}")

    return nnx.merge(graph_def, jax_state)
