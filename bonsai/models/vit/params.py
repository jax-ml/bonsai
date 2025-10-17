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

import logging
import re

import jax
import safetensors.flax as safetensors
from etils import epath
from flax import nnx

from bonsai.models.vit import modeling as model_lib


def _get_key_and_transform_mapping():
    # Mapping st_keys -> (nnx_keys, (permute_rule, reshape_rule)).
    embed_dim, num_heads, head_dim = 768, 12, 64
    return {
        r"^classifier.bias$": (r"classifier.bias", None),
        r"^classifier.weight$": (r"classifier.kernel", ((1, 0), None)),
        r"^vit.embeddings.cls_token$": (r"pos_embeddings.cls_token", None),
        r"^vit.embeddings.patch_embeddings.projection.bias$": (r"pos_embeddings.projection.bias", None),
        r"^vit.embeddings.patch_embeddings.projection.weight$": (
            r"pos_embeddings.projection.kernel",
            ((2, 3, 1, 0), None),
        ),
        r"^vit.embeddings.position_embeddings$": (r"pos_embeddings.pos_embeddings", None),
        r"^vit.encoder.layer.([0-9]+).attention.attention.key.bias$": (
            r"layers.layers.\1.attention.key.bias",
            (None, (num_heads, head_dim)),
        ),
        r"^vit.encoder.layer.([0-9]+).attention.attention.key.weight$": (
            r"layers.layers.\1.attention.key.kernel",
            ((1, 0), (embed_dim, num_heads, head_dim)),
        ),
        r"^vit.encoder.layer.([0-9]+).attention.attention.query.bias$": (
            r"layers.layers.\1.attention.query.bias",
            (None, (num_heads, head_dim)),
        ),
        r"^vit.encoder.layer.([0-9]+).attention.attention.query.weight$": (
            r"layers.layers.\1.attention.query.kernel",
            ((1, 0), (embed_dim, num_heads, head_dim)),
        ),
        r"^vit.encoder.layer.([0-9]+).attention.attention.value.bias$": (
            r"layers.layers.\1.attention.value.bias",
            (None, (num_heads, head_dim)),
        ),
        r"^vit.encoder.layer.([0-9]+).attention.attention.value.weight$": (
            r"layers.layers.\1.attention.value.kernel",
            ((1, 0), (embed_dim, num_heads, head_dim)),
        ),
        r"^vit.encoder.layer.([0-9]+).attention.output.dense.bias$": (r"layers.layers.\1.attention.out.bias", None),
        r"^vit.encoder.layer.([0-9]+).attention.output.dense.weight$": (
            r"layers.layers.\1.attention.out.kernel",
            ((1, 0), (num_heads, head_dim, embed_dim)),
        ),
        r"^vit.encoder.layer.([0-9]+).intermediate.dense.bias$": (r"layers.layers.\1.linear1.bias", None),
        r"^vit.encoder.layer.([0-9]+).intermediate.dense.weight$": (r"layers.layers.\1.linear1.kernel", ((1, 0), None)),
        r"^vit.encoder.layer.([0-9]+).layernorm_after.bias$": (r"layers.layers.\1.layernorm_after.bias", None),
        r"^vit.encoder.layer.([0-9]+).layernorm_after.weight$": (r"layers.layers.\1.layernorm_after.scale", None),
        r"^vit.encoder.layer.([0-9]+).layernorm_before.bias$": (r"layers.layers.\1.layernorm_before.bias", None),
        r"^vit.encoder.layer.([0-9]+).layernorm_before.weight$": (r"layers.layers.\1.layernorm_before.scale", None),
        r"^vit.encoder.layer.([0-9]+).output.dense.bias$": (r"layers.layers.\1.linear2.bias", None),
        r"^vit.encoder.layer.([0-9]+).output.dense.weight$": (r"layers.layers.\1.linear2.kernel", ((1, 0), None)),
        r"^vit.layernorm.bias$": (r"ln.bias", None),
        r"^vit.layernorm.weight$": (r"ln.scale", None),
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


def _assign_weights(keys, tensor, state_dict, st_key, transform):
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
        state_dict[key] = tensor
    else:
        _assign_weights(rest, tensor, state_dict[key], st_key, transform)


def _stoi(s):
    try:
        return int(s)
    except ValueError:
        return s


def create_vit_from_pretrained(
    file_dir: str,
    num_classes: int = 1000,
    *,
    mesh: jax.sharding.Mesh | None = None,
):
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

    # vit = nnx.eval_shape(lambda: model_lib.ViT(num_classes=num_classes, rngs=nnx.Rngs(0)))
    vit = model_lib.ViT(num_classes=num_classes, rngs=nnx.Rngs(0))
    graph_def, abs_state = nnx.split(vit)
    jax_state = abs_state.to_pure_dict()

    mapping = _get_key_and_transform_mapping()
    for st_key, tensor in tensor_dict.items():
        jax_key, transform = _st_key_to_jax_key(mapping, st_key)
        if jax_key is None:
            continue
        keys = [_stoi(k) for k in jax_key.split(".")]
        _assign_weights(keys, tensor, jax_state, st_key, transform)

    return nnx.merge(graph_def, jax_state)
