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

"""Utils for loading and converting LLaDA safetensors weights."""

import logging
import re
from enum import Enum

import jax
import jax.numpy as np
import safetensors.flax as safetensors
from etils import epath
from flax import nnx

from bonsai.models.llada_8b import modeling as model_lib


def _get_key_and_transform_mapping():
    # Maps safetensor keys → (JAX nnx key template, no transform needed)
    class Transform(Enum):
        """Transformations for model parameters"""

        BIAS = None
        LINEAR = ((1, 0), None)
        EMBED = None
        LN_SCALE = None

    mapping = {
        # Embeddings
        r"^wte\.weight$": ("wte.embedding", Transform.EMBED),
        r"^wpe\.weight$": ("wpe.embedding", Transform.EMBED),
        # Final LayerNorm
        r"^ln_f\.weight$": ("ln_f.scale", Transform.LN_SCALE),
        r"^ln_f\.bias$": ("ln_f.bias", Transform.BIAS),
        # Transformer Blocks
        # Block-level norms
        r"^blocks\.([0-9]+)\.attn_norm\.weight$": (r"blocks.\1.attn_norm.scale", Transform.LN_SCALE),
        r"^blocks\.([0-9]+)\.attn_norm\.bias$": (r"blocks.\1.attn_norm.bias", Transform.BIAS),
        r"^blocks\.([0-9]+)\.ff_norm\.weight$": (r"blocks.\1.ff_norm.scale", Transform.LN_SCALE),
        r"^blocks\.([0-9]+)\.ff_norm\.bias$": (r"blocks.\1.ff_norm.bias", Transform.BIAS),
        # Optional per-head Q/K norms (if attention_layer_norm=True)
        r"^blocks\.([0-9]+)\.q_norm\.weight$": (r"blocks.\1.q_norm.scale", Transform.LN_SCALE),
        r"^blocks\.([0-9]+)\.q_norm\.bias$": (r"blocks.\1.q_norm.bias", Transform.BIAS),
        r"^blocks\.([0-9]+)\.k_norm\.weight$": (r"blocks.\1.k_norm.scale", Transform.LN_SCALE),
        r"^blocks\.([0-9]+)\.k_norm\.bias$": (r"blocks.\1.k_norm.bias", Transform.BIAS),
        # Attention projections
        # SEQUENTIAL variant: fused Q‖K‖V
        r"^blocks\.([0-9]+)\.att_proj\.weight$": (r"blocks.\1.att_proj.kernel", Transform.LINEAR),
        r"^blocks\.([0-9]+)\.att_proj\.bias$": (r"blocks.\1.att_proj.bias", Transform.BIAS),
        # LLAMA variant: split Q / K / V
        r"^blocks\.([0-9]+)\.q_proj\.weight$": (r"blocks.\1.q_proj.kernel", Transform.LINEAR),
        r"^blocks\.([0-9]+)\.q_proj\.bias$": (r"blocks.\1.q_proj.bias", Transform.BIAS),
        r"^blocks\.([0-9]+)\.k_proj\.weight$": (r"blocks.\1.k_proj.kernel", Transform.LINEAR),
        r"^blocks\.([0-9]+)\.k_proj\.bias$": (r"blocks.\1.k_proj.bias", Transform.BIAS),
        r"^blocks\.([0-9]+)\.v_proj\.weight$": (r"blocks.\1.v_proj.kernel", Transform.LINEAR),
        r"^blocks\.([0-9]+)\.v_proj\.bias$": (r"blocks.\1.v_proj.bias", Transform.BIAS),
        # Attention output projection (common)
        r"^blocks\.([0-9]+)\.attn_out\.weight$": (r"blocks.\1.attn_out.kernel", Transform.LINEAR),
        r"^blocks\.([0-9]+)\.attn_out\.bias$": (r"blocks.\1.attn_out.bias", Transform.BIAS),
        # MLP / Feed-Forward
        # SEQUENTIAL (plain FFN or SwiGLU single-proj input)
        r"^blocks\.([0-9]+)\.ff_proj\.weight$": (r"blocks.\1.ff_proj.kernel", Transform.LINEAR),
        r"^blocks\.([0-9]+)\.ff_proj\.bias$": (r"blocks.\1.ff_proj.bias", Transform.BIAS),
        r"^blocks\.([0-9]+)\.ff_out\.weight$": (r"blocks.\1.ff_out.kernel", Transform.LINEAR),
        r"^blocks\.([0-9]+)\.ff_out\.bias$": (r"blocks.\1.ff_out.bias", Transform.BIAS),
        # LLAMA (gate/up style)
        r"^blocks\.([0-9]+)\.up_proj\.weight$": (r"blocks.\1.up_proj.kernel", Transform.LINEAR),
        r"^blocks\.([0-9]+)\.up_proj\.bias$": (r"blocks.\1.up_proj.bias", Transform.BIAS),  # include if present
        # Untied LM head (only if weight_tying == False)
        r"^ff_out\.weight$": ("ff_out.kernel", Transform.LINEAR),
        r"^ff_out\.bias$": ("ff_out.bias", Transform.BIAS),
    }
    return mapping


def find_non_array_keys(tree):
    """
    Walk `tree` (nested dicts/lists/tuples) and return a list of
    “full key paths” whose leaves are not numpy or JAX arrays.
    """
    bad = []

    def _recurse(subtree, path):
        if isinstance(subtree, dict):
            for k, v in subtree.items():
                _recurse(v, f"{path}.{k}" if path else k)
        elif isinstance(subtree, (list, tuple)):
            for i, v in enumerate(subtree):
                _recurse(v, f"{path}[{i}]")
        else:
            # treat JAX Arrays as “good”
            if not isinstance(subtree, jax.Array):
                bad.append(path)

    _recurse(tree, "")
    return bad


def _st_key_to_jax_key(mapping, source_key):
    """Map a safetensors key to exactly one model key & transform, else warn/error."""
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


def assign_nan_at_path(tree, dotted_path):
    """Assign NaN array at the specified dotted path in the tree."""

    def parse_key(k):
        return int(k) if k.isdigit() else k

    parts = [parse_key(p) for p in dotted_path.split(".")]
    subtree = tree
    for p in parts[:-1]:
        subtree = subtree[p]
    leaf_key = parts[-1]

    # Get shape of current placeholder (if present), else default
    value = subtree.get(leaf_key, None)
    if hasattr(value, "shape"):
        shape = value.shape
    else:
        shape = (1,)  # fallback

    subtree[leaf_key] = np.full(shape, np.nan, dtype=np.bfloat16)


def create_llada_from_pretrained(
    file_dir: str,
    config: model_lib.ModelConfig,
    *,
    mesh: jax.sharding.Mesh | None = None,
):
    """
    Load safetensors weights and initialize remaining missing weights with NaNs.
    """
    # 1. Load safetensor weights
    files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors found in {file_dir}")

    state_dict = {}
    for f in files:
        state_dict |= safetensors.load_file(f)

    state_dict = {k.removeprefix("model.transformer."): v for k, v in state_dict.items()}

    # 2. Create uninitialized model
    model = nnx.eval_shape(lambda: model_lib.LLaDAModel(cfg=config, rngs=nnx.Rngs(params=0, dropout=0)))
    graph_def, abs_state = nnx.split(model)
    jax_state = nnx.to_pure_dict(abs_state)

    # 3. Assign known weights
    mapping = _get_key_and_transform_mapping()
    for st_key, tensor in state_dict.items():
        jax_key, transform = _st_key_to_jax_key(mapping, st_key)
        if jax_key is None:
            continue
        keys = [_stoi(k) for k in jax_key.split(".")]
        _assign_weights(keys, tensor, jax_state, st_key, transform)

    # 4. Fill in missing keys with NaNs
    missing_keys = find_non_array_keys(jax_state)
    for path in missing_keys:
        # logging.warning(f"Missing param at: {path} - initializing with NaNs")
        assign_nan_at_path(jax_state, path)

    # 5. Device placement
    if mesh is not None:
        sharding = nnx.to_pure_dict(nnx.get_named_sharding(abs_state, mesh))
        jax_state = jax.device_put(jax_state, sharding)
    else:
        jax_state = jax.device_put(jax_state, jax.devices()[0])

    # 6. Merge & return
    return nnx.merge(graph_def, jax_state)
