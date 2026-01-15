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

import logging
import re
from enum import Enum

import jax
import jax.numpy as jnp
import safetensors.flax as safetensors
from etils import epath
from flax import nnx
import gc

from bonsai.models.llada import modeling as model_lib


def _get_key_and_transform_mapping():
    class Transform(Enum):
        """Specifies default transformation types for model parameter names."""

        EMBED = None
        NORM = None
        LINEAR = ((1, 0), None)

    # Mapping st_keys -> (nnx_keys, (permute_rule, reshape_rule)).
    return {
        r"^model\.transformer\.blocks\.(\d+)\.attn_norm\.weight$": (r"blocks\.\1\.attn_norm\.weight", Transform.NORM),
        r"^model\.transformer\.blocks\.(\d+)\.attn_out\.weight$": (r"blocks\.\1\.attn_out\.kernel", Transform.LINEAR),
        r"^model\.transformer\.blocks\.(\d+)\.ff_norm\.weight$": (r"blocks\.\1\.ff_norm\.weight", Transform.NORM),
        r"^model\.transformer\.blocks\.(\d+)\.ff_out\.weight$": (r"blocks\.\1\.ff_out\.kernel", Transform.LINEAR),
        r"^model\.transformer\.blocks\.(\d+)\.ff_proj\.weight$": (r"blocks\.\1\.ff_proj\.kernel", Transform.LINEAR),
        r"^model\.transformer\.blocks\.(\d+)\.k_proj\.weight$": (r"blocks\.\1\.k_proj\.kernel", Transform.LINEAR),
        r"^model\.transformer\.blocks\.(\d+)\.q_proj\.weight$": (r"blocks\.\1\.q_proj\.kernel", Transform.LINEAR),
        r"^model\.transformer\.blocks\.(\d+)\.up_proj\.weight$": (r"blocks\.\1\.up_proj\.kernel", Transform.LINEAR),
        r"^model\.transformer\.blocks\.(\d+)\.v_proj\.weight$": (r"blocks\.\1\.v_proj\.kernel", Transform.LINEAR),
        r"^model\.transformer\.ff_out\.weight$": (r"ff_out\.kernel", Transform.LINEAR),
        r"^model\.transformer\.ln_f\.weight$": (r"ln_f\.weight", Transform.NORM),
        r"^model\.transformer\.wte\.weight$": (r"wte\.embedding", Transform.EMBED),
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


def create_llada_from_pretrained(file_dir: str, cfg: model_lib.ModelConfig, *, mesh: jax.sharding.Mesh | None = None):
    """
    Load safetensor weights from a file, then convert & merge into a flax.nnx ViT model.

    Returns:
      A flax.nnx.Model instance with loaded parameters.
    """
    files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors found in {file_dir}")

    llada = model_lib.LLaDAModel(cfg, rngs=nnx.Rngs(0))
    graph_def, abs_state = nnx.split(llada)
    jax_state = abs_state.to_pure_dict()
    sharding = nnx.get_named_sharding(abs_state, mesh).to_pure_dict() if mesh is not None else None

    mapping = _get_key_and_transform_mapping()
    conversion_errors = []
    for f in files:
        with safetensors.safe_open(f, framework="numpy") as sf:
            for torch_key in sf.keys():
                tensor = jnp.array(sf.get_tensor(torch_key))
                jax_key, transform = _st_key_to_jax_key(mapping, torch_key)
                if jax_key is None:
                    continue
                keys = [_stoi(k) for k in jax_key.split(r"\.")]
                try:
                    _assign_weights(keys, tensor, jax_state, torch_key, transform.value, sharding)
                except Exception as e:
                    full_jax_key = ".".join([str(k) for k in keys])
                    conversion_errors.append(
                        f"Failed to assign '{torch_key}' to '{full_jax_key}': {type(e).__name__}: {e}"
                    )
        gc.collect()

    return nnx.merge(graph_def, jax_state)
