# Copyright 2026 The JAX Authors.
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
import re

import jax
import jax.numpy as jnp
import safetensors
from etils import epath
from flax import nnx
import logging
from typing import Any, TypeAlias

ModelType: TypeAlias = Any
ModelConfigType: TypeAlias = Any
TransformValueType: TypeAlias = None | tuple[tuple[int, ...], tuple[int, ...] | None, bool]
TransformType: TypeAlias = Any  # Enum with values of type TransformValueType
KeyMapType: TypeAlias = tuple[str, TransformType]


def map_to_bonsai_key(mapping: dict[str, KeyMapType], source_key: str) -> KeyMapType | tuple[None, None]:
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


def stoi(s: str) -> int | str:
    """Convert a string to an int if possible, otherwise return the string."""
    try:
        return int(s)
    except ValueError:
        return s


def assign_weights(
    keys: list[str],
    tensor: jnp.ndarray,
    state_dict: dict,
    st_key: str,
    transform: TransformType,
    sharding_dict: dict | None,
):
    """Recursively descend into state_dict and assign the (possibly permuted/reshaped) tensor.
    Assumes that the state_dict values are of type Array.
    """
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
        # Only apply sharding if sharding_dict is provided
        if sharding_dict is not None:
            state_dict[key] = jax.device_put(tensor, sharding_dict[key])
        else:
            state_dict[key] = jax.device_put(tensor)
    else:
        next_sharding = sharding_dict[key] if sharding_dict is not None else None
        assign_weights(rest, tensor, state_dict[key], st_key, transform, next_sharding)


def assign_weights_from_eval_shape(
    keys: list[str], tensor: jnp.ndarray, state_dict: dict, st_key: str, transform: TransformType
):
    """Recursively descend into state_dict and assign the (possibly permuted/reshaped) tensor.
    Assumes that the state_dict values are of type ShapeDtypeStruct.
    """
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

        tensor = tensor.astype(state_dict[key].dtype)
        if hasattr(state_dict[key], "sharding") and state_dict[key].sharding is not None:
            tensor = jax.device_put(tensor, state_dict[key].sharding.spec)
        state_dict[key] = tensor
    else:
        assign_weights_from_eval_shape(rest, tensor, state_dict[key], st_key, transform)


def create_model_from_safe_tensors(
    file_dir: str, model_cls: ModelType, cfg: ModelConfigType, key_mapping: dict, mesh: jax.sharding.Mesh | None = None
) -> ModelType:
    """
    Load tensors from the safetensors file and create a model (memory-optimized).
    We currently define this separately for each model, but this may be a useful tool later
    NOTE: This is not yet implemented.
    """
    raise NotImplementedError("This is in progress.")

    files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors found in {file_dir}")

    model = nnx.eval_shape(lambda: model_cls(cfg, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(model)
    state_dict = nnx.to_pure_dict(abs_state)

    conversion_errors = []
    for f in files:
        with safetensors.safe_open(f, framework="numpy") as sf:
            for torch_key in sf.keys():
                tensor = jnp.array(sf.get_tensor(torch_key))

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

    return nnx.merge(graph_def, state_dict)
