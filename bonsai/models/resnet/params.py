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
from enum import Enum

import jax
import jax.numpy as jnp
import safetensors.flax as safetensors
from etils import epath
from flax import nnx

from bonsai.models.resnet import modeling as model_lib


def _get_key_and_transform_mapping(cfg: model_lib.ModelConfig):
    class Transform(Enum):
        """Transformations for model parameters"""

        BIAS = None
        LINEAR = ((1, 0), None)
        CONV2D = ((2, 3, 1, 0), None)
        DEFAULT = None

    # Mapping st_keys -> (nnx_keys, (permute_rule, reshape_rule)).
    return {
        r"^resnet\.embedder\.embedder\.convolution\.weight$": ("stem.conv.kernel", Transform.CONV2D),
        r"^resnet\.embedder\.embedder\.normalization\.weight$": ("stem.bn.scale", Transform.DEFAULT),
        r"^resnet\.embedder\.embedder\.normalization\.bias$": ("stem.bn.bias", Transform.BIAS),
        r"^resnet\.embedder\.embedder\.normalization\.running_mean$": ("stem.bn.mean", Transform.DEFAULT),
        r"^resnet\.embedder\.embedder\.normalization\.running_var$": ("stem.bn.var", Transform.DEFAULT),
        r"^resnet\.encoder\.stages\.([0-3])\.layers\.([0-9]+)\.layer\.([0-2])\.convolution\.weight$": (
            r"layer\1.blocks.\2.conv\3.kernel",
            Transform.CONV2D,
        ),
        r"^resnet\.encoder\.stages\.([0-3])\.layers\.([0-9]+)\.layer\.([0-2])\.normalization\.weight$": (
            r"layer\1.blocks.\2.bn\3.scale",
            Transform.DEFAULT,
        ),
        r"^resnet\.encoder\.stages\.([0-3])\.layers\.([0-9]+)\.layer\.([0-2])\.normalization\.bias$": (
            r"layer\1.blocks.\2.bn\3.bias",
            Transform.BIAS,
        ),
        r"^resnet\.encoder\.stages\.([0-3])\.layers\.([0-9]+)\.layer\.([0-2])\.normalization\.running_mean$": (
            r"layer\1.blocks.\2.bn\3.mean",
            Transform.DEFAULT,
        ),
        r"^resnet\.encoder\.stages\.([0-3])\.layers\.([0-9]+)\.layer\.([0-2])\.normalization\.running_var$": (
            r"layer\1.blocks.\2.bn\3.var",
            Transform.DEFAULT,
        ),
        r"^resnet\.encoder\.stages\.([0-3])\.layers\.([0-9]+)\.shortcut\.convolution\.weight$": (
            r"layer\1.blocks.\2.downsample.conv.kernel",
            Transform.CONV2D,
        ),
        r"^resnet\.encoder\.stages\.([0-3])\.layers\.([0-9]+)\.shortcut\.normalization\.weight$": (
            r"layer\1.blocks.\2.downsample.bn.scale",
            Transform.DEFAULT,
        ),
        r"^resnet\.encoder\.stages\.([0-3])\.layers\.([0-9]+)\.shortcut\.normalization\.bias$": (
            r"layer\1.blocks.\2.downsample.bn.bias",
            Transform.BIAS,
        ),
        r"^resnet\.encoder\.stages\.([0-3])\.layers\.([0-9]+)\.shortcut\.normalization\.running_mean$": (
            r"layer\1.blocks.\2.downsample.bn.mean",
            Transform.DEFAULT,
        ),
        r"^resnet\.encoder\.stages\.([0-3])\.layers\.([0-9]+)\.shortcut\.normalization\.running_var$": (
            r"layer\1.blocks.\2.downsample.bn.var",
            Transform.DEFAULT,
        ),
        r"^classifier\.1\.weight$": ("fc.kernel", Transform.LINEAR),
        r"^classifier\.1\.bias$": ("fc.bias", Transform.BIAS),
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
        state_dict[key] = jnp.array(tensor)
    else:
        _assign_weights(rest, tensor, state_dict[key], st_key, transform)


def _stoi(s):
    try:
        return int(s)
    except ValueError:
        return s


def create_resnet_from_pretrained(
    file_dir: str,
    config: model_lib.ModelConfig,
    *,
    mesh: jax.sharding.Mesh | None = None,
):
    """
    Load safetensor weights from a file, then convert & merge into a flax.nnx ResNet model.

    Returns:
      A flax.nnx.Model instance with loaded parameters.
    """
    files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors found in {file_dir}")

    state_dict = {}
    for f in files:
        state_dict |= safetensors.load_file(f)

    model = nnx.eval_shape(lambda: model_lib.ResNet(config, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(model)
    jax_state = nnx.to_pure_dict(abs_state)

    mapping = _get_key_and_transform_mapping(config)
    conversion_errors = []
    for st_key, tensor in state_dict.items():
        jax_key, transform = _st_key_to_jax_key(mapping, st_key)
        if jax_key is None:
            continue
        keys = [_stoi(k) for k in jax_key.split(".")]

        try:
            _assign_weights(keys, tensor, jax_state, st_key, transform.value)
        except Exception as e:
            full_jax_key = ".".join([str(k) for k in keys])
            conversion_errors.append(f"Failed to assign '{st_key}' to '{full_jax_key}': {type(e).__name__}: {e}")

    if conversion_errors:
        full_error_log = "\n".join(conversion_errors)
        raise RuntimeError(f"Encountered {len(conversion_errors)} weight conversion errors. Log:\n{full_error_log}")

    if mesh is not None:
        sharding = nnx.to_pure_dict(nnx.get_named_sharding(abs_state, mesh))
        jax_state = jax.device_put(jax_state, sharding)
    else:
        jax_state = jax.device_put(jax_state, jax.devices()[0])

    return nnx.merge(graph_def, jax_state)
