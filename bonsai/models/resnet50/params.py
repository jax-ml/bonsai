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
from typing import Callable

import jax
import safetensors.flax as safetensors
from etils import epath
from flax import nnx

from bonsai.models.resnet50 import modeling as model_lib


def _get_key_and_transform_mapping():
    # Mapping st_keys -> (nnx_keys, (permute_rule, reshape_rule)).
    return {
        # stem
        r"^resnet\.embedder\.embedder\.convolution\.weight$": (
            "stem.conv.kernel",
            ((2, 3, 1, 0), None),
        ),
        r"^resnet\.embedder\.embedder\.normalization\.weight$": ("stem.bn.scale", None),
        r"^resnet\.embedder\.embedder\.normalization\.bias$": ("stem.bn.bias", None),
        r"^resnet\.embedder\.embedder\.normalization\.running_mean$": (
            "stem.bn.mean",
            None,
        ),
        r"^resnet\.embedder\.embedder\.normalization\.running_var$": (
            "stem.bn.var",
            None,
        ),
        # any of conv1/conv2/conv3 kernels
        r"^resnet\.encoder\.stages\.([0-3])\.layers\.([0-9]+)\.layer\.([0-2])\.convolution\.weight$": (
            r"layer\1.blocks.\2.conv\3.kernel",
            ((2, 3, 1, 0), None),
        ),
        # BN scale (formerly 'weight')
        r"^resnet\.encoder\.stages\.([0-3])\.layers\.([0-9]+)\.layer\.([0-2])\.normalization\.weight$": (
            r"layer\1.blocks.\2.bn\3.scale",
            None,
        ),
        # BN bias
        r"^resnet\.encoder\.stages\.([0-3])\.layers\.([0-9]+)\.layer\.([0-2])\.normalization\.bias$": (
            r"layer\1.blocks.\2.bn\3.bias",
            None,
        ),
        # BN running mean
        r"^resnet\.encoder\.stages\.([0-3])\.layers\.([0-9]+)\.layer\.([0-2])\.normalization\.running_mean$": (
            r"layer\1.blocks.\2.bn\3.mean",
            None,
        ),
        # BN running var
        r"^resnet\.encoder\.stages\.([0-3])\.layers\.([0-9]+)\.layer\.([0-2])\.normalization\.running_var$": (
            r"layer\1.blocks.\2.bn\3.var",
            None,
        ),
        # shortcut conv
        r"^resnet\.encoder\.stages\.([0-3])\.layers\.([0-9]+)\.shortcut\.convolution\.weight$": (
            r"layer\1.blocks.\2.downsample.conv.kernel",
            ((2, 3, 1, 0), None),
        ),
        # shortcut BN scale
        r"^resnet\.encoder\.stages\.([0-3])\.layers\.([0-9]+)\.shortcut\.normalization\.weight$": (
            r"layer\1.blocks.\2.downsample.bn.scale",
            None,
        ),
        # shortcut BN bias
        r"^resnet\.encoder\.stages\.([0-3])\.layers\.([0-9]+)\.shortcut\.normalization\.bias$": (
            r"layer\1.blocks.\2.downsample.bn.bias",
            None,
        ),
        # shortcut BN running mean
        r"^resnet\.encoder\.stages\.([0-3])\.layers\.([0-9]+)\.shortcut\.normalization\.running_mean$": (
            r"layer\1.blocks.\2.downsample.bn.mean",
            None,
        ),
        # shortcut BN running var
        r"^resnet\.encoder\.stages\.([0-3])\.layers\.([0-9]+)\.shortcut\.normalization\.running_var$": (
            r"layer\1.blocks.\2.downsample.bn.var",
            None,
        ),
        # final classifier
        r"^classifier\.1\.weight$": ("fc.kernel", ((1, 0), None)),
        r"^classifier\.1\.bias$": ("fc.bias", None),
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


def _create_resnet_from_pretrained(
    model_cls: Callable[..., model_lib.ResNet],
    file_dir: str,
    num_classes: int = 1000,
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

    model = nnx.eval_shape(lambda: model_cls(num_classes=num_classes, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(model)
    jax_state = abs_state.to_pure_dict()

    mapping = _get_key_and_transform_mapping()
    for st_key, tensor in state_dict.items():
        jax_key, transform = _st_key_to_jax_key(mapping, st_key)
        if jax_key is None:
            continue
        keys = [_stoi(k) for k in jax_key.split(".")]
        _assign_weights(keys, tensor, jax_state, st_key, transform)

    if mesh is not None:
        sharding = nnx.get_named_sharding(abs_state, mesh).to_pure_dict()
        jax_state = jax.device_put(jax_state, sharding)
    else:
        jax_state = jax.device_put(jax_state, jax.devices()[0])

    return nnx.merge(graph_def, jax_state)


def create_resnet50_from_pretrained(
    file_dir: str,
    num_classes: int = 1000,
    *,
    mesh: jax.sharding.Mesh | None = None,
):
    """Loads ResNet50 weights."""
    return _create_resnet_from_pretrained(
        model_lib.ResNet50,
        file_dir=file_dir,
        num_classes=num_classes,
        mesh=mesh,
    )


def create_resnet152_from_pretrained(
    file_dir: str,
    num_classes: int = 1000,
    *,
    mesh: jax.sharding.Mesh | None = None,
):
    """Loads ResNet152 weights."""
    return _create_resnet_from_pretrained(
        model_lib.ResNet152,
        file_dir=file_dir,
        num_classes=num_classes,
        mesh=mesh,
    )
