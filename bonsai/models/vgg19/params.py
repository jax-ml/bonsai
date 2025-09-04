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

import h5py
import jax
import numpy as np
from flax import nnx

from bonsai.models.vgg19 import modeling as model_lib


def _load_h5_file(file_path: str):
    """Load weights from an HDF5 file into a flat dictionary."""
    state_dict = {}
    with h5py.File(file_path, "r") as f:
        # Recursively visit all items in the HDF5 file
        def visit_items(name, obj):
            if isinstance(obj, h5py.Dataset):
                # Load the data as a numpy array
                state_dict[name] = np.array(obj)

        f.visititems(visit_items)
    return state_dict


def _get_key_and_transform_mapping():
    # Mapping st_keys -> (nnx_keys, (permute_rule, reshape_rule)).
    return {
        # conv_block 0
        r"^layers/vgg_backbone/layers/conv2d/vars/0$": ("conv_block0.conv_layers.0.kernel", None),
        r"^layers/vgg_backbone/layers/conv2d/vars/1$": ("conv_block0.conv_layers.0.bias", None),
        r"^layers/vgg_backbone/layers/conv2d_1/vars/0$": ("conv_block0.conv_layers.1.kernel", None),
        r"^layers/vgg_backbone/layers/conv2d_1/vars/1$": ("conv_block0.conv_layers.1.bias", None),
        # conv_block 1
        r"^layers/vgg_backbone/layers/conv2d_2/vars/0$": ("conv_block1.conv_layers.0.kernel", None),
        r"^layers/vgg_backbone/layers/conv2d_2/vars/1$": ("conv_block1.conv_layers.0.bias", None),
        r"^layers/vgg_backbone/layers/conv2d_3/vars/0$": ("conv_block1.conv_layers.1.kernel", None),
        r"^layers/vgg_backbone/layers/conv2d_3/vars/1$": ("conv_block1.conv_layers.1.bias", None),
        # conv_block 2
        r"^layers/vgg_backbone/layers/conv2d_4/vars/0$": ("conv_block2.conv_layers.0.kernel", None),
        r"^layers/vgg_backbone/layers/conv2d_4/vars/1$": ("conv_block2.conv_layers.0.bias", None),
        r"^layers/vgg_backbone/layers/conv2d_5/vars/0$": ("conv_block2.conv_layers.1.kernel", None),
        r"^layers/vgg_backbone/layers/conv2d_5/vars/1$": ("conv_block2.conv_layers.1.bias", None),
        r"^layers/vgg_backbone/layers/conv2d_6/vars/0$": ("conv_block2.conv_layers.2.kernel", None),
        r"^layers/vgg_backbone/layers/conv2d_6/vars/1$": ("conv_block2.conv_layers.2.bias", None),
        r"^layers/vgg_backbone/layers/conv2d_7/vars/0$": ("conv_block2.conv_layers.3.kernel", None),
        r"^layers/vgg_backbone/layers/conv2d_7/vars/1$": ("conv_block2.conv_layers.3.bias", None),
        # conv_block 3
        r"^layers/vgg_backbone/layers/conv2d_8/vars/0$": ("conv_block3.conv_layers.0.kernel", None),
        r"^layers/vgg_backbone/layers/conv2d_8/vars/1$": ("conv_block3.conv_layers.0.bias", None),
        r"^layers/vgg_backbone/layers/conv2d_9/vars/0$": ("conv_block3.conv_layers.1.kernel", None),
        r"^layers/vgg_backbone/layers/conv2d_9/vars/1$": ("conv_block3.conv_layers.1.bias", None),
        r"^layers/vgg_backbone/layers/conv2d_10/vars/0$": ("conv_block3.conv_layers.2.kernel", None),
        r"^layers/vgg_backbone/layers/conv2d_10/vars/1$": ("conv_block3.conv_layers.2.bias", None),
        r"^layers/vgg_backbone/layers/conv2d_11/vars/0$": ("conv_block3.conv_layers.3.kernel", None),
        r"^layers/vgg_backbone/layers/conv2d_11/vars/1$": ("conv_block3.conv_layers.3.bias", None),
        # conv_block 4
        r"^layers/vgg_backbone/layers/conv2d_12/vars/0$": ("conv_block4.conv_layers.0.kernel", None),
        r"^layers/vgg_backbone/layers/conv2d_12/vars/1$": ("conv_block4.conv_layers.0.bias", None),
        r"^layers/vgg_backbone/layers/conv2d_13/vars/0$": ("conv_block4.conv_layers.1.kernel", None),
        r"^layers/vgg_backbone/layers/conv2d_13/vars/1$": ("conv_block4.conv_layers.1.bias", None),
        r"^layers/vgg_backbone/layers/conv2d_14/vars/0$": ("conv_block4.conv_layers.2.kernel", None),
        r"^layers/vgg_backbone/layers/conv2d_14/vars/1$": ("conv_block4.conv_layers.2.bias", None),
        r"^layers/vgg_backbone/layers/conv2d_15/vars/0$": ("conv_block4.conv_layers.3.kernel", None),
        r"^layers/vgg_backbone/layers/conv2d_15/vars/1$": ("conv_block4.conv_layers.3.bias", None),
        # Classifier
        r"^layers/sequential/layers/conv2d/vars/0$": ("classifier.layers.0.kernel", (None, (-1, 4096))),
        r"^layers/sequential/layers/conv2d/vars/1$": ("classifier.layers.0.bias", None),
        r"^layers/sequential/layers/conv2d_1/vars/0$": ("classifier.layers.2.kernel", (None, (4096, 4096))),
        r"^layers/sequential/layers/conv2d_1/vars/1$": ("classifier.layers.2.bias", None),
        r"^layers/sequential/layers/dense/vars/0$": ("classifier.layers.4.kernel", None),
        r"^layers/sequential/layers/dense/vars/1$": ("classifier.layers.4.bias", None),
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


def create_vgg19_from_pretrained(
    file_path: str,
    num_classes: int = 1000,
    *,
    mesh: jax.sharding.Mesh | None = None,
):
    """
    Load safetensor weights from a file, then convert & merge into a flax.nnx VGG19 model.

    Returns:
      A flax.nnx.Model instance with loaded parameters.
    """
    state_dict = _load_h5_file(file_path)

    vgg19 = nnx.eval_shape(lambda: model_lib.VGG19(num_classes=num_classes, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(vgg19)
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
