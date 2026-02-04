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


import h5py
import numpy as np
from etils import epath
from flax import nnx

from bonsai.models.vgg19 import modeling as model_lib
from bonsai.utils.params import stoi, map_to_bonsai_key, assign_weights_from_eval_shape


def _load_h5_file(file_path: str):
    """Load weights from an HDF5 file into a flat dictionary."""
    tensor_dict = {}
    with h5py.File(file_path, "r") as f:
        # Recursively visit all items in the HDF5 file
        def visit_items(name, obj):
            if isinstance(obj, h5py.Dataset):
                # Load the data as a numpy array
                tensor_dict[name] = np.array(obj)

        f.visititems(visit_items)
    return tensor_dict


def _get_key_and_transform_mapping(cfg: model_lib.ModelConfig):
    # Mapping st_keys -> (nnx_keys, (permute_rule, reshape_rule, reshape_first)).
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
        r"^layers/sequential/layers/conv2d/vars/0$": ("classifier.layers.0.kernel", None),
        r"^layers/sequential/layers/conv2d/vars/1$": ("classifier.layers.0.bias", None),
        r"^layers/sequential/layers/conv2d_1/vars/0$": ("classifier.layers.1.kernel", None),
        r"^layers/sequential/layers/conv2d_1/vars/1$": ("classifier.layers.1.bias", None),
        r"^layers/sequential/layers/dense/vars/0$": ("classifier.layers.3.kernel", None),
        r"^layers/sequential/layers/dense/vars/1$": ("classifier.layers.3.bias", None),
    }


def create_model_from_h5(file_dir: str, cfg: model_lib.ModelConfig) -> model_lib.VGG:
    """
    Load h5 weights from a file, then convert & merge into a flax.nnx VGG19 model.

    Returns:
      A flax.nnx.Model instance with loaded parameters.
    """
    file = epath.Path(file_dir).expanduser() / "task.weights.h5"
    if not file:
        raise ValueError(f"No h5 found in {file_dir}")

    tensor_dict = _load_h5_file(file)

    vgg = nnx.eval_shape(lambda: model_lib.VGG(cfg, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(vgg)
    state_dict = nnx.to_pure_dict(abs_state)

    mapping = _get_key_and_transform_mapping(cfg)
    conversion_errors = []
    for st_key, tensor in tensor_dict.items():
        jax_key, transform = map_to_bonsai_key(mapping, st_key)
        if jax_key is None:
            continue
        keys = [stoi(k) for k in jax_key.split(".")]
        try:
            assign_weights_from_eval_shape(keys, tensor, state_dict, st_key, transform)
        except Exception as e:
            full_jax_key = ".".join([str(k) for k in keys])
            conversion_errors.append(f"Failed to assign '{st_key}' to '{full_jax_key}': {type(e).__name__}: {e}")

    if conversion_errors:
        full_error_log = "\n".join(conversion_errors)
        raise RuntimeError(f"Encountered {len(conversion_errors)} weight conversion errors. Log:\n{full_error_log}")

    return nnx.merge(graph_def, state_dict)
