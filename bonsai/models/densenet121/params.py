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

from bonsai.models.densenet121 import modeling as model_lib
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
    # Mapping st_keys -> (nnx_keys, (permute_rule, reshape_rule)).
    bn_params = ["scale", "bias", "mean", "var"]
    mapping = {}

    # init conv
    mapping[r"^layers/dense_net_backbone/layers/conv2d/vars/0$"] = ("init_conv.kernel", None)

    # init bn
    for i in range(len(bn_params)):
        key = rf"^layers/dense_net_backbone/layers/batch_normalization/vars/{i}$"
        mapping[key] = (f"init_bn.{bn_params[i]}", None)

    str_index = 0
    for i in range(len(cfg.dense_block_layers)):
        end_index = str_index + (cfg.dense_block_layers[i] * 2)

        # i-th dense block
        for layer_index in range(str_index, end_index):
            # dense block.bn
            for bn_params_index in range(len(bn_params)):
                key = (
                    rf"^layers/dense_net_backbone/layers/batch_normalization_{layer_index + 1}/vars/{bn_params_index}$"
                )
                val = (f"blocks.layers.{i * 2}.bn_layers.{layer_index - str_index}.{bn_params[bn_params_index]}", None)
                mapping[key] = val

            # dense block.conv
            key = rf"^layers/dense_net_backbone/layers/conv2d_{layer_index + 1}/vars/0$"
            val = (f"blocks.layers.{i * 2}.conv_layers.{layer_index - str_index}.kernel", None)
            mapping[key] = val

        if i < len(cfg.dense_block_layers) - 1:
            # i-th transition
            for bn_params_index in range(len(bn_params)):
                # transition.bn
                key = rf"^layers/dense_net_backbone/layers/batch_normalization_{end_index + 1}/vars/{bn_params_index}$"
                val = (f"blocks.layers.{i * 2 + 1}.bn.{bn_params[bn_params_index]}", None)
                mapping[key] = val

            # transition.conv
            key = rf"^layers/dense_net_backbone/layers/conv2d_{end_index + 1}/vars/0$"
            val = (f"blocks.layers.{i * 2 + 1}.conv.kernel", None)
            mapping[key] = val

            str_index = end_index + 1

    # final bn
    for i in range(len(bn_params)):
        key = rf"^layers/dense_net_backbone/layers/batch_normalization_120/vars/{i}$"
        mapping[key] = (f"final_bn.{bn_params[i]}", None)

    # linear
    mapping[r"^layers/dense/vars/0"] = ("linear.kernel", None)
    mapping[r"^layers/dense/vars/1"] = ("linear.bias", None)

    return mapping


def create_model_from_h5(file_dir: str, cfg: model_lib.ModelConfig) -> model_lib.DenseNet:
    """
    Load h5 weights from a file, then convert & merge into a flax.nnx DenseNet121 model.

    Returns:
      A flax.nnx.Model instance with loaded parameters.
    """
    file = epath.Path(file_dir).expanduser() / "task.weights.h5"
    if not file:
        raise ValueError(f"No h5 found in {file_dir}")

    tensor_dict = _load_h5_file(file)

    densenet = nnx.eval_shape(lambda: model_lib.DenseNet(cfg, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(densenet)
    state_dict = nnx.to_pure_dict(abs_state)

    mapping = _get_key_and_transform_mapping(cfg)
    for st_key, tensor in tensor_dict.items():
        jax_key, transform = map_to_bonsai_key(mapping, st_key)
        if jax_key is None:
            continue
        keys = [stoi(k) for k in jax_key.split(".")]
        assign_weights_from_eval_shape(keys, tensor, state_dict, st_key, transform)

    return nnx.merge(graph_def, state_dict)
