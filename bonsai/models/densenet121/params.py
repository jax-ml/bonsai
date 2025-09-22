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
from etils import epath
from flax import nnx

from bonsai.models.densenet121 import modeling as model_lib


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


def _get_key_and_transform_mapping():
    # Mapping st_keys -> (nnx_keys, (permute_rule, reshape_rule)).
    return {
        # init conv
        r"^layers/dense_net_backbone/layers/conv2d/vars/0$": ("init_conv.kernel", None),
        # init bn
        r"^layers/dense_net_backbone/layers/batch_normalization/vars/0$": ("init_bn.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization/vars/1$": ("init_bn.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization/vars/2$": ("init_bn.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization/vars/3$": ("init_bn.var", None),
        # dense block 0
        r"^layers/dense_net_backbone/layers/batch_normalization_1/vars/0$": ("dense_blocks.0.bn_layers.0.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_1/vars/1$": ("dense_blocks.0.bn_layers.0.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_1/vars/2$": ("dense_blocks.0.bn_layers.0.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_1/vars/3$": ("dense_blocks.0.bn_layers.0.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_1/vars/0$": ("dense_blocks.0.conv_layers.0.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_2/vars/0$": ("dense_blocks.0.bn_layers.1.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_2/vars/1$": ("dense_blocks.0.bn_layers.1.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_2/vars/2$": ("dense_blocks.0.bn_layers.1.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_2/vars/3$": ("dense_blocks.0.bn_layers.1.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_2/vars/0$": ("dense_blocks.0.conv_layers.1.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_3/vars/0$": ("dense_blocks.0.bn_layers.2.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_3/vars/1$": ("dense_blocks.0.bn_layers.2.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_3/vars/2$": ("dense_blocks.0.bn_layers.2.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_3/vars/3$": ("dense_blocks.0.bn_layers.2.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_3/vars/0$": ("dense_blocks.0.conv_layers.2.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_4/vars/0$": ("dense_blocks.0.bn_layers.3.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_4/vars/1$": ("dense_blocks.0.bn_layers.3.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_4/vars/2$": ("dense_blocks.0.bn_layers.3.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_4/vars/3$": ("dense_blocks.0.bn_layers.3.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_4/vars/0$": ("dense_blocks.0.conv_layers.3.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_5/vars/0$": ("dense_blocks.0.bn_layers.4.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_5/vars/1$": ("dense_blocks.0.bn_layers.4.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_5/vars/2$": ("dense_blocks.0.bn_layers.4.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_5/vars/3$": ("dense_blocks.0.bn_layers.4.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_5/vars/0$": ("dense_blocks.0.conv_layers.4.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_6/vars/0$": ("dense_blocks.0.bn_layers.5.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_6/vars/1$": ("dense_blocks.0.bn_layers.5.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_6/vars/2$": ("dense_blocks.0.bn_layers.5.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_6/vars/3$": ("dense_blocks.0.bn_layers.5.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_6/vars/0$": ("dense_blocks.0.conv_layers.5.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_7/vars/0$": ("dense_blocks.0.bn_layers.6.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_7/vars/1$": ("dense_blocks.0.bn_layers.6.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_7/vars/2$": ("dense_blocks.0.bn_layers.6.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_7/vars/3$": ("dense_blocks.0.bn_layers.6.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_7/vars/0$": ("dense_blocks.0.conv_layers.6.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_8/vars/0$": ("dense_blocks.0.bn_layers.7.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_8/vars/1$": ("dense_blocks.0.bn_layers.7.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_8/vars/2$": ("dense_blocks.0.bn_layers.7.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_8/vars/3$": ("dense_blocks.0.bn_layers.7.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_8/vars/0$": ("dense_blocks.0.conv_layers.7.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_9/vars/0$": ("dense_blocks.0.bn_layers.8.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_9/vars/1$": ("dense_blocks.0.bn_layers.8.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_9/vars/2$": ("dense_blocks.0.bn_layers.8.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_9/vars/3$": ("dense_blocks.0.bn_layers.8.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_9/vars/0$": ("dense_blocks.0.conv_layers.8.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_10/vars/0$": ("dense_blocks.0.bn_layers.9.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_10/vars/1$": ("dense_blocks.0.bn_layers.9.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_10/vars/2$": ("dense_blocks.0.bn_layers.9.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_10/vars/3$": ("dense_blocks.0.bn_layers.9.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_10/vars/0$": ("dense_blocks.0.conv_layers.9.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_11/vars/0$": (
            "dense_blocks.0.bn_layers.10.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_11/vars/1$": ("dense_blocks.0.bn_layers.10.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_11/vars/2$": ("dense_blocks.0.bn_layers.10.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_11/vars/3$": ("dense_blocks.0.bn_layers.10.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_11/vars/0$": ("dense_blocks.0.conv_layers.10.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_12/vars/0$": (
            "dense_blocks.0.bn_layers.11.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_12/vars/1$": ("dense_blocks.0.bn_layers.11.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_12/vars/2$": ("dense_blocks.0.bn_layers.11.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_12/vars/3$": ("dense_blocks.0.bn_layers.11.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_12/vars/0$": ("dense_blocks.0.conv_layers.11.kernel", None),
        # transition 0
        r"^layers/dense_net_backbone/layers/batch_normalization_13/vars/0$": ("transition_layers.0.bn.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_13/vars/1$": ("transition_layers.0.bn.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_13/vars/2$": ("transition_layers.0.bn.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_13/vars/3$": ("transition_layers.0.bn.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_13/vars/0$": ("transition_layers.0.conv.kernel", None),
        # dense block 1
        r"^layers/dense_net_backbone/layers/batch_normalization_14/vars/0$": ("dense_blocks.1.bn_layers.0.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_14/vars/1$": ("dense_blocks.1.bn_layers.0.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_14/vars/2$": ("dense_blocks.1.bn_layers.0.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_14/vars/3$": ("dense_blocks.1.bn_layers.0.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_14/vars/0$": ("dense_blocks.1.conv_layers.0.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_15/vars/0$": ("dense_blocks.1.bn_layers.1.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_15/vars/1$": ("dense_blocks.1.bn_layers.1.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_15/vars/2$": ("dense_blocks.1.bn_layers.1.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_15/vars/3$": ("dense_blocks.1.bn_layers.1.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_15/vars/0$": ("dense_blocks.1.conv_layers.1.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_16/vars/0$": ("dense_blocks.1.bn_layers.2.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_16/vars/1$": ("dense_blocks.1.bn_layers.2.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_16/vars/2$": ("dense_blocks.1.bn_layers.2.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_16/vars/3$": ("dense_blocks.1.bn_layers.2.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_16/vars/0$": ("dense_blocks.1.conv_layers.2.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_17/vars/0$": ("dense_blocks.1.bn_layers.3.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_17/vars/1$": ("dense_blocks.1.bn_layers.3.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_17/vars/2$": ("dense_blocks.1.bn_layers.3.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_17/vars/3$": ("dense_blocks.1.bn_layers.3.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_17/vars/0$": ("dense_blocks.1.conv_layers.3.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_18/vars/0$": ("dense_blocks.1.bn_layers.4.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_18/vars/1$": ("dense_blocks.1.bn_layers.4.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_18/vars/2$": ("dense_blocks.1.bn_layers.4.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_18/vars/3$": ("dense_blocks.1.bn_layers.4.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_18/vars/0$": ("dense_blocks.1.conv_layers.4.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_19/vars/0$": ("dense_blocks.1.bn_layers.5.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_19/vars/1$": ("dense_blocks.1.bn_layers.5.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_19/vars/2$": ("dense_blocks.1.bn_layers.5.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_19/vars/3$": ("dense_blocks.1.bn_layers.5.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_19/vars/0$": ("dense_blocks.1.conv_layers.5.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_20/vars/0$": ("dense_blocks.1.bn_layers.6.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_20/vars/1$": ("dense_blocks.1.bn_layers.6.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_20/vars/2$": ("dense_blocks.1.bn_layers.6.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_20/vars/3$": ("dense_blocks.1.bn_layers.6.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_20/vars/0$": ("dense_blocks.1.conv_layers.6.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_21/vars/0$": ("dense_blocks.1.bn_layers.7.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_21/vars/1$": ("dense_blocks.1.bn_layers.7.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_21/vars/2$": ("dense_blocks.1.bn_layers.7.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_21/vars/3$": ("dense_blocks.1.bn_layers.7.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_21/vars/0$": ("dense_blocks.1.conv_layers.7.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_22/vars/0$": ("dense_blocks.1.bn_layers.8.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_22/vars/1$": ("dense_blocks.1.bn_layers.8.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_22/vars/2$": ("dense_blocks.1.bn_layers.8.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_22/vars/3$": ("dense_blocks.1.bn_layers.8.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_22/vars/0$": ("dense_blocks.1.conv_layers.8.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_23/vars/0$": ("dense_blocks.1.bn_layers.9.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_23/vars/1$": ("dense_blocks.1.bn_layers.9.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_23/vars/2$": ("dense_blocks.1.bn_layers.9.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_23/vars/3$": ("dense_blocks.1.bn_layers.9.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_23/vars/0$": ("dense_blocks.1.conv_layers.9.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_24/vars/0$": (
            "dense_blocks.1.bn_layers.10.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_24/vars/1$": ("dense_blocks.1.bn_layers.10.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_24/vars/2$": ("dense_blocks.1.bn_layers.10.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_24/vars/3$": ("dense_blocks.1.bn_layers.10.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_24/vars/0$": ("dense_blocks.1.conv_layers.10.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_25/vars/0$": (
            "dense_blocks.1.bn_layers.11.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_25/vars/1$": ("dense_blocks.1.bn_layers.11.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_25/vars/2$": ("dense_blocks.1.bn_layers.11.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_25/vars/3$": ("dense_blocks.1.bn_layers.11.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_25/vars/0$": ("dense_blocks.1.conv_layers.11.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_26/vars/0$": (
            "dense_blocks.1.bn_layers.12.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_26/vars/1$": ("dense_blocks.1.bn_layers.12.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_26/vars/2$": ("dense_blocks.1.bn_layers.12.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_26/vars/3$": ("dense_blocks.1.bn_layers.12.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_26/vars/0$": ("dense_blocks.1.conv_layers.12.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_27/vars/0$": (
            "dense_blocks.1.bn_layers.13.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_27/vars/1$": ("dense_blocks.1.bn_layers.13.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_27/vars/2$": ("dense_blocks.1.bn_layers.13.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_27/vars/3$": ("dense_blocks.1.bn_layers.13.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_27/vars/0$": ("dense_blocks.1.conv_layers.13.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_28/vars/0$": (
            "dense_blocks.1.bn_layers.14.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_28/vars/1$": ("dense_blocks.1.bn_layers.14.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_28/vars/2$": ("dense_blocks.1.bn_layers.14.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_28/vars/3$": ("dense_blocks.1.bn_layers.14.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_28/vars/0$": ("dense_blocks.1.conv_layers.14.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_29/vars/0$": (
            "dense_blocks.1.bn_layers.15.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_29/vars/1$": ("dense_blocks.1.bn_layers.15.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_29/vars/2$": ("dense_blocks.1.bn_layers.15.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_29/vars/3$": ("dense_blocks.1.bn_layers.15.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_29/vars/0$": ("dense_blocks.1.conv_layers.15.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_30/vars/0$": (
            "dense_blocks.1.bn_layers.16.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_30/vars/1$": ("dense_blocks.1.bn_layers.16.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_30/vars/2$": ("dense_blocks.1.bn_layers.16.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_30/vars/3$": ("dense_blocks.1.bn_layers.16.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_30/vars/0$": ("dense_blocks.1.conv_layers.16.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_31/vars/0$": (
            "dense_blocks.1.bn_layers.17.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_31/vars/1$": ("dense_blocks.1.bn_layers.17.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_31/vars/2$": ("dense_blocks.1.bn_layers.17.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_31/vars/3$": ("dense_blocks.1.bn_layers.17.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_31/vars/0$": ("dense_blocks.1.conv_layers.17.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_32/vars/0$": (
            "dense_blocks.1.bn_layers.18.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_32/vars/1$": ("dense_blocks.1.bn_layers.18.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_32/vars/2$": ("dense_blocks.1.bn_layers.18.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_32/vars/3$": ("dense_blocks.1.bn_layers.18.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_32/vars/0$": ("dense_blocks.1.conv_layers.18.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_33/vars/0$": (
            "dense_blocks.1.bn_layers.19.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_33/vars/1$": ("dense_blocks.1.bn_layers.19.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_33/vars/2$": ("dense_blocks.1.bn_layers.19.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_33/vars/3$": ("dense_blocks.1.bn_layers.19.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_33/vars/0$": ("dense_blocks.1.conv_layers.19.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_34/vars/0$": (
            "dense_blocks.1.bn_layers.20.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_34/vars/1$": ("dense_blocks.1.bn_layers.20.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_34/vars/2$": ("dense_blocks.1.bn_layers.20.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_34/vars/3$": ("dense_blocks.1.bn_layers.20.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_34/vars/0$": ("dense_blocks.1.conv_layers.20.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_35/vars/0$": (
            "dense_blocks.1.bn_layers.21.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_35/vars/1$": ("dense_blocks.1.bn_layers.21.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_35/vars/2$": ("dense_blocks.1.bn_layers.21.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_35/vars/3$": ("dense_blocks.1.bn_layers.21.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_35/vars/0$": ("dense_blocks.1.conv_layers.21.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_36/vars/0$": (
            "dense_blocks.1.bn_layers.22.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_36/vars/1$": ("dense_blocks.1.bn_layers.22.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_36/vars/2$": ("dense_blocks.1.bn_layers.22.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_36/vars/3$": ("dense_blocks.1.bn_layers.22.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_36/vars/0$": ("dense_blocks.1.conv_layers.22.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_37/vars/0$": (
            "dense_blocks.1.bn_layers.23.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_37/vars/1$": ("dense_blocks.1.bn_layers.23.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_37/vars/2$": ("dense_blocks.1.bn_layers.23.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_37/vars/3$": ("dense_blocks.1.bn_layers.23.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_37/vars/0$": ("dense_blocks.1.conv_layers.23.kernel", None),
        # transition 1
        r"^layers/dense_net_backbone/layers/batch_normalization_38/vars/0$": ("transition_layers.1.bn.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_38/vars/1$": ("transition_layers.1.bn.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_38/vars/2$": ("transition_layers.1.bn.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_38/vars/3$": ("transition_layers.1.bn.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_38/vars/0$": ("transition_layers.1.conv.kernel", None),
        # dense block 2
        r"^layers/dense_net_backbone/layers/batch_normalization_39/vars/0$": ("dense_blocks.2.bn_layers.0.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_39/vars/1$": ("dense_blocks.2.bn_layers.0.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_39/vars/2$": ("dense_blocks.2.bn_layers.0.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_39/vars/3$": ("dense_blocks.2.bn_layers.0.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_39/vars/0$": ("dense_blocks.2.conv_layers.0.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_40/vars/0$": ("dense_blocks.2.bn_layers.1.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_40/vars/1$": ("dense_blocks.2.bn_layers.1.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_40/vars/2$": ("dense_blocks.2.bn_layers.1.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_40/vars/3$": ("dense_blocks.2.bn_layers.1.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_40/vars/0$": ("dense_blocks.2.conv_layers.1.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_41/vars/0$": ("dense_blocks.2.bn_layers.2.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_41/vars/1$": ("dense_blocks.2.bn_layers.2.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_41/vars/2$": ("dense_blocks.2.bn_layers.2.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_41/vars/3$": ("dense_blocks.2.bn_layers.2.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_41/vars/0$": ("dense_blocks.2.conv_layers.2.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_42/vars/0$": ("dense_blocks.2.bn_layers.3.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_42/vars/1$": ("dense_blocks.2.bn_layers.3.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_42/vars/2$": ("dense_blocks.2.bn_layers.3.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_42/vars/3$": ("dense_blocks.2.bn_layers.3.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_42/vars/0$": ("dense_blocks.2.conv_layers.3.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_43/vars/0$": ("dense_blocks.2.bn_layers.4.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_43/vars/1$": ("dense_blocks.2.bn_layers.4.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_43/vars/2$": ("dense_blocks.2.bn_layers.4.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_43/vars/3$": ("dense_blocks.2.bn_layers.4.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_43/vars/0$": ("dense_blocks.2.conv_layers.4.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_44/vars/0$": ("dense_blocks.2.bn_layers.5.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_44/vars/1$": ("dense_blocks.2.bn_layers.5.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_44/vars/2$": ("dense_blocks.2.bn_layers.5.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_44/vars/3$": ("dense_blocks.2.bn_layers.5.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_44/vars/0$": ("dense_blocks.2.conv_layers.5.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_45/vars/0$": ("dense_blocks.2.bn_layers.6.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_45/vars/1$": ("dense_blocks.2.bn_layers.6.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_45/vars/2$": ("dense_blocks.2.bn_layers.6.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_45/vars/3$": ("dense_blocks.2.bn_layers.6.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_45/vars/0$": ("dense_blocks.2.conv_layers.6.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_46/vars/0$": ("dense_blocks.2.bn_layers.7.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_46/vars/1$": ("dense_blocks.2.bn_layers.7.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_46/vars/2$": ("dense_blocks.2.bn_layers.7.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_46/vars/3$": ("dense_blocks.2.bn_layers.7.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_46/vars/0$": ("dense_blocks.2.conv_layers.7.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_47/vars/0$": ("dense_blocks.2.bn_layers.8.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_47/vars/1$": ("dense_blocks.2.bn_layers.8.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_47/vars/2$": ("dense_blocks.2.bn_layers.8.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_47/vars/3$": ("dense_blocks.2.bn_layers.8.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_47/vars/0$": ("dense_blocks.2.conv_layers.8.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_48/vars/0$": ("dense_blocks.2.bn_layers.9.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_48/vars/1$": ("dense_blocks.2.bn_layers.9.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_48/vars/2$": ("dense_blocks.2.bn_layers.9.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_48/vars/3$": ("dense_blocks.2.bn_layers.9.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_48/vars/0$": ("dense_blocks.2.conv_layers.9.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_49/vars/0$": (
            "dense_blocks.2.bn_layers.10.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_49/vars/1$": ("dense_blocks.2.bn_layers.10.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_49/vars/2$": ("dense_blocks.2.bn_layers.10.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_49/vars/3$": ("dense_blocks.2.bn_layers.10.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_49/vars/0$": ("dense_blocks.2.conv_layers.10.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_50/vars/0$": (
            "dense_blocks.2.bn_layers.11.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_50/vars/1$": ("dense_blocks.2.bn_layers.11.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_50/vars/2$": ("dense_blocks.2.bn_layers.11.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_50/vars/3$": ("dense_blocks.2.bn_layers.11.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_50/vars/0$": ("dense_blocks.2.conv_layers.11.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_51/vars/0$": (
            "dense_blocks.2.bn_layers.12.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_51/vars/1$": ("dense_blocks.2.bn_layers.12.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_51/vars/2$": ("dense_blocks.2.bn_layers.12.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_51/vars/3$": ("dense_blocks.2.bn_layers.12.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_51/vars/0$": ("dense_blocks.2.conv_layers.12.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_52/vars/0$": (
            "dense_blocks.2.bn_layers.13.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_52/vars/1$": ("dense_blocks.2.bn_layers.13.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_52/vars/2$": ("dense_blocks.2.bn_layers.13.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_52/vars/3$": ("dense_blocks.2.bn_layers.13.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_52/vars/0$": ("dense_blocks.2.conv_layers.13.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_53/vars/0$": (
            "dense_blocks.2.bn_layers.14.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_53/vars/1$": ("dense_blocks.2.bn_layers.14.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_53/vars/2$": ("dense_blocks.2.bn_layers.14.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_53/vars/3$": ("dense_blocks.2.bn_layers.14.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_53/vars/0$": ("dense_blocks.2.conv_layers.14.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_54/vars/0$": (
            "dense_blocks.2.bn_layers.15.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_54/vars/1$": ("dense_blocks.2.bn_layers.15.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_54/vars/2$": ("dense_blocks.2.bn_layers.15.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_54/vars/3$": ("dense_blocks.2.bn_layers.15.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_54/vars/0$": ("dense_blocks.2.conv_layers.15.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_55/vars/0$": (
            "dense_blocks.2.bn_layers.16.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_55/vars/1$": ("dense_blocks.2.bn_layers.16.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_55/vars/2$": ("dense_blocks.2.bn_layers.16.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_55/vars/3$": ("dense_blocks.2.bn_layers.16.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_55/vars/0$": ("dense_blocks.2.conv_layers.16.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_56/vars/0$": (
            "dense_blocks.2.bn_layers.17.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_56/vars/1$": ("dense_blocks.2.bn_layers.17.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_56/vars/2$": ("dense_blocks.2.bn_layers.17.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_56/vars/3$": ("dense_blocks.2.bn_layers.17.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_56/vars/0$": ("dense_blocks.2.conv_layers.17.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_57/vars/0$": (
            "dense_blocks.2.bn_layers.18.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_57/vars/1$": ("dense_blocks.2.bn_layers.18.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_57/vars/2$": ("dense_blocks.2.bn_layers.18.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_57/vars/3$": ("dense_blocks.2.bn_layers.18.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_57/vars/0$": ("dense_blocks.2.conv_layers.18.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_58/vars/0$": (
            "dense_blocks.2.bn_layers.19.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_58/vars/1$": ("dense_blocks.2.bn_layers.19.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_58/vars/2$": ("dense_blocks.2.bn_layers.19.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_58/vars/3$": ("dense_blocks.2.bn_layers.19.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_58/vars/0$": ("dense_blocks.2.conv_layers.19.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_59/vars/0$": (
            "dense_blocks.2.bn_layers.20.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_59/vars/1$": ("dense_blocks.2.bn_layers.20.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_59/vars/2$": ("dense_blocks.2.bn_layers.20.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_59/vars/3$": ("dense_blocks.2.bn_layers.20.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_59/vars/0$": ("dense_blocks.2.conv_layers.20.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_60/vars/0$": (
            "dense_blocks.2.bn_layers.21.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_60/vars/1$": ("dense_blocks.2.bn_layers.21.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_60/vars/2$": ("dense_blocks.2.bn_layers.21.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_60/vars/3$": ("dense_blocks.2.bn_layers.21.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_60/vars/0$": ("dense_blocks.2.conv_layers.21.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_61/vars/0$": (
            "dense_blocks.2.bn_layers.22.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_61/vars/1$": ("dense_blocks.2.bn_layers.22.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_61/vars/2$": ("dense_blocks.2.bn_layers.22.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_61/vars/3$": ("dense_blocks.2.bn_layers.22.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_61/vars/0$": ("dense_blocks.2.conv_layers.22.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_62/vars/0$": (
            "dense_blocks.2.bn_layers.23.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_62/vars/1$": ("dense_blocks.2.bn_layers.23.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_62/vars/2$": ("dense_blocks.2.bn_layers.23.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_62/vars/3$": ("dense_blocks.2.bn_layers.23.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_62/vars/0$": ("dense_blocks.2.conv_layers.23.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_63/vars/0$": (
            "dense_blocks.2.bn_layers.24.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_63/vars/1$": ("dense_blocks.2.bn_layers.24.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_63/vars/2$": ("dense_blocks.2.bn_layers.24.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_63/vars/3$": ("dense_blocks.2.bn_layers.24.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_63/vars/0$": ("dense_blocks.2.conv_layers.24.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_64/vars/0$": (
            "dense_blocks.2.bn_layers.25.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_64/vars/1$": ("dense_blocks.2.bn_layers.25.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_64/vars/2$": ("dense_blocks.2.bn_layers.25.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_64/vars/3$": ("dense_blocks.2.bn_layers.25.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_64/vars/0$": ("dense_blocks.2.conv_layers.25.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_65/vars/0$": (
            "dense_blocks.2.bn_layers.26.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_65/vars/1$": ("dense_blocks.2.bn_layers.26.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_65/vars/2$": ("dense_blocks.2.bn_layers.26.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_65/vars/3$": ("dense_blocks.2.bn_layers.26.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_65/vars/0$": ("dense_blocks.2.conv_layers.26.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_66/vars/0$": (
            "dense_blocks.2.bn_layers.27.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_66/vars/1$": ("dense_blocks.2.bn_layers.27.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_66/vars/2$": ("dense_blocks.2.bn_layers.27.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_66/vars/3$": ("dense_blocks.2.bn_layers.27.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_66/vars/0$": ("dense_blocks.2.conv_layers.27.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_67/vars/0$": (
            "dense_blocks.2.bn_layers.28.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_67/vars/1$": ("dense_blocks.2.bn_layers.28.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_67/vars/2$": ("dense_blocks.2.bn_layers.28.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_67/vars/3$": ("dense_blocks.2.bn_layers.28.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_67/vars/0$": ("dense_blocks.2.conv_layers.28.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_68/vars/0$": (
            "dense_blocks.2.bn_layers.29.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_68/vars/1$": ("dense_blocks.2.bn_layers.29.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_68/vars/2$": ("dense_blocks.2.bn_layers.29.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_68/vars/3$": ("dense_blocks.2.bn_layers.29.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_68/vars/0$": ("dense_blocks.2.conv_layers.29.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_69/vars/0$": (
            "dense_blocks.2.bn_layers.30.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_69/vars/1$": ("dense_blocks.2.bn_layers.30.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_69/vars/2$": ("dense_blocks.2.bn_layers.30.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_69/vars/3$": ("dense_blocks.2.bn_layers.30.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_69/vars/0$": ("dense_blocks.2.conv_layers.30.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_70/vars/0$": (
            "dense_blocks.2.bn_layers.31.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_70/vars/1$": ("dense_blocks.2.bn_layers.31.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_70/vars/2$": ("dense_blocks.2.bn_layers.31.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_70/vars/3$": ("dense_blocks.2.bn_layers.31.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_70/vars/0$": ("dense_blocks.2.conv_layers.31.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_71/vars/0$": (
            "dense_blocks.2.bn_layers.32.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_71/vars/1$": ("dense_blocks.2.bn_layers.32.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_71/vars/2$": ("dense_blocks.2.bn_layers.32.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_71/vars/3$": ("dense_blocks.2.bn_layers.32.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_71/vars/0$": ("dense_blocks.2.conv_layers.32.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_72/vars/0$": (
            "dense_blocks.2.bn_layers.33.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_72/vars/1$": ("dense_blocks.2.bn_layers.33.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_72/vars/2$": ("dense_blocks.2.bn_layers.33.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_72/vars/3$": ("dense_blocks.2.bn_layers.33.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_72/vars/0$": ("dense_blocks.2.conv_layers.33.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_73/vars/0$": (
            "dense_blocks.2.bn_layers.34.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_73/vars/1$": ("dense_blocks.2.bn_layers.34.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_73/vars/2$": ("dense_blocks.2.bn_layers.34.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_73/vars/3$": ("dense_blocks.2.bn_layers.34.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_73/vars/0$": ("dense_blocks.2.conv_layers.34.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_74/vars/0$": (
            "dense_blocks.2.bn_layers.35.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_74/vars/1$": ("dense_blocks.2.bn_layers.35.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_74/vars/2$": ("dense_blocks.2.bn_layers.35.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_74/vars/3$": ("dense_blocks.2.bn_layers.35.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_74/vars/0$": ("dense_blocks.2.conv_layers.35.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_75/vars/0$": (
            "dense_blocks.2.bn_layers.36.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_75/vars/1$": ("dense_blocks.2.bn_layers.36.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_75/vars/2$": ("dense_blocks.2.bn_layers.36.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_75/vars/3$": ("dense_blocks.2.bn_layers.36.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_75/vars/0$": ("dense_blocks.2.conv_layers.36.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_76/vars/0$": (
            "dense_blocks.2.bn_layers.37.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_76/vars/1$": ("dense_blocks.2.bn_layers.37.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_76/vars/2$": ("dense_blocks.2.bn_layers.37.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_76/vars/3$": ("dense_blocks.2.bn_layers.37.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_76/vars/0$": ("dense_blocks.2.conv_layers.37.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_77/vars/0$": (
            "dense_blocks.2.bn_layers.38.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_77/vars/1$": ("dense_blocks.2.bn_layers.38.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_77/vars/2$": ("dense_blocks.2.bn_layers.38.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_77/vars/3$": ("dense_blocks.2.bn_layers.38.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_77/vars/0$": ("dense_blocks.2.conv_layers.38.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_78/vars/0$": (
            "dense_blocks.2.bn_layers.39.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_78/vars/1$": ("dense_blocks.2.bn_layers.39.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_78/vars/2$": ("dense_blocks.2.bn_layers.39.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_78/vars/3$": ("dense_blocks.2.bn_layers.39.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_78/vars/0$": ("dense_blocks.2.conv_layers.39.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_79/vars/0$": (
            "dense_blocks.2.bn_layers.40.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_79/vars/1$": ("dense_blocks.2.bn_layers.40.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_79/vars/2$": ("dense_blocks.2.bn_layers.40.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_79/vars/3$": ("dense_blocks.2.bn_layers.40.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_79/vars/0$": ("dense_blocks.2.conv_layers.40.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_80/vars/0$": (
            "dense_blocks.2.bn_layers.41.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_80/vars/1$": ("dense_blocks.2.bn_layers.41.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_80/vars/2$": ("dense_blocks.2.bn_layers.41.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_80/vars/3$": ("dense_blocks.2.bn_layers.41.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_80/vars/0$": ("dense_blocks.2.conv_layers.41.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_81/vars/0$": (
            "dense_blocks.2.bn_layers.42.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_81/vars/1$": ("dense_blocks.2.bn_layers.42.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_81/vars/2$": ("dense_blocks.2.bn_layers.42.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_81/vars/3$": ("dense_blocks.2.bn_layers.42.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_81/vars/0$": ("dense_blocks.2.conv_layers.42.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_82/vars/0$": (
            "dense_blocks.2.bn_layers.43.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_82/vars/1$": ("dense_blocks.2.bn_layers.43.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_82/vars/2$": ("dense_blocks.2.bn_layers.43.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_82/vars/3$": ("dense_blocks.2.bn_layers.43.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_82/vars/0$": ("dense_blocks.2.conv_layers.43.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_83/vars/0$": (
            "dense_blocks.2.bn_layers.44.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_83/vars/1$": ("dense_blocks.2.bn_layers.44.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_83/vars/2$": ("dense_blocks.2.bn_layers.44.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_83/vars/3$": ("dense_blocks.2.bn_layers.44.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_83/vars/0$": ("dense_blocks.2.conv_layers.44.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_84/vars/0$": (
            "dense_blocks.2.bn_layers.45.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_84/vars/1$": ("dense_blocks.2.bn_layers.45.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_84/vars/2$": ("dense_blocks.2.bn_layers.45.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_84/vars/3$": ("dense_blocks.2.bn_layers.45.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_84/vars/0$": ("dense_blocks.2.conv_layers.45.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_85/vars/0$": (
            "dense_blocks.2.bn_layers.46.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_85/vars/1$": ("dense_blocks.2.bn_layers.46.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_85/vars/2$": ("dense_blocks.2.bn_layers.46.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_85/vars/3$": ("dense_blocks.2.bn_layers.46.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_85/vars/0$": ("dense_blocks.2.conv_layers.46.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_86/vars/0$": (
            "dense_blocks.2.bn_layers.47.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_86/vars/1$": ("dense_blocks.2.bn_layers.47.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_86/vars/2$": ("dense_blocks.2.bn_layers.47.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_86/vars/3$": ("dense_blocks.2.bn_layers.47.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_86/vars/0$": ("dense_blocks.2.conv_layers.47.kernel", None),
        # transition 2
        r"^layers/dense_net_backbone/layers/batch_normalization_87/vars/0$": ("transition_layers.2.bn.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_87/vars/1$": ("transition_layers.2.bn.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_87/vars/2$": ("transition_layers.2.bn.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_87/vars/3$": ("transition_layers.2.bn.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_87/vars/0$": ("transition_layers.2.conv.kernel", None),
        # dense block 3
        r"^layers/dense_net_backbone/layers/batch_normalization_88/vars/0$": ("dense_blocks.3.bn_layers.0.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_88/vars/1$": ("dense_blocks.3.bn_layers.0.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_88/vars/2$": ("dense_blocks.3.bn_layers.0.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_88/vars/3$": ("dense_blocks.3.bn_layers.0.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_88/vars/0$": ("dense_blocks.3.conv_layers.0.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_89/vars/0$": ("dense_blocks.3.bn_layers.1.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_89/vars/1$": ("dense_blocks.3.bn_layers.1.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_89/vars/2$": ("dense_blocks.3.bn_layers.1.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_89/vars/3$": ("dense_blocks.3.bn_layers.1.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_89/vars/0$": ("dense_blocks.3.conv_layers.1.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_90/vars/0$": ("dense_blocks.3.bn_layers.2.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_90/vars/1$": ("dense_blocks.3.bn_layers.2.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_90/vars/2$": ("dense_blocks.3.bn_layers.2.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_90/vars/3$": ("dense_blocks.3.bn_layers.2.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_90/vars/0$": ("dense_blocks.3.conv_layers.2.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_91/vars/0$": ("dense_blocks.3.bn_layers.3.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_91/vars/1$": ("dense_blocks.3.bn_layers.3.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_91/vars/2$": ("dense_blocks.3.bn_layers.3.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_91/vars/3$": ("dense_blocks.3.bn_layers.3.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_91/vars/0$": ("dense_blocks.3.conv_layers.3.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_92/vars/0$": ("dense_blocks.3.bn_layers.4.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_92/vars/1$": ("dense_blocks.3.bn_layers.4.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_92/vars/2$": ("dense_blocks.3.bn_layers.4.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_92/vars/3$": ("dense_blocks.3.bn_layers.4.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_92/vars/0$": ("dense_blocks.3.conv_layers.4.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_93/vars/0$": ("dense_blocks.3.bn_layers.5.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_93/vars/1$": ("dense_blocks.3.bn_layers.5.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_93/vars/2$": ("dense_blocks.3.bn_layers.5.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_93/vars/3$": ("dense_blocks.3.bn_layers.5.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_93/vars/0$": ("dense_blocks.3.conv_layers.5.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_94/vars/0$": ("dense_blocks.3.bn_layers.6.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_94/vars/1$": ("dense_blocks.3.bn_layers.6.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_94/vars/2$": ("dense_blocks.3.bn_layers.6.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_94/vars/3$": ("dense_blocks.3.bn_layers.6.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_94/vars/0$": ("dense_blocks.3.conv_layers.6.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_95/vars/0$": ("dense_blocks.3.bn_layers.7.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_95/vars/1$": ("dense_blocks.3.bn_layers.7.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_95/vars/2$": ("dense_blocks.3.bn_layers.7.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_95/vars/3$": ("dense_blocks.3.bn_layers.7.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_95/vars/0$": ("dense_blocks.3.conv_layers.7.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_96/vars/0$": ("dense_blocks.3.bn_layers.8.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_96/vars/1$": ("dense_blocks.3.bn_layers.8.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_96/vars/2$": ("dense_blocks.3.bn_layers.8.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_96/vars/3$": ("dense_blocks.3.bn_layers.8.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_96/vars/0$": ("dense_blocks.3.conv_layers.8.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_97/vars/0$": ("dense_blocks.3.bn_layers.9.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_97/vars/1$": ("dense_blocks.3.bn_layers.9.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_97/vars/2$": ("dense_blocks.3.bn_layers.9.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_97/vars/3$": ("dense_blocks.3.bn_layers.9.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_97/vars/0$": ("dense_blocks.3.conv_layers.9.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_98/vars/0$": (
            "dense_blocks.3.bn_layers.10.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_98/vars/1$": ("dense_blocks.3.bn_layers.10.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_98/vars/2$": ("dense_blocks.3.bn_layers.10.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_98/vars/3$": ("dense_blocks.3.bn_layers.10.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_98/vars/0$": ("dense_blocks.3.conv_layers.10.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_99/vars/0$": (
            "dense_blocks.3.bn_layers.11.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_99/vars/1$": ("dense_blocks.3.bn_layers.11.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_99/vars/2$": ("dense_blocks.3.bn_layers.11.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_99/vars/3$": ("dense_blocks.3.bn_layers.11.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_99/vars/0$": ("dense_blocks.3.conv_layers.11.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_100/vars/0$": (
            "dense_blocks.3.bn_layers.12.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_100/vars/1$": (
            "dense_blocks.3.bn_layers.12.bias",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_100/vars/2$": (
            "dense_blocks.3.bn_layers.12.mean",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_100/vars/3$": ("dense_blocks.3.bn_layers.12.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_100/vars/0$": ("dense_blocks.3.conv_layers.12.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_101/vars/0$": (
            "dense_blocks.3.bn_layers.13.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_101/vars/1$": (
            "dense_blocks.3.bn_layers.13.bias",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_101/vars/2$": (
            "dense_blocks.3.bn_layers.13.mean",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_101/vars/3$": ("dense_blocks.3.bn_layers.13.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_101/vars/0$": ("dense_blocks.3.conv_layers.13.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_102/vars/0$": (
            "dense_blocks.3.bn_layers.14.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_102/vars/1$": (
            "dense_blocks.3.bn_layers.14.bias",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_102/vars/2$": (
            "dense_blocks.3.bn_layers.14.mean",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_102/vars/3$": ("dense_blocks.3.bn_layers.14.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_102/vars/0$": ("dense_blocks.3.conv_layers.14.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_103/vars/0$": (
            "dense_blocks.3.bn_layers.15.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_103/vars/1$": (
            "dense_blocks.3.bn_layers.15.bias",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_103/vars/2$": (
            "dense_blocks.3.bn_layers.15.mean",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_103/vars/3$": ("dense_blocks.3.bn_layers.15.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_103/vars/0$": ("dense_blocks.3.conv_layers.15.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_104/vars/0$": (
            "dense_blocks.3.bn_layers.16.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_104/vars/1$": (
            "dense_blocks.3.bn_layers.16.bias",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_104/vars/2$": (
            "dense_blocks.3.bn_layers.16.mean",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_104/vars/3$": ("dense_blocks.3.bn_layers.16.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_104/vars/0$": ("dense_blocks.3.conv_layers.16.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_105/vars/0$": (
            "dense_blocks.3.bn_layers.17.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_105/vars/1$": (
            "dense_blocks.3.bn_layers.17.bias",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_105/vars/2$": (
            "dense_blocks.3.bn_layers.17.mean",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_105/vars/3$": ("dense_blocks.3.bn_layers.17.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_105/vars/0$": ("dense_blocks.3.conv_layers.17.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_106/vars/0$": (
            "dense_blocks.3.bn_layers.18.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_106/vars/1$": (
            "dense_blocks.3.bn_layers.18.bias",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_106/vars/2$": (
            "dense_blocks.3.bn_layers.18.mean",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_106/vars/3$": ("dense_blocks.3.bn_layers.18.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_106/vars/0$": ("dense_blocks.3.conv_layers.18.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_107/vars/0$": (
            "dense_blocks.3.bn_layers.19.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_107/vars/1$": (
            "dense_blocks.3.bn_layers.19.bias",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_107/vars/2$": (
            "dense_blocks.3.bn_layers.19.mean",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_107/vars/3$": ("dense_blocks.3.bn_layers.19.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_107/vars/0$": ("dense_blocks.3.conv_layers.19.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_108/vars/0$": (
            "dense_blocks.3.bn_layers.20.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_108/vars/1$": (
            "dense_blocks.3.bn_layers.20.bias",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_108/vars/2$": (
            "dense_blocks.3.bn_layers.20.mean",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_108/vars/3$": ("dense_blocks.3.bn_layers.20.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_108/vars/0$": ("dense_blocks.3.conv_layers.20.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_109/vars/0$": (
            "dense_blocks.3.bn_layers.21.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_109/vars/1$": (
            "dense_blocks.3.bn_layers.21.bias",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_109/vars/2$": (
            "dense_blocks.3.bn_layers.21.mean",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_109/vars/3$": ("dense_blocks.3.bn_layers.21.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_109/vars/0$": ("dense_blocks.3.conv_layers.21.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_110/vars/0$": (
            "dense_blocks.3.bn_layers.22.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_110/vars/1$": (
            "dense_blocks.3.bn_layers.22.bias",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_110/vars/2$": (
            "dense_blocks.3.bn_layers.22.mean",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_110/vars/3$": ("dense_blocks.3.bn_layers.22.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_110/vars/0$": ("dense_blocks.3.conv_layers.22.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_111/vars/0$": (
            "dense_blocks.3.bn_layers.23.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_111/vars/1$": (
            "dense_blocks.3.bn_layers.23.bias",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_111/vars/2$": (
            "dense_blocks.3.bn_layers.23.mean",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_111/vars/3$": ("dense_blocks.3.bn_layers.23.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_111/vars/0$": ("dense_blocks.3.conv_layers.23.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_112/vars/0$": (
            "dense_blocks.3.bn_layers.24.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_112/vars/1$": (
            "dense_blocks.3.bn_layers.24.bias",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_112/vars/2$": (
            "dense_blocks.3.bn_layers.24.mean",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_112/vars/3$": ("dense_blocks.3.bn_layers.24.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_112/vars/0$": ("dense_blocks.3.conv_layers.24.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_113/vars/0$": (
            "dense_blocks.3.bn_layers.25.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_113/vars/1$": (
            "dense_blocks.3.bn_layers.25.bias",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_113/vars/2$": (
            "dense_blocks.3.bn_layers.25.mean",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_113/vars/3$": ("dense_blocks.3.bn_layers.25.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_113/vars/0$": ("dense_blocks.3.conv_layers.25.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_114/vars/0$": (
            "dense_blocks.3.bn_layers.26.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_114/vars/1$": (
            "dense_blocks.3.bn_layers.26.bias",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_114/vars/2$": (
            "dense_blocks.3.bn_layers.26.mean",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_114/vars/3$": ("dense_blocks.3.bn_layers.26.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_114/vars/0$": ("dense_blocks.3.conv_layers.26.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_115/vars/0$": (
            "dense_blocks.3.bn_layers.27.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_115/vars/1$": (
            "dense_blocks.3.bn_layers.27.bias",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_115/vars/2$": (
            "dense_blocks.3.bn_layers.27.mean",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_115/vars/3$": ("dense_blocks.3.bn_layers.27.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_115/vars/0$": ("dense_blocks.3.conv_layers.27.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_116/vars/0$": (
            "dense_blocks.3.bn_layers.28.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_116/vars/1$": (
            "dense_blocks.3.bn_layers.28.bias",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_116/vars/2$": (
            "dense_blocks.3.bn_layers.28.mean",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_116/vars/3$": ("dense_blocks.3.bn_layers.28.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_116/vars/0$": ("dense_blocks.3.conv_layers.28.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_117/vars/0$": (
            "dense_blocks.3.bn_layers.29.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_117/vars/1$": (
            "dense_blocks.3.bn_layers.29.bias",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_117/vars/2$": (
            "dense_blocks.3.bn_layers.29.mean",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_117/vars/3$": ("dense_blocks.3.bn_layers.29.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_117/vars/0$": ("dense_blocks.3.conv_layers.29.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_118/vars/0$": (
            "dense_blocks.3.bn_layers.30.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_118/vars/1$": (
            "dense_blocks.3.bn_layers.30.bias",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_118/vars/2$": (
            "dense_blocks.3.bn_layers.30.mean",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_118/vars/3$": ("dense_blocks.3.bn_layers.30.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_118/vars/0$": ("dense_blocks.3.conv_layers.30.kernel", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_119/vars/0$": (
            "dense_blocks.3.bn_layers.31.scale",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_119/vars/1$": (
            "dense_blocks.3.bn_layers.31.bias",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_119/vars/2$": (
            "dense_blocks.3.bn_layers.31.mean",
            None,
        ),
        r"^layers/dense_net_backbone/layers/batch_normalization_119/vars/3$": ("dense_blocks.3.bn_layers.31.var", None),
        r"^layers/dense_net_backbone/layers/conv2d_119/vars/0$": ("dense_blocks.3.conv_layers.31.kernel", None),
        # final bn
        r"^layers/dense_net_backbone/layers/batch_normalization_120/vars/0$": ("final_bn.scale", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_120/vars/1$": ("final_bn.bias", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_120/vars/2$": ("final_bn.mean", None),
        r"^layers/dense_net_backbone/layers/batch_normalization_120/vars/3$": ("final_bn.var", None),
        # linear
        r"^layers/dense/vars/0$": ("linear.kernel", None),
        r"^layers/dense/vars/1$": ("linear.bias", None),
    }


def _st_key_to_jax_key(mapping, source_key):
    """Map a h5 key to exactly one JAX key & transform, else warn/error."""
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


def create_model_from_h5(
    file_dir: str,
    cfg: model_lib.ModelCfg,
    mesh: jax.sharding.Mesh | None = None,
) -> model_lib.DenseNet:
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
    state_dict = abs_state.to_pure_dict()

    mapping = _get_key_and_transform_mapping()
    for st_key, tensor in tensor_dict.items():
        jax_key, transform = _st_key_to_jax_key(mapping, st_key)
        if jax_key is None:
            continue
        keys = [_stoi(k) for k in jax_key.split(".")]
        _assign_weights(keys, tensor, state_dict, st_key, transform)

    if mesh is not None:
        sharding = nnx.get_named_sharding(abs_state, mesh).to_pure_dict()
        state_dict = jax.device_put(state_dict, sharding)
    else:
        state_dict = jax.device_put(state_dict, jax.devices()[0])

    return nnx.merge(graph_def, state_dict)
