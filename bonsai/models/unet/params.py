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

import jax
from flax import nnx

from bonsai.models.unet import modeling as model_lib


def create_model(
    cfg: model_lib.ModelCfg,
    rngs: nnx.Rngs,
    mesh: jax.sharding.Mesh | None = None,
) -> model_lib.UNet:
    """
    Create a U-Net model with initialized parameters.

    Returns:
      A flax.nnx.Module instance with random parameters.
    """
    model = model_lib.UNet(cfg, rngs=rngs)

    if mesh is not None:
        # This part is for distributed execution, if needed.
        graph_def, state = nnx.split(model)
        sharding = nnx.get_named_sharding(model, mesh)
        state = jax.device_put(state, sharding)
        return nnx.merge(graph_def, state)
    else:
        return model
