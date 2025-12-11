# Copyright 2025 The JAX Authors.
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

"""Parameter utilities for Mamba2 models."""

import jax
from flax import nnx

from bonsai.models.mamba2 import modeling


def create_random_model(cfg: modeling.Mamba2Config, seed: int = 0) -> modeling.Mamba2ForCausalLM:
    """Create a randomly initialized Mamba2ForCausalLM.

    Args:
        cfg: Mamba2Config for the model.
        seed: Random seed for initialization.

    Returns:
        Randomly initialized Mamba2ForCausalLM.
    """
    return modeling.Mamba2ForCausalLM(cfg, rngs=nnx.Rngs(seed))


def create_random_forecaster(
    input_dim: int,
    d_model: int = 768,
    n_layers: int = 4,
    output_dim: int = 1,
    forecast_horizon: int = 24,
    seed: int = 0,
    **kwargs,
) -> modeling.Mamba2Forecaster:
    """Create a randomly initialized Mamba2Forecaster.

    Args:
        input_dim: Number of input features per timestep.
        d_model: Hidden dimension of the model.
        n_layers: Number of Mamba2 layers.
        output_dim: Number of output features per timestep.
        forecast_horizon: Number of future timesteps to predict.
        seed: Random seed for initialization.
        **kwargs: Additional arguments passed to Mamba2Forecaster.

    Returns:
        Randomly initialized Mamba2Forecaster.
    """
    return modeling.Mamba2Forecaster(
        input_dim=input_dim,
        d_model=d_model,
        n_layers=n_layers,
        output_dim=output_dim,
        forecast_horizon=forecast_horizon,
        rngs=nnx.Rngs(seed),
        **kwargs,
    )


def count_parameters(model: nnx.Module) -> int:
    """Count the total number of trainable parameters in a model.

    Args:
        model: NNX module to count parameters for.

    Returns:
        Total number of parameters.
    """
    _____graphdef, state = nnx.split(model)
    params = state.filter(nnx.Param)
    return sum(p.size for p in jax.tree.leaves(params))
