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

"""Wan2.1-T2V-1.3B: Text-to-Video Diffusion Transformer.

This module implements the Wan2.1-T2V-1.3B model, a 1.3B parameter diffusion transformer
for text-to-video generation using Flow Matching framework.

Example usage:
    >>> from bonsai.models.wan2 import modeling, params
    >>> from flax import nnx
    >>>
    >>> # Create model configuration
    >>> config = modeling.ModelConfig.wan2_1_1_3b()
    >>>
    >>> # Initialize model (with random weights)
    >>> model = modeling.Wan2DiT(config, rngs=nnx.Rngs(params=0))
    >>>
    >>> # Or load from checkpoint
    >>> model = params.create_model_from_safe_tensors(
    ...     "/path/to/checkpoint",
    ...     config,
    ...     load_transformer_only=True
    ... )
    >>>
    >>> # Generate video from text embeddings
    >>> latents = modeling.generate_video(
    ...     model=model,
    ...     text_embeds=text_embeddings,
    ...     num_frames=81,
    ...     guidance_scale=5.5
    ... )
"""

from bonsai.models.wan2 import modeling, params

__all__ = ["modeling", "params"]
