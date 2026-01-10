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

"""Qwen3-VL model implementation in JAX/Flax NNX."""

from .modeling import (
    # Configs
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
    # Vision components
    Qwen3VLPatchEmbed,
    Qwen3VLVisionMLP,
    Qwen3VLVisionAttention,
    Qwen3VLVisionBlock,
    Qwen3VLPatchMerger,
    Qwen3VLVisionModel,
    # Text components
    RMSNorm,
    Qwen3VLMLP,
    Qwen3VLAttention,
    Qwen3VLDecoderLayer,
    Qwen3VLTextModel,
    # Cache
    LayerCache,
    Cache,
    init_cache,
    # Full model
    Qwen3VLModel,
    Qwen3VLForConditionalGeneration,
    # Helpers
    forward,
    make_causal_mask,
)

from .params import (
    create_model_from_safe_tensors,
    get_pretrained_config,
)

__all__ = [
    # Configs
    "Qwen3VLConfig",
    "Qwen3VLTextConfig",
    "Qwen3VLVisionConfig",
    # Vision
    "Qwen3VLPatchEmbed",
    "Qwen3VLVisionMLP",
    "Qwen3VLVisionAttention",
    "Qwen3VLVisionBlock",
    "Qwen3VLPatchMerger",
    "Qwen3VLVisionModel",
    # Text
    "RMSNorm",
    "Qwen3VLMLP",
    "Qwen3VLAttention",
    "Qwen3VLDecoderLayer",
    "Qwen3VLTextModel",
    # Cache
    "LayerCache",
    "Cache",
    "init_cache",
    # Full model
    "Qwen3VLModel",
    "Qwen3VLForConditionalGeneration",
    # Helpers
    "forward",
    "make_causal_mask",
    # Params
    "create_model_from_safe_tensors",
    "get_pretrained_config",
]
