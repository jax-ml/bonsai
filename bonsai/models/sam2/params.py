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

"""Utils for loading and converting SAM 2 safetensors weights."""

import logging
import re

import jax
import numpy as np
import safetensors.flax as safetensors
from flax import nnx

from bonsai.models.sam2 import modeling as model_lib

PYTORCH_TO_JAX_CONV_2D_KERNEL = (2, 3, 1, 0)  # (C_out, C_in, kH, kW) -> (kH, kW, C_in, C_out)
PYTORCH_TO_JAX_CONV_2D_TRANSPOSE_KERNEL = (2, 3, 0, 1)  # (C_in, C_out, kH, kW) -> (kH, kW, C_in, C_out)
PYTORCH_TO_JAX_LINEAR = (1, 0)  # (D_in, D_out) -> (D_out, D_in)


def _get_key_and_transform_mapping():
    # Maps safetensor keys → (JAX nnx key template, no transform needed)

    image_encoder_mapping = {
        # === Patch Embedding ===
        r"^image_encoder\.trunk\.patch_embed\.proj\.weight$": (
            "image_encoder.trunk.patch_embed.proj.kernel",
            (PYTORCH_TO_JAX_CONV_2D_KERNEL, None),
        ),
        r"^image_encoder\.trunk\.patch_embed\.proj\.bias$": (
            "image_encoder.trunk.patch_embed.proj.bias",
            None,
        ),
        # === Positional Embeddings ===
        r"^image_encoder\.trunk\.pos_embed$": ("image_encoder.trunk.pos_embed", None),
        r"^image_encoder\.trunk\.pos_embed_window$": (
            "image_encoder.trunk.pos_embed_window",
            None,
        ),
        # === Transformer Blocks ===
        r"^image_encoder\.trunk\.blocks\.([0-9]+)\.norm1\.weight$": (
            r"image_encoder.trunk.blocks.\1.norm1.scale",
            None,
        ),
        r"^image_encoder\.trunk\.blocks\.([0-9]+)\.norm1\.bias$": (
            r"image_encoder.trunk.blocks.\1.norm1.bias",
            None,
        ),
        r"^image_encoder\.trunk\.blocks\.([0-9]+)\.attn\.qkv\.weight$": (
            r"image_encoder.trunk.blocks.\1.attn.qkv.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^image_encoder\.trunk\.blocks\.([0-9]+)\.attn\.qkv\.bias$": (
            r"image_encoder.trunk.blocks.\1.attn.qkv.bias",
            None,
        ),
        r"^image_encoder\.trunk\.blocks\.([0-9]+)\.attn\.proj\.weight$": (
            r"image_encoder.trunk.blocks.\1.attn.proj.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^image_encoder\.trunk\.blocks\.([0-9]+)\.attn\.proj\.bias$": (
            r"image_encoder.trunk.blocks.\1.attn.proj.bias",
            None,
        ),
        r"^image_encoder\.trunk\.blocks\.([0-9]+)\.norm2\.weight$": (
            r"image_encoder.trunk.blocks.\1.norm2.scale",
            None,
        ),
        r"^image_encoder\.trunk\.blocks\.([0-9]+)\.norm2\.bias$": (
            r"image_encoder.trunk.blocks.\1.norm2.bias",
            None,
        ),
        r"^image_encoder\.trunk\.blocks\.([0-9]+)\.mlp\.layers\.0\.weight$": (
            r"image_encoder.trunk.blocks.\1.mlp.layers.0.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^image_encoder\.trunk\.blocks\.([0-9]+)\.mlp\.layers\.0\.bias$": (
            r"image_encoder.trunk.blocks.\1.mlp.layers.0.bias",
            None,
        ),
        r"^image_encoder\.trunk\.blocks\.([0-9]+)\.mlp\.layers\.1\.weight$": (
            r"image_encoder.trunk.blocks.\1.mlp.layers.1.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^image_encoder\.trunk\.blocks\.([0-9]+)\.mlp\.layers\.1\.bias$": (
            r"image_encoder.trunk.blocks.\1.mlp.layers.1.bias",
            None,
        ),
        r"^image_encoder\.trunk\.blocks\.([0-9]+)\.proj\.weight$": (
            r"image_encoder.trunk.blocks.\1.proj.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^image_encoder\.trunk\.blocks\.([0-9]+)\.proj\.bias$": (
            r"image_encoder.trunk.blocks.\1.proj.bias",
            None,
        ),
        # === Neck Convolutions ===
        r"^image_encoder\.neck\.convs\.([0-3])\.conv\.weight$": (
            r"image_encoder.neck.convs.\1.kernel",
            (PYTORCH_TO_JAX_CONV_2D_KERNEL, None),
        ),
        r"^image_encoder\.neck\.convs\.([0-3])\.conv\.bias$": (
            r"image_encoder.neck.convs.\1.bias",
            None,
        ),
    }
    memory_attention_mapping = {
        # === Norms ===
        r"^memory_attention\.layers\.([0-3])\.norm1\.weight$": (
            r"memory_attention.layers.\1.norm1.scale",
            None,
        ),
        r"^memory_attention\.layers\.([0-3])\.norm1\.bias$": (
            r"memory_attention.layers.\1.norm1.bias",
            None,
        ),
        r"^memory_attention\.layers\.([0-3])\.norm2\.weight$": (
            r"memory_attention.layers.\1.norm2.scale",
            None,
        ),
        r"^memory_attention\.layers\.([0-3])\.norm2\.bias$": (
            r"memory_attention.layers.\1.norm2.bias",
            None,
        ),
        r"^memory_attention\.layers\.([0-3])\.norm3\.weight$": (
            r"memory_attention.layers.\1.norm3.scale",
            None,
        ),
        r"^memory_attention\.layers\.([0-3])\.norm3\.bias$": (
            r"memory_attention.layers.\1.norm3.bias",
            None,
        ),
        # === Feedforward MLP ===
        r"^memory_attention\.layers\.([0-3])\.linear1\.weight$": (
            r"memory_attention.layers.\1.linear1.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^memory_attention\.layers\.([0-3])\.linear1\.bias$": (
            r"memory_attention.layers.\1.linear1.bias",
            None,
        ),
        r"^memory_attention\.layers\.([0-3])\.linear2\.weight$": (
            r"memory_attention.layers.\1.linear2.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^memory_attention\.layers\.([0-3])\.linear2\.bias$": (
            r"memory_attention.layers.\1.linear2.bias",
            None,
        ),
        # === Self Attention ===
        r"^memory_attention\.layers\.([0-3])\.self_attn\.q_proj\.weight$": (
            r"memory_attention.layers.\1.self_attn.q_proj.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^memory_attention\.layers\.([0-3])\.self_attn\.q_proj\.bias$": (
            r"memory_attention.layers.\1.self_attn.q_proj.bias",
            None,
        ),
        r"^memory_attention\.layers\.([0-3])\.self_attn\.k_proj\.weight$": (
            r"memory_attention.layers.\1.self_attn.k_proj.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^memory_attention\.layers\.([0-3])\.self_attn\.k_proj\.bias$": (
            r"memory_attention.layers.\1.self_attn.k_proj.bias",
            None,
        ),
        r"^memory_attention\.layers\.([0-3])\.self_attn\.v_proj\.weight$": (
            r"memory_attention.layers.\1.self_attn.v_proj.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^memory_attention\.layers\.([0-3])\.self_attn\.v_proj\.bias$": (
            r"memory_attention.layers.\1.self_attn.v_proj.bias",
            None,
        ),
        r"^memory_attention\.layers\.([0-3])\.self_attn\.out_proj\.weight$": (
            r"memory_attention.layers.\1.self_attn.out_proj.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^memory_attention\.layers\.([0-3])\.self_attn\.out_proj\.bias$": (
            r"memory_attention.layers.\1.self_attn.out_proj.bias",
            None,
        ),
        # === Cross Attention ===
        r"^memory_attention\.layers\.([0-3])\.cross_attn_image\.q_proj\.weight$": (
            r"memory_attention.layers.\1.cross_attn_image.q_proj.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^memory_attention\.layers\.([0-3])\.cross_attn_image\.q_proj\.bias$": (
            r"memory_attention.layers.\1.cross_attn_image.q_proj.bias",
            None,
        ),
        r"^memory_attention\.layers\.([0-3])\.cross_attn_image\.k_proj\.weight$": (
            r"memory_attention.layers.\1.cross_attn_image.k_proj.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^memory_attention\.layers\.([0-3])\.cross_attn_image\.k_proj\.bias$": (
            r"memory_attention.layers.\1.cross_attn_image.k_proj.bias",
            None,
        ),
        r"^memory_attention\.layers\.([0-3])\.cross_attn_image\.v_proj\.weight$": (
            r"memory_attention.layers.\1.cross_attn_image.v_proj.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^memory_attention\.layers\.([0-3])\.cross_attn_image\.v_proj\.bias$": (
            r"memory_attention.layers.\1.cross_attn_image.v_proj.bias",
            None,
        ),
        r"^memory_attention\.layers\.([0-3])\.cross_attn_image\.out_proj\.weight$": (
            r"memory_attention.layers.\1.cross_attn_image.out_proj.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^memory_attention\.layers\.([0-3])\.cross_attn_image\.out_proj\.bias$": (
            r"memory_attention.layers.\1.cross_attn_image.out_proj.bias",
            None,
        ),
        # === Final Norm ===
        r"^memory_attention\.norm\.weight$": ("memory_attention.norm.scale", None),
        r"^memory_attention\.norm\.bias$": ("memory_attention.norm.bias", None),
    }

    memory_encoder_mapping = {
        # === MaskDownSampler Encoder ===
        # === Conv2d layers (even indices: 0, 3, 6, 9, 12) ===
        r"^memory_encoder\.mask_downsampler\.encoder\.(0|3|6|9|12)\.weight$": (
            r"memory_encoder.mask_downsampler.encoder.layers.\1.kernel",
            (PYTORCH_TO_JAX_CONV_2D_KERNEL, None),
        ),
        r"^memory_encoder\.mask_downsampler\.encoder\.(0|3|6|9|12)\.bias$": (
            r"memory_encoder.mask_downsampler.encoder.layers.\1.bias",
            None,
        ),
        # === LayerNorm layers (odd indices: 1, 4, 7, 10) ===
        r"^memory_encoder\.mask_downsampler\.encoder\.(1|4|7|10)\.weight$": (
            r"memory_encoder.mask_downsampler.encoder.layers.\1.weight",
            None,
        ),
        r"^memory_encoder\.mask_downsampler\.encoder\.(1|4|7|10)\.bias$": (
            r"memory_encoder.mask_downsampler.encoder.layers.\1.bias",
            None,
        ),
        # === Pixel Feature Projection ===
        r"^memory_encoder\.pix_feat_proj\.weight$": (
            "memory_encoder.pix_feat_proj.kernel",
            (PYTORCH_TO_JAX_CONV_2D_KERNEL, None),
        ),
        r"^memory_encoder\.pix_feat_proj\.bias$": (
            "memory_encoder.pix_feat_proj.bias",
            None,
        ),
        # === Fuser: ConvNeXt-like layers ===
        r"^memory_encoder\.fuser\.layers\.([0-1])\.gamma$": (
            r"memory_encoder.fuser.layers.\1.gamma",
            None,
        ),
        r"^memory_encoder\.fuser\.layers\.([0-1])\.weight$": (
            r"memory_encoder.fuser.layers.\1.gamma",
            None,
        ),
        r"^memory_encoder\.fuser\.layers\.([0-1])\.dwconv\.weight$": (
            r"memory_encoder.fuser.layers.\1.dwconv.kernel",
            (PYTORCH_TO_JAX_CONV_2D_KERNEL, None),
        ),
        r"^memory_encoder\.fuser\.layers\.([0-1])\.dwconv\.bias$": (
            r"memory_encoder.fuser.layers.\1.dwconv.bias",
            None,
        ),
        r"^memory_encoder\.fuser\.layers\.([0-1])\.norm\.weight$": (
            r"memory_encoder.fuser.layers.\1.norm.weight",
            None,
        ),
        r"^memory_encoder\.fuser\.layers\.([0-1])\.norm\.bias$": (
            r"memory_encoder.fuser.layers.\1.norm.bias",
            None,
        ),
        r"^memory_encoder\.fuser\.layers\.([0-1])\.pwconv1\.weight$": (
            r"memory_encoder.fuser.layers.\1.pwconv1.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^memory_encoder\.fuser\.layers\.([0-1])\.pwconv1\.bias$": (
            r"memory_encoder.fuser.layers.\1.pwconv1.bias",
            None,
        ),
        r"^memory_encoder\.fuser\.layers\.([0-1])\.pwconv2\.weight$": (
            r"memory_encoder.fuser.layers.\1.pwconv2.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^memory_encoder\.fuser\.layers\.([0-1])\.pwconv2\.bias$": (
            r"memory_encoder.fuser.layers.\1.pwconv2.bias",
            None,
        ),
        # === Final Output Projection ===
        r"^memory_encoder\.out_proj\.weight$": (
            "memory_encoder.out_proj.kernel",
            (PYTORCH_TO_JAX_CONV_2D_KERNEL, None),
        ),
        r"^memory_encoder\.out_proj\.bias$": ("memory_encoder.out_proj.bias", None),
    }
    sam_mask_decoder = {
        # Self-attention
        r"^sam_mask_decoder\.transformer\.layers\.([0-9]+)\.self_attn\.q_proj\.weight$": (
            r"sam_mask_decoder.transformer.blocks.\1.self_attn.q_proj.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^sam_mask_decoder\.transformer\.layers\.([0-9]+)\.self_attn\.q_proj\.bias$": (
            r"sam_mask_decoder.transformer.blocks.\1.self_attn.q_proj.bias",
            None,
        ),
        r"^sam_mask_decoder\.transformer\.layers\.([0-9]+)\.self_attn\.k_proj\.weight$": (
            r"sam_mask_decoder.transformer.blocks.\1.self_attn.k_proj.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^sam_mask_decoder\.transformer\.layers\.([0-9]+)\.self_attn\.k_proj\.bias$": (
            r"sam_mask_decoder.transformer.blocks.\1.self_attn.k_proj.bias",
            None,
        ),
        r"^sam_mask_decoder\.transformer\.layers\.([0-9]+)\.self_attn\.v_proj\.weight$": (
            r"sam_mask_decoder.transformer.blocks.\1.self_attn.v_proj.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^sam_mask_decoder\.transformer\.layers\.([0-9]+)\.self_attn\.v_proj\.bias$": (
            r"sam_mask_decoder.transformer.blocks.\1.self_attn.v_proj.bias",
            None,
        ),
        r"^sam_mask_decoder\.transformer\.layers\.([0-9]+)\.self_attn\.out_proj\.weight$": (
            r"sam_mask_decoder.transformer.blocks.\1.self_attn.out_proj.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^sam_mask_decoder\.transformer\.layers\.([0-9]+)\.self_attn\.out_proj\.bias$": (
            r"sam_mask_decoder.transformer.blocks.\1.self_attn.out_proj.bias",
            None,
        ),
        # Norms
        r"^sam_mask_decoder\.transformer\.layers\.([0-9]+)\.norm([1234])\.weight$": (
            r"sam_mask_decoder.transformer.blocks.\1.norm\2.scale",
            None,
        ),
        r"^sam_mask_decoder\.transformer\.layers\.([0-9]+)\.norm([1234])\.bias$": (
            r"sam_mask_decoder.transformer.blocks.\1.norm\2.bias",
            None,
        ),
        # MLP
        r"^sam_mask_decoder\.transformer\.layers\.([0-9]+)\.mlp\.layers\.([01])\.weight$": (
            r"sam_mask_decoder.transformer.blocks.\1.mlp.layers.\2.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^sam_mask_decoder\.transformer\.layers\.([0-9]+)\.mlp\.layers\.([01])\.bias$": (
            r"sam_mask_decoder.transformer.blocks.\1.mlp.layers.\2.bias",
            None,
        ),
        # Cross-attn: token → image
        r"^sam_mask_decoder\.transformer\.layers\.([0-9]+)\.cross_attn_token_to_image\.q_proj\.weight$": (
            r"sam_mask_decoder.transformer.blocks.\1.cross_attn_token_to_image.q_proj.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^sam_mask_decoder\.transformer\.layers\.([0-9]+)\.cross_attn_token_to_image\.q_proj\.bias$": (
            r"sam_mask_decoder.transformer.blocks.\1.cross_attn_token_to_image.q_proj.bias",
            None,
        ),
        r"^sam_mask_decoder\.transformer\.layers\.([0-9]+)\.cross_attn_token_to_image\.k_proj\.weight$": (
            r"sam_mask_decoder.transformer.blocks.\1.cross_attn_token_to_image.k_proj.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^sam_mask_decoder\.transformer\.layers\.([0-9]+)\.cross_attn_token_to_image\.k_proj\.bias$": (
            r"sam_mask_decoder.transformer.blocks.\1.cross_attn_token_to_image.k_proj.bias",
            None,
        ),
        r"^sam_mask_decoder\.transformer\.layers\.([0-9]+)\.cross_attn_token_to_image\.v_proj\.weight$": (
            r"sam_mask_decoder.transformer.blocks.\1.cross_attn_token_to_image.v_proj.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^sam_mask_decoder\.transformer\.layers\.([0-9]+)\.cross_attn_token_to_image\.v_proj\.bias$": (
            r"sam_mask_decoder.transformer.blocks.\1.cross_attn_token_to_image.v_proj.bias",
            None,
        ),
        r"^sam_mask_decoder\.transformer\.layers\.([0-9]+)\.cross_attn_token_to_image\.out_proj\.weight$": (
            r"sam_mask_decoder.transformer.blocks.\1.cross_attn_token_to_image.out_proj.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^sam_mask_decoder\.transformer\.layers\.([0-9]+)\.cross_attn_token_to_image\.out_proj\.bias$": (
            r"sam_mask_decoder.transformer.blocks.\1.cross_attn_token_to_image.out_proj.bias",
            None,
        ),
        # Cross-attn: image → token
        r"^sam_mask_decoder\.transformer\.layers\.([0-9]+)\.cross_attn_image_to_token\.q_proj\.weight$": (
            r"sam_mask_decoder.transformer.blocks.\1.cross_attn_image_to_token.q_proj.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^sam_mask_decoder\.transformer\.layers\.([0-9]+)\.cross_attn_image_to_token\.q_proj\.bias$": (
            r"sam_mask_decoder.transformer.blocks.\1.cross_attn_image_to_token.q_proj.bias",
            None,
        ),
        r"^sam_mask_decoder\.transformer\.layers\.([0-9]+)\.cross_attn_image_to_token\.k_proj\.weight$": (
            r"sam_mask_decoder.transformer.blocks.\1.cross_attn_image_to_token.k_proj.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^sam_mask_decoder\.transformer\.layers\.([0-9]+)\.cross_attn_image_to_token\.k_proj\.bias$": (
            r"sam_mask_decoder.transformer.blocks.\1.cross_attn_image_to_token.k_proj.bias",
            None,
        ),
        r"^sam_mask_decoder\.transformer\.layers\.([0-9]+)\.cross_attn_image_to_token\.v_proj\.weight$": (
            r"sam_mask_decoder.transformer.blocks.\1.cross_attn_image_to_token.v_proj.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^sam_mask_decoder\.transformer\.layers\.([0-9]+)\.cross_attn_image_to_token\.v_proj\.bias$": (
            r"sam_mask_decoder.transformer.blocks.\1.cross_attn_image_to_token.v_proj.bias",
            None,
        ),
        r"^sam_mask_decoder\.transformer\.layers\.([0-9]+)\.cross_attn_image_to_token\.out_proj\.weight$": (
            r"sam_mask_decoder.transformer.blocks.\1.cross_attn_image_to_token.out_proj.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^sam_mask_decoder\.transformer\.layers\.([0-9]+)\.cross_attn_image_to_token\.out_proj\.bias$": (
            r"sam_mask_decoder.transformer.blocks.\1.cross_attn_image_to_token.out_proj.bias",
            None,
        ),
        # Final cross-attn
        r"^sam_mask_decoder\.transformer\.final_attn_token_to_image\.q_proj\.weight$": (
            "sam_mask_decoder.transformer.final_attn_token_to_image.q_proj.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^sam_mask_decoder\.transformer\.final_attn_token_to_image\.q_proj\.bias$": (
            "sam_mask_decoder.transformer.final_attn_token_to_image.q_proj.bias",
            None,
        ),
        r"^sam_mask_decoder\.transformer\.final_attn_token_to_image\.k_proj\.weight$": (
            "sam_mask_decoder.transformer.final_attn_token_to_image.k_proj.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^sam_mask_decoder\.transformer\.final_attn_token_to_image\.k_proj\.bias$": (
            "sam_mask_decoder.transformer.final_attn_token_to_image.k_proj.bias",
            None,
        ),
        r"^sam_mask_decoder\.transformer\.final_attn_token_to_image\.v_proj\.weight$": (
            "sam_mask_decoder.transformer.final_attn_token_to_image.v_proj.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^sam_mask_decoder\.transformer\.final_attn_token_to_image\.v_proj\.bias$": (
            "sam_mask_decoder.transformer.final_attn_token_to_image.v_proj.bias",
            None,
        ),
        r"^sam_mask_decoder\.transformer\.final_attn_token_to_image\.out_proj\.weight$": (
            "sam_mask_decoder.transformer.final_attn_token_to_image.out_proj.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^sam_mask_decoder\.transformer\.final_attn_token_to_image\.out_proj\.bias$": (
            "sam_mask_decoder.transformer.final_attn_token_to_image.out_proj.bias",
            None,
        ),
        # Final norm
        r"^sam_mask_decoder\.transformer\.norm_final_attn\.weight$": (
            "sam_mask_decoder.transformer.norm_final_attn.scale",
            None,
        ),
        r"^sam_mask_decoder\.transformer\.norm_final_attn\.bias$": (
            "sam_mask_decoder.transformer.norm_final_attn.bias",
            None,
        ),
        # Tokens
        r"^sam_mask_decoder\.iou_token\.weight$": (
            "sam_mask_decoder.iou_token.embedding",
            None,
        ),
        r"^sam_mask_decoder\.mask_tokens\.weight$": (
            "sam_mask_decoder.mask_tokens.embedding",
            None,
        ),
        r"^sam_mask_decoder\.obj_score_token\.weight$": (
            "sam_mask_decoder.obj_score_token.embedding",
            None,
        ),
        # Upscaling layers
        r"^sam_mask_decoder\.output_upscaling\.0\.weight$": (
            "sam_mask_decoder.output_upscaling.layers.0.kernel",
            (PYTORCH_TO_JAX_CONV_2D_TRANSPOSE_KERNEL, None),
        ),
        r"^sam_mask_decoder\.output_upscaling\.0\.bias$": (
            "sam_mask_decoder.output_upscaling.layers.0.bias",
            None,
        ),
        r"^sam_mask_decoder\.output_upscaling\.1\.weight$": (
            "sam_mask_decoder.output_upscaling.layers.1.weight",
            None,
        ),
        r"^sam_mask_decoder\.output_upscaling\.1\.bias$": (
            "sam_mask_decoder.output_upscaling.layers.1.bias",
            None,
        ),
        r"^sam_mask_decoder\.output_upscaling\.3\.weight$": (
            "sam_mask_decoder.output_upscaling.layers.3.kernel",
            (PYTORCH_TO_JAX_CONV_2D_TRANSPOSE_KERNEL, None),
        ),
        r"^sam_mask_decoder\.output_upscaling\.3\.bias$": (
            "sam_mask_decoder.output_upscaling.layers.3.bias",
            None,
        ),
        # Convs
        r"^sam_mask_decoder\.conv_s0\.weight$": (
            "sam_mask_decoder.conv_s0.kernel",
            (PYTORCH_TO_JAX_CONV_2D_KERNEL, None),
        ),
        r"^sam_mask_decoder\.conv_s0\.bias$": ("sam_mask_decoder.conv_s0.bias", None),
        r"^sam_mask_decoder\.conv_s1\.weight$": (
            "sam_mask_decoder.conv_s1.kernel",
            (PYTORCH_TO_JAX_CONV_2D_KERNEL, None),
        ),
        r"^sam_mask_decoder\.conv_s1\.bias$": ("sam_mask_decoder.conv_s1.bias", None),
        # Output Hypernetwork MLPs
        r"^sam_mask_decoder\.output_hypernetworks_mlps\.([0-3])\.layers\.([0-2])\.weight$": (
            r"sam_mask_decoder.output_hypernetworks_mlps.\1.layers.\2.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^sam_mask_decoder\.output_hypernetworks_mlps\.([0-3])\.layers\.([0-2])\.bias$": (
            r"sam_mask_decoder.output_hypernetworks_mlps.\1.layers.\2.bias",
            None,
        ),
        # IoU prediction head
        r"^sam_mask_decoder\.iou_prediction_head\.layers\.([0-2])\.weight$": (
            r"sam_mask_decoder.iou_prediction_head.layers.\1.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^sam_mask_decoder\.iou_prediction_head\.layers\.([0-2])\.bias$": (
            r"sam_mask_decoder.iou_prediction_head.layers.\1.bias",
            None,
        ),
        # Predicted object score head
        r"^sam_mask_decoder\.pred_obj_score_head\.layers\.([0-2])\.weight$": (
            r"sam_mask_decoder.pred_obj_score_head.layers.\1.kernel",
            (PYTORCH_TO_JAX_LINEAR, None),
        ),
        r"^sam_mask_decoder\.pred_obj_score_head\.layers\.([0-2])\.bias$": (
            r"sam_mask_decoder.pred_obj_score_head.layers.\1.bias",
            None,
        ),
    }

    sam_prompt_encoder = {
        # Positional Encoding
        r"^sam_prompt_encoder\.pe_layer\.positional_encoding_gaussian_matrix$": (
            "sam_prompt_encoder.pe_layer.gaussian_matrix",
            None,
        ),
        # Point Embeddings
        r"^sam_prompt_encoder\.point_embeddings\.([0-3])\.weight$": (
            r"sam_prompt_encoder.point_embeddings.\1.embedding",
            None,
        ),
        # Special Point Embeds
        r"^sam_prompt_encoder\.not_a_point_embed\.weight$": (
            "sam_prompt_encoder.not_a_point_embed.embedding",
            None,
        ),
        r"^sam_prompt_encoder\.no_mask_embed\.weight$": (
            "sam_prompt_encoder.no_mask_embed.embedding",
            None,
        ),
        # Mask Downscaling
        r"^sam_prompt_encoder\.mask_downscaling\.([0|3|6])\.weight$": (
            r"sam_prompt_encoder.mask_downscaling.layers.\1.kernel",
            (PYTORCH_TO_JAX_CONV_2D_KERNEL, None),
        ),
        r"^sam_prompt_encoder\.mask_downscaling\.([0|3|6])\.bias$": (
            r"sam_prompt_encoder.mask_downscaling.layers.\1.bias",
            None,
        ),
        r"^sam_prompt_encoder\.mask_downscaling\.([1|4])\.weight$": (
            r"sam_prompt_encoder.mask_downscaling.layers.\1.weight",
            None,
        ),
        r"^sam_prompt_encoder\.mask_downscaling\.([1|4])\.bias$": (
            r"sam_prompt_encoder.mask_downscaling.layers.\1.bias",
            None,
        ),
    }
    global_mapping = {
        # Embeddings & Positional Encodings
        r"^maskmem_tpos_enc$": ("maskmem_tpos_enc", None),
        r"^no_mem_embed$": ("no_mem_embed", None),
        r"^no_mem_pos_enc$": ("no_mem_pos_enc", None),
        r"^no_obj_ptr$": ("no_obj_ptr", None),
        # Mask Downsample
        r"^mask_downsample\.weight$": ("mask_downsample.kernel", (PYTORCH_TO_JAX_CONV_2D_KERNEL, None)),
        r"^mask_downsample\.bias$": ("mask_downsample.bias", None),
        # Object Pointer Projection MLP
        r"^obj_ptr_proj\.layers\.0\.weight$": ("obj_ptr_proj.layers.0.kernel", (PYTORCH_TO_JAX_LINEAR, None)),
        r"^obj_ptr_proj\.layers\.0\.bias$": ("obj_ptr_proj.layers.0.bias", None),
        r"^obj_ptr_proj\.layers\.1\.weight$": ("obj_ptr_proj.layers.1.kernel", (PYTORCH_TO_JAX_LINEAR, None)),
        r"^obj_ptr_proj\.layers\.1\.bias$": ("obj_ptr_proj.layers.1.bias", None),
        r"^obj_ptr_proj\.layers\.2\.weight$": ("obj_ptr_proj.layers.2.kernel", (PYTORCH_TO_JAX_LINEAR, None)),
        r"^obj_ptr_proj\.layers\.2\.bias$": ("obj_ptr_proj.layers.2.bias", None),
        # Object TPos Encoding
        r"^obj_ptr_tpos_proj\.weight$": ("obj_ptr_tpos_proj.kernel", (PYTORCH_TO_JAX_LINEAR, None)),
        r"^obj_ptr_tpos_proj\.bias$": ("obj_ptr_tpos_proj.bias", None),
    }
    return (
        image_encoder_mapping
        | memory_attention_mapping
        | memory_encoder_mapping
        | sam_mask_decoder
        | sam_prompt_encoder
        | global_mapping
    )


def find_non_array_keys(tree):
    """
    Walk `tree` (nested dicts/lists/tuples) and return a list of
    “full key paths” whose leaves are not numpy or JAX arrays.
    """
    bad = []

    def _recurse(subtree, path):
        if isinstance(subtree, dict):
            for k, v in subtree.items():
                _recurse(v, f"{path}.{k}" if path else k)
        elif isinstance(subtree, (list, tuple)):
            for i, v in enumerate(subtree):
                _recurse(v, f"{path}[{i}]")
        else:
            # treat both numpy ndarrays and JAX Arrays as “good”
            if not isinstance(subtree, (np.ndarray, jax.Array)):
                bad.append(path)

    _recurse(tree, "")
    return bad


def _st_key_to_jax_key(mapping, source_key):
    """Map a safetensors key to exactly one model key & transform, else warn/error."""
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


def assign_nan_at_path(tree, dotted_path):
    """Assign NaN array at the specified dotted path in the tree."""

    def parse_key(k):
        return int(k) if k.isdigit() else k

    parts = [parse_key(p) for p in dotted_path.split(".")]
    subtree = tree
    for p in parts[:-1]:
        subtree = subtree[p]
    leaf_key = parts[-1]

    # Get shape of current placeholder (if present), else default
    value = subtree.get(leaf_key, None)
    if hasattr(value, "shape"):
        shape = value.shape
    else:
        shape = (1,)  # fallback

    subtree[leaf_key] = np.full(shape, np.nan, dtype=np.float32)


def create_sam2_from_pretrained(
    file_path: str,
    config: model_lib.SAM2Config,
    *,
    mesh: jax.sharding.Mesh | None = None,
):
    """
    Load safetensors weights and initialize remaining missing weights with NaNs.
    """
    # 1. Load safetensor weights
    state_dict = safetensors.load_file(file_path)

    # 2. Create uninitialized SAM2 nnx model
    sam2 = nnx.eval_shape(lambda: model_lib.build_sam2_model_from_config(config, rngs=nnx.Rngs(params=0, dropout=0)))
    graph_def, abs_state = nnx.split(sam2)
    jax_state = nnx.to_pure_dict(abs_state)

    # 3. Assign known weights
    mapping = _get_key_and_transform_mapping()
    for st_key, tensor in state_dict.items():
        jax_key, transform = _st_key_to_jax_key(mapping, st_key)
        if jax_key is None:
            continue
        keys = [_stoi(k) for k in jax_key.split(".")]
        _assign_weights(keys, tensor, jax_state, st_key, transform)

    # 4. Fill in missing keys with NaNs
    missing_keys = find_non_array_keys(jax_state)
    for path in missing_keys:
        logging.warning(f"Missing param at: {path} - initializing with NaNs")
        assign_nan_at_path(jax_state, path)

    # 5. Device placement
    if mesh is not None:
        sharding = nnx.to_pure_dict(nnx.get_named_sharding(abs_state, mesh))
        jax_state = jax.device_put(jax_state, sharding)
    else:
        jax_state = jax.device_put(jax_state, jax.devices()[0])

    # 6. Merge & return
    return nnx.merge(graph_def, jax_state)
