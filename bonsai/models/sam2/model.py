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

"""SAM 2 model."""

from dataclasses import dataclass

import jax
import jax.image as jimage
import jax.numpy as jnp
import numpy as np
from flax import nnx
from PIL.Image import Image as PILImage

from bonsai.models.sam2.model_hiera_det import Hiera
from bonsai.models.sam2.model_image_encoder import FPNNeck, ImageEncoder
from bonsai.models.sam2.model_mask_decoder import MaskDecoder
from bonsai.models.sam2.model_memory_attention import MemoryAttention, MemoryAttentionLayer
from bonsai.models.sam2.model_memory_encoder import ConvNextBlock, Fuser, MaskDownSampler, MemoryEncoder
from bonsai.models.sam2.model_positional_encoding import PositionEmbeddingSine, get_1d_sine_pe
from bonsai.models.sam2.model_prompt_encoder import PromptEncoder
from bonsai.models.sam2.model_sam2_transforms import SAM2Transforms
from bonsai.models.sam2.model_transformer import RoPEAttention, TwoWayTransformer
from bonsai.models.sam2.model_utils import MLP, Identity, select_closest_cond_frames


@dataclass(frozen=True)
class RoPEAttentionConfig:
    rope_theta: float
    feat_sizes: tuple[int, int]
    embedding_dim: int
    num_heads: int
    downsample_rate: int
    dropout: float
    rope_k_repeat: bool = False
    kv_in_dim: int | None = None


@dataclass(frozen=True)
class MemoryAttentionLayerConfig:
    activation: str
    dim_feedforward: int
    dropout: float
    pos_enc_at_attn: bool
    self_attention: RoPEAttentionConfig
    cross_attention: RoPEAttentionConfig
    d_model: int
    pos_enc_at_cross_attn_keys: bool
    pos_enc_at_cross_attn_queries: bool


@dataclass(frozen=True)
class MemoryAttentionConfig:
    d_model: int
    pos_enc_at_input: bool
    layer: MemoryAttentionLayerConfig
    num_layers: int


@dataclass(frozen=True)
class PositionEncodingConfig:
    num_pos_feats: int
    normalize: bool
    scale: float | None
    temperature: float


@dataclass(frozen=True)
class FPNNeckConfig:
    position_encoding: PositionEncodingConfig
    d_model: int
    backbone_channel_list: list[int]
    fpn_top_down_levels: list[int]
    fpn_interp_model: str


@dataclass(frozen=True)
class HieraConfig:
    embed_dim: int
    num_heads: int
    stages: tuple[int, ...] = (2, 3, 16, 3)
    global_att_blocks: tuple[int, ...] = (12, 16, 20)
    window_pos_embed_bkg_spatial_size: tuple[int, ...] = (14, 14)
    window_spec: tuple[int, ...] = (8, 4, 14, 7)


@dataclass(frozen=True)
class ImageEncoderConfig:
    scalp: int
    trunk: HieraConfig
    neck: FPNNeckConfig


@dataclass(frozen=True)
class MaskDownSamplerConfig:
    kernel_size: int
    stride: int
    padding: int


@dataclass(frozen=True)
class CXBlockConfig:
    dim: int
    kernel_size: int
    padding: int
    layer_scale_init_value: float
    use_dwconv: bool


@dataclass(frozen=True)
class FuserConfig:
    layer: CXBlockConfig
    num_layers: int


@dataclass(frozen=True)
class MemoryEncoderConfig:
    out_dim: int
    position_encoding: PositionEncodingConfig
    mask_downsampler: MaskDownSamplerConfig
    fuser: FuserConfig


@dataclass(frozen=True)
class SAM2Config:
    image_encoder: ImageEncoderConfig
    memory_attention: MemoryAttentionConfig
    memory_encoder: MemoryEncoderConfig
    num_maskmem: int
    image_size: int
    sigmoid_scale_for_mem_enc: float
    sigmoid_bias_for_mem_enc: float
    use_mask_input_as_output_without_sam: bool
    directly_add_no_mem_embed: bool
    no_obj_embed_spatial: bool
    use_high_res_features_in_sam: bool
    multimask_output_in_sam: bool
    iou_prediction_use_sigmoid: bool
    use_obj_ptrs_in_encoder: bool
    add_tpos_enc_to_obj_ptrs: bool
    proj_tpos_enc_in_obj_ptrs: bool
    use_signed_tpos_enc_to_obj_ptrs: bool
    only_obj_ptrs_in_the_past_for_eval: bool
    pred_obj_scores: bool
    pred_obj_scores_mlp: bool
    fixed_no_obj_ptr: bool
    multimask_output_for_tracking: bool
    use_multimask_token_for_obj_ptr: bool
    multimask_min_pt_num: int
    multimask_max_pt_num: int
    use_mlp_for_obj_ptr_proj: bool
    compile_image_encoder: bool

    @classmethod
    def sam2_tiny(cls):
        return cls(
            image_encoder=ImageEncoderConfig(
                scalp=1,
                trunk=HieraConfig(
                    embed_dim=96,
                    num_heads=1,
                    stages=(1, 2, 7, 2),
                    global_att_blocks=(5, 7, 9),
                    window_pos_embed_bkg_spatial_size=(7, 7),
                ),
                neck=FPNNeckConfig(
                    position_encoding=PositionEncodingConfig(
                        num_pos_feats=256,
                        normalize=True,
                        scale=None,
                        temperature=10000,
                    ),
                    d_model=256,
                    backbone_channel_list=[768, 384, 192, 96],
                    fpn_top_down_levels=[2, 3],
                    fpn_interp_model="nearest",
                ),
            ),
            memory_attention=MemoryAttentionConfig(
                d_model=256,
                pos_enc_at_input=True,
                layer=MemoryAttentionLayerConfig(
                    activation="relu",
                    dim_feedforward=2048,
                    dropout=0.1,
                    pos_enc_at_attn=False,
                    self_attention=RoPEAttentionConfig(
                        rope_theta=10000.0,
                        feat_sizes=(64, 64),
                        embedding_dim=256,
                        num_heads=1,
                        downsample_rate=1,
                        dropout=0.1,
                    ),
                    cross_attention=RoPEAttentionConfig(
                        rope_theta=10000.0,
                        feat_sizes=(64, 64),
                        embedding_dim=256,
                        num_heads=1,
                        downsample_rate=1,
                        dropout=0.1,
                        rope_k_repeat=True,
                        kv_in_dim=64,
                    ),
                    d_model=256,
                    pos_enc_at_cross_attn_keys=True,
                    pos_enc_at_cross_attn_queries=False,
                ),
                num_layers=4,
            ),
            memory_encoder=MemoryEncoderConfig(
                out_dim=64,
                position_encoding=PositionEncodingConfig(
                    num_pos_feats=64,
                    normalize=True,
                    scale=None,
                    temperature=10000,
                ),
                mask_downsampler=MaskDownSamplerConfig(
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                fuser=FuserConfig(
                    layer=CXBlockConfig(
                        dim=256,
                        kernel_size=7,
                        padding=3,
                        layer_scale_init_value=1e-6,
                        use_dwconv=True,
                    ),
                    num_layers=2,
                ),
            ),
            num_maskmem=7,
            image_size=1024,
            sigmoid_scale_for_mem_enc=20.0,
            sigmoid_bias_for_mem_enc=-10.0,
            use_mask_input_as_output_without_sam=True,
            directly_add_no_mem_embed=True,
            no_obj_embed_spatial=True,
            use_high_res_features_in_sam=True,
            multimask_output_in_sam=True,
            iou_prediction_use_sigmoid=True,
            use_obj_ptrs_in_encoder=True,
            add_tpos_enc_to_obj_ptrs=True,
            proj_tpos_enc_in_obj_ptrs=True,
            use_signed_tpos_enc_to_obj_ptrs=True,
            only_obj_ptrs_in_the_past_for_eval=True,
            pred_obj_scores=True,
            pred_obj_scores_mlp=True,
            fixed_no_obj_ptr=True,
            multimask_output_for_tracking=True,
            use_multimask_token_for_obj_ptr=True,
            multimask_min_pt_num=0,
            multimask_max_pt_num=1,
            use_mlp_for_obj_ptr_proj=True,
            compile_image_encoder=False,
        )

    @classmethod
    def sam2_small(cls):
        return cls(
            image_encoder=ImageEncoderConfig(
                scalp=1,
                trunk=HieraConfig(
                    embed_dim=96,
                    num_heads=1,
                    stages=(1, 2, 11, 2),
                    global_att_blocks=(7, 10, 13),
                    window_pos_embed_bkg_spatial_size=(7, 7),
                ),
                neck=FPNNeckConfig(
                    position_encoding=PositionEncodingConfig(
                        num_pos_feats=256,
                        normalize=True,
                        scale=None,
                        temperature=10000,
                    ),
                    d_model=256,
                    backbone_channel_list=[768, 384, 192, 96],
                    fpn_top_down_levels=[2, 3],
                    fpn_interp_model="nearest",
                ),
            ),
            memory_attention=MemoryAttentionConfig(
                d_model=256,
                pos_enc_at_input=True,
                layer=MemoryAttentionLayerConfig(
                    activation="relu",
                    dim_feedforward=2048,
                    dropout=0.1,
                    pos_enc_at_attn=False,
                    self_attention=RoPEAttentionConfig(
                        rope_theta=10000.0,
                        feat_sizes=(64, 64),
                        embedding_dim=256,
                        num_heads=1,
                        downsample_rate=1,
                        dropout=0.1,
                    ),
                    cross_attention=RoPEAttentionConfig(
                        rope_theta=10000.0,
                        feat_sizes=(64, 64),
                        embedding_dim=256,
                        num_heads=1,
                        downsample_rate=1,
                        dropout=0.1,
                        rope_k_repeat=True,
                        kv_in_dim=64,
                    ),
                    d_model=256,
                    pos_enc_at_cross_attn_keys=True,
                    pos_enc_at_cross_attn_queries=False,
                ),
                num_layers=4,
            ),
            memory_encoder=MemoryEncoderConfig(
                out_dim=64,
                position_encoding=PositionEncodingConfig(
                    num_pos_feats=64,
                    normalize=True,
                    scale=None,
                    temperature=10000,
                ),
                mask_downsampler=MaskDownSamplerConfig(
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                fuser=FuserConfig(
                    layer=CXBlockConfig(
                        dim=256,
                        kernel_size=7,
                        padding=3,
                        layer_scale_init_value=1e-6,
                        use_dwconv=True,
                    ),
                    num_layers=2,
                ),
            ),
            num_maskmem=7,
            image_size=1024,
            sigmoid_scale_for_mem_enc=20.0,
            sigmoid_bias_for_mem_enc=-10.0,
            use_mask_input_as_output_without_sam=True,
            directly_add_no_mem_embed=True,
            no_obj_embed_spatial=True,
            use_high_res_features_in_sam=True,
            multimask_output_in_sam=True,
            iou_prediction_use_sigmoid=True,
            use_obj_ptrs_in_encoder=True,
            add_tpos_enc_to_obj_ptrs=True,
            proj_tpos_enc_in_obj_ptrs=True,
            use_signed_tpos_enc_to_obj_ptrs=True,
            only_obj_ptrs_in_the_past_for_eval=True,
            pred_obj_scores=True,
            pred_obj_scores_mlp=True,
            fixed_no_obj_ptr=True,
            multimask_output_for_tracking=True,
            use_multimask_token_for_obj_ptr=True,
            multimask_min_pt_num=0,
            multimask_max_pt_num=1,
            use_mlp_for_obj_ptr_proj=True,
            compile_image_encoder=False,
        )

    @classmethod
    def sam2_baseplus(cls):
        return cls(
            image_encoder=ImageEncoderConfig(
                scalp=1,
                trunk=HieraConfig(
                    embed_dim=112,
                    num_heads=2,
                    stages=(1, 2, 11, 2),
                    global_att_blocks=(7, 10, 13),
                    window_pos_embed_bkg_spatial_size=(7, 7),
                ),
                neck=FPNNeckConfig(
                    position_encoding=PositionEncodingConfig(
                        num_pos_feats=256,
                        normalize=True,
                        scale=None,
                        temperature=10000,
                    ),
                    d_model=256,
                    backbone_channel_list=[896, 448, 224, 112],
                    fpn_top_down_levels=[2, 3],
                    fpn_interp_model="nearest",
                ),
            ),
            memory_attention=MemoryAttentionConfig(
                d_model=256,
                pos_enc_at_input=True,
                layer=MemoryAttentionLayerConfig(
                    activation="relu",
                    dim_feedforward=2048,
                    dropout=0.1,
                    pos_enc_at_attn=False,
                    self_attention=RoPEAttentionConfig(
                        rope_theta=10000.0,
                        feat_sizes=(64, 64),
                        embedding_dim=256,
                        num_heads=1,
                        downsample_rate=1,
                        dropout=0.1,
                    ),
                    cross_attention=RoPEAttentionConfig(
                        rope_theta=10000.0,
                        feat_sizes=(64, 64),
                        embedding_dim=256,
                        num_heads=1,
                        downsample_rate=1,
                        dropout=0.1,
                        rope_k_repeat=True,
                        kv_in_dim=64,
                    ),
                    d_model=256,
                    pos_enc_at_cross_attn_keys=True,
                    pos_enc_at_cross_attn_queries=False,
                ),
                num_layers=4,
            ),
            memory_encoder=MemoryEncoderConfig(
                out_dim=64,
                position_encoding=PositionEncodingConfig(
                    num_pos_feats=64,
                    normalize=True,
                    scale=None,
                    temperature=10000,
                ),
                mask_downsampler=MaskDownSamplerConfig(
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                fuser=FuserConfig(
                    layer=CXBlockConfig(
                        dim=256,
                        kernel_size=7,
                        padding=3,
                        layer_scale_init_value=1e-6,
                        use_dwconv=True,
                    ),
                    num_layers=2,
                ),
            ),
            num_maskmem=7,
            image_size=1024,
            sigmoid_scale_for_mem_enc=20.0,
            sigmoid_bias_for_mem_enc=-10.0,
            use_mask_input_as_output_without_sam=True,
            directly_add_no_mem_embed=True,
            no_obj_embed_spatial=True,
            use_high_res_features_in_sam=True,
            multimask_output_in_sam=True,
            iou_prediction_use_sigmoid=True,
            use_obj_ptrs_in_encoder=True,
            add_tpos_enc_to_obj_ptrs=True,
            proj_tpos_enc_in_obj_ptrs=True,
            use_signed_tpos_enc_to_obj_ptrs=True,
            only_obj_ptrs_in_the_past_for_eval=True,
            pred_obj_scores=True,
            pred_obj_scores_mlp=True,
            fixed_no_obj_ptr=True,
            multimask_output_for_tracking=True,
            use_multimask_token_for_obj_ptr=True,
            multimask_min_pt_num=0,
            multimask_max_pt_num=1,
            use_mlp_for_obj_ptr_proj=True,
            compile_image_encoder=False,
        )

    @classmethod
    def sam2_large(cls):
        return cls(
            image_encoder=ImageEncoderConfig(
                scalp=1,
                trunk=HieraConfig(
                    embed_dim=144,
                    num_heads=2,
                    stages=(2, 6, 36, 4),
                    global_att_blocks=(23, 33, 43),
                    window_pos_embed_bkg_spatial_size=(7, 7),
                    window_spec=(8, 4, 16, 8),
                ),
                neck=FPNNeckConfig(
                    position_encoding=PositionEncodingConfig(
                        num_pos_feats=256,
                        normalize=True,
                        scale=None,
                        temperature=10000,
                    ),
                    d_model=256,
                    backbone_channel_list=[1152, 576, 288, 144],
                    fpn_top_down_levels=[2, 3],
                    fpn_interp_model="nearest",
                ),
            ),
            memory_attention=MemoryAttentionConfig(
                d_model=256,
                pos_enc_at_input=True,
                layer=MemoryAttentionLayerConfig(
                    activation="relu",
                    dim_feedforward=2048,
                    dropout=0.1,
                    pos_enc_at_attn=False,
                    self_attention=RoPEAttentionConfig(
                        rope_theta=10000.0,
                        feat_sizes=(64, 64),
                        embedding_dim=256,
                        num_heads=1,
                        downsample_rate=1,
                        dropout=0.1,
                    ),
                    cross_attention=RoPEAttentionConfig(
                        rope_theta=10000.0,
                        feat_sizes=(64, 64),
                        rope_k_repeat=True,
                        embedding_dim=256,
                        num_heads=1,
                        downsample_rate=1,
                        dropout=0.1,
                        kv_in_dim=64,
                    ),
                    d_model=256,
                    pos_enc_at_cross_attn_keys=True,
                    pos_enc_at_cross_attn_queries=False,
                ),
                num_layers=4,
            ),
            memory_encoder=MemoryEncoderConfig(
                out_dim=64,
                position_encoding=PositionEncodingConfig(
                    num_pos_feats=64,
                    normalize=True,
                    scale=None,
                    temperature=10000,
                ),
                mask_downsampler=MaskDownSamplerConfig(
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                fuser=FuserConfig(
                    layer=CXBlockConfig(
                        dim=256,
                        kernel_size=7,
                        padding=3,
                        layer_scale_init_value=1e-6,
                        use_dwconv=True,
                    ),
                    num_layers=2,
                ),
            ),
            num_maskmem=7,
            image_size=1024,
            sigmoid_scale_for_mem_enc=20.0,
            sigmoid_bias_for_mem_enc=-10.0,
            use_mask_input_as_output_without_sam=True,
            directly_add_no_mem_embed=True,
            no_obj_embed_spatial=True,
            use_high_res_features_in_sam=True,
            multimask_output_in_sam=True,
            iou_prediction_use_sigmoid=True,
            use_obj_ptrs_in_encoder=True,
            add_tpos_enc_to_obj_ptrs=True,
            proj_tpos_enc_in_obj_ptrs=True,
            use_signed_tpos_enc_to_obj_ptrs=True,
            only_obj_ptrs_in_the_past_for_eval=True,
            pred_obj_scores=True,
            pred_obj_scores_mlp=True,
            fixed_no_obj_ptr=True,
            multimask_output_for_tracking=True,
            use_multimask_token_for_obj_ptr=True,
            multimask_min_pt_num=0,
            multimask_max_pt_num=1,
            use_mlp_for_obj_ptr_proj=True,
            compile_image_encoder=False,
        )


# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0


class SAM2Base(nnx.Module):
    def __init__(
        self,
        image_encoder,
        memory_attention,
        memory_encoder,
        num_maskmem=7,
        image_size=512,
        backbone_stride=16,
        sigmoid_scale_for_mem_enc=1.0,
        sigmoid_bias_for_mem_enc=0.0,
        binarize_mask_from_pts_for_mem_enc=False,
        use_mask_input_as_output_without_sam=False,
        max_cond_frames_in_attn=-1,
        directly_add_no_mem_embed=False,
        use_high_res_features_in_sam=False,
        multimask_output_in_sam=False,
        multimask_min_pt_num=1,
        multimask_max_pt_num=1,
        multimask_output_for_tracking=False,
        use_multimask_token_for_obj_ptr: bool = False,
        iou_prediction_use_sigmoid=False,
        memory_temporal_stride_for_eval=1,
        non_overlap_masks_for_mem_enc=False,
        use_obj_ptrs_in_encoder=False,
        max_obj_ptrs_in_encoder=16,
        add_tpos_enc_to_obj_ptrs=True,
        proj_tpos_enc_in_obj_ptrs=False,
        use_signed_tpos_enc_to_obj_ptrs=False,
        only_obj_ptrs_in_the_past_for_eval=False,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        fixed_no_obj_ptr: bool = False,
        soft_no_obj_ptr: bool = False,
        use_mlp_for_obj_ptr_proj: bool = False,
        no_obj_embed_spatial: bool = False,
        sam_mask_decoder_extra_args=None,
        compile_image_encoder: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.training = False  # Inference only for now

        # Part 1: the image backbone
        self.image_encoder = image_encoder
        self.use_high_res_features_in_sam = use_high_res_features_in_sam
        self.num_feature_levels = 3 if use_high_res_features_in_sam else 1
        self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
        if use_obj_ptrs_in_encoder:
            self.mask_downsample = nnx.Conv(1, 1, kernel_size=(4, 4), strides=(4, 4), rngs=rngs)

        # Part 2: memory attention
        self.memory_attention = memory_attention
        self.hidden_dim = image_encoder.neck.d_model

        # Part 3: memory encoder
        self.memory_encoder = memory_encoder
        self.mem_dim = self.hidden_dim
        if hasattr(self.memory_encoder, "out_proj") and hasattr(self.memory_encoder.out_proj, "kernel"):
            # if there is compression of memories along channel dim
            self.mem_dim = self.memory_encoder.out_proj.kernel.shape[-1]

        # Positional embeddings
        self.num_maskmem = num_maskmem
        self.maskmem_tpos_enc = nnx.Param(jnp.zeros((num_maskmem, 1, 1, self.mem_dim)))
        self.no_mem_embed = nnx.Param(jnp.zeros((1, 1, self.hidden_dim)))
        self.no_mem_pos_enc = nnx.Param(jnp.zeros((1, 1, self.hidden_dim)))

        # boolean flags
        self.directly_add_no_mem_embed = directly_add_no_mem_embed
        self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc
        self.binarize_mask_from_pts_for_mem_enc = binarize_mask_from_pts_for_mem_enc
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval
        self.use_mask_input_as_output_without_sam = use_mask_input_as_output_without_sam
        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid

        # temporal pointer and obj flags
        self.add_tpos_enc_to_obj_ptrs = add_tpos_enc_to_obj_ptrs
        self.proj_tpos_enc_in_obj_ptrs = proj_tpos_enc_in_obj_ptrs
        if proj_tpos_enc_in_obj_ptrs:
            assert add_tpos_enc_to_obj_ptrs, "proj_tpos_enc_in_obj_ptrs requires add_tpos_enc_to_obj_ptrs"
        self.use_signed_tpos_enc_to_obj_ptrs = use_signed_tpos_enc_to_obj_ptrs
        self.only_obj_ptrs_in_the_past_for_eval = only_obj_ptrs_in_the_past_for_eval
        self.fixed_no_obj_ptr = fixed_no_obj_ptr
        self.soft_no_obj_ptr = soft_no_obj_ptr
        self.use_mlp_for_obj_ptr_proj = use_mlp_for_obj_ptr_proj

        # Part 4: SAM heads
        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args
        self.pred_obj_scores = pred_obj_scores
        self.pred_obj_scores_mlp = pred_obj_scores_mlp

        if self.fixed_no_obj_ptr:
            assert self.pred_obj_scores and self.use_obj_ptrs_in_encoder
        if self.pred_obj_scores and self.use_obj_ptrs_in_encoder:
            self.no_obj_ptr = nnx.Param(jnp.zeros((1, self.hidden_dim)))

        if no_obj_embed_spatial:
            self.no_obj_embed_spatial = nnx.Param(jnp.zeros((1, self.mem_dim)))
        else:
            self.no_obj_embed_spatial = None

        self._build_sam_heads(rngs)
        self.max_cond_frames_in_attn = max_cond_frames_in_attn

        if compile_image_encoder:
            self.image_encoder = nnx.jit(self.image_encoder)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Please use the corresponding methods in SAM2VideoPredictor for inference or SAM2Train for training/fine-tuning"
        )

    def _build_sam_heads(self, rngs: nnx.Rngs):
        # set embedding dims
        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride

        # Prompt encoder
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
            rngs=rngs,
        )

        # Mask decoder
        self.sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2, embedding_dim=self.sam_prompt_embed_dim, mlp_dim=2048, num_heads=8, rngs=rngs
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            **(self.sam_mask_decoder_extra_args or {}),
            rngs=rngs,
        )

        # object pointer projection
        if self.use_obj_ptrs_in_encoder:
            if self.use_mlp_for_obj_ptr_proj:
                self.obj_ptr_proj = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 3, rngs=rngs)
            else:
                self.obj_ptr_proj = nnx.Linear(self.hidden_dim, self.hidden_dim, rngs=rngs)
        else:
            self.obj_ptr_proj = Identity()

        # temporal pos encoding projection
        if self.proj_tpos_enc_in_obj_ptrs:
            self.obj_ptr_tpos_proj = nnx.Linear(self.hidden_dim, self.mem_dim, rngs=rngs)
        else:
            self.obj_ptr_tpos_proj = Identity()

    def _forward_sam_heads(
        self,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
    ):
        """
        Forward SAM prompt encoders and mask heads.

        Inputs:
        - backbone_features: image features of [B, H, W, C] shape
        - point_inputs: a dictionary with "point_coords" and "point_labels", where
          1) "point_coords" has [B, P, 2] shape and float32 dtype and contains the
             absolute pixel-unit coordinate in (x, y) format of the P input points
          2) "point_labels" has shape [B, P] and int32 dtype, where 1 means
             positive clicks, 0 means negative clicks, and -1 means padding
        - mask_inputs: a mask of [B, 1, H*16, W*16] shape, float or bool, with the
          same spatial size as the image.
        - high_res_features: either 1) None or 2) or a list of length 2 containing
          two feature maps of [B, 4*H, 4*W, C] and [B, 2*H, 2*W, C] shapes respectively,
          which will be used as high-resolution feature maps for SAM decoder.
        - multimask_output: if it's True, we output 3 candidate masks and their 3
          corresponding IoU estimates, and if it's False, we output only 1 mask and
          its corresponding IoU estimate.

        Outputs:
        - low_res_multimasks: [B, M, H*4, W*4] shape (where M = 3 if
          `multimask_output=True` and M = 1 if `multimask_output=False`), the SAM
          output mask logits (before sigmoid) for the low-resolution masks, with 4x
          the resolution (1/4 stride) of the input backbone_features.
        - high_res_multimasks: [B, M, H*16, W*16] shape (where M = 3
          if `multimask_output=True` and M = 1 if `multimask_output=False`),
          upsampled from the low-resolution masks, with shape size as the image
          (stride is 1 pixel).
        - ious, [B, M] shape, where (where M = 3 if `multimask_output=True` and M = 1
          if `multimask_output=False`), the estimated IoU of each output mask.
        - low_res_masks: [B, 1, H*4, W*4] shape, the best mask in `low_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `low_res_multimasks`.
        - high_res_masks: [B, 1, H*16, W*16] shape, the best mask in `high_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `high_res_multimasks`.
        - obj_ptr: [B, C] shape, the object pointer vector for the output mask, extracted
          based on the output token from the SAM mask decoder.
        """
        B = backbone_features.shape[0]
        # a) Handle point prompts
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
        else:
            sam_point_coords = jnp.zeros((B, 1, 2), dtype=jnp.float32)
            sam_point_labels = jnp.full((B, 1), -1, dtype=jnp.int32)

        # b) Handle mask prompts
        if mask_inputs is not None:
            assert mask_inputs.ndim == 4 and mask_inputs.shape[:2] == (B, 1)
            target_size = self.sam_prompt_encoder.mask_input_size
            if mask_inputs.shape[-2:] != target_size:
                new_shape = (B, 1, *target_size)
                sam_mask_prompt = jimage.resize(mask_inputs.astype(jnp.float32), new_shape, method="bilinear")
            else:
                sam_mask_prompt = mask_inputs
        else:
            sam_mask_prompt = None

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )
        low_res_multimasks, ious, sam_output_tokens, object_score_logits = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        if self.pred_obj_scores:
            is_obj = object_score_logits > 0
            low_res_multimasks = jnp.where(is_obj[:, None, None], low_res_multimasks, NO_OBJ_SCORE)

        # ensure float32
        low_res_multimasks = low_res_multimasks.astype(jnp.float32)
        high_res_multimasks = jimage.resize(
            low_res_multimasks,
            (B, low_res_multimasks.shape[1], self.image_size, self.image_size),
            method="bilinear",
        )

        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            best_iou_inds = jnp.argmax(ious, axis=-1)
            batch_inds = jnp.arange(B)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds][..., None, :, :]
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds][..., None, :, :]
            sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks = low_res_multimasks
            high_res_masks = high_res_multimasks

        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.pred_obj_scores:
            if self.soft_no_obj_ptr:
                lam = jax.nn.sigmoid(object_score_logits)
            else:
                lam = is_obj.astype(jnp.float32)
            if self.fixed_no_obj_ptr:
                obj_ptr = lam[:, None] * obj_ptr
            obj_ptr = obj_ptr + (1 - lam[:, None]) * self.no_obj_ptr

        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs):
        """
        Directly turn binary `mask_inputs` into output mask logits without using SAM.
        """
        # Use -10/+10 as logits for neg/pos pixels
        out_scale, out_bias = 20.0, -10.0
        mask_inputs_float = mask_inputs.astype(jnp.float32)
        # high-res logits
        high_res_masks = mask_inputs_float * out_scale + out_bias
        # downsample to low-res
        B, H, W, C = high_res_masks.shape
        low_H, low_W = H // 4, W // 4
        low_res_masks = jimage.resize(high_res_masks, (B, low_H, low_W, C), method="bilinear")
        # dummy IoU predictions
        ious = jnp.ones((B, 1), dtype=jnp.float32)
        # object pointer
        if not self.use_obj_ptrs_in_encoder:
            obj_ptr = jnp.zeros((B, self.hidden_dim), dtype=jnp.float32)
        else:
            (_, _, _, _, _, obj_ptr, _) = self._forward_sam_heads(
                backbone_features=backbone_features,
                mask_inputs=self.mask_downsample(mask_inputs_float),
                high_res_features=high_res_features,
            )
        # decide object presence
        is_obj = jnp.any(mask_inputs.reshape((B, -1)) > 0, axis=1)
        lam = is_obj.astype(jnp.float32)[..., None]
        object_score_logits = out_scale * lam + out_bias
        if self.pred_obj_scores:
            if self.fixed_no_obj_ptr:
                obj_ptr = lam * obj_ptr
            obj_ptr = obj_ptr + (1 - lam) * self.no_obj_ptr
        return (
            low_res_masks,
            high_res_masks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def forward_image(self, img_batch):
        """Get the image feature on the input batch."""
        backbone_out = self.image_encoder(img_batch)
        if self.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(backbone_out["backbone_fpn"][0])
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(backbone_out["backbone_fpn"][1])
        return backbone_out

    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        # shallow copy
        backbone_out = dict(backbone_out)
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        # record spatial sizes
        feat_sizes = [(x.shape[-3], x.shape[-2]) for x in vision_pos_embeds]
        # flatten and transpose each map: (B,H,W, C) -> (H*W, B, C)
        vision_feats = [jnp.transpose(jnp.reshape(x, (x.shape[0], -1, x.shape[-1])), (1, 0, 2)) for x in feature_maps]
        vision_pos_embeds = [
            jnp.transpose(jnp.reshape(x, (x.shape[0], -1, x.shape[-1])), (1, 0, 2)) for x in vision_pos_embeds
        ]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,
    ):
        """
        Fuse the current frame's visual feature map with previous memory.
        """
        # extract shapes
        feat = current_vision_feats[-1]  # shape (seq_len, B, C)
        B = feat.shape[1]
        C = self.hidden_dim
        H, W = feat_sizes[-1]

        # if no memory, reshape and return
        if self.num_maskmem == 0:
            pix = jnp.transpose(feat, (1, 0, 2)).reshape((B, H, W, C))
            return pix

        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1

        if not is_init_cond_frame:
            to_cat_memory = []
            to_cat_memory_pos_embed = []

            # conditioning frames
            cond_outputs = output_dict["cond_frame_outputs"]
            selected, unselected = select_closest_cond_frames(frame_idx, cond_outputs, self.max_cond_frames_in_attn)
            t_pos_and_prevs = [(0, out) for out in selected.values()]

            # non-cond frames
            stride = 1 if self.training else self.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos
                if t_rel == 1:
                    prev_idx = frame_idx - t_rel if not track_in_reverse else frame_idx + t_rel
                else:
                    if not track_in_reverse:
                        prev_idx = ((frame_idx - 2) // stride) * stride
                        prev_idx -= (t_rel - 2) * stride
                    else:
                        prev_idx = -(-(frame_idx + 2) // stride) * stride
                        prev_idx += (t_rel - 2) * stride
                out = output_dict["non_cond_frame_outputs"].get(prev_idx, None)
                if out is None:
                    out = unselected.get(prev_idx, None)
                t_pos_and_prevs.append((t_pos, out))

            # encode features
            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue
                feats = prev["maskmem_features"]  # shape (B,C,H,W)
                # flatten -> (B,H*W, C) and transpose to (H*W,B,C)
                fm = feats.reshape((B, -1, C)).transpose((1, 0, 2))
                to_cat_memory.append(fm)
                enc = prev["maskmem_pos_enc"][-1]
                enc = enc.reshape((1, -1, C)).transpose((1, 0, 2))
                enc = enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                to_cat_memory_pos_embed.append(enc)

            # object pointers
            if self.use_obj_ptrs_in_encoder:
                max_ptrs = min(num_frames, self.max_obj_ptrs_in_encoder)
                ptr_cond = selected
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond = {
                        t: out
                        for t, out in selected.items()
                        if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                    }
                pos_and_ptrs = []
                for t, out in ptr_cond.items():
                    diff = (
                        (frame_idx - t) * tpos_sign_mul if self.use_signed_tpos_enc_to_obj_ptrs else abs(frame_idx - t)
                    )
                    pos_and_ptrs.append((diff, out["obj_ptr"]))
                for t_diff in range(1, max_ptrs):
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                    out = output_dict["non_cond_frame_outputs"].get(t, unselected.get(t, None))
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"]))
                if pos_and_ptrs:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    obj_ptrs = jnp.stack(ptrs_list, axis=0)
                    if self.add_tpos_enc_to_obj_ptrs:
                        t_diff_max = max_ptrs - 1
                        dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = get_1d_sine_pe(jnp.array(pos_list) / t_diff_max, dim=dim)
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos[:, None, :].repeat((1, B, 1))
                    else:
                        obj_pos = jnp.zeros((len(pos_list), B, self.mem_dim))
                    if self.mem_dim < C:
                        obj_ptrs = obj_ptrs.reshape((-1, B, C // self.mem_dim, self.mem_dim))
                        obj_ptrs = obj_ptrs.transpose((0, 2, 1, 3)).reshape((-1, B, self.mem_dim))
                        obj_pos = jnp.repeat(obj_pos, C // self.mem_dim, axis=0)
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0
        else:
            # initial frame
            if self.directly_add_no_mem_embed:
                base = current_vision_feats[-1] + self.no_mem_embed
                return base.transpose((1, 0, 2)).reshape((B, H, W, C))
            to_cat_memory = [jnp.repeat(self.no_mem_embed, B, axis=1)]
            to_cat_memory_pos_embed = [jnp.repeat(self.no_mem_pos_enc, B, axis=1)]

        # concatenate
        memory = jnp.concatenate(to_cat_memory, axis=0)
        memory_pos = jnp.concatenate(to_cat_memory_pos_embed, axis=0)

        pix_feat = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        # reshape to (B,H,W,C)
        pix_feat = pix_feat.transpose((1, 0, 2)).reshape((B, H, W, C))
        return pix_feat

    def _encode_new_memory(
        self,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        is_mask_from_pts,
    ):
        """Encode the current image and its prediction into a memory feature."""
        # batch and dims
        seq, B, C = current_vision_feats[-1].shape
        H, W = feat_sizes[-1]
        # reshape to BCHW
        pix_feat = current_vision_feats[-1].transpose((1, 0, 2)).reshape((B, H, W, C))

        # optional non-overlap
        if self.non_overlap_masks_for_mem_enc and not self.training:
            pred_masks_high_res = self._apply_non_overlapping_constraints(pred_masks_high_res)

        # sigmoid or binarize
        if self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).astype(jnp.float32)
        else:
            mask_for_mem = jax.nn.sigmoid(pred_masks_high_res)

        # scale & bias
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc

        # encode memory
        maskmem_out = self.memory_encoder(pix_feat, mask_for_mem, skip_mask_sigmoid=True)
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]

        # add no-object embedding
        if self.no_obj_embed_spatial is not None:
            is_obj = object_score_logits > 0
            lam = is_obj.astype(jnp.float32)[..., None, None, None]
            no_obj = self.no_obj_embed_spatial[..., None, None]
            maskmem_features = maskmem_features + (1 - lam) * no_obj

        return maskmem_features, maskmem_pos_enc

    def _track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,
        prev_sam_mask_logits=None,
    ):
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.transpose((1, 2, 0)).reshape((x.shape[1], x.shape[2], *s))
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None
        # mask as output
        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            pix = current_vision_feats[-1].transpose((1, 2, 0)).reshape((-1, self.hidden_dim, *feat_sizes[-1]))
            sam_outputs = self._use_mask_as_output(pix, high_res_features, mask_inputs)
        else:
            pix = self._prepare_memory_conditioned_features(
                frame_idx,
                is_init_cond_frame,
                current_vision_feats[-1:],
                current_vision_pos_embeds[-1:],
                feat_sizes[-1:],
                output_dict,
                num_frames,
                track_in_reverse,
            )
            if prev_sam_mask_logits is not None:
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            mm = self._use_multimask(is_init_cond_frame, point_inputs)
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=mm,
            )
        return current_out, sam_outputs, high_res_features, pix

    def _encode_memory_in_output(
        self,
        current_vision_feats,
        feat_sizes,
        point_inputs,
        run_mem_encoder,
        high_res_masks,
        object_score_logits,
        current_out,
    ):
        if run_mem_encoder and self.num_maskmem > 0:
            high_res_masks_for_mem_enc = high_res_masks
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks_for_mem_enc,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,
        run_mem_encoder=True,
        prev_sam_mask_logits=None,
    ):
        # 1) run the core track step
        current_out, sam_outputs, _, _ = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )

        # 2) unpack SAM outputs
        _, _, _, low_res_masks, high_res_masks, obj_ptr, obj_logits = sam_outputs

        # 3) fill in predictions
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        if not self.training:
            current_out["object_score_logits"] = obj_logits

        # 4) optionally encode into memory
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            obj_logits,
            current_out,
        )

        return current_out

    def _use_multimask(self, is_init_cond_frame, point_inputs) -> bool:
        """Decide whether to output multiple SAM masks."""
        num_pts = 0
        if point_inputs is not None:
            # point_labels has shape (B, P)
            num_pts = int(point_inputs["point_labels"].shape[1])
        return (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
        )

    def _apply_non_overlapping_constraints(
        self,
        pred_masks: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Keep only the highest-scoring mask at each spatial location.
        Expects pred_masks.shape == (O, H, W) or (O, H, W,…) where O is 'object' dim.
        """
        object_dim = pred_masks.shape[0]
        if object_dim == 1:
            return pred_masks

        # find which object has max logit at each pixel
        # max_inds shape == (H, W, …)
        max_inds = jnp.argmax(pred_masks, axis=0)

        # build a boolean mask of shape (O, H, W, …)
        # where keep[o, ...] == True iff max_inds == o
        # first, expand max_inds with an object axis
        # we broadcast compare to turn into (O, H, W, …)
        obj_range = jnp.arange(object_dim)[:, None, None]
        keep = max_inds[None, ...] == obj_range

        # clamp all other scores to ≤ -10.0
        clamped = jnp.clip(pred_masks, a_max=-10.0)
        return jnp.where(keep, pred_masks, clamped)


def build_sam2_model_from_config(cfg: SAM2Config, rngs: nnx.Rngs) -> SAM2Base:
    # === Position Encodings ===
    pos_enc_backbone = PositionEmbeddingSine(
        num_pos_feats=cfg.image_encoder.neck.position_encoding.num_pos_feats,
        normalize=cfg.image_encoder.neck.position_encoding.normalize,
        scale=cfg.image_encoder.neck.position_encoding.scale,
        temperature=cfg.image_encoder.neck.position_encoding.temperature,
    )

    pos_enc_memory = PositionEmbeddingSine(
        num_pos_feats=cfg.memory_encoder.position_encoding.num_pos_feats,
        normalize=cfg.memory_encoder.position_encoding.normalize,
        scale=cfg.memory_encoder.position_encoding.scale,
        temperature=cfg.memory_encoder.position_encoding.temperature,
    )

    # === Fuser ===
    fuser_layer = ConvNextBlock(
        dim=cfg.memory_encoder.fuser.layer.dim,
        kernel_size=cfg.memory_encoder.fuser.layer.kernel_size,
        padding=cfg.memory_encoder.fuser.layer.padding,
        layer_scale_init_value=cfg.memory_encoder.fuser.layer.layer_scale_init_value,
        use_dwconv=cfg.memory_encoder.fuser.layer.use_dwconv,
        rngs=rngs,
    )

    fuser = Fuser(layer=fuser_layer, num_layers=cfg.memory_encoder.fuser.num_layers, rngs=rngs)

    memory_encoder = MemoryEncoder(
        out_dim=cfg.memory_encoder.out_dim,
        position_encoding=pos_enc_memory,
        mask_downsampler=MaskDownSampler(
            kernel_size=cfg.memory_encoder.mask_downsampler.kernel_size,
            stride=cfg.memory_encoder.mask_downsampler.stride,
            padding=cfg.memory_encoder.mask_downsampler.padding,
            rngs=rngs,
        ),
        fuser=fuser,
        rngs=rngs,
    )

    # === Memory Attention ===
    attention_layers = []
    for _ in range(cfg.memory_attention.num_layers):
        self_attn = RoPEAttention(
            rope_theta=cfg.memory_attention.layer.self_attention.rope_theta,
            feat_sizes=cfg.memory_attention.layer.self_attention.feat_sizes,
            embedding_dim=cfg.memory_attention.layer.self_attention.embedding_dim,
            num_heads=cfg.memory_attention.layer.self_attention.num_heads,
            downsample_rate=cfg.memory_attention.layer.self_attention.downsample_rate,
            dropout=cfg.memory_attention.layer.self_attention.dropout,
            rngs=rngs,
        )

        cross_attn = RoPEAttention(
            rope_theta=cfg.memory_attention.layer.cross_attention.rope_theta,
            feat_sizes=cfg.memory_attention.layer.cross_attention.feat_sizes,
            embedding_dim=cfg.memory_attention.layer.cross_attention.embedding_dim,
            num_heads=cfg.memory_attention.layer.cross_attention.num_heads,
            downsample_rate=cfg.memory_attention.layer.cross_attention.downsample_rate,
            dropout=cfg.memory_attention.layer.cross_attention.dropout,
            rope_k_repeat=cfg.memory_attention.layer.cross_attention.rope_k_repeat,
            kv_in_dim=cfg.memory_attention.layer.cross_attention.kv_in_dim,
            rngs=rngs,
        )

        attention_layers.append(
            MemoryAttentionLayer(
                activation=cfg.memory_attention.layer.activation,
                dim_feedforward=cfg.memory_attention.layer.dim_feedforward,
                dropout=cfg.memory_attention.layer.dropout,
                pos_enc_at_attn=cfg.memory_attention.layer.pos_enc_at_attn,
                self_attention=self_attn,
                cross_attention=cross_attn,
                d_model=cfg.memory_attention.layer.d_model,
                pos_enc_at_cross_attn_keys=cfg.memory_attention.layer.pos_enc_at_cross_attn_keys,
                pos_enc_at_cross_attn_queries=cfg.memory_attention.layer.pos_enc_at_cross_attn_queries,
                rngs=rngs,
            )
        )

    memory_attention = MemoryAttention(
        d_model=cfg.memory_attention.d_model,
        pos_enc_at_input=cfg.memory_attention.pos_enc_at_input,
        layers=attention_layers,
        rngs=rngs,
    )

    # === Image Encoder ===
    trunk = Hiera(
        embed_dim=cfg.image_encoder.trunk.embed_dim,
        num_heads=cfg.image_encoder.trunk.num_heads,
        stages=cfg.image_encoder.trunk.stages,
        global_att_blocks=cfg.image_encoder.trunk.global_att_blocks,
        window_pos_embed_bkg_spatial_size=cfg.image_encoder.trunk.window_pos_embed_bkg_spatial_size,
        window_spec=cfg.image_encoder.trunk.window_spec,
        rngs=rngs,
    )

    neck = FPNNeck(
        position_encoding=pos_enc_backbone,
        d_model=cfg.image_encoder.neck.d_model,
        backbone_channel_list=cfg.image_encoder.neck.backbone_channel_list,
        fpn_top_down_levels=cfg.image_encoder.neck.fpn_top_down_levels,
        fpn_interp_model=cfg.image_encoder.neck.fpn_interp_model,
        rngs=rngs,
    )

    image_encoder = ImageEncoder(
        scalp=cfg.image_encoder.scalp,
        trunk=trunk,
        neck=neck,
    )

    # === SAM2 ===
    return SAM2Base(
        image_encoder=image_encoder,
        memory_attention=memory_attention,
        memory_encoder=memory_encoder,
        num_maskmem=cfg.num_maskmem,
        image_size=cfg.image_size,
        sigmoid_scale_for_mem_enc=cfg.sigmoid_scale_for_mem_enc,
        sigmoid_bias_for_mem_enc=cfg.sigmoid_bias_for_mem_enc,
        use_mask_input_as_output_without_sam=cfg.use_mask_input_as_output_without_sam,
        directly_add_no_mem_embed=cfg.directly_add_no_mem_embed,
        no_obj_embed_spatial=cfg.no_obj_embed_spatial,
        use_high_res_features_in_sam=cfg.use_high_res_features_in_sam,
        multimask_output_in_sam=cfg.multimask_output_in_sam,
        iou_prediction_use_sigmoid=cfg.iou_prediction_use_sigmoid,
        use_obj_ptrs_in_encoder=cfg.use_obj_ptrs_in_encoder,
        add_tpos_enc_to_obj_ptrs=cfg.add_tpos_enc_to_obj_ptrs,
        proj_tpos_enc_in_obj_ptrs=cfg.proj_tpos_enc_in_obj_ptrs,
        use_signed_tpos_enc_to_obj_ptrs=cfg.use_signed_tpos_enc_to_obj_ptrs,
        only_obj_ptrs_in_the_past_for_eval=cfg.only_obj_ptrs_in_the_past_for_eval,
        pred_obj_scores=cfg.pred_obj_scores,
        pred_obj_scores_mlp=cfg.pred_obj_scores_mlp,
        fixed_no_obj_ptr=cfg.fixed_no_obj_ptr,
        multimask_output_for_tracking=cfg.multimask_output_for_tracking,
        use_multimask_token_for_obj_ptr=cfg.use_multimask_token_for_obj_ptr,
        multimask_min_pt_num=cfg.multimask_min_pt_num,
        multimask_max_pt_num=cfg.multimask_max_pt_num,
        use_mlp_for_obj_ptr_proj=cfg.use_mlp_for_obj_ptr_proj,
        compile_image_encoder=cfg.compile_image_encoder,
        rngs=rngs,
    )


class SAM2ImagePredictor(nnx.Module):
    def __init__(
        self,
        sam_model: SAM2Base,
        mask_threshold: float = 0.0,
        max_hole_area: float = 0.0,
        max_sprinkle_area: float = 0.0,
        **kwargs,
    ) -> None:
        """
        Uses SAM-2 to calculate the image embedding for an image, and then
        allows repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (SAM2Base): The model to use for mask prediction.
          mask_threshold (float): Threshold to convert mask logits to binary.
          max_hole_area (float): Max hole area to fill in mask.
          max_sprinkle_area (float): Max sprinkle area to remove in mask.
        """
        self.model: SAM2Base = sam_model
        self._transforms: SAM2Transforms = SAM2Transforms(
            resolution=self.model.image_size,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )

        # Predictor state
        self._is_image_set = False
        self._features = {}
        self._orig_hw = None
        self._is_batch = False

        # Predictor config
        self.mask_threshold = mask_threshold

        # Spatial dim for backbone feature maps (low → high)
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

    def set_image(self, image: np.ndarray | PILImage) -> None:
        """
        Calculates the image embeddings for the provided image.

        Arguments:
          image (np.ndarray or PIL.Image): RGB image in HWC format or PIL.Image.
        """
        self.reset_predictor()

        # Store original image size
        if isinstance(image, np.ndarray):
            self._orig_hw = [image.shape[:2]]  # (H, W)
        elif isinstance(image, jax.Array):
            self._orig_hw = [image.shape[:2]]  # (H, W)
        elif isinstance(image, PILImage):
            w, h = image.size
            self._orig_hw = [(h, w)]
        else:
            raise ValueError("Unsupported image format.")

        # Preprocess and add batch dim
        input_image = self._transforms(image)  # Returns jnp.ndarray
        input_image = input_image[None, ...]  # Shape: (1, H, W, 3)

        # Forward pass through model
        backbone_out = self.model.forward_image(input_image)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)

        # Add no_mem_embed if needed
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        # Format features
        feats = [
            jnp.transpose(f, (0, 2, 3, 1)).reshape(1, *feat_size)
            for f, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        self._features = {
            "image_embed": feats[-1],
            "high_res_feats": feats[:-1],
        }
        self._is_image_set = True

    def set_image_batch(self, image_list: list[np.ndarray]) -> None:
        """
        Calculates the image embeddings for a batch of images.

        Arguments:
          image_list (list[np.ndarray]): RGB images in HWC format with dtype uint8.
        """
        self.reset_predictor()
        assert isinstance(image_list, list) and all(
            isinstance(im, np.ndarray) or isinstance(im, jax.Array) for im in image_list
        )

        # Save original (H, W) for each image
        self._orig_hw = [img.shape[:2] for img in image_list]

        # Apply batched transform (assumes returns jnp.ndarray in NHWC)
        img_batch = self._transforms.forward_batch(image_list)  # Shape: (B, H, W, 3)

        batch_size = img_batch.shape[0]
        assert (
            img_batch.ndim == 4 and img_batch.shape[-1] == 3
        ), f"img_batch must be (B, H, W, 3), got {img_batch.shape}"

        # Forward pass
        backbone_out = self.model.forward_image(img_batch)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)

        # Add no_mem_embed if needed
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        # Format features from low to high resolution
        feats = [
            jnp.transpose(f, (1, 2, 0)).reshape(batch_size, -1, *feat_size)
            for f, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]

        self._features = {
            "image_embed": feats[-1],
            "high_res_feats": feats[:-1],
        }
        self._is_image_set = True
        self._is_batch = True

    def predict_batch(
        self,
        point_coords_batch: list[np.ndarray] | None = None,
        point_labels_batch: list[np.ndarray] | None = None,
        box_batch: list[np.ndarray] | None = None,
        mask_input_batch: list[np.ndarray] | None = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords: bool = True,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """
        Predict masks for a batch of images using corresponding batched prompts.

        Returns:
          Tuple of:
            - masks: List of (C, H, W) arrays
            - ious: List of (C,) arrays
            - low_res_masks: List of (C, 256, 256) arrays
        """
        if not self._is_batch or not self._is_image_set:
            raise RuntimeError("Call set_image_batch(...) before predict_batch.")

        num_images = len(self._features["image_embed"])
        masks_all, ious_all, lowres_all = [], [], []

        for i in range(num_images):
            # Slice prompts for image i
            point_coords = point_coords_batch[i] if point_coords_batch else None
            point_labels = point_labels_batch[i] if point_labels_batch else None
            box = box_batch[i] if box_batch else None
            mask_input = mask_input_batch[i] if mask_input_batch else None

            # Transform prompts
            mask_input_tensor, coords, labels, box_tensor = self._prep_prompts(
                point_coords,
                point_labels,
                box,
                mask_input,
                normalize_coords,
                img_idx=i,
            )

            # Predict
            masks, ious, lowres = self._predict(
                point_coords=coords,
                point_labels=labels,
                boxes=box_tensor,
                mask_input=mask_input_tensor,
                multimask_output=multimask_output,
                return_logits=return_logits,
                img_idx=i,
            )

            # Convert from JAX to NumPy
            masks_all.append(jnp.squeeze(masks, axis=0))
            ious_all.append(jnp.squeeze(ious, axis=0))
            lowres_all.append(jnp.squeeze(lowres, axis=0))

        return masks_all, ious_all, lowres_all

    def predict(
        self,
        point_coords: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
        box: np.ndarray | None = None,
        mask_input: np.ndarray | None = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict a mask for a single image and prompt.

        Returns:
          Tuple of:
            - masks: np.ndarray of shape (1, C, H, W)
            - ious: np.ndarray of shape (1, C)
            - low_res_masks: np.ndarray of shape (1, C, 256, 256)
        """
        if self._is_batch:
            raise RuntimeError("predict(...) cannot be used after set_image_batch(...)")

        if not self._is_image_set:
            raise RuntimeError("set_image(...) must be called before predict(...)")

        mask_input_tensor, coords, labels, box_tensor = self._prep_prompts(
            point_coords,
            point_labels,
            box,
            mask_input,
            normalize_coords,
        )

        # Forward pass
        masks, ious, lowres_masks = self._predict(
            point_coords=coords,
            point_labels=labels,
            boxes=box_tensor,
            mask_input=mask_input_tensor,
            multimask_output=multimask_output,
            return_logits=return_logits,
        )

        # Convert from JAX to NumPy arrays (and keep batch dimension)
        masks = np.asarray(masks)
        ious = np.asarray(ious)
        lowres_masks = np.asarray(lowres_masks)

        return masks, ious, lowres_masks

    def _prep_prompts(
        self,
        point_coords: np.ndarray | None,
        point_labels: np.ndarray | None,
        box: np.ndarray | None,
        mask_logits: np.ndarray | None,
        normalize_coords: bool,
        img_idx: int = -1,
    ):
        """
        Prepares prompt inputs (points, labels, boxes, masks) for the model.

        Returns:
            Tuple of (mask_input, coords, labels, boxes)
        """
        unnorm_coords = labels = unnorm_box = mask_input = None

        if point_coords is not None:
            assert point_labels is not None, "point_labels must be supplied if point_coords is supplied."

            point_coords = jnp.array(point_coords, dtype=jnp.float32)
            labels = jnp.array(point_labels, dtype=jnp.int32)

            unnorm_coords = self._transforms.transform_coords(
                point_coords,
                normalize=normalize_coords,
                orig_hw=self._orig_hw[img_idx],
            )

            if unnorm_coords.ndim == 2:
                # Add batch dimension if missing
                unnorm_coords = unnorm_coords[None, :, :]
                labels = labels[None, :]

        if box is not None:
            box = jnp.array(box, dtype=jnp.float32)
            unnorm_box = self._transforms.transform_boxes(
                box,
                normalize=normalize_coords,
                orig_hw=self._orig_hw[img_idx],
            )

        if mask_logits is not None:
            mask_input = jnp.array(mask_logits, dtype=jnp.float32)
            if mask_input.ndim == 3:
                mask_input = mask_input[None, :, :, :]  # add batch dimension

        return mask_input, unnorm_coords, labels, unnorm_box

    def _predict(
        self,
        point_coords: jnp.ndarray | None,
        point_labels: jnp.ndarray | None,
        boxes: jnp.ndarray | None = None,
        mask_input: jnp.ndarray | None = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        img_idx: int = -1,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Predict masks for the given input prompts using the currently set image.
        All inputs are expected to be pre-transformed.

        Returns:
            masks (jnp.ndarray): BxCxHxW masks at original resolution
            iou_predictions (jnp.ndarray): BxC quality scores
            low_res_masks (jnp.ndarray): BxCx256x256 mask logits
        """
        if not self._is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before prediction.")

        concat_points = None
        if point_coords is not None:
            concat_points = (point_coords, point_labels)

        if boxes is not None:
            box_coords = boxes.reshape(-1, 2, 2)
            box_labels = jnp.array([[2, 3]], dtype=jnp.int32)
            box_labels = jnp.tile(box_labels, (boxes.shape[0], 1))
            if concat_points is not None:
                concat_coords = jnp.concatenate([box_coords, concat_points[0]], axis=1)
                concat_labels = jnp.concatenate([box_labels, concat_points[1]], axis=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)

        # Prompt encoding
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=mask_input,
        )

        # Get features
        batched_mode = concat_points is not None and concat_points[0].shape[0] > 1
        high_res_features = [
            jnp.expand_dims(feat_level[img_idx], axis=0) for feat_level in self._features["high_res_feats"]
        ]

        image_embed = jnp.expand_dims(self._features["image_embed"][img_idx], axis=0)
        image_pe = self.model.sam_prompt_encoder.get_dense_pe()

        # Mask decoding
        low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )

        # Post-process
        masks = self._transforms.postprocess_masks(low_res_masks, self._orig_hw[img_idx])
        low_res_masks = jnp.clip(low_res_masks, -32.0, 32.0)

        if not return_logits:
            masks = masks > self.mask_threshold

        return masks, iou_predictions, low_res_masks

    def get_image_embedding(self):
        """
        Returns the image embeddings for the currently set image,
        shape: [1, C, H, W], typically C=256, H=W=64.
        """
        if not self._is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) to generate an embedding.")
        assert self._features is not None, "Features must exist if an image has been set."
        return self._features["image_embed"]

    def reset_predictor(self) -> None:
        """
        Resets the image embeddings and other state variables.
        """
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        self._is_batch = False
