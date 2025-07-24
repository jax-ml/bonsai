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

from typing import Callable

import flax.nnx as nnx
import jax.numpy as jnp

from bonsai.models.sam2.model_utils import MLP, LayerNorm2d


class MaskDecoder(nnx.Module):
    def __init__(
        self,
        transformer_dim: int,
        transformer: nnx.Module,
        num_multimask_outputs: int = 3,
        activation: Callable = nnx.gelu,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid: bool = False,
        dynamic_multimask_via_stability: bool = False,
        dynamic_multimask_stability_delta: float = 0.05,
        dynamic_multimask_stability_thresh: float = 0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nnx.Embed(1, self.transformer_dim, rngs=rngs)
        self.num_mask_tokens = self.num_multimask_outputs + 1
        self.mask_tokens = nnx.Embed(self.num_mask_tokens, self.transformer_dim, rngs=rngs)

        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = nnx.Embed(1, self.transformer_dim, rngs=rngs)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        self.output_upscaling = nnx.Sequential(
            nnx.ConvTranspose(
                self.transformer_dim,
                self.transformer_dim // 4,
                kernel_size=(2, 2),
                strides=(2, 2),
                rngs=rngs,
            ),
            LayerNorm2d(self.transformer_dim // 4),
            activation,
            nnx.ConvTranspose(
                self.transformer_dim // 4,
                self.transformer_dim // 8,
                kernel_size=(2, 2),
                strides=(2, 2),
                rngs=rngs,
            ),
            activation,
        )

        self.use_high_res_features = use_high_res_features
        if self.use_high_res_features:
            self.conv_s0 = nnx.Conv(
                self.transformer_dim,
                self.transformer_dim // 8,
                kernel_size=(1, 1),
                strides=(1, 1),
                rngs=rngs,
            )
            self.conv_s1 = nnx.Conv(
                self.transformer_dim,
                self.transformer_dim // 4,
                kernel_size=(1, 1),
                strides=(1, 1),
                rngs=rngs,
            )

        self.output_hypernetworks_mlps = [
            MLP(
                self.transformer_dim,
                self.transformer_dim,
                self.transformer_dim // 8,
                3,
                rngs=rngs,
            )
            for _ in range(self.num_mask_tokens)
        ]

        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
            rngs=rngs,
        )

        if self.pred_obj_scores:
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(self.transformer_dim, self.transformer_dim, 1, 3, rngs=rngs)
            else:
                self.pred_obj_score_head = nnx.Linear(self.transformer_dim, 1, rngs=rngs)

        # When outputting a single mask, optionally we can dynamically fall back to the best
        # multimask output token if the single mask output token gives low stability scores.
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

    def __call__(
        self,
        image_embeddings: jnp.ndarray,
        image_pe: jnp.ndarray,
        sparse_prompt_embeddings: jnp.ndarray,
        dense_prompt_embeddings: jnp.ndarray,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: list[jnp.ndarray] | None = None,
        training: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Predict masks given image and prompt embeddings.
        """
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
        )

        if multimask_output:
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not training:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]

        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]
        else:
            # Take the mask output token. Here we *always* use the token for single mask output.
            # At test time, even if we track after 1-click (and using multimask_output=True),
            # we still take the single mask token here. The rationale is that we always track
            # after multiple clicks during training, so the past tokens seen during training
            # are always the single mask token (and we'll let it be the object-memory token).
            sam_tokens_out = mask_tokens_out[:, 0:1]

        return masks, iou_pred, sam_tokens_out, object_score_logits

    def predict_masks(
        self,
        image_embeddings: jnp.ndarray,
        image_pe: jnp.ndarray,
        sparse_prompt_embeddings: jnp.ndarray,
        dense_prompt_embeddings: jnp.ndarray,
        repeat_image: bool,
        high_res_features: list[jnp.ndarray] | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Predicts masks. See '__call__' for more details."""
        s = 0
        if self.pred_obj_scores:
            output_tokens = jnp.concatenate(
                [
                    self.obj_score_token.embedding,  # [1, C]
                    self.iou_token.embedding,  # [1, C]
                    self.mask_tokens.embedding,  # [num_mask_tokens, C]
                ],
                axis=0,
            )
            s = 1
        else:
            output_tokens = jnp.concatenate([self.iou_token.embedding, self.mask_tokens.embedding], axis=0)

        batch_size = sparse_prompt_embeddings.shape[0]
        output_tokens = jnp.broadcast_to(
            output_tokens[None, :, :], (batch_size, *output_tokens.shape)
        )  # [B, num_tokens, C]
        tokens = jnp.concatenate([output_tokens, sparse_prompt_embeddings], axis=1)  # [B, T, C]

        # Repeat image and PE if needed
        if repeat_image:
            src = jnp.repeat(image_embeddings, tokens.shape[0], axis=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0], "Mismatch in batch size"
            src = image_embeddings
        src = src + dense_prompt_embeddings
        B, C, H, W = src.shape

        pos_src = jnp.repeat(image_pe, tokens.shape[0], axis=0)

        # Transformer forward
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, (s + 1) : (s + 1 + self.num_mask_tokens), :]

        # Reshape src for mask decoding
        src = src.reshape(B, H, W, C)  # B H W C for conv

        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling.layers
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = dc1(src)
            upscaled_embedding += feat_s1.transpose(0, 2, 3, 1)
            upscaled_embedding = act1(ln1(upscaled_embedding))
            upscaled_embedding = dc2(upscaled_embedding)
            upscaled_embedding += feat_s0.transpose(0, 2, 3, 1)
            upscaled_embedding = act2(upscaled_embedding)

        # Apply hypernetworks
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper = self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])  # [B, C]
            hyper_in_list.append(hyper)
        hyper_in = jnp.stack(hyper_in_list, axis=1)  # [B, num_tokens, C]

        b, h, w, c = upscaled_embedding.shape
        flat_features = upscaled_embedding.transpose(0, 3, 1, 2).reshape((b, c, -1))  # [B, C, H*W]
        masks = jnp.einsum("bqc,bch->bqh", hyper_in, flat_features)  # [B, num_tokens, H*W]
        masks = masks.reshape((b, self.num_mask_tokens, h, w))

        # Predict IoU and object score logits
        iou_pred = self.iou_prediction_head(iou_token_out)

        if self.pred_obj_scores:
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            object_score_logits = jnp.full((iou_pred.shape[0], 1), 10.0, dtype=iou_pred.dtype)

        return masks, iou_pred, mask_tokens_out, object_score_logits

    def _get_stability_scores(self, mask_logits: jnp.ndarray) -> jnp.ndarray:
        """
        Compute stability scores of the mask logits based on the IoU between upper and
        lower thresholds.
        """
        mask_logits = mask_logits.reshape(*mask_logits.shape[:-2], -1)  # flatten H x W
        stability_delta = self.dynamic_multimask_stability_delta

        area_i = jnp.sum(mask_logits > stability_delta, axis=-1).astype(jnp.float32)
        area_u = jnp.sum(mask_logits > -stability_delta, axis=-1).astype(jnp.float32)

        stability_scores = jnp.where(area_u > 0, area_i / area_u, 1.0)
        return stability_scores

    def _dynamic_multimask_via_stability(
        self,
        all_mask_logits: jnp.ndarray,  # [B, T, H, W]
        all_iou_scores: jnp.ndarray,  # [B, T]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        If the stability score from the single-mask output (token 0) is low,
        replace it with the best scoring multi-mask output (tokens 1~3).
        """
        # Select best multimask prediction (tokens 1~)
        multimask_logits = all_mask_logits[:, 1:, :, :]  # [B, T-1, H, W]
        multimask_iou_scores = all_iou_scores[:, 1:]  # [B, T-1]

        best_scores_inds = jnp.argmax(multimask_iou_scores, axis=-1)  # [B]
        batch_inds = jnp.arange(all_iou_scores.shape[0])  # [B]

        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]  # [B, H, W]
        best_multimask_logits = best_multimask_logits[:, None, :, :]  # [B, 1, H, W]
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
        best_multimask_iou_scores = best_multimask_iou_scores[:, None]  # [B, 1]

        # Single-mask prediction (token 0)
        singlemask_logits = all_mask_logits[:, 0:1, :, :]  # [B, 1, H, W]
        singlemask_iou_scores = all_iou_scores[:, 0:1]  # [B, 1]

        stability_scores = self._get_stability_scores(singlemask_logits)  # [B, 1]
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh  # [B, 1]

        # Select between single and multimask based on stability
        mask_logits_out = jnp.where(is_stable[:, :, None, None], singlemask_logits, best_multimask_logits)
        iou_scores_out = jnp.where(is_stable, singlemask_iou_scores, best_multimask_iou_scores)

        return mask_logits_out, iou_scores_out
