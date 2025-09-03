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

from bonsai.models.sam2.model_positional_encoding import PositionEmbeddingRandom
from bonsai.models.sam2.model_utils import LayerNorm2d


class PromptEncoder(nnx.Module):
    """
    PromptEncoder (NNX)

    Encodes sparse (points, boxes) and dense (masks) prompts for input to
    the SAM2 mask decoder. Produces both sparse embeddings for prompts
    and dense embeddings for mask features.

    Args:
        embed_dim: Dimension of output embeddings.
        image_embedding_size: Spatial (H, W) of image encoder output.
        input_image_size: Original padded image input size (H, W).
        mask_in_chans: Hidden channels used for mask embedding tower.
        activation: Activation used in mask tower (default: GELU).
        rngs: NNX random seed container.
    """

    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: tuple[int, int],
        input_image_size: tuple[int, int],
        mask_in_chans: int,
        activation: Callable = nnx.gelu,
        *,
        rngs: nnx.Rngs,
    ):
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size

        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2, rngs=rngs)

        # Four embeddings for points: pos, neg, box-corner1, box-corner2
        self.point_embeddings = [nnx.Embed(1, embed_dim, rngs=rngs) for _ in range(4)]
        self.not_a_point_embed = nnx.Embed(1, embed_dim, rngs=rngs)  # Used when label == -1

        # Conv tower for mask downscaling
        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nnx.Sequential(
            nnx.Conv(1, mask_in_chans // 4, kernel_size=(2, 2), strides=(2, 2), rngs=rngs),
            LayerNorm2d(mask_in_chans // 4),
            activation,
            nnx.Conv(mask_in_chans // 4, mask_in_chans, kernel_size=(2, 2), strides=(2, 2), rngs=rngs),
            LayerNorm2d(mask_in_chans),
            activation,
            nnx.Conv(mask_in_chans, embed_dim, kernel_size=(1, 1), rngs=rngs),
        )

        self.no_mask_embed = nnx.Embed(1, embed_dim, rngs=rngs)

    def get_dense_pe(self) -> jnp.ndarray:
        """
        Returns positional encoding over the spatial image embedding grid.
        Shape: (1, embed_dim, H, W)
        """
        return self.pe_layer(self.image_embedding_size)[None, ...]

    def _embed_points(self, points, labels, pad=True) -> jnp.ndarray:
        """
        Embed point prompts with label-specific offsets.
        Label -1: no point
              0: neg point
              1: pos point
              2: box corner 1
              3: box corner 2
        """
        points = points + 0.5  # center of pixel
        B = points.shape[0]
        if pad:
            points = jnp.concatenate([points, jnp.zeros((B, 1, 2))], axis=1)
            labels = jnp.concatenate([labels, -jnp.ones((B, 1))], axis=1)

        pe = self.pe_layer.forward_with_coords(points, self.input_image_size)

        # Zero index tensor for embedding lookup
        zero_idx = jnp.zeros_like(labels, dtype=jnp.int32)

        # Route based on label
        out = jnp.where(labels[..., None] == -1, self.not_a_point_embed(zero_idx), pe)
        for i in range(4):
            out = jnp.where(labels[..., None] == i, pe + self.point_embeddings[i](zero_idx), out)
        return out  # [B, N, D]

    def _embed_boxes(self, boxes) -> jnp.ndarray:
        """
        Embed boxes by treating corners as special points.
        Input: [B, 4] → [B, 2, 2]
        Output: [B, 2, D]
        """
        boxes = boxes + 0.5
        coords = boxes.reshape((-1, 2, 2))  # [B, 2, 2]
        corner_pe = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        zero_idx = jnp.zeros((corner_pe.shape[0],), dtype=jnp.int32)
        corner_pe = corner_pe.at[:, 0, :].add(self.point_embeddings[2](zero_idx))
        corner_pe = corner_pe.at[:, 1, :].add(self.point_embeddings[3](zero_idx))
        return corner_pe  # [B, 2, D]

    def _embed_masks(self, masks: jnp.ndarray) -> jnp.ndarray:
        """Downscale binary masks into dense feature maps."""
        return self.mask_downscaling(masks)

    def _get_batch_size(self, points, boxes, masks) -> int:
        """Resolve batch size from any prompt input."""
        if points is not None:
            return points[0].shape[0]
        if boxes is not None:
            return boxes.shape[0]
        if masks is not None:
            return masks.shape[0]
        return 1

    def __call__(
        self, points: tuple[jnp.ndarray, jnp.ndarray] | None, boxes: jnp.ndarray | None, masks: jnp.ndarray | None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Embed all input prompts.

        Args:
          points: Tuple of (coords: [B, N, 2], labels: [B, N]) or None
          boxes: [B, 4] or None
          masks: [B, 1, H, W] or None

        Returns:
          sparse_embeddings: [B, N_pts+N_boxes, D]
          dense_embeddings: [B, D, H, W]
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = jnp.zeros((bs, 0, self.embed_dim))

        if points is not None:
            coords, labels = points
            pe = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = jnp.concatenate([sparse_embeddings, pe], axis=1)

        if boxes is not None:
            be = self._embed_boxes(boxes)
            sparse_embeddings = jnp.concatenate([sparse_embeddings, be], axis=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = jnp.broadcast_to(
                self.no_mask_embed(jnp.zeros((bs,), dtype=jnp.int32))[:, :, None, None],
                (bs, self.embed_dim, *self.image_embedding_size),
            )

        return sparse_embeddings, dense_embeddings
