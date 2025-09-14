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

from functools import partial
from typing import Sequence

import jax.image
import jax.numpy as jnp
import numpy as np
from PIL import Image
from scipy.ndimage import label as cc_label


def _postprocess_mask_numpy(mask, threshold, max_hole_area, max_sprinkle_area):
    filled = mask.copy()

    if max_hole_area > 0:
        bg_mask = (mask <= threshold).astype(np.uint8)
        labels, num = cc_label(bg_mask)
        areas = np.zeros_like(labels, dtype=np.float32)
        for l in range(1, num + 1):
            area = (labels == l).sum()
            areas[labels == l] = area
        is_hole = (labels > 0) & (areas <= max_hole_area)
        filled[is_hole] = threshold + 10.0

    if max_sprinkle_area > 0:
        fg_mask = (mask > threshold).astype(np.uint8)
        labels, num = cc_label(fg_mask)
        areas = np.zeros_like(labels, dtype=np.float32)
        for l in range(1, num + 1):
            area = (labels == l).sum()
            areas[labels == l] = area
        is_speckle = (labels > 0) & (areas <= max_sprinkle_area)
        filled[is_speckle] = threshold - 10.0

    return filled


class SAM2Transforms:
    def __init__(
        self, resolution: int, mask_threshold: float, max_hole_area: float = 0.0, max_sprinkle_area: float = 0.0
    ):
        self.resolution = resolution
        self.mask_threshold = mask_threshold
        self.max_hole_area = max_hole_area
        self.max_sprinkle_area = max_sprinkle_area
        self.mean = jnp.array([0.485, 0.456, 0.406])
        self.std = jnp.array([0.229, 0.224, 0.225])

    def _normalize(self, image: jnp.ndarray) -> jnp.ndarray:
        # Normalize per channel
        return (image - self.mean[None, None, :]) / self.std[None, None, :]

    def _resize(self, image: jnp.ndarray, shape: tuple[int, int]) -> jnp.ndarray:
        return jax.image.resize(image, (*shape, image.shape[-1]), method="bilinear")

    def __call__(self, img: Image.Image) -> jnp.ndarray:
        """Transforms a single PIL image into normalized, resized tensor."""
        image = jnp.array(np.array(img).astype(np.float32) / 255.0)
        image = self._resize(image, (self.resolution, self.resolution))
        image = self._normalize(image)
        return image

    def forward_batch(self, img_list: Sequence[Image.Image]) -> jnp.ndarray:
        """Apply transform to a list of PIL images and stack into a batch."""
        return jnp.stack([self(img) for img in img_list], axis=0)

    def transform_coords(
        self, coords: jnp.ndarray, normalize: bool = False, orig_hw: tuple[int, int] | None = None
    ) -> jnp.ndarray:
        if normalize:
            assert orig_hw is not None
            h, w = orig_hw
            coords = coords.at[..., 0].set(coords[..., 0] / w)
            coords = coords.at[..., 1].set(coords[..., 1] / h)
        coords = coords * self.resolution
        return coords

    def transform_boxes(
        self, boxes: jnp.ndarray, normalize: bool = False, orig_hw: tuple[int, ...] | None = None
    ) -> jnp.ndarray:
        boxes = boxes.reshape(-1, 2, 2)
        return self.transform_coords(boxes, normalize, orig_hw)

    @partial(jax.jit, static_argnames=["self", "orig_hw", "do_postprocess"])
    def postprocess_masks(
        self, masks: jnp.ndarray, orig_hw: tuple[int, int], *, do_postprocess: bool = False
    ) -> jnp.ndarray:
        """
        Perform post-processing on output masks. Safe for JIT if do_postprocess=False.
        """
        if do_postprocess:
            try:
                masks_np = np.array(masks)  # Move to CPU
                for b in range(masks_np.shape[0]):
                    for m in range(masks_np.shape[1]):
                        mask = masks_np[b, m]
                        processed = _postprocess_mask_numpy(
                            mask,
                            threshold=self.mask_threshold,
                            max_hole_area=self.max_hole_area,
                            max_sprinkle_area=self.max_sprinkle_area,
                        )
                        masks_np[b, m] = processed
            except Exception as e:
                import warnings

                warnings.warn(
                    f"{e}\n\nSkipping post-processing due to the error above.", category=UserWarning, stacklevel=2
                )
                masks_np = np.array(masks)
            masks = jnp.array(masks_np)  # Back to JAX

        # Resize to original shape (always in JAX)
        resized = jax.image.resize(
            masks, shape=(masks.shape[0], masks.shape[1], orig_hw[0], orig_hw[1]), method="bilinear"
        )
        return resized
