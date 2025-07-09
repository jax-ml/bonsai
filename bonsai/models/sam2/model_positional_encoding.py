# Copyright 2025 Google LLC
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

import math

import flax.nnx as nnx
import jax
import jax.numpy as jnp


class PositionEmbeddingSine(nnx.Module):
    """
    Sine-based 2D positional encoding, analogous to “Attention Is All You Need” but
    generalized to images.

    Args:
      num_pos_feats: output channels will be 2 * num_pos_feats
      temperature: frequency scaling
      normalize: whether to normalize coordinates to [0, 1]
      scale: if normalize, multiply normalized coords by this (defaults to 2π)
    """

    def __init__(
        self,
        num_pos_feats: int,
        temperature: float = 10000.0,
        normalize: bool = True,
        scale: float | None = None,
    ):
        super().__init__()
        assert num_pos_feats % 2 == 0, "num_pos_feats must be even"
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale if (scale is not None) else 2 * math.pi

    def _encode_xy(self, x: jnp.ndarray, y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # x, y are 1D arrays of length N in [0, 1] or in pixel coords if normalize=False
        x_embed = x * self.scale
        y_embed = y * self.scale

        dim_t = jnp.arange(self.num_pos_feats, dtype=jnp.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t  # [N, D]
        pos_y = y_embed[:, None] / dim_t

        # interleave sin/cos on even/odd dims
        pos_x = jnp.stack([jnp.sin(pos_x[:, 0::2]), jnp.cos(pos_x[:, 1::2])], axis=2).reshape(x.shape[0], -1)
        pos_y = jnp.stack([jnp.sin(pos_y[:, 0::2]), jnp.cos(pos_y[:, 1::2])], axis=2).reshape(y.shape[0], -1)

        return pos_x, pos_y

    def encode_boxes(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        w: jnp.ndarray,
        h: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Encode box centers (x, y) and sizes (w, h):
          x, y, w, h have shape [B] or [N] → outputs [B, 2*num_pos_feats + 2]
        """
        pos_x, pos_y = self._encode_xy(x, y)
        # concatenate [pos_y, pos_x, h, w]
        return jnp.concatenate([pos_y, pos_x, h[:, None], w[:, None]], axis=-1)

    def encode_points(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        labels: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Encode point prompts with labels:
          x, y, labels shapes: [B, N]
          output: [B, N, 2*num_pos_feats + 1]
        """
        B, N = x.shape
        flat_x = x.reshape(-1)
        flat_y = y.reshape(-1)
        pos_x, pos_y = self._encode_xy(flat_x, flat_y)
        pos_x = pos_x.reshape(B, N, -1)
        pos_y = pos_y.reshape(B, N, -1)
        lbl = labels[..., None]  # [B, N, 1]
        return jnp.concatenate([pos_y, pos_x, lbl], axis=-1)

    def __call__(self, feature_map: jnp.ndarray) -> jnp.ndarray:
        """
        Given feature_map shape [B, C, H, W], return pos encoding [B, 2*num_pos_feats, H, W].
        """
        B, _, H, W = feature_map.shape

        # create normalized coordinate grids in [0,1]
        y = jnp.linspace(0.5 / H, 1.0 - 0.5 / H, H)
        x = jnp.linspace(0.5 / W, 1.0 - 0.5 / W, W)
        yy, xx = jnp.meshgrid(y, x, indexing="ij")  # [H, W]

        if self.normalize:
            yy = yy
            xx = xx
        else:
            # scale to pixel coords
            yy = yy * H
            xx = xx * W

        # flatten and encode
        flat_x = xx.reshape(-1)
        flat_y = yy.reshape(-1)
        pos_x, pos_y = self._encode_xy(flat_x, flat_y)

        # reassemble into [H, W, D]
        D = pos_x.shape[-1]
        pos_x = pos_x.reshape(H, W, D)
        pos_y = pos_y.reshape(H, W, D)

        # concatenate along channel
        pe = jnp.concatenate([pos_y, pos_x], axis=-1)  # [H, W, 2D]
        pe = pe.transpose(2, 0, 1)  # [2D, H, W]
        pe = jnp.broadcast_to(pe[None, ...], (B, *pe.shape))  # [B, 2D, H, W]

        return pe


class PositionEmbeddingRandom(nnx.Module):
    """
    Positional encoding using random spatial frequencies.

    This module encodes 2D coordinates using a shared Gaussian matrix
    as in the original SAM2 implementation.

    Args:
        num_pos_feats: Output embedding dim will be 2 * num_pos_feats.
        scale: Multiplier for random frequencies. Defaults to 1.0.
        rngs: NNX random seed container.
    """

    def __init__(self, num_pos_feats: int, scale: float = 1.0, *, rngs: nnx.Rngs):
        key = rngs.params()
        gaussian_matrix = scale * jax.random.normal(key, (2, num_pos_feats))  # shape [2, C]
        self.gaussian_matrix = nnx.Param(gaussian_matrix)  # shape [2, C]

    def _pe_encoding(self, coords: jnp.ndarray) -> jnp.ndarray:
        """
        Encodes coordinates in [0, 1]^2 using random sinusoidal projection.

        Args:
            coords: [..., 2] normalized in [0, 1]

        Returns:
            [..., 2C] positional embeddings
        """
        coords = 2.0 * coords - 1.0  # rescale to [-1, 1]
        projected = coords @ self.gaussian_matrix  # [..., C]
        projected = 2.0 * jnp.pi * projected
        return jnp.concatenate([jnp.sin(projected), jnp.cos(projected)], axis=-1)  # [..., 2C]

    def __call__(self, size: tuple[int, int]) -> jnp.ndarray:
        """
        Generate positional encoding grid for shape (H, W).

        Args:
            size: (H, W)

        Returns:
            [2C, H, W] positional encoding map
        """
        H, W = size
        y_embed = (jnp.arange(H) + 0.5) / H  # [H]
        x_embed = (jnp.arange(W) + 0.5) / W  # [W]

        # Create meshgrid [H, W, 2]
        yy, xx = jnp.meshgrid(y_embed, x_embed, indexing="ij")
        coords = jnp.stack([xx, yy], axis=-1)  # [H, W, 2]
        pe = self._pe_encoding(coords)  # [H, W, 2C]
        return jnp.transpose(pe, (2, 0, 1))  # [2C, H, W]

    def forward_with_coords(self, coords_input: jnp.ndarray, image_size: tuple[int, int]) -> jnp.ndarray:
        """
        Encode arbitrary input coordinates, normalized by image size.

        Args:
            coords_input: [B, N, 2] in pixel space
            image_size: (H, W)

        Returns:
            [B, N, 2C] positional embeddings
        """
        H, W = image_size
        coords = coords_input / jnp.array([W, H])
        return self._pe_encoding(coords)


def init_t_xy(end_x: int, end_y: int):
    """Initializes flattened x and y position coordinates on a 2D grid."""
    t = jnp.arange(end_x * end_y)
    t_x = jnp.mod(t, end_x).astype(jnp.float32)
    t_y = (t // end_x).astype(jnp.float32)
    return t_x, t_y


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 10000.0):
    """
    Computes axial rotary embeddings for 2D inputs.

    Returns:
      freqs_cis: complex sinusoidal matrix of shape [end_x * end_y, dim]
    """
    idx = jnp.arange(0, dim, 4)[: dim // 4]
    freqs = 1.0 / (theta ** (idx / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = jnp.outer(t_x, freqs)
    freqs_y = jnp.outer(t_y, freqs)

    cis_x = jax.lax.complex(jnp.cos(freqs_x), jnp.sin(freqs_x))
    cis_y = jax.lax.complex(jnp.cos(freqs_y), jnp.sin(freqs_y))

    return jnp.concatenate([cis_x, cis_y], axis=-1)  # [HW, C]


def reshape_for_broadcast(freqs_cis: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """
    Reshapes freqs_cis to broadcast over query/key shape [B, H, T, C].
    """
    print(freqs_cis.shape)
    assert freqs_cis.shape == (
        x.shape[2],
        x.shape[3],
    ), f"Expected freqs_cis shape {(x.shape[2], x.shape[3])}, got {freqs_cis.shape}"
    return freqs_cis[jnp.newaxis, jnp.newaxis, :, :]  # [1, 1, T, C]


def apply_rotary_enc(
    q: jnp.ndarray,
    k: jnp.ndarray,
    freqs_cis: jnp.ndarray,
    repeat_freqs_k: bool = False,
):
    """
    Applies rotary position encoding to queries and keys.

    Args:
      q, k: [B, H, T, C] real-valued inputs with even C
      freqs_cis: [T, C] complex rotations
      repeat_freqs_k: whether to repeat freqs_cis to match k length

    Returns:
      q_rot, k_rot: same shape as input
    """
    assert q.shape[-1] % 2 == 0, "q last dim must be even"
    assert k.shape[-1] % 2 == 0, "k last dim must be even"

    q_c = jax.lax.complex(q[..., 0::2], q[..., 1::2])
    k_c = jax.lax.complex(k[..., 0::2], k[..., 1::2])
    print(q_c.shape, k_c.shape)

    freqs_cis = reshape_for_broadcast(freqs_cis, q_c)

    q_rot = q_c * freqs_cis

    if repeat_freqs_k:
        r = k.shape[2] // q.shape[2]
        freqs_cis = jnp.repeat(freqs_cis, r, axis=2)

    k_rot = k_c * freqs_cis

    def to_real(x_c):
        return jnp.stack([jnp.real(x_c), jnp.imag(x_c)], axis=-1).reshape(*x_c.shape[:-1], -1)

    return to_real(q_rot), to_real(k_rot)


def get_1d_sine_pe(pos_inds: jnp.ndarray, dim: int, temperature: float = 10000.0) -> jnp.ndarray:
    """
    Generate 1D sine-cosine positional embeddings as in the Transformer paper.
    """
    pe_dim = dim // 2
    # compute inverse frequencies
    dim_t = jnp.arange(pe_dim, dtype=jnp.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)
    # scale positions by inverse frequencies
    pos = jnp.expand_dims(pos_inds, axis=-1) / dim_t
    # interleave sin and cos
    return jnp.concatenate([jnp.sin(pos), jnp.cos(pos)], axis=-1)
