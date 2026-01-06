import dataclasses
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array


@dataclasses.dataclass
class VJEPA2EncoderOutput:
    """Output from VJEPA2 Encoder."""

    last_hidden_state: Array


@dataclasses.dataclass
class VJEPA2PredictorOutput:
    """Output from VJEPA2 Predictor."""

    last_hidden_state: Array
    target_hidden_state: Optional[Array] = None


@dataclasses.dataclass
class VJEPA2ModelOutput:
    """Output from VJEPA2 Model."""

    last_hidden_state: Array
    masked_hidden_state: Optional[Array] = None
    predictor_output: Optional[VJEPA2PredictorOutput] = None


@dataclasses.dataclass
class VJEPA2ClassificationOutput:
    """Output from VJEPA2 Classification Model."""

    logits: Array
    last_hidden_state: Array


@dataclasses.dataclass(frozen=True)
class VJEPA2FlaxConfig:
    """Configuration for VJEPA2 Flax model."""

    model_type: str = "vjepa2"

    # Patch embedding params
    patch_size: int = 16
    tubelet_size: int = 2
    in_chans: int = 3

    # Input dimensions
    crop_size: int = 256
    frames_per_clip: int = 64

    # Encoder params
    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    mlp_ratio: float = 4.0
    layer_norm_eps: float = 1e-6
    qkv_bias: bool = True
    hidden_act: str = "gelu"

    # Predictor params
    pred_hidden_size: int = 384
    pred_num_attention_heads: int = 12
    pred_num_hidden_layers: int = 12
    pred_num_mask_tokens: int = 10
    pred_zero_init_mask_tokens: bool = True
    pred_mlp_ratio: float = 4.0

    # Pooler params
    num_pooler_layers: int = 3

    # Classification params (optional)
    num_labels: int = 174  # SSv2 default, 48 for diving48

    @classmethod
    def vitl_fpc64_256(cls):
        """ViT-Large with 64 frames per clip, 256 crop size."""
        return cls()

    @classmethod
    def vith_fpc64_256(cls):
        return cls(
            hidden_size=2180,
            num_attention_heads=16,
            num_hidden_layers=32,
        )

    @classmethod
    def vitg_fpc64_256(cls):
        return cls(
            hidden_size=1408,
            num_attention_heads=22,
            num_hidden_layers=40,
            mlp_ratio=4.363636363636363,
        )

    @classmethod
    def vitg_fpc64_384(cls):
        return cls(
            crop_size=384,
            hidden_size=1408,
            num_attention_heads=22,
            num_hidden_layers=40,
            mlp_ratio=4.363636363636363,
        )

    @classmethod
    def vitl_fpc16_256(cls):
        return cls(frames_per_clip=16)

    @classmethod
    def vitl_fpc32_256(cls):
        return cls(
            frames_per_clip=32,
            num_labels=48,
        )

    @classmethod
    def vitg_fpc32_384(cls):
        return cls(
            frames_per_clip=32,
            crop_size=384,
            hidden_size=1408,
            num_attention_heads=22,
            num_hidden_layers=40,
            mlp_ratio=4.363636363636363,
        )

    @classmethod
    def standard_test(cls):
        """Small config for testing."""
        return cls(
            hidden_size=64,
            num_attention_heads=2,
            num_hidden_layers=2,
            pred_hidden_size=32,
            pred_num_attention_heads=2,
            pred_num_hidden_layers=2,
            crop_size=64,
            frames_per_clip=4,
            num_pooler_layers=1,
        )


ACT2FN = {
    "gelu": nnx.gelu,
    "silu": nnx.silu,
    "relu": nnx.relu,
}


class VJEPA2PatchEmbeddings3D(nnx.Module):
    """Image to Patch Embedding using 3D convolution.

    Input: (batch_size, num_frames, height, width, channels)
    Output: (batch_size, num_patches, hidden_size)
    """

    def __init__(self, config: VJEPA2FlaxConfig, hidden_size: int, rngs: nnx.Rngs):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size

        # 3D conv for video patches
        kernel = (config.tubelet_size, config.patch_size, config.patch_size)
        self.proj = nnx.Conv(
            in_features=config.in_chans,
            out_features=hidden_size,
            kernel_size=kernel,
            strides=kernel,
            rngs=rngs,
        )

    def __call__(self, pixel_values_videos: Array) -> Array:
        """
        Args:
            pixel_values_videos: (B, T, H, W, C) - already transposed from PyTorch format
        Returns:
            embeddings: (B, num_patches, hidden_size)
        """
        batch_size = pixel_values_videos.shape[0]

        # Conv3d expects (B, T, H, W, C) which is what we have
        x = self.proj(pixel_values_videos)  # (B, T', H', W', hidden_size)

        # Flatten spatial and temporal dimensions
        x = x.reshape(batch_size, -1, self.hidden_size)  # (B, num_patches, hidden_size)

        return x


class VJEPA2Embeddings(nnx.Module):
    """Construct patch embeddings for video input."""

    def __init__(self, config: VJEPA2FlaxConfig, hidden_size: int, rngs: nnx.Rngs):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.patch_embeddings = VJEPA2PatchEmbeddings3D(config, hidden_size=hidden_size, rngs=rngs)

    def __call__(self, pixel_values_videos: Array) -> Array:
        """
        Args:
            pixel_values_videos: (B, T, H, W, C) - already transposed
        Returns:
            embeddings: (B, num_patches, hidden_size)
        """
        num_frames = pixel_values_videos.shape[1]  # T dimension

        # Handle case where frames < tubelet_size by duplicating
        if num_frames < self.config.tubelet_size:
            repeats = self.config.tubelet_size
            pixel_values_videos = jnp.tile(pixel_values_videos, (1, repeats, 1, 1, 1))

        embeddings = self.patch_embeddings(pixel_values_videos)
        return embeddings


def rotate_queries_or_keys(x: Array, pos: Array, dim: int) -> Array:
    """Apply rotary position embeddings to queries or keys.

    Args:
        x: Input tensor of shape (B, num_heads, N, head_dim)
        pos: Position indices of shape (B, num_heads, N) or (N,) for broadcasting
        dim: Dimension size for this component (d_dim, h_dim, or w_dim)

    Returns:
        Rotated tensor of same shape as input but only for first `dim` dimensions
    """
    _, _, _, D = x.shape

    # Compute frequencies - use input dtype like PyTorch
    omega = jnp.arange(D // 2, dtype=x.dtype)
    omega = omega / (D / 2.0)
    omega = 1.0 / (10000**omega)  # (D/2,)

    # Compute angles: pos * omega
    freq = pos[..., None] * omega  # (..., N, D/2)

    # Build rotation matrix
    emb_sin = jnp.sin(freq)  # (..., N, D/2)
    emb_cos = jnp.cos(freq)  # (..., N, D/2)

    # Repeat for full dimension
    emb_sin = jnp.tile(emb_sin, (1, 1, 1, 2))  # (..., N, D)
    emb_cos = jnp.tile(emb_cos, (1, 1, 1, 2))  # (..., N, D)

    # Split into pairs and rotate like PyTorch
    y = x.reshape(*x.shape[:-1], -1, 2)  # (..., N, D/2, 2)
    y1, y2 = y[..., 0], y[..., 1]  # Each (..., N, D/2)

    # Stack as (-y2, y1) and flatten
    y_rotated = jnp.stack((-y2, y1), axis=-1)  # (..., N, D/2, 2)
    y_rotated = y_rotated.reshape(x.shape)  # (..., N, D)

    # Apply rotation: x * cos + rotated * sin
    rotated = (x * emb_cos) + (y_rotated * emb_sin)

    return rotated


class VJEPA2RopeAttention(nnx.Module):
    """Multi-head attention with 3D RoPE for video."""

    def __init__(
        self,
        config: VJEPA2FlaxConfig,
        hidden_size: int,
        num_attention_heads: int,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

        if hidden_size % num_attention_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} is not divisible by num_attention_heads {num_attention_heads}")

        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = num_attention_heads * self.attention_head_size

        # Q, K, V projections
        self.query = nnx.Linear(hidden_size, self.all_head_size, use_bias=config.qkv_bias, rngs=rngs)
        self.key = nnx.Linear(hidden_size, self.all_head_size, use_bias=config.qkv_bias, rngs=rngs)
        self.value = nnx.Linear(hidden_size, self.all_head_size, use_bias=config.qkv_bias, rngs=rngs)

        # Output projection
        self.proj = nnx.Linear(hidden_size, hidden_size, rngs=rngs)

        # Grid dimensions for position encoding
        self.grid_size = config.crop_size // config.patch_size
        self.grid_depth = config.frames_per_clip // config.tubelet_size

        # RoPE dimension splits (divide head_dim into 3 parts for d, h, w)
        self.d_dim = int(2 * ((self.attention_head_size // 3) // 2))
        self.h_dim = int(2 * ((self.attention_head_size // 3) // 2))
        self.w_dim = int(2 * ((self.attention_head_size // 3) // 2))

        self.scaling = self.attention_head_size**-0.5

    def _get_frame_pos(self, ids: Array) -> Array:
        """Get frame position from token ids."""
        tokens_per_frame = self.grid_size * self.grid_size
        return ids // tokens_per_frame

    def _get_height_pos(self, ids: Array) -> Array:
        """Get height position from token ids."""
        tokens_per_frame = self.grid_size * self.grid_size
        frame_ids = self._get_frame_pos(ids)
        ids = ids - tokens_per_frame * frame_ids
        tokens_per_row = self.grid_size
        return ids // tokens_per_row

    def get_position_ids(self, x: Array, masks: Optional[Array] = None):
        """Compute frame, height, width position ids.

        Args:
            x: Input tensor (B, N, D)
            masks: Optional position masks (B, N)

        Returns:
            Tuple of (frame_ids, height_ids, width_ids)
        """
        token_size = x.shape[1]

        if masks is not None:
            # Use provided masks, expand for heads
            ids = jnp.broadcast_to(
                masks[:, None, :],
                (masks.shape[0], self.num_attention_heads, masks.shape[1]),
            )
        else:
            # Create sequential ids
            ids = jnp.arange(token_size)

        tokens_per_frame = self.grid_size * self.grid_size
        frame_ids = self._get_frame_pos(ids)

        tokens_per_row = self.grid_size
        height_ids = self._get_height_pos(ids)

        # Width = ids - frame_component - height_component
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids

        return frame_ids, height_ids, width_ids

    def apply_rotary_embeddings(self, qk: Array, pos_ids: Tuple[Array, Array, Array]) -> Array:
        """Apply 3D rotary embeddings.

        Args:
            qk: Query or key tensor (B, num_heads, N, head_dim)
            pos_ids: Tuple of (frame_ids, height_ids, width_ids)

        Returns:
            Rotated tensor
        """
        d_mask, h_mask, w_mask = pos_ids

        s = 0
        # Rotate depth component
        qkd = rotate_queries_or_keys(qk[..., s : s + self.d_dim], pos=d_mask, dim=self.d_dim)
        s += self.d_dim

        # Rotate height component
        qkh = rotate_queries_or_keys(qk[..., s : s + self.h_dim], pos=h_mask, dim=self.h_dim)
        s += self.h_dim

        # Rotate width component
        qkw = rotate_queries_or_keys(qk[..., s : s + self.w_dim], pos=w_mask, dim=self.w_dim)
        s += self.w_dim

        # Combine rotated dimensions with any remaining unrotated dimensions
        if s < self.attention_head_size:
            qkr = qk[..., s:]
            qk = jnp.concatenate([qkd, qkh, qkw, qkr], axis=-1)
        else:
            qk = jnp.concatenate([qkd, qkh, qkw], axis=-1)

        return qk

    def __call__(
        self,
        hidden_states: Array,
        position_mask: Optional[Array] = None,
    ) -> Array:
        """
        Args:
            hidden_states: (B, N, hidden_size)
            position_mask: Optional position mask (B, N)

        Returns:
            output: (B, N, hidden_size)
        """
        batch_size, seq_length, _ = hidden_states.shape

        # Project to Q, K, V
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        # Reshape to (B, N, num_heads, head_dim) then transpose to (B, num_heads, N, head_dim)
        query_layer = query_layer.reshape(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        query_layer = query_layer.transpose(0, 2, 1, 3)

        key_layer = key_layer.reshape(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        key_layer = key_layer.transpose(0, 2, 1, 3)

        value_layer = value_layer.reshape(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        value_layer = value_layer.transpose(0, 2, 1, 3)

        # Get position IDs and apply rotary embeddings
        pos_ids = self.get_position_ids(hidden_states, masks=position_mask)
        query_layer = self.apply_rotary_embeddings(query_layer, pos_ids)
        key_layer = self.apply_rotary_embeddings(key_layer, pos_ids)

        # Compute attention: (B, H, N, D) @ (B, H, D, N) -> (B, H, N, N)
        attn_weights = jnp.matmul(query_layer, key_layer.transpose(0, 1, 3, 2)) * self.scaling

        # Softmax
        attn_weights = nnx.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(query_layer.dtype)

        # Apply attention to values: (B, H, N, N) @ (B, H, N, D) -> (B, H, N, D)
        context_layer = jnp.matmul(attn_weights, value_layer)

        # Transpose back: (B, H, N, D) -> (B, N, H, D) -> (B, N, H*D)
        context_layer = context_layer.transpose(0, 2, 1, 3)
        context_layer = context_layer.reshape(batch_size, seq_length, self.all_head_size)

        # Output projection
        output = self.proj(context_layer)

        return output


class VJEPA2MLP(nnx.Module):
    """Feed-forward network."""

    def __init__(
        self,
        config: VJEPA2FlaxConfig,
        hidden_size: int,
        mlp_ratio: float,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        hidden_features = int(hidden_size * mlp_ratio)
        self.fc1 = nnx.Linear(hidden_size, hidden_features, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_features, hidden_size, rngs=rngs)
        self.activation = ACT2FN[config.hidden_act]

    def __call__(self, hidden_states: Array) -> Array:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class VJEPA2Layer(nnx.Module):
    """Single transformer layer with pre-norm."""

    def __init__(
        self,
        config: VJEPA2FlaxConfig,
        hidden_size: int,
        num_attention_heads: int,
        mlp_ratio: float,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.norm1 = nnx.LayerNorm(hidden_size, epsilon=config.layer_norm_eps, rngs=rngs)
        self.attention = VJEPA2RopeAttention(
            config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(hidden_size, epsilon=config.layer_norm_eps, rngs=rngs)
        self.mlp = VJEPA2MLP(config, hidden_size=hidden_size, mlp_ratio=mlp_ratio, rngs=rngs)

    def __call__(
        self,
        hidden_states: Array,
        position_mask: Optional[Array] = None,
    ) -> Array:
        # Self-attention with pre-norm
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attention(hidden_states, position_mask=position_mask)
        hidden_states = hidden_states + residual

        # MLP with pre-norm
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states


class VJEPA2Encoder(nnx.Module):
    """VJEPA2 Encoder consisting of embeddings and transformer layers."""

    def __init__(self, config: VJEPA2FlaxConfig, rngs: nnx.Rngs):
        super().__init__()
        self.config = config

        self.embeddings = VJEPA2Embeddings(config, hidden_size=config.hidden_size, rngs=rngs)

        self.layer = nnx.List(
            [
                VJEPA2Layer(
                    config,
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    mlp_ratio=config.mlp_ratio,
                    rngs=rngs,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.layernorm = nnx.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps, rngs=rngs)

    def __call__(self, pixel_values_videos: Array) -> VJEPA2EncoderOutput:
        """
        Args:
            pixel_values_videos: (B, T, H, W, C)

        Returns:
            VJEPA2EncoderOutput with last_hidden_state
        """
        hidden_states = self.embeddings(pixel_values_videos)

        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, position_mask=None)

        hidden_states = self.layernorm(hidden_states)

        return VJEPA2EncoderOutput(last_hidden_state=hidden_states)


def apply_masks(tensor: Array, masks: list) -> Array:
    """Apply masks to tensor.

    Args:
        tensor: (B, num_patches, D)
        masks: List of arrays of shape (B, num_keep) with indices to keep

    Returns:
        Concatenated masked tensors
    """
    all_masked = []
    for mask in masks:
        # Use advanced indexing to gather
        batch_size = tensor.shape[0]
        feature_dim = tensor.shape[-1]

        # Expand mask for gathering: (B, num_keep) -> (B, num_keep, D)
        mask_expanded = jnp.broadcast_to(mask[..., None], (batch_size, mask.shape[1], feature_dim))

        # Gather along sequence dimension
        gathered = jnp.take_along_axis(tensor, mask_expanded.astype(jnp.int32), axis=1)
        all_masked.append(gathered)

    return jnp.concatenate(all_masked, axis=0)


class VJEPA2PredictorEmbeddings(nnx.Module):
    """Predictor embeddings with mask tokens."""

    def __init__(self, config: VJEPA2FlaxConfig, rngs: nnx.Rngs):
        super().__init__()
        self.config = config

        self.predictor_embeddings = nnx.Linear(config.hidden_size, config.pred_hidden_size, rngs=rngs)

        # Mask tokens: (num_mask_tokens, 1, 1, pred_hidden_size)
        if config.pred_zero_init_mask_tokens:
            mask_tokens = jnp.zeros((config.pred_num_mask_tokens, 1, 1, config.pred_hidden_size))
        else:
            mask_tokens = (
                jax.random.normal(
                    rngs.params(),
                    (config.pred_num_mask_tokens, 1, 1, config.pred_hidden_size),
                )
                * 0.02
            )
        self.mask_tokens = nnx.Param(mask_tokens)

    def __call__(
        self,
        hidden_states: Array,
        context_mask: list,
        target_mask: list,
        mask_index: int = 1,
    ) -> Tuple[Array, Array]:
        """
        Args:
            hidden_states: Encoder outputs (context)
            context_mask: List of masks for context tokens
            target_mask: List of masks for target tokens
            mask_index: Index of mask token to use

        Returns:
            Tuple of (embeddings, position_masks)
        """
        batch_size = hidden_states.shape[0]
        context = self.predictor_embeddings(hidden_states)

        # Get mask token
        mask_index = mask_index % self.config.pred_num_mask_tokens
        target_token = self.mask_tokens[mask_index]  # (1, 1, pred_hidden_size)

        # Determine number of patches from target mask
        max_patch_num = int(jnp.max(target_mask[0])) + 1
        target = jnp.tile(target_token, (batch_size, max_patch_num, 1))
        target = apply_masks(target, target_mask)

        # Concatenate context and target
        context = jnp.tile(context, (len(context_mask), 1, 1))
        embeddings = jnp.concatenate([context, target], axis=1)

        # Position masks
        cm = jnp.concatenate(context_mask, axis=0)
        tm = jnp.concatenate(target_mask, axis=0)
        masks = jnp.concatenate([cm, tm], axis=1)

        return embeddings, masks


class VJEPA2Predictor(nnx.Module):
    """VJEPA2 Predictor module."""

    def __init__(self, config: VJEPA2FlaxConfig, rngs: nnx.Rngs):
        super().__init__()
        self.config = config

        self.embeddings = VJEPA2PredictorEmbeddings(config, rngs=rngs)

        self.layer = nnx.List(
            [
                VJEPA2Layer(
                    config,
                    hidden_size=config.pred_hidden_size,
                    num_attention_heads=config.pred_num_attention_heads,
                    mlp_ratio=config.pred_mlp_ratio,
                    rngs=rngs,
                )
                for _ in range(config.pred_num_hidden_layers)
            ]
        )

        self.layernorm = nnx.LayerNorm(config.pred_hidden_size, epsilon=config.layer_norm_eps, rngs=rngs)
        self.proj = nnx.Linear(config.pred_hidden_size, config.hidden_size, rngs=rngs)

    def sort_tokens(
        self,
        hidden_states: Array,
        position_masks: Array,
        argsort: Array,
    ) -> Tuple[Array, Array]:
        """Sort tokens by position."""
        # Gather position masks
        position_masks = jnp.take_along_axis(position_masks, argsort, axis=1)

        # Gather hidden states
        hidden_states_argsort = jnp.broadcast_to(argsort[..., None], hidden_states.shape)
        hidden_states = jnp.take_along_axis(hidden_states, hidden_states_argsort, axis=1)

        return hidden_states, position_masks

    def unsort_tokens(self, hidden_states: Array, argsort: Array) -> Array:
        """Unsort tokens back to original order."""
        reverse_argsort = jnp.argsort(argsort, axis=1)
        reverse_argsort = jnp.broadcast_to(reverse_argsort[..., None], hidden_states.shape)
        hidden_states = jnp.take_along_axis(hidden_states, reverse_argsort, axis=1)
        return hidden_states

    def __call__(
        self,
        encoder_hidden_states: Array,
        context_mask: list,
        target_mask: list,
    ) -> VJEPA2PredictorOutput:
        """
        Args:
            encoder_hidden_states: (B, N, hidden_size)
            context_mask: List of context masks
            target_mask: List of target masks

        Returns:
            VJEPA2PredictorOutput
        """
        # Apply masks to encoder hidden states
        encoder_hidden_states = apply_masks(encoder_hidden_states, context_mask)
        _, n_ctxt, _ = encoder_hidden_states.shape

        # Get predictor embeddings
        hidden_states, position_masks = self.embeddings(encoder_hidden_states, context_mask, target_mask)

        # Sort tokens by position
        argsort = jnp.argsort(position_masks, axis=1)
        hidden_states, position_masks = self.sort_tokens(hidden_states, position_masks, argsort)

        # Apply transformer layers
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, position_mask=position_masks)

        hidden_states = self.layernorm(hidden_states)

        # Unsort and extract predicted tokens
        hidden_states = self.unsort_tokens(hidden_states, argsort)
        hidden_states = hidden_states[:, n_ctxt:]

        # Project back to encoder dimension
        hidden_states = self.proj(hidden_states)

        return VJEPA2PredictorOutput(last_hidden_state=hidden_states)


class VJEPA2PoolerSelfAttention(nnx.Module):
    """Self-attention for pooler (no RoPE)."""

    def __init__(self, config: VJEPA2FlaxConfig, rngs: nnx.Rngs):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nnx.Linear(self.embed_dim, self.embed_dim, rngs=rngs)
        self.k_proj = nnx.Linear(self.embed_dim, self.embed_dim, rngs=rngs)
        self.v_proj = nnx.Linear(self.embed_dim, self.embed_dim, rngs=rngs)
        self.out_proj = nnx.Linear(self.embed_dim, self.embed_dim, rngs=rngs)

    def __call__(self, hidden_states: Array) -> Array:
        batch_size, seq_length, _ = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        # Reshape to (B, N, H, D) then transpose to (B, H, N, D)
        queries = queries.reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Attention
        attn_weights = jnp.matmul(queries, keys.transpose(0, 1, 3, 2)) * self.scale
        attn_weights = nnx.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(queries.dtype)

        attn_output = jnp.matmul(attn_weights, values)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_length, self.embed_dim)

        return self.out_proj(attn_output)


class VJEPA2PoolerCrossAttention(nnx.Module):
    """Cross-attention for pooler (no output projection)."""

    def __init__(self, config: VJEPA2FlaxConfig, rngs: nnx.Rngs):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nnx.Linear(self.embed_dim, self.embed_dim, rngs=rngs)
        self.k_proj = nnx.Linear(self.embed_dim, self.embed_dim, rngs=rngs)
        self.v_proj = nnx.Linear(self.embed_dim, self.embed_dim, rngs=rngs)

    def __call__(self, queries: Array, keys: Array, values: Array) -> Array:
        batch_size, q_seq_length, _ = queries.shape
        kv_seq_length = keys.shape[1]

        queries = self.q_proj(queries)
        keys = self.k_proj(keys)
        values = self.v_proj(values)

        # Reshape and transpose
        queries = queries.reshape(batch_size, q_seq_length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(batch_size, kv_seq_length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(batch_size, kv_seq_length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Attention
        attn_weights = jnp.matmul(queries, keys.transpose(0, 1, 3, 2)) * self.scale
        attn_weights = nnx.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(queries.dtype)

        attn_output = jnp.matmul(attn_weights, values)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, q_seq_length, self.embed_dim)

        return attn_output


class VJEPA2PoolerSelfAttentionLayer(nnx.Module):
    """Pooler self-attention layer with MLP."""

    def __init__(self, config: VJEPA2FlaxConfig, rngs: nnx.Rngs):
        super().__init__()
        self.layer_norm1 = nnx.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps, rngs=rngs)
        self.self_attn = VJEPA2PoolerSelfAttention(config, rngs=rngs)
        self.layer_norm2 = nnx.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps, rngs=rngs)
        self.mlp = VJEPA2MLP(
            config,
            hidden_size=config.hidden_size,
            mlp_ratio=config.mlp_ratio,
            rngs=rngs,
        )

    def __call__(self, hidden_states: Array) -> Array:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class VJEPA2PoolerCrossAttentionLayer(nnx.Module):
    """Pooler cross-attention layer with MLP."""

    def __init__(self, config: VJEPA2FlaxConfig, rngs: nnx.Rngs):
        super().__init__()
        self.layer_norm1 = nnx.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps, rngs=rngs)
        self.cross_attn = VJEPA2PoolerCrossAttention(config, rngs=rngs)
        self.layer_norm2 = nnx.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps, rngs=rngs)
        self.mlp = VJEPA2MLP(
            config,
            hidden_size=config.hidden_size,
            mlp_ratio=config.mlp_ratio,
            rngs=rngs,
        )

    def __call__(self, queries: Array, hidden_state: Array) -> Array:
        residual = queries
        hidden_state_normed = self.layer_norm1(hidden_state)
        hidden_state = self.cross_attn(queries, hidden_state_normed, hidden_state_normed)
        hidden_state = residual + hidden_state

        residual = hidden_state
        hidden_state = self.layer_norm2(hidden_state)
        hidden_state = self.mlp(hidden_state)
        hidden_state = residual + hidden_state

        return hidden_state


class VJEPA2AttentivePooler(nnx.Module):
    """Attentive pooler for classification."""

    def __init__(self, config: VJEPA2FlaxConfig, rngs: nnx.Rngs):
        super().__init__()
        self.query_tokens = nnx.Param(jnp.zeros((1, 1, config.hidden_size)))

        self.cross_attention_layer = VJEPA2PoolerCrossAttentionLayer(config, rngs=rngs)

        self.self_attention_layers = nnx.List(
            [VJEPA2PoolerSelfAttentionLayer(config, rngs=rngs) for _ in range(config.num_pooler_layers)]
        )

    def __call__(self, hidden_state: Array) -> Array:
        """
        Args:
            hidden_state: (B, N, hidden_size)

        Returns:
            pooled: (B, hidden_size)
        """
        batch_size = hidden_state.shape[0]

        # Apply self-attention layers
        for layer in self.self_attention_layers:
            hidden_state = layer(hidden_state)

        # Cross-attention with query tokens
        queries = jnp.tile(self.query_tokens[...], (batch_size, 1, 1))
        hidden_state = self.cross_attention_layer(queries, hidden_state)

        # Squeeze the sequence dimension
        return hidden_state.squeeze(1)


class VJEPA2Model(nnx.Module):
    """VJEPA2 Model with Encoder and Predictor."""

    def __init__(self, config: VJEPA2FlaxConfig, rngs: nnx.Rngs):
        super().__init__()
        self.config = config

        self.encoder = VJEPA2Encoder(config, rngs=rngs)
        self.predictor = VJEPA2Predictor(config, rngs=rngs)

    def __call__(
        self,
        pixel_values_videos: Array,
        context_mask: Optional[list] = None,
        target_mask: Optional[list] = None,
        skip_predictor: bool = False,
    ) -> VJEPA2ModelOutput:
        """
        Args:
            pixel_values_videos: (B, T, H, W, C)
            context_mask: Optional list of context masks
            target_mask: Optional list of target masks
            skip_predictor: If True, skip predictor forward

        Returns:
            VJEPA2ModelOutput
        """
        # Encoder forward
        encoder_outputs = self.encoder(pixel_values_videos)
        sequence_output = encoder_outputs.last_hidden_state

        batch_size = pixel_values_videos.shape[0]
        num_patches = sequence_output.shape[1]

        # Create default masks if not provided
        if context_mask is None and target_mask is None:
            default_mask = jnp.broadcast_to(jnp.arange(num_patches)[None, :], (batch_size, num_patches))
            context_mask = [default_mask]
            target_mask = [default_mask]

        predictor_output = None
        if not skip_predictor:
            predictor_outputs = self.predictor(
                encoder_hidden_states=sequence_output,
                context_mask=context_mask,
                target_mask=target_mask,
            )
            target_hidden_state = apply_masks(sequence_output, target_mask)
            predictor_output = VJEPA2PredictorOutput(
                last_hidden_state=predictor_outputs.last_hidden_state,
                target_hidden_state=target_hidden_state,
            )

        masked_hidden_state = apply_masks(sequence_output, context_mask) if context_mask else None

        return VJEPA2ModelOutput(
            last_hidden_state=sequence_output,
            masked_hidden_state=masked_hidden_state,
            predictor_output=predictor_output,
        )

    def get_vision_features(self, pixel_values_videos: Array) -> Array:
        """Get encoder features without running predictor."""
        outputs = self(pixel_values_videos, skip_predictor=True)
        return outputs.last_hidden_state


class VJEPA2ForVideoClassification(nnx.Module):
    """VJEPA2 with classification head for video classification."""

    def __init__(self, config: VJEPA2FlaxConfig, rngs: nnx.Rngs):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels

        self.vjepa2 = VJEPA2Model(config, rngs=rngs)
        self.pooler = VJEPA2AttentivePooler(config, rngs=rngs)
        self.classifier = nnx.Linear(config.hidden_size, config.num_labels, rngs=rngs)

    def __call__(self, pixel_values_videos: Array) -> VJEPA2ClassificationOutput:
        """
        Args:
            pixel_values_videos: (B, T, H, W, C)

        Returns:
            VJEPA2ClassificationOutput with logits
        """
        outputs = self.vjepa2(pixel_values_videos, skip_predictor=True)
        last_hidden_state = outputs.last_hidden_state

        pooler_output = self.pooler(last_hidden_state)
        logits = self.classifier(pooler_output)

        return VJEPA2ClassificationOutput(logits=logits, last_hidden_state=last_hidden_state)
