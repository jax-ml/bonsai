import dataclasses
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array


@dataclasses.dataclass(frozen=True)
class DINOv3ViTFlaxConfig:
    model_type = "dinov3_ViT"
    patch_size: Tuple[int, int] = (16, 16)
    hidden_size: int = 384
    intermediate_size: int = 1536
    num_hidden_layers: int = 12
    num_attention_heads: int = 6
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-5
    rope_theta: float = 100.0
    image_size: int = 224
    num_channels: int = 3
    query_bias: bool = True
    key_bias: bool = False
    value_bias: bool = True
    proj_bias: bool = True
    mlp_bias: bool = True
    layerscale_value: float = 1.0
    use_gated_mlp: bool = False
    num_register_tokens: int = 4

    @classmethod
    def dinov3_vits16(cls):
        return cls()

    @classmethod
    def dinov3_vits16plus(cls):
        return cls(
            hidden_size=384,
            intermediate_size=1536,
            num_hidden_layers=12,
            num_attention_heads=6,
            hidden_act="silu",
            use_gated_mlp=True,
        )

    @classmethod
    def dinov3_vitb16(cls):
        return cls(
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            hidden_act="gelu",
            use_gated_mlp=False,
        )

    @classmethod
    def dinov3_vitl16(cls):
        return cls(
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=24,
            num_attention_heads=16,
            hidden_act="gelu",
            use_gated_mlp=False,
        )

    @classmethod
    def dinov3_vith16plus(cls):
        return cls(
            hidden_size=1280,
            intermediate_size=5120,
            num_hidden_layers=32,
            num_attention_heads=20,
            hidden_act="silu",
            use_gated_mlp=True,
        )

    @classmethod
    def dinov3_vit7b16(cls):
        return cls(
            hidden_size=4096,
            intermediate_size=8192,
            num_hidden_layers=40,
            num_attention_heads=32,
            hidden_act="silu",
            use_gated_mlp=True,
        )


class DINOv3ViTEmbeddings(nnx.Module):
    def __init__(self, config: DINOv3ViTFlaxConfig, rngs: nnx.Rngs):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.cls_token = nnx.Param(jnp.ones((1, 1, self.hidden_size), dtype=jnp.float32))
        self.mask_token = nnx.Param(jnp.zeros((1, 1, self.hidden_size), dtype=jnp.float32))
        self.register_tokens = nnx.Param(
            jnp.zeros((1, config.num_register_tokens, config.hidden_size), dtype=jnp.float32)
        )
        self.patch_embeddings = nnx.Conv(
            in_features=config.num_channels,
            out_features=config.hidden_size,
            kernel_size=config.patch_size,
            strides=config.patch_size,
            rngs=rngs,
        )

    def __call__(self, pixel_values: Array) -> Array:
        b, _, _, _ = pixel_values.shape

        # (batch_size, num_channels, height, width) -> (batch_size, num_patches, hidden_size)
        pixel_values = pixel_values.transpose(0, 2, 3, 1)
        patch_embeddings = self.patch_embeddings(pixel_values)
        patch_embeddings = patch_embeddings.reshape(b, -1, self.hidden_size)

        cls_token = jnp.broadcast_to(self.cls_token[...], (b, 1, self.hidden_size))
        register_tokens = jnp.broadcast_to(
            self.register_tokens[...], (b, self.config.num_register_tokens, self.hidden_size)
        )
        return jnp.concat([cls_token, register_tokens, patch_embeddings], axis=1)


class Dinov3ViTRopePositionEmbedding(nnx.Module):
    def __init__(self, config: DINOv3ViTFlaxConfig):
        super().__init__()
        self.config = config
        self.base = config.rope_theta
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_patches_h = config.image_size // config.patch_size[0]
        self.num_patches_w = config.image_size // config.patch_size[0]

    def __call__(self, pixel_values: Array) -> Tuple[Array, Array]:
        _, _, height, width = pixel_values.shape
        num_patches_h = height // self.config.patch_size[0]
        num_patches_w = width // self.config.patch_size[0]

        coords_h = jnp.arange(0.5, num_patches_h, dtype=jnp.float32) / num_patches_h  # [H]
        coords_w = jnp.arange(0.5, num_patches_w, dtype=jnp.float32) / num_patches_w  # [W]
        coords = jnp.stack(jnp.meshgrid(coords_h, coords_w, indexing="ij"), axis=-1)  # [H, W, 2]
        coords = coords.reshape(-1, 2)
        coords = 2 * coords - 1.0

        inv_freq = 1.0 / self.base ** jnp.arange(0.0, 1.0, 4.0 / self.head_dim, dtype=jnp.float32)  # [head_dim // 4]
        angles = 2 * jnp.pi * coords[:, :, None] * inv_freq[None, None, :]  # (HW, 2, D//4)
        angles = angles.reshape(coords.shape[0], -1)  # (HW, D//2)
        angles = jnp.tile(angles, (1, 2))  # (HW, D)

        cos = jnp.cos(angles)
        sin = jnp.sin(angles)

        return (cos, sin)


class Dinov3LayerScale(nnx.Module):
    def __init__(self, config: DINOv3ViTFlaxConfig):
        super().__init__()
        self.lambda1 = nnx.Param(jnp.full((config.hidden_size,), config.layerscale_value, dtype=jnp.float32))

    def __call__(self, x: Array) -> Array:
        return x * self.lambda1


def rotate_half(x: Array) -> Array:
    d = x.shape[-1]
    assert d % 2 == 0
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q: Array, k: Array, cos: Array, sin: Array) -> Tuple[Array, Array]:
    q = q.astype(jnp.bfloat16)
    k = k.astype(jnp.bfloat16)
    cos = cos.astype(jnp.bfloat16)
    sin = sin.astype(jnp.bfloat16)
    num_tokens = q.shape[-2]
    num_patches = cos.shape[-2]
    num_prefix = num_tokens - num_patches
    q_prefix, q_patches = jnp.split(q, [num_prefix], axis=-2)
    k_prefix, k_patches = jnp.split(k, [num_prefix], axis=-2)
    cos_b = cos[None, None, ...]
    sin_b = sin[None, None, ...]
    # Rotation
    q_patches = (q_patches * cos_b) + (rotate_half(q_patches) * sin_b)
    k_patches = (k_patches * cos_b) + (rotate_half(k_patches) * sin_b)
    q = jnp.concatenate([q_prefix, q_patches], axis=-2)
    k = jnp.concatenate([k_prefix, k_patches], axis=-2)
    q = q.astype(jnp.float32)
    k = k.astype(jnp.float32)
    return (q, k)


class Dinov3ViTAttention(nnx.Module):
    def __init__(self, config: DINOv3ViTFlaxConfig, rngs: nnx.Rngs):
        super().__init__()
        self.config = config

        self.q_proj = nnx.Linear(
            in_features=config.hidden_size, out_features=config.hidden_size, use_bias=config.query_bias, rngs=rngs
        )
        self.k_proj = nnx.Linear(
            in_features=config.hidden_size, out_features=config.hidden_size, use_bias=config.key_bias, rngs=rngs
        )
        self.v_proj = nnx.Linear(
            in_features=config.hidden_size, out_features=config.hidden_size, use_bias=config.value_bias, rngs=rngs
        )
        self.o_proj = nnx.Linear(
            in_features=config.hidden_size, out_features=config.hidden_size, use_bias=config.proj_bias, rngs=rngs
        )

    def __call__(self, hidden_states: Array, position_embeddings: Tuple[Array, Array]) -> Array:
        batch_size, patches, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        n_heads = self.config.num_attention_heads
        head_dim = self.config.hidden_size // n_heads

        query_states = query_states.reshape(batch_size, patches, n_heads, head_dim).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(batch_size, patches, n_heads, head_dim).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(batch_size, patches, n_heads, head_dim).transpose(0, 2, 1, 3)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        scale = self.config.hidden_size // self.config.num_attention_heads
        scale = 1.0 / jnp.sqrt(scale)

        # (B, H, P, D) @ (B, H, D, P) -> (B, H, P, P)
        attn_weights = jnp.matmul(query_states, key_states.transpose(0, 1, 3, 2)) * scale
        attn_weights = nnx.softmax(attn_weights, axis=-1)

        # (B, H, P, P) @ (B, H, P, D) -> (B, H, P, D)
        hidden_states = jnp.matmul(attn_weights, value_states)

        hidden_states = hidden_states.transpose(0, 2, 1, 3).reshape(batch_size, patches, -1)
        hidden_states = self.o_proj(hidden_states)
        return hidden_states


class Dinov3MLP(nnx.Module):
    def __init__(self, config: DINOv3ViTFlaxConfig, rngs: nnx.Rngs):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.up_proj = nnx.Linear(self.hidden_size, self.intermediate_size, rngs=rngs)
        self.down_proj = nnx.Linear(self.intermediate_size, self.hidden_size, rngs=rngs)
        if config.hidden_act == "gelu":
            self.act_fn = nnx.gelu
        elif config.hidden_act == "silu":
            self.act_fn = nnx.silu

    def __call__(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))


class Dinov3GatedMLP(nnx.Module):
    def __init__(self, config: DINOv3ViTFlaxConfig, rngs: nnx.Rngs):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nnx.Linear(self.hidden_size, self.intermediate_size, use_bias=config.mlp_bias, rngs=rngs)
        self.up_proj = nnx.Linear(self.hidden_size, self.intermediate_size, use_bias=config.mlp_bias, rngs=rngs)
        self.down_proj = nnx.Linear(self.intermediate_size, self.hidden_size, use_bias=config.mlp_bias, rngs=rngs)
        if config.hidden_act == "gelu":
            self.act_fn = nnx.gelu
        elif config.hidden_act == "silu":
            self.act_fn = nnx.silu

    def __call__(self, x):
        x = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return x


class Dinov3ViTLayer(nnx.Module):
    def __init__(self, config: DINOv3ViTFlaxConfig, rngs: nnx.Rngs):
        self.norm1 = nnx.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps, rngs=rngs)
        self.attention = Dinov3ViTAttention(config, rngs=rngs)
        self.layer_scale1 = Dinov3LayerScale(config)
        self.norm2 = nnx.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps, rngs=rngs)
        if config.use_gated_mlp:
            self.mlp = Dinov3GatedMLP(config, rngs=rngs)
        else:
            self.mlp = Dinov3MLP(config, rngs=rngs)

        self.layer_scale2 = Dinov3LayerScale(config)

    def __call__(self, hidden_states: Array, position_embeddings: Tuple[Array, Array]) -> Array:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attention(hidden_states, position_embeddings)
        hidden_states = self.layer_scale1(hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.layer_scale2(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states


class Dinov3ViTModel(nnx.Module):
    def __init__(self, config: DINOv3ViTFlaxConfig, rngs: nnx.Rngs):
        super().__init__()
        self.config = config
        self.embeddings = DINOv3ViTEmbeddings(config, rngs=rngs)
        self.rope_embeddings = Dinov3ViTRopePositionEmbedding(config)
        self.layer = nnx.List([Dinov3ViTLayer(config, rngs=rngs) for _ in range(config.num_hidden_layers)])
        self.norm = nnx.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps, rngs=rngs)

    def __call__(self, pixel_values: Array):
        hidden_states = self.embeddings(pixel_values)
        position_embeddings = self.rope_embeddings(pixel_values)

        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, position_embeddings)

        sequence_output = self.norm(hidden_states)
        pooled_output = sequence_output[:, 0, :]

        return {"last_hidden_state": sequence_output, "pooler_output": pooled_output}


@jax.jit()
def forward(model: Dinov3ViTModel, inputs: Array):
    return model(inputs)
