import dataclasses

import jax
import jax.numpy as jnp
from flax import nnx


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    image_size: tuple[int, int]
    patch_size: tuple[int, int]
    num_channels: int
    hidden_dim: int
    num_heads: int
    mlp_dim: int
    num_layers: int
    attn_dropout_prob: float = 0.0
    dropout_prob: float = 0.0
    eps: float = 1e-12
    num_labels: int = 1000
    out_channels: int | None = None
    feature_size: int | None = None
    encoder_channels: tuple[int, ...] = (1, 2, 4, 8)
    encoder_num_layers: tuple[int, ...] = (0, 2, 1, 0)
    decoder_channels: tuple[tuple[int, int], ...] = ((16, 8), (8, 4), (4, 2), (2, 1))

    @classmethod
    def unetr(cls):
        return cls(
            image_size=(256, 256),
            patch_size=(16, 16),
            num_channels=3,
            hidden_dim=768,
            num_heads=12,
            mlp_dim=3072,
            num_layers=12,
            out_channels=3,
            feature_size=16,
        )


def interpolate_posembed(posemb: jnp.ndarray, num_tokens: int, has_class_token: bool) -> jnp.ndarray:
    assert posemb.shape[0] == 1
    if has_class_token:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        num_tokens -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0, 0:]

    gs_old = int(jnp.sqrt(len(posemb_grid)))
    gs_new = int(jnp.sqrt(num_tokens))

    assert gs_old**2 == len(posemb_grid), f"{gs_old**2} != {len(posemb_grid)}"
    assert gs_new**2 == num_tokens, f"{gs_new**2} != {num_tokens}"

    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1)

    zoom = (1, gs_new, gs_new, posemb_grid.shape[-1])
    posemb_grid = jax.image.resize(posemb_grid, zoom, method="bicubic")
    posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)

    return jnp.array(jnp.concatenate([posemb_tok, posemb_grid], axis=1))


class Embeddings(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        num_patches = (cfg.image_size[0] // cfg.patch_size[0]) * (cfg.image_size[1] // cfg.patch_size[1])

        self.projection = nnx.Conv(
            cfg.num_channels, cfg.hidden_dim, kernel_size=cfg.patch_size, strides=cfg.patch_size, rngs=rngs
        )
        self.cls_token = nnx.Param(jax.random.normal(rngs.params(), (1, 1, cfg.hidden_dim)))
        self.pos_embeddings = nnx.Param(jax.random.normal(rngs.params(), (1, num_patches + 1, cfg.hidden_dim)))
        self.dropout = nnx.Dropout(cfg.dropout_prob)

    def __call__(self, pixel_values: jnp.ndarray, *, rngs: nnx.Rngs | None) -> jnp.ndarray:
        embeddings = self.projection(pixel_values)
        b, h, w, c = embeddings.shape
        embeddings = embeddings.reshape(b, h * w, c)

        num_new_patch_patches = h * w

        stored_pos_embeddings = self.pos_embeddings.value
        num_stored_patches = stored_pos_embeddings.shape[1]

        if num_stored_patches == num_new_patch_patches + 1:
            current_pos_embeddings = stored_pos_embeddings
        else:
            current_pos_embeddings = interpolate_posembed(
                stored_pos_embeddings, num_tokens=num_new_patch_patches + 1, has_class_token=True
            )

        cls_tokens = jnp.tile(self.cls_token.value, (b, 1, 1))
        embeddings = jnp.concatenate((cls_tokens, embeddings), axis=1)

        embeddings = embeddings + current_pos_embeddings
        embeddings = self.dropout(embeddings, rngs=rngs)
        return embeddings


class TransformerEncoder(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.attention = nnx.MultiHeadAttention(
            num_heads=cfg.num_heads,
            in_features=cfg.hidden_dim,
            dropout_rate=cfg.attn_dropout_prob,
            decode=False,
            rngs=rngs,
        )
        self.linear1 = nnx.Linear(cfg.hidden_dim, cfg.mlp_dim, rngs=rngs)
        self.linear2 = nnx.Linear(cfg.mlp_dim, cfg.hidden_dim, rngs=rngs)
        self.dropout = nnx.Dropout(cfg.dropout_prob)
        self.layernorm_before = nnx.LayerNorm(cfg.hidden_dim, epsilon=cfg.eps, rngs=rngs)
        self.layernorm_after = nnx.LayerNorm(cfg.hidden_dim, epsilon=cfg.eps, rngs=rngs)

    def __call__(self, hidden_states, head_mask=None, *, rngs: nnx.Rngs | None):
        hidden_states_norm = self.layernorm_before(hidden_states)
        attention_output = self.attention(hidden_states_norm, head_mask, rngs=rngs)
        hidden_states = attention_output + hidden_states
        layer_output = self.layernorm_after(hidden_states)
        layer_output = jax.nn.gelu(self.linear1(layer_output))
        layer_output = self.linear2(layer_output)
        layer_output = self.dropout(layer_output, rngs=rngs)
        layer_output += hidden_states
        return layer_output


class ViTClassificationModel(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.pos_embeddings = Embeddings(cfg, rngs=rngs)
        self.layers = nnx.List([TransformerEncoder(cfg, rngs=rngs) for _ in range(cfg.num_layers)])
        self.ln = nnx.LayerNorm(cfg.hidden_dim, epsilon=cfg.eps, rngs=rngs)
        self.classifier = nnx.Linear(cfg.hidden_dim, cfg.num_labels, rngs=rngs)

    def __call__(self, x, *, rngs: nnx.Rngs | None):
        x = self.pos_embeddings(x, rngs=rngs)
        for layer in self.layers:
            x = layer(x, rngs=rngs)
        x = self.ln(x)
        x = self.classifier(x[:, 0, :])
        return x


@jax.jit
def forward(graphdef: nnx.GraphDef[nnx.Module], state: nnx.State, x: jax.Array, rngs: nnx.Rngs) -> jax.Array:
    model = nnx.merge(graphdef, state)
    return model(x, rngs=rngs)
