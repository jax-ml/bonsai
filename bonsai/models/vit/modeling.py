import jax
import jax.numpy as jnp
from flax import nnx


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
    def __init__(
        self,
        image_size: tuple[int, int],
        patch_size: tuple[int, int],
        num_channels: int,
        hidden_dim: int,
        dropout_prob: float,
        *,
        rngs: nnx.Rngs,
    ):
        num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])

        self.projection = nnx.Conv(num_channels, hidden_dim, kernel_size=patch_size, strides=patch_size, rngs=rngs)
        self.cls_token = nnx.Variable(jax.random.normal(rngs.params(), (1, 1, hidden_dim)))
        self.pos_embeddings = nnx.Variable(jax.random.normal(rngs.params(), (1, num_patches + 1, hidden_dim)))
        self.dropout = nnx.Dropout(dropout_prob, rngs=rngs)

    def __call__(self, pixel_values: jnp.ndarray) -> jnp.ndarray:
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
        embeddings = self.dropout(embeddings)
        return embeddings


class TransformerEncoder(nnx.Module):
    def __init__(self, num_heads: int, attn_dim: int, mlp_dim: int, dropout_prob: float, eps: float, *, rngs: nnx.Rngs):
        self.attention = nnx.MultiHeadAttention(num_heads=num_heads, in_features=attn_dim, decode=False, rngs=rngs)
        self.linear1 = nnx.Linear(attn_dim, mlp_dim, rngs=rngs)
        self.linear2 = nnx.Linear(mlp_dim, attn_dim, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_prob, rngs=rngs)
        self.layernorm_before = nnx.LayerNorm(attn_dim, epsilon=eps, rngs=rngs)
        self.layernorm_after = nnx.LayerNorm(attn_dim, epsilon=eps, rngs=rngs)

    def __call__(self, hidden_states, head_mask=None):
        hidden_states_norm = self.layernorm_before(hidden_states)
        attention_output = self.attention(hidden_states_norm, head_mask)
        hidden_states = attention_output + hidden_states
        layer_output = self.layernorm_after(hidden_states)
        layer_output = jax.nn.gelu(self.linear1(layer_output))
        layer_output = self.linear2(layer_output)
        layer_output = self.dropout(layer_output)
        layer_output += hidden_states
        return layer_output


class ViTClassificationModel(nnx.Module):
    def __init__(
        self,
        image_size: tuple[int, int],
        patch_size: tuple[int, int],
        num_channels: int,
        hidden_dim: int,
        dropout_prob: float,
        num_heads: int,
        mlp_dim: int,
        eps: float,
        num_layers: int,
        num_labels: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.pos_embeddings = Embeddings(image_size, patch_size, num_channels, hidden_dim, dropout_prob, rngs=rngs)
        self.layers = nnx.Sequential(
            *[
                TransformerEncoder(num_heads, hidden_dim, mlp_dim, dropout_prob, eps, rngs=rngs)
                for _ in range(num_layers)
            ]
        )
        self.ln = nnx.LayerNorm(hidden_dim, epsilon=eps, rngs=rngs)
        self.classifier = nnx.Linear(hidden_dim, num_labels, rngs=rngs)

    def __call__(self, x):
        x = self.pos_embeddings(x)
        x = self.layers(x)
        x = self.ln(x)
        x = self.classifier(x[:, 0, :])
        return x


def ViT(num_classes: int, *, rngs: nnx.Rngs):
    return ViTClassificationModel((224, 224), (16, 16), 3, 768, 0.0, 12, 3072, 1e-12, 12, num_classes, rngs=rngs)


@jax.jit
def forward(graphdef: nnx.GraphDef[nnx.Module], state: nnx.State, x: jax.Array) -> jax.Array:
    model = nnx.merge(graphdef, state)
    return model(x)
