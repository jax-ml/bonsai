import jax
import jax.numpy as jnp
from flax import nnx
from typing import Sequence, Optional

class DropPath(nnx.Module):
    """
    Stochastic depth (DropPath) module, compatible with JAX jit.
    """
    def __init__(self, drop_prob: float = 0.0):
        self.drop_prob = drop_prob

    def __call__(self, x, rng: Optional[jax.Array] = None, train: bool = True):
        train_flag = jnp.asarray(train)

        if rng is None:
            rng = jax.random.PRNGKey(0)

        def apply_drop(_):
            keep_prob = jnp.asarray(1.0) - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            mask = jax.random.bernoulli(rng, p=keep_prob, shape=shape)
            return (x * mask) / keep_prob

        def no_drop(_):
            return x

        cond = jnp.logical_or(self.drop_prob == 0.0, jnp.logical_not(train_flag))
        return jax.lax.cond(cond, no_drop, apply_drop, operand=None)


class Block(nnx.Module):

    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init_value=1e-6, *, rngs: nnx.Rngs):
        self.dwconv = nnx.Conv(
            in_features=dim,
            out_features=dim,
            kernel_size=(7, 7),
            padding=3,
            feature_group_count=dim,
            rngs=rngs,
        )
        self.norm = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.pwconv1 = nnx.Linear(dim, 4 * dim, rngs=rngs)
        self.activation = nnx.gelu
        self.pwconv2 = nnx.Linear(4 * dim, dim, rngs=rngs)

        self.gamma = nnx.Param(layer_scale_init_value * jnp.ones((dim))) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path)

    def __call__(self, x, rng: Optional[jax.Array] = None, train: bool = True):

        if rng is None:
            rng = jax.random.PRNGKey(0)

        input_ = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.activation(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma.value * x

        x = input_ + self.drop_path(x, rng=rng, train=train)
        return x


class EmbeddingsWrapper(nnx.Module):
    """
    Thin wrapper so `embeddings` returns NCHW (PyTorch style) like HF ConvNeXt 'embeddings'.
    Internally uses a provided `stem` module (which expects NHWC input and returns NHWC).
    """
    def __init__(self, stem: nnx.Module):
        self.stem = stem

    def __call__(self, x):
        # stem expects NHWC (JAX image layout). If test provides NHWC, OK.
        # Convert stem output (NHWC) -> NCHW to match HF torch embeddings.
        out = self.stem(x)
        return jnp.transpose(out, (0, 3, 1, 2))


class ConvNeXt(nnx.Module):
    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        depths: Sequence[int] = (3, 3, 27, 3),
        dims: Sequence[int] = (192, 384, 768, 1536),
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        head_init_scale: float = 1.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.downsample_layers = nnx.List()
        self.depths = depths

        # stem: produces NHWC
        stem = nnx.Sequential(
            nnx.Conv(in_features=in_chans, out_features=dims[0], kernel_size=(4, 4), strides=(4, 4), rngs=rngs),
            nnx.LayerNorm(dims[0], epsilon=1e-6, rngs=rngs),
        )        
        
        self.embeddings = EmbeddingsWrapper(stem)

        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nnx.Sequential(
                nnx.LayerNorm(dims[i], epsilon=1e-6, rngs=rngs),
                nnx.Conv(
                    in_features=dims[i],
                    out_features=dims[i + 1],
                    kernel_size=(2, 2),
                    strides=(2, 2),
                    rngs=rngs,
                ),
            )
            self.downsample_layers.append(downsample_layer)

        # stages and blocks
        self.stages = nnx.List()
        dp_rates = list(jnp.linspace(0, drop_path_rate, sum(depths)))
        curr = 0
        for i in range(4):
            stage_blocks = nnx.List()
            for j in range(depths[i]):
                stage_blocks.append(
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[curr + j],
                        layer_scale_init_value=layer_scale_init_value,
                        rngs=rngs,
                    )
                )
            self.stages.append(stage_blocks)
            curr += depths[i]

        self.norm = nnx.LayerNorm(dims[-1], epsilon=1e-6, rngs=rngs)
        self.head = nnx.Linear(dims[-1], num_classes, rngs=rngs)

    def __call__(self, x, rng: Optional[jax.Array] = None, train: bool = False):
        """
        Forward pass.
        `x` expected in NHWC (JAX default). This function keeps the model internally NHWC.
        """
        if rng is None:
            rng = jax.random.PRNGKey(0)

        for i in range(4):
            x = self.downsample_layers[i](x)

            for block in self.stages[i]:
                rng, block_rng = jax.random.split(rng)
                x = block(x, rng=block_rng, train=train)

        # global average pool over H, W (NHWC)
        x = jnp.mean(x, axis=(1, 2))

        x = self.norm(x)
        x = self.head(x)
        return x

@jax.jit
def forward(
    graph_def: nnx.GraphDef,
    state: nnx.State,
    x: jax.Array,
    *,
    rng: Optional[jax.Array] = None,
    train: bool = False,
):
    model = nnx.merge(graph_def, state)
    return model(x, rng=rng, train=train)