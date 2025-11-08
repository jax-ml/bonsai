import dataclasses
import logging
from functools import partial
from itertools import pairwise
from typing import Sequence

import jax
import jax.numpy as jnp
from flax import nnx


@dataclasses.dataclass(frozen=True)
class ModelCfg:
    """Configuration for the Variational Autoencoder (VAE) model."""

    input_dim: int = 784  # 28*28 for MNIST
    hidden_dims: Sequence[int] = (512, 256)
    latent_dim: int = 20


class Encoder(nnx.Module):
    """Encodes the input into latent space parameters (mu and logvar)."""

    def __init__(self, cfg: ModelCfg, *, rngs: nnx.Rngs):
        self.hidden_layers = [
            nnx.Linear(in_features, out_features, rngs=rngs)
            for in_features, out_features in zip([cfg.input_dim, *list(cfg.hidden_dims)], cfg.hidden_dims)
        ]
        self.fc_mu = nnx.Linear(cfg.hidden_dims[-1], cfg.latent_dim, rngs=rngs)
        self.fc_logvar = nnx.Linear(cfg.hidden_dims[-1], cfg.latent_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        # Flatten the image
        x = x.reshape((x.shape[0], -1))
        for layer in self.hidden_layers:
            x = nnx.relu(layer(x))

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nnx.Module):
    """Decodes the latent vector back into the original input space."""

    def __init__(self, cfg: ModelCfg, *, rngs: nnx.Rngs):
        # Mirrored architecture of the encoder
        dims = [cfg.latent_dim, *list(reversed(cfg.hidden_dims))]
        self.hidden_layers = [
            nnx.Linear(in_features, out_features, rngs=rngs) for in_features, out_features in pairwise(dims, dims[1:])
        ]
        self.fc_out = nnx.Linear(dims[-1], cfg.input_dim, rngs=rngs)

    def __call__(self, z: jax.Array) -> jax.Array:
        for layer in self.hidden_layers:
            z = nnx.relu(layer(z))

        reconstruction_logits = self.fc_out(z)
        return reconstruction_logits


class VAE(nnx.Module):
    """Full Variational Autoencoder model."""

    def __init__(self, cfg: ModelCfg, *, rngs: nnx.Rngs):
        logging.warning("This model does not load weights from a reference implementation.")
        self.cfg = cfg
        self.encoder = Encoder(cfg, rngs=rngs)
        self.decoder = Decoder(cfg, rngs=rngs)

    def reparameterize(self, mu: jax.Array, logvar: jax.Array, key: jax.Array) -> jax.Array:
        """Performs the reparameterization trick to sample from the latent space."""
        std = jnp.exp(0.5 * logvar)
        epsilon = jax.random.normal(key, std.shape)
        return mu + epsilon * std

    def __call__(self, x: jax.Array, sample_key: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Defines the forward pass of the VAE."""
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar, sample_key)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar


@partial(jax.jit, static_argnums=(0,))
def forward(model, x, key):
    return model(x, key)
