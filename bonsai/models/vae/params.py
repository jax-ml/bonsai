import jax
from flax import nnx

from bonsai.models.vae import modeling as vae_lib


def create_model(
    cfg: vae_lib.ModelCfg,
    rngs: nnx.Rngs,
    mesh: jax.sharding.Mesh | None = None,
) -> vae_lib.VAE:
    """
    Create a VAE model with initialized parameters.

    Returns:
      A flax.nnx.Module instance with random parameters.
    """
    model = vae_lib.VAE(cfg, rngs=rngs)

    if mesh is not None:
        # This part is for distributed execution, if needed.
        graph_def, state = nnx.split(model)
        sharding = nnx.get_named_sharding(model, mesh)
        state = jax.device_put(state, sharding)
        return nnx.merge(graph_def, state)
    else:
        return model
