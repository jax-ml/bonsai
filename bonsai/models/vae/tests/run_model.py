import jax
import jax.numpy as jnp
from flax import nnx

from bonsai.models.vae import modeling, params


def run_model():
    # 1. Create model and PRNG keys
    rngs = nnx.Rngs(params=0, sample=1)
    config = modeling.ModelCfg(input_dim=28 * 28, hidden_dims=(512, 256), latent_dim=20)
    model = params.create_model(cfg=config, rngs=rngs)

    # 2. Prepare dummy input
    batch_size = 4
    dummy_input = jnp.ones((batch_size, 28, 28, 1), dtype=jnp.float32)
    sample_key = rngs.sample()

    # 3. Run a forward pass
    print("Running forward pass...")
    reconstruction, mu, logvar = modeling.forward(model, dummy_input, sample_key)
    print("Forward pass complete.")

    # 4. Show output shapes
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"LogVar shape: {logvar.shape}")

    # The reconstruction is flattened, let's show its intended image shape
    recon_img_shape = (batch_size, 28, 28, 1)
    print(f"Reshaped Reconstruction: {reconstruction.reshape(recon_img_shape).shape}")


if __name__ == "__main__":
    run_model()
