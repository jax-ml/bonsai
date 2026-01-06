import jax
import jax.numpy as jnp
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoVideoProcessor

from bonsai.models.vjepa2.modeling import VJEPA2FlaxConfig
from bonsai.models.vjepa2.params import create_model_from_safe_tensors


def main():
    # Use 16-frame model (lower memory footprint)
    hf_repo = "facebook/vjepa2-vitl-fpc64-256"

    print(f"Downloading model from {hf_repo}...")
    model_dir = snapshot_download(hf_repo)

    # Load processor and model
    processor = AutoVideoProcessor.from_pretrained(hf_repo)

    print("Loading Flax model...")
    config = VJEPA2FlaxConfig.vitl_fpc64_256()
    model = create_model_from_safe_tensors(model_dir, cfg=config, classifier=False)
    model.eval()

    # Create random video input (16 frames, 3 channels, 256x256)
    print("Running inference...")
    np.random.seed(42)
    video = np.random.randn(16, 3, 256, 256).astype(np.float32)

    # Process video
    inputs = processor(video, return_tensors="pt")
    pixel_values_videos = inputs.pixel_values_videos

    # Convert to JAX format: (B, T, H, W, C)
    video_jax = jnp.asarray(pixel_values_videos.numpy())
    video_jax = video_jax.transpose(0, 1, 3, 4, 2)

    # Forward pass
    output = model(video_jax)

    print(f"✓ Forward pass successful!")
    print(f"  Input shape: {video_jax.shape}")
    print(f"  Output shape: {output.last_hidden_state.shape}")


if __name__ == "__main__":
    main()
