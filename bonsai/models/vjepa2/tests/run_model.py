import jax
import jax.numpy as jnp
import numpy as np
from huggingface_hub import snapshot_download
from torchcodec.decoders import VideoDecoder
from transformers import AutoConfig, AutoVideoProcessor

from bonsai.models.vjepa2.modeling import VJEPA2FlaxConfig
from bonsai.models.vjepa2.params import create_model_from_safe_tensors


def main():
    # Load model and video preprocessor (SSv2 - 174 action classes)
    hf_repo = "facebook/vjepa2-vitl-fpc16-256-ssv2"

    print(f"Loading VJEPA2 from {hf_repo}...")
    model_dir = snapshot_download(hf_repo)

    # Load HF config for label names
    hf_config = AutoConfig.from_pretrained(hf_repo)
    processor = AutoVideoProcessor.from_pretrained(hf_repo)

    # Load Flax model
    config = VJEPA2FlaxConfig.vitl_fpc16_256()
    model = create_model_from_safe_tensors(model_dir, cfg=config, classifier=True)
    model.eval()

    # Load actual video frames
    video_url = (
        "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/bowling/-WH-lxmGJVY_000005_000015.mp4"
    )
    vr = VideoDecoder(video_url)
    frame_idx = np.arange(0, hf_config.frames_per_clip, 8)
    video = vr.get_frames_at(indices=frame_idx).data  # frames x channels x height x width

    # Preprocess and run inference
    inputs = processor(video, return_tensors="pt")
    pixel_values_videos = inputs.pixel_values_videos

    # Convert to JAX format: (B, T, C, H, W) -> (B, T, H, W, C)
    video_jax = jnp.asarray(pixel_values_videos.numpy())
    video_jax = video_jax.transpose(0, 1, 3, 4, 2)

    # Run inference
    outputs = model(video_jax)
    logits = np.asarray(jax.device_get(outputs.logits))

    # Get top 5 predictions with label names
    print("\nTop 5 predicted class names:")
    probs = jax.nn.softmax(jnp.array(logits), axis=-1)
    probs_np = np.asarray(jax.device_get(probs))

    top5_indices = np.argsort(logits[0])[-5:][::-1]
    top5_probs = probs_np[0][top5_indices]

    for idx, prob in zip(top5_indices, top5_probs):
        text_label = hf_config.id2label[idx]
        print(f" - {text_label}: {prob:.2f}")


if __name__ == "__main__":
    main()
