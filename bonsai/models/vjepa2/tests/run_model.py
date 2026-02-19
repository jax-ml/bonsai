import jax
import jax.numpy as jnp
import numpy as np
import cv2
from huggingface_hub import snapshot_download, hf_hub_download
from transformers import AutoConfig, AutoVideoProcessor

from bonsai.models.vjepa2.modeling import ModelConfig, forward
from bonsai.models.vjepa2.params import create_model_from_safe_tensors


def load_video_frames_cv2(video_path, indices):
    """Robust frame loading using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []
    sorted_indices = sorted(set(indices))  # For optimized frame access
    current_idx = 0

    for target_idx in sorted_indices:
        while current_idx < target_idx:
            ret = cap.grab()  # grab() is faster than read() for skipping
            if not ret:
                break
            current_idx += 1
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV loads as BGR, JAX/Models expect RGB
        frames.append(frame)
        current_idx += 1

    cap.release()
    return np.array(frames)


def main():
    hf_repo = "facebook/vjepa2-vitl-fpc16-256-ssv2"
    print(f"Loading VJEPA2 from {hf_repo}...")
    model_dir = snapshot_download(hf_repo)
    hf_config = AutoConfig.from_pretrained(hf_repo)
    processor = AutoVideoProcessor.from_pretrained(hf_repo)

    config = ModelConfig.vitl_fpc16_256()
    model = create_model_from_safe_tensors(model_dir, cfg=config, classifier=True)
    model.eval()

    print("Downloading video...")
    video_path = hf_hub_download(
        repo_id="nateraw/kinetics-mini", filename="val/bowling/-WH-lxmGJVY_000005_000015.mp4", repo_type="dataset"
    )

    # Decode video
    frame_indices = np.arange(0, 16 * 8, 8)
    print("Decoding frames with OpenCV...")
    video_data = load_video_frames_cv2(video_path, frame_indices)

    # The processor forces PyTorch return types, so we convert to numpy immediately.
    inputs = processor(list(video_data), return_tensors="pt")
    pixel_values = inputs.pixel_values_videos.numpy()  # (B, T, C, H, W)

    # Transpose (B, T, C, H, W) -> (B, T, H, W, C)
    pixel_values = pixel_values.transpose(0, 1, 3, 4, 2)

    video_jax = jnp.array(pixel_values)

    outputs = forward(model, video_jax)
    logits = np.asarray(jax.device_get(outputs["logits"]))

    print("\nTop 5 predicted class names:")
    probs = jax.nn.softmax(jnp.array(logits), axis=-1)
    probs_np = np.asarray(jax.device_get(probs))
    top5_indices = np.argsort(logits[0])[-5:][::-1]

    for idx in top5_indices:
        text_label = hf_config.id2label[idx]
        print(f" - {text_label}: {probs_np[0][idx]:.2f}")


if __name__ == "__main__":
    main()
