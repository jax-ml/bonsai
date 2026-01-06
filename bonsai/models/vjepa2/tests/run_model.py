import jax
import jax.numpy as jnp
import numpy as np
import torch
from torchcodec.decoders import VideoDecoder

from bonsai.models.vjepa2.modeling import VJEPA2FlaxConfig, VJEPA2Model, VJEPA2ForVideoClassification
from bonsai.models.vjepa2.params import create_model_from_safe_tensors

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModel, AutoVideoProcessor

hf_repo = "facebook/vjepa2-vitl-fpc64-256"

processor = AutoVideoProcessor.from_pretrained(hf_repo)
# torch_model = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256")

model_dir = snapshot_download(hf_repo)

video_url = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/archery/-Qz25rXdMjE_000014_000024.mp4"
vr = VideoDecoder(video_url)
frame_idx = np.arange(0, 64) # choosing some frames. here, you can define more complex sampling strategy
video = vr.get_frames_at(indices=frame_idx).data  # T x C x H x W
video = processor(video, return_tensors="pt")

config = VJEPA2FlaxConfig.vitl_fpc64_256()
model = create_model_from_safe_tensors(model_dir, cfg = config, classifier=False)
