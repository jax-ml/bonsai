import jax.numpy as jnp 
from huggingface_hub import snapshot_download
from transformers import AutoImageProcessor

from PIL import Image
import requests 

from bonsai.models.dinov3 import params
from bonsai.models.dinov3 import modeling as model_lib 

def run_model():
    model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
    
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/image_processor_example.png" # Replace with your image URL
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB") 
    processor = AutoImageProcessor.from_pretrained(model_name)

    model_ckpt_path = snapshot_download(model_name, allow_patterns=["*.safetensors", "*.json"])
    config = model_lib.DINOv3ViTFlaxConfig.dinov3_vits16()

    model = params.create_model_from_safe_tensors(
        file_dir = model_ckpt_path,
        cfg = config
    )

    inputs = processor(image, return_tensors="pt")
    print(f"Input image/s shape: {inputs['pixel_values'].shape}")
    outputs = model(jnp.asarray(inputs['pixel_values'])).pooler_output
    print(f"Output shape: {outputs.shape}")

if __name__ == "__main__":
    run_model()

__all__ = ["run_model"]



