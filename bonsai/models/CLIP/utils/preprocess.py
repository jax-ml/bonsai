import numpy as np, jax.numpy as jnp
from PIL import Image
from transformers import CLIPTokenizerFast

tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")

def preprocess_image(pil_image, image_size=224):
    img = pil_image.convert("RGB").resize((image_size, image_size))
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    arr = (arr - mean) / std
    return jnp.array(arr)

def tokenize_text(texts, max_len=32):
    out = tokenizer(texts, padding="max_length", truncation=True,
                    max_length=max_len, return_tensors="np")
    return out["input_ids"]
