from dataclasses import dataclass
from typing import Optional

from huggingface_hub import hf_hub_download


@dataclass
class CLIPConfig:
    embed_dim: int = 512

    image_size: int = 224
    patch_size: int = 32
    vision_width: int = 768
    vision_layers: int = 12
    vision_heads: int = 12

    vocab_size: int = 49408
    context_length: int = 77
    text_width: int = 512
    text_layers: int = 12
    text_heads: int = 8


def load_pretrained_clip(
    model_name: str = "openai/clip-vit-base-patch32",
    revision: Optional[str] = None,
):
    raise NotImplementedError(
        
    )
