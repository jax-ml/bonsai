from dataclasses import dataclass

@dataclass
class CLIPConfig:
    # Vision Encoder
    image_size: int = 224
    patch_size: int = 16
    vit_dim: int = 256
    vit_depth: int = 6
    vit_heads: int = 8

    # Text Encoder
    text_vocab_size: int = 49408
    text_max_len: int = 32
    text_dim: int = 256
    text_depth: int = 4
    text_heads: int = 8

    # Optimization
    lr: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 8
