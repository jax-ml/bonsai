from dataclasses import dataclass
from typing import Literal

@dataclass
class CLIPConfig:
    image_size: int = 224
    encoder_type: Literal["vit", "resnet"] = "vit"
    model_size: Literal["ViT-B/32", "ViT-L/14"] = "ViT-B/32"
    dtype: str = "float32"   

    patch_size: int = 32
    image_embed_dim: int = 768
    vit_num_layers: int = 12
    vit_num_heads: int = 12
    vit_mlp_dim: int = 3072

    resnet_stem_channels: int = 64
    resnet_block_channels: tuple = (64, 128, 256, 512)
    resnet_block_repeats: tuple = (3, 4, 6, 3)

    # text encoder
    text_embed_dim: int = 512
    text_vocab_size: int = 49408
    text_max_length: int = 77
    text_num_layers: int = 12
    text_num_heads: int = 8
    text_mlp_dim: int = 2048

    proj_dim: int = 512

    def apply_model_size_presets(self):
        if self.model_size == "ViT-B/32":
            self.patch_size = 32
            self.image_embed_dim = 768
            self.vit_num_layers = 12
            self.vit_num_heads = 12
            self.vit_mlp_dim = 3072
            self.text_embed_dim = 512
            self.proj_dim = 512
        elif self.model_size == "ViT-L/14":
            self.patch_size = 14
            self.image_embed_dim = 1024
            self.vit_num_layers = 24
            self.vit_num_heads = 16
            self.vit_mlp_dim = 4096
            self.text_embed_dim = 1024
            self.proj_dim = 1024
        else:
            raise ValueError("Unknown model_size: " + str(self.model_size))
