import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange

from .params import CLIPConfig

class VisionTransformer(nn.Module):
    cfg: CLIPConfig

    @nn.compact
    def __call__(self, x):
        B = x.shape[0]

        x = nn.Conv(
            features=self.cfg.vision_width,
            kernel_size=(self.cfg.patch_size, self.cfg.patch_size),
            strides=(self.cfg.patch_size, self.cfg.patch_size),
            padding="VALID",
            name="patch_embed",
        )(x)

        x = rearrange(x, "b h w c -> b (h w) c")

        cls = self.param(
            "cls_token",
            nn.initializers.zeros,
            (1, 1, self.cfg.vision_width),
        )
        cls = jnp.tile(cls, (B, 1, 1))
        x = jnp.concatenate([cls, x], axis=1)

        pos = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.01),
            (1, x.shape[1], self.cfg.vision_width),
        )
        x = x + pos

        for i in range(self.cfg.vision_layers):
            h = nn.LayerNorm(name=f"ln1_{i}")(x)
            h = nn.SelfAttention(
                num_heads=self.cfg.vision_heads,
                qkv_features=self.cfg.vision_width,
                name=f"attn_{i}",
            )(h)
            x = x + h

        x = nn.LayerNorm(name="ln_post")(x[:, 0])
        return x

class TextTransformer(nn.Module):
    cfg: CLIPConfig

    @nn.compact
    def __call__(self, tokens):

        x = nn.Embed(
            num_embeddings=self.cfg.vocab_size,
            features=self.cfg.text_width,
            name="token_embed",
        )(tokens)
        
        pos = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.01),
            (1, self.cfg.context_length, self.cfg.text_width),
        )
        x = x + pos
        
        causal_mask = nn.attention.make_causal_mask(tokens)
        for i in range(self.cfg.text_layers):
            h = nn.LayerNorm(name=f"ln1_{i}")(x)
            h = nn.SelfAttention(
                num_heads=self.cfg.text_heads,
                qkv_features=self.cfg.text_width,
                name=f"attn_{i}",
            )(h, mask=causal_mask)
            x = x + h

        x = nn.LayerNorm(name="ln_post")(x[:, -1])
        return x

class CLIP(nn.Module):
    cfg: CLIPConfig

    @nn.compact
    def __call__(self, images, texts):
        image_features = VisionTransformer(self.cfg)(images)
        text_features = TextTransformer(self.cfg)(texts)

        image_emb = nn.Dense(
            self.cfg.embed_dim,
            name="image_projection",
        )(image_features)

        text_emb = nn.Dense(
            self.cfg.embed_dim,
            name="text_projection",
        )(text_features)

        image_emb = image_emb / jnp.linalg.norm(image_emb, axis=-1, keepdims=True)
        text_emb = text_emb / jnp.linalg.norm(text_emb, axis=-1, keepdims=True)

        logit_scale = self.param(
            "logit_scale",
            lambda k: jnp.ones(()) * jnp.log(1 / 0.07),
        )
        logit_scale = jnp.exp(logit_scale)

        logits = logit_scale * image_emb @ text_emb.T
        return logits

__all__ = [
    "VisionTransformer",
    "TextTransformer",
    "CLIP",
]
