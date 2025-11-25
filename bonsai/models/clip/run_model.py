from typing import Any
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import initializers
from .params import CLIPConfig

def _get_dtype(cfg: CLIPConfig):
    return jnp.float32 if cfg.dtype == "float32" else jnp.float16

class MLPBlock(nn.Module):
    mlp_dim: int
    out_dim: int
    act = nn.gelu
    dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.mlp_dim, dtype=self.dtype)(x)
        x = self.act(x)
        x = nn.Dense(self.out_dim, dtype=self.dtype)(x)
        return x

class AddPositionEmbs(nn.Module):
    max_len: int
    emb_dim: int
    dtype = jnp.float32

    def setup(self):
        self.pos_emb = self.param("pos_emb", initializers.normal(0.02), (1, self.max_len, self.emb_dim))

    def __call__(self, x):
        return x + self.pos_emb

class TransformerEncoderBlock(nn.Module):
    num_heads: int
    mlp_dim: int
    dtype = jnp.float32

    @nn.compact
    def __call__(self, x, deterministic=True):
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = nn.SelfAttention(num_heads=self.num_heads, dtype=self.dtype, deterministic=deterministic)(y)
        x = x + y
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = MLPBlock(self.mlp_dim, x.shape[-1], dtype=self.dtype)(y)
        return x + y

class SimplePatchEmbed(nn.Module):
    patch_size: int
    emb_dim: int
    dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        ps = self.patch_size
        x = nn.Conv(self.emb_dim, (ps,ps), strides=(ps,ps), padding='VALID', dtype=self.dtype)(x)
        b,h,w,c = x.shape
        return jnp.reshape(x, (b, h*w, c))

class ImageEncoderViT(nn.Module):
    cfg: CLIPConfig
    dtype = jnp.float32

    @nn.compact
    def __call__(self, images, deterministic=True):
        cfg = self.cfg
        x = SimplePatchEmbed(cfg.patch_size, cfg.image_embed_dim, dtype=self.dtype)(images)
        cls = self.param('cls', initializers.zeros, (1,1,cfg.image_embed_dim))
        cls_b = jnp.tile(cls, (x.shape[0],1,1))
        x = jnp.concatenate([cls_b, x], axis=1)
        x = AddPositionEmbs(x.shape[1], cfg.image_embed_dim, dtype=self.dtype)(x)
        for _ in range(cfg.vit_num_layers):
            x = TransformerEncoderBlock(cfg.vit_num_heads, cfg.vit_mlp_dim, dtype=self.dtype)(x, deterministic=deterministic)
        cls_out = x[:,0]
        cls_out = nn.LayerNorm(dtype=self.dtype)(cls_out)
        img_feat = nn.Dense(cfg.image_embed_dim, dtype=self.dtype)(cls_out)
        return img_feat

# small ResNet-like encoder (kept light)
class ResNetStem(nn.Module):
    out_ch: int
    dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.out_ch, (7,7), strides=(2,2), padding='SAME', use_bias=False, dtype=self.dtype)(x)
        x = nn.BatchNorm(use_running_average=True, dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3,3), strides=(2,2), padding='SAME')
        return x

class ResidualBlock(nn.Module):
    out_ch: int
    strides: tuple = (1,1)
    dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        residual = x
        y = nn.Conv(self.out_ch, (3,3), strides=self.strides, padding='SAME', use_bias=False, dtype=self.dtype)(x)
        y = nn.BatchNorm(use_running_average=True, dtype=self.dtype)(y)
        y = nn.relu(y)
        y = nn.Conv(self.out_ch, (3,3), padding='SAME', use_bias=False, dtype=self.dtype)(y)
        y = nn.BatchNorm(use_running_average=True, dtype=self.dtype)(y)
        if residual.shape[-1] != self.out_ch or self.strides != (1,1):
            residual = nn.Conv(self.out_ch, (1,1), strides=self.strides, padding='SAME', use_bias=False, dtype=self.dtype)(residual)
            residual = nn.BatchNorm(use_running_average=True, dtype=self.dtype)(residual)
        return nn.relu(residual + y)

class ImageEncoderResNet(nn.Module):
    cfg: CLIPConfig
    dtype = jnp.float32

    @nn.compact
    def __call__(self, images, deterministic=True):
        cfg = self.cfg
        x = ResNetStem(cfg.resnet_stem_channels, dtype=self.dtype)(images)
        for ch, repeats in zip(cfg.resnet_block_channels, cfg.resnet_block_repeats):
            for i in range(repeats):
                strides = (2,2) if i == 0 else (1,1)
                x = ResidualBlock(ch, strides=strides, dtype=self.dtype)(x)
        x = x.mean(axis=(1,2))
        x = nn.LayerNorm(dtype=self.dtype)(x)
        img_feat = nn.Dense(cfg.image_embed_dim, dtype=self.dtype)(x)
        return img_feat

class TextEncoder(nn.Module):
    cfg: CLIPConfig
    dtype = jnp.float32

    @nn.compact
    def __call__(self, token_ids, deterministic=True):
        cfg = self.cfg
        tok_emb = nn.Embed(num_embeddings=cfg.text_vocab_size, features=cfg.text_embed_dim, dtype=self.dtype)(token_ids)
        tok_emb = AddPositionEmbs(tok_emb.shape[1], cfg.text_embed_dim, dtype=self.dtype)(tok_emb)
        x = tok_emb
        for _ in range(cfg.text_num_layers):
            x = TransformerEncoderBlock(cfg.text_num_heads, cfg.text_mlp_dim, dtype=self.dtype)(x, deterministic=deterministic)
        eos_feat = x[:, -1, :]
        eos_feat = nn.LayerNorm(dtype=self.dtype)(eos_feat)
        txt_feat = nn.Dense(cfg.text_embed_dim, dtype=self.dtype)(eos_feat)
        return txt_feat

class CLIPModel(nn.Module):
    cfg: CLIPConfig
    dtype = jnp.float32

    def setup(self):
        self.cfg.apply_model_size_presets()
        self._dtype = _get_dtype(self.cfg)
        if self.cfg.encoder_type == 'vit':
            self.image_encoder = ImageEncoderViT(self.cfg, dtype=self._dtype)
        else:
            self.image_encoder = ImageEncoderResNet(self.cfg, dtype=self._dtype)
        self.text_encoder = TextEncoder(self.cfg, dtype=self._dtype)
        self.img_proj = nn.Dense(self.cfg.proj_dim, dtype=self._dtype, use_bias=False)
        self.txt_proj = nn.Dense(self.cfg.proj_dim, dtype=self._dtype, use_bias=False)
        self.logit_scale = self.param('logit_scale', lambda rng, shape: jnp.array(1.0), ())

    def encode_image(self, images, deterministic=True):
        feats = self.image_encoder(images, deterministic=deterministic)
        proj = self.img_proj(feats)
        proj = proj / (jnp.linalg.norm(proj, axis=-1, keepdims=True) + 1e-10)
        return proj

    def encode_text(self, token_ids, deterministic=True):
        feats = self.text_encoder(token_ids, deterministic=deterministic)
        proj = self.txt_proj(feats)
        proj = proj / (jnp.linalg.norm(proj, axis=-1, keepdims=True) + 1e-10)
        return proj

    def __call__(self, images, token_ids, deterministic=True):
        i_e = self.encode_image(images, deterministic=deterministic)
        t_e = self.encode_text(token_ids, deterministic=deterministic)
        scale = jnp.exp(self.logit_scale)
        logits = jnp.matmul(i_e, t_e.T) * scale
        return logits, i_e, t_e, scale

def clip_contrastive_loss(logits: jnp.ndarray):
    n = logits.shape[0]
    labels = jnp.arange(n)
    loss_i = jnp.mean(nn.softmax_cross_entropy(logits=logits, labels=jax.nn.one_hot(labels, n), axis=1))
    loss_t = jnp.mean(nn.softmax_cross_entropy(logits=logits.T, labels=jax.nn.one_hot(labels, n), axis=1))
    return 0.5 * (loss_i + loss_t)
