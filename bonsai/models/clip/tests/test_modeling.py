import jax
import jax.numpy as jnp
from clip.params import CLIPConfig
from clip.modeling import CLIPModel, clip_contrastive_loss
from clip.tokenizer import simple_whitespace_tokenizer

def test_clip_forward_and_loss():
    cfg = CLIPConfig()
    cfg.model_size = "ViT-B/32"
    cfg.dtype = "float32"
    cfg.apply_model_size_presets()
    cfg.image_size = 64
    cfg.text_max_length = 16
    model = CLIPModel(cfg)
    rng = jax.random.PRNGKey(0)
    images = jax.random.normal(rng, (2, cfg.image_size, cfg.image_size, 3))
    tokens, _ = simple_whitespace_tokenizer(["a cat", "a dog"], max_length=cfg.text_max_length)
    params = model.init(rng, images, tokens)
    logits, i_e, t_e, scale = model.apply(params, images, tokens, deterministic=True)
    assert logits.shape == (2,2)
    assert i_e.shape[1] == cfg.proj_dim
    assert t_e.shape[1] == cfg.proj_dim
    loss = clip_contrastive_loss(logits)
    assert jnp.isfinite(loss)
