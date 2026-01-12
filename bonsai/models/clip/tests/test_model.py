import jax
import jax.numpy as jnp

from bonsai.models.clip.modeling import CLIP
from bonsai.models.clip.params import CLIPConfig



def test_clip_forward():
    cfg = CLIPConfig()
    model = CLIP(cfg)

    key = jax.random.PRNGKey(0)

    images = jnp.ones((2, 224, 224, 3))
    texts = jnp.ones((2, cfg.context_length), dtype=jnp.int32)

    params = model.init(key, images, texts)
    logits = model.apply(params, images, texts)

    assert logits.shape == (2, 2)
