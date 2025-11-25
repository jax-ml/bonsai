"""
Training entrypoint for CLIP reproduction.

- Supports ViT-B/32 and ViT-L/14 via CLIPConfig.model_size
- Supports loading a pretrained tokenizer (tokenizers lib) via tokenizer_path
- Mixed precision: set CLIPConfig.dtype = "float16" to run model in float16 where supported
- Uses TFDS CIFAR-10 as toy dataset mapping labels -> captions
"""

import os
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds
from flax.training import train_state

from .params import CLIPConfig
from .modeling import CLIPModel, clip_contrastive_loss
from .tokenizer import simple_whitespace_tokenizer, load_tokenizer


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 3
    lr: float = 3e-4
    workdir: str = "/tmp/clip_run"
    tokenizer_path: Optional[str] = None
    image_size: Optional[int] = None  
    
CIFAR10_LABELS = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]
TEMPLATES = ["a photo of a {}", "a close-up photo of a {}"]

def preprocess(example, image_size):
    img = example["image"]
    img = tfds.as_numpy(img) / 255.0

    if image_size != 32:
        import cv2 
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    # normalize to [-1,1]
    img = (img - 0.5) * 2.0
    label = int(example["label"])
    return img.astype(np.float32), label

def make_datasets(batch_size, image_size):
    ds_train = tfds.load("cifar10", split="train", as_supervised=False)
    ds_train = ds_train.shuffle(1024).map(lambda ex: tfds.as_numpy(ex))
    ds_train = ds_train.batch(batch_size)
    ds_val = tfds.load("cifar10", split="test", as_supervised=False).map(lambda ex: tfds.as_numpy(ex)).batch(batch_size)
    return ds_train, ds_val

def create_state(key, cfg: CLIPConfig, lr: float):
    cfg.apply_model_size_presets()
    model = CLIPModel(cfg)
    dummy_img = jnp.zeros((1, cfg.image_size, cfg.image_size, 3), dtype=jnp.float32)
    dummy_txt = jnp.zeros((1, cfg.text_max_length), dtype=jnp.int32)
    params = model.init(key, dummy_img, dummy_txt)
    tx = optax.adamw(lr)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state, model

@jax.jit
def train_step(state, model, images, tokens):
    def loss_fn(params):
        logits, _, _, _ = model.apply(params, images, tokens, deterministic=False)
        loss = clip_contrastive_loss(logits)
        return loss
    grads = jax.grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state

def zero_shot_eval(state_params, model, tokenizer_fn, image_batch):
    prompts = [t.format(c) for c in CIFAR10_LABELS for t in TEMPLATES]
    if tokenizer_fn is None:
        token_ids, _ = simple_whitespace_tokenizer(prompts, max_length=model.cfg.text_max_length)
    else:
        token_ids = tokenizer_fn(prompts, max_length=model.cfg.text_max_length)
    text_embs = model.apply(state_params, jnp.zeros((1, model.cfg.image_size, model.cfg.image_size, 3)), token_ids, method=CLIPModel.encode_text, deterministic=True)
    n_templates = len(TEMPLATES)
    num_classes = len(CIFAR10_LABELS)
    text_embs = text_embs.reshape(num_classes, n_templates, -1).mean(axis=1)  

    img_embs = model.apply(state_params, jnp.array(image_batch), jnp.zeros((len(image_batch), model.cfg.text_max_length), dtype=jnp.int32), method=CLIPModel.encode_image, deterministic=True)
    sims = jnp.matmul(img_embs, text_embs.T) 
    preds = jnp.argmax(sims, axis=-1)
    return preds

def train_main(train_cfg: TrainConfig, model_cfg: CLIPConfig):

    model_cfg.apply_model_size_presets()

    tokenizer_fn = None
    if train_cfg.tokenizer_path is not None:
        tokenizer_fn = load_tokenizer(train_cfg.tokenizer_path)
        if tokenizer_fn is None:
            print("Tokenizer load failed; falling back to simple_whitespace_tokenizer")

    key = jax.random.PRNGKey(0)
    state, model = create_state(key, model_cfg, train_cfg.lr)
    model.cfg = model_cfg  

    ds_train, ds_val = make_datasets(train_cfg.batch_size, model_cfg.image_size)

    os.makedirs(train_cfg.workdir, exist_ok=True)

    print("Starting training loop (toy CIFAR-10) ...")
    for epoch in range(1, train_cfg.epochs + 1):
        t0 = time.time()

        for batch in ds_train:
            imgs = []
            labels = []
            for ex in batch:
                img = ex["image"].astype("float32") / 255.0
                if model_cfg.image_size != 32:
                    import cv2
                    img = cv2.resize(img, (model_cfg.image_size, model_cfg.image_size), interpolation=cv2.INTER_LINEAR)
                img = (img - 0.5) * 2.0
                imgs.append(img)
                labels.append(int(ex["label"]))
            imgs = jnp.array(imgs)
            captions = [f"a photo of a {CIFAR10_LABELS[l]}" for l in labels]
            if tokenizer_fn is None:
                tokens, _ = simple_whitespace_tokenizer(captions, max_length=model_cfg.text_max_length)
            else:
                tokens = tokenizer_fn(captions, max_length=model_cfg.text_max_length)
            state = train_step(state, model, imgs, jnp.array(tokens))
        t1 = time.time()
        
        val_images = []
        val_labels = []
        for i, ex in enumerate(ds_val):
            for elem in ex:
                img = elem["image"].astype("float32") / 255.0
                if model_cfg.image_size != 32:
                    import cv2
                    img = cv2.resize(img, (model_cfg.image_size, model_cfg.image_size), interpolation=cv2.INTER_LINEAR)
                img = (img - 0.5) * 2.0
                val_images.append(img)
                val_labels.append(int(elem["label"]))
                if len(val_images) >= 256:
                    break
            break
        if len(val_images) > 0:
            preds = zero_shot_eval(state.params, model, tokenizer_fn, val_images)
            # calculate toy accuracy
            preds = list(map(int, list(preds)))
            acc = sum([p == l for p, l in zip(preds, val_labels[:len(preds)])]) / max(1, len(preds))
        else:
            acc = 0.0

        ckpt_path = os.path.join(train_cfg.workdir, f"clip_epoch{epoch}.npz")
        flat = jax.tree_map(lambda x: np.array(x), state.params)

        np_save_dict = {}
        def collect(prefix, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    collect(prefix + "/" + k, v)
                else:
                    np_save_dict[prefix + "/" + k] = v
        try:
            collect("", flat)
            np.savez_compressed(ckpt_path, **np_save_dict)
            print("Saved checkpoint:", ckpt_path)
        except Exception as e:
            print("Checkpoint save failed:", e)

        print(f"Epoch {epoch} done — time {t1-t0:.1f}s — toy zero-shot acc: {acc:.4f}")

    return state

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workdir", type=str, default="/tmp/clip_run")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--model_size", type=str, default="ViT-B/32")
    parser.add_argument("--dtype", type=str, default="float32")
    args = parser.parse_args()

    model_cfg = CLIPConfig()
    model_cfg.model_size = args.model_size
    model_cfg.dtype = args.dtype
    model_cfg.apply_model_size_presets()

    train_cfg = TrainConfig(batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, workdir=args.workdir, tokenizer_path=args.tokenizer_path)

    train_main(train_cfg, model_cfg)
