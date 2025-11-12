# Copyright 2025 The Bonsai AI Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ADE20K Data Loader for CLIP-JAX training and validation."""

import os
import random
import numpy as np
from PIL import Image
import jax.numpy as jnp

from bonsai.models.clip_jax.utils.preprocess import preprocess_image, tokenize_text


# --- Dataset path ---
ADE_PATH = "datasets/ADEChallengeData2016"


def load_scene_labels():
    """
    Load ADE20K scene label mappings from sceneCategories.txt.
    Returns:
        dict: {image_id -> scene_class}
    """
    scene_map = {}
    txt_path = os.path.join(ADE_PATH, "sceneCategories.txt")

    if not os.path.exists(txt_path):
        print("⚠️ sceneCategories.txt not found — using synthetic captions.")
        return scene_map

    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                scene_map[parts[0]] = parts[1]

    return scene_map


def list_images(split: str = "training", max_samples: int = 200):
    """
    List ADE20K image paths for a given split.

    Args:
        split (str): "training" or "validation"
        max_samples (int): Limit for samples to load

    Returns:
        list: List of image file paths
    """
    img_dir = os.path.join(ADE_PATH, "images", split)
    imgs = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg")]
    random.shuffle(imgs)
    return imgs[:max_samples]


def generate_caption(img_path: str, scene_labels: dict):
    """
    Generate a synthetic or real caption for an ADE20K image.

    Args:
        img_path (str): Path to image
        scene_labels (dict): Mapping of image_id to scene class

    Returns:
        str: Caption describing the scene
    """
    base = os.path.basename(img_path)
    label = scene_labels.get(os.path.splitext(base)[0], None)

    if label:
        templates = [
            f"a photo of a {label}",
            f"an image of a {label}",
            f"a picture showing a {label}",
        ]
        return random.choice(templates)

    # fallback if label not found
    cls = random.randint(0, 150)
    return f"a photo of a scene containing class_{cls}"


def data_generator(split: str = "training", batch_size: int = 8, image_size: int = 224, max_len: int = 32):
    """
    Infinite generator yielding batches of (image, token_ids) for CLIP-JAX training.

    Args:
        split (str): "training" or "validation"
        batch_size (int): Number of samples per batch
        image_size (int): Resize dimension
        max_len (int): Max text token length

    Yields:
        Tuple (jax.numpy.ndarray, jax.numpy.ndarray)
    """
    imgs = list_images(split)
    scene_labels = load_scene_labels()

    while True:
        batch_imgs, batch_toks = [], []

        for _ in range(batch_size):
            img_path = random.choice(imgs)
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"⚠️ Skipping broken image {img_path}: {e}")
                continue

            arr = preprocess_image(img, image_size)
            caption = generate_caption(img_path, scene_labels)
            toks = tokenize_text([caption], max_len)[0].astype(np.int32)

            batch_imgs.append(arr)
            batch_toks.append(toks)

        yield jnp.stack(batch_imgs), jnp.stack(batch_toks)
