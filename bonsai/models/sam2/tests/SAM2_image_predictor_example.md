---
jupytext:
  cell_metadata_filter: -all
  formats: ipynb,md:myst
  main_language: python
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

<a href="https://colab.research.google.com/github/jax-ml/bonsai/blob/main/bonsai/models/sam2/tests/SAM2_image_predictor_example.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

+++

Suggested runtime: TPU v2-8

+++

# **Object masks in images from prompts with SAM 2**

+++

Segment Anything Model 2 (SAM 2) predicts object masks given prompts that indicate the desired object. The model first converts the image into an image embedding that allows high quality masks to be efficiently produced from a prompt.

*This colab is modified from the original [SAM2 colab](https://github.com/facebookresearch/sam2/blob/main/notebooks/image_predictor_example.ipynb).*

+++

## **Set-up**

```{code-cell} ipython3
!pip install -q git+https://github.com/jax-ml/bonsai@sam2-faulty-masks
!pip install -q opencv-python
```

```{code-cell} ipython3
import os
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image
```

```{code-cell} ipython3
!wget -q -P ./images/ truck.jpg "https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/images/truck.jpg"
!wget -q -P ./images/ cars.jpg "https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/images/cars.jpg"
!wget -q -P ./images/ groceries.jpg "https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/images/groceries.jpg"

" ".join(os.listdir("./images"))
```

```{code-cell} ipython3
key = jax.random.PRNGKey(3)


def show_mask(mask, ax, *, random_color=False, borders=True):
    if random_color:
        rgb = jax.random.uniform(key, (3,))
        color = jnp.concatenate([rgb, jnp.array([0.6])])
    else:
        color = jnp.array([30 / 255, 144 / 255, 1.0, 0.6])
    h, w = mask.shape[-2:]
    mask_uint8 = jax.device_get(mask.astype(jnp.uint8))  # JAX → np broadcast
    mask_img = mask_uint8.reshape(h, w, 1) * color.reshape(1, 1, -1)

    if borders:
        import cv2

        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(c, epsilon=0.01, closed=True) for c in contours]
        mask_img = cv2.drawContours(mask_img.copy(), contours, -1, color=(1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_img)


def show_points(coords, labels, ax, marker_size=375):
    coords, labels = jax.device_get(coords), jax.device_get(labels)
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0], pos_points[:, 1], color="green", marker="*", s=marker_size, edgecolor="white", linewidth=1.25
    )
    ax.scatter(
        neg_points[:, 0], neg_points[:, 1], color="red", marker="*", s=marker_size, edgecolor="white", linewidth=1.25
    )


def show_box(box, ax):
    x0, y0 = float(box[0]), float(box[1])
    w, h = float(box[2] - box[0]), float(box[3] - box[1])
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))


def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(jax.device_get(image))
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None, "Need labels for each point"
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        plt.axis("off")
        plt.show()
```

```{code-cell} ipython3
image = Image.open("images/truck.jpg")
image = np.array(image.convert("RGB"))
```

```{code-cell} ipython3
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis("on")
plt.show()
```

## **Selecting objects with SAM 2**

```{code-cell} ipython3
from bonsai.models.sam2 import modeling, params

model_name = "facebook/sam2-hiera-small-hf"
MODEL_CP_PATH = "./checkpoints/" + model_name.split("/")[1]
snapshot_download(model_name, local_dir=MODEL_CP_PATH)

config = modeling.SAM2Config.sam2_small()
model_obj = params.create_sam2_from_pretrained(MODEL_CP_PATH + "/model.safetensors", config)

predictor = modeling.SAM2ImagePredictor(model_obj)
```

```{code-cell} ipython3
predictor.set_image(image)
```

```{code-cell} ipython3
points = jnp.array([[500, 375]])
labels = jnp.array([1.0])
```

```{code-cell} ipython3
@partial(jax.jit, static_argnames=["model"])
def forward(model, points, labels):
    return model.predict(points, labels)


masks, scores, logits = forward(predictor, points, labels)
jax.block_until_ready(masks)
```

```{code-cell} ipython3
sorted_ind = np.argsort(scores)[::-1][0]
masks = masks[:, sorted_ind, :, :]
scores = scores[:, sorted_ind]
logits = logits[:, sorted_ind, :, :]
```

```{code-cell} ipython3
print(type(image))
print(image.shape)
```

```{code-cell} ipython3
print(type(masks[0]))
print(masks[0].shape)
```

```{code-cell} ipython3
for i in range(masks.shape[1]):
    print(i)
    mask = masks[0][i]
    score = scores[i]
    print(f"mask {i} shape before reshape:", mask.shape)
    print(f"score {i} shape:", score.shape)
    # Convert (3, 256, 256) to (256, 256)
    if mask.shape[0] == 3:
        mask = mask[0]  # or mask = mask.mean(axis=0) for average
    # Use only the first score value
    score_scalar = float(score[i]) if hasattr(score, "__getitem__") else float(score)
    plt.figure(figsize=(8, 8))
    plt.imshow(jax.device_get(image))
    plt.imshow(jax.device_get(mask), alpha=0.5, cmap="Blues")
    plt.scatter(
        jax.device_get(points)[:, 0],
        jax.device_get(points)[:, 1],
        c=["lime" if l == 1 else "red" for l in jax.device_get(labels)],
        s=200,
        marker="*",
        edgecolors="white",
        linewidths=1.5,
    )
    plt.title(f"Score: {score_scalar:.3f}")
    plt.axis("off")
    plt.show()
```
