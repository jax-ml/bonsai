---
jupytext:
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

+++ {"colab_type": "text", "id": "view-in-github"}

<a href="https://colab.research.google.com/github/jax-ml/bonsai/blob/main/bonsai/models/sam2/tests/SAM2_image_predictor_example.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

+++ {"id": "qL9hFclxkPLg"}

Suggested runtime: TPU v2-8

+++ {"id": "tLcHRPOLGtRn"}

# **Object masks in images from prompts with SAM 2**

+++ {"id": "gBTt5rAEG5Aq"}

Segment Anything Model 2 (SAM 2) predicts object masks given prompts that indicate the desired object. The model first converts the image into an image embedding that allows high quality masks to be efficiently produced from a prompt.

*This colab is modified from the original [SAM2 colab](https://github.com/facebookresearch/sam2/blob/main/notebooks/image_predictor_example.ipynb).*

+++ {"id": "Yueb7mukJFf-"}

## **Set-up**

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: CawGpos5-mUh
outputId: da510d7f-6a8e-4b98-c361-f32ba1cb3cef
---
!pip install -q git+https://github.com/jax-ml/bonsai@sam2-faulty-masks
!pip install -q opencv-python
```

```{code-cell} ipython3
:id: kOuBDT900JUv

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
---
colab:
  base_uri: https://localhost:8080/
  height: 53
id: 4Y9xtbHZkWnK
outputId: 5549f134-255b-4241-dd1c-f7912b19d86e
---
!wget -q -P ./images/ truck.jpg "https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/images/truck.jpg"
!wget -q -P ./images/ cars.jpg "https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/images/cars.jpg"
!wget -q -P ./images/ groceries.jpg "https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/images/groceries.jpg"

" ".join(os.listdir("./images"))
```

```{code-cell} ipython3
:id: Cn355b3ukZLG

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
:id: wQWRhOq0ka7s

image = Image.open("images/truck.jpg")
image = np.array(image.convert("RGB"))
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 582
id: aux8UU-n0Tql
outputId: 9b88a32b-20f8-4cf9-bd48-28227122d008
---
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis("on")
plt.show()
```

+++ {"id": "Vsc5SV2GLlFi"}

## **Selecting objects with SAM 2**

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 503
  referenced_widgets: [6cf65da55ddc463fa6809763166be434, 900bf85d6f1547de8048bd4eaa0f90e1,
    01948cbc6af54894819bc6dce1ac550a, 4ca44faf71954c65bf5fc34a76676445, d1edfe227cfa497ba1f946bfa4d0cec5,
    4244255e135d4c839311d533913a4cb2, 7c6fa9872f004a8481600f3cd6c23826, c4ca99320536430b8b3e4c40ffa0cfca,
    eb1c80c1cf2244be838f0c7d4bc29f38, e11958583297483c8a8e94e1f7056e6c, 871f6d2959414fbdbc704dfff51f743c]
id: FV1cHo9K0VEf
outputId: 39c79b56-3c9c-4776-8dc4-522632196b8e
---
from bonsai.models.sam2 import modeling, params

model_name = "facebook/sam2-hiera-small-hf"
MODEL_CP_PATH = "./checkpoints/" + model_name.split("/")[1]
snapshot_download(model_name, local_dir=MODEL_CP_PATH)

config = modeling.SAM2Config.sam2_small()
model_obj = params.create_sam2_from_pretrained(MODEL_CP_PATH + "/model.safetensors", config)

predictor = modeling.SAM2ImagePredictor(model_obj)
```

```{code-cell} ipython3
:id: cN-q5J0tQHlO

predictor.set_image(image)
```

```{code-cell} ipython3
:id: L4FU3Iy6XaMU

points = jnp.array([[500, 375]])
labels = jnp.array([1.0])
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: gEH10HyuN64A
outputId: d5c11dad-160a-45bd-e346-7f0898a28f26
---
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
---
colab:
  base_uri: https://localhost:8080/
id: 6All6OmoX_N9
outputId: 3d68fd42-1d6b-4b13-d1f2-d4fdf2d33c96
---
print(type(image))
print(image.shape)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: WZTellIEYzX7
outputId: a691ab1e-d579-4ff2-f9e5-e154bd9b7043
---
print(type(masks[0]))
print(masks[0].shape)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 1000
id: 5p3NK2XDZWau
outputId: 41f83857-10c2-4815-8e62-094392c9159b
---
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
