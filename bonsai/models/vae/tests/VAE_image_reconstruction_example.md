---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
---

<a href="https://colab.research.google.com/github/jax-ml/bonsai/blob/main/bonsai/models/vae/tests/VAE_image_reconstruction_example.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

+++

# **Image Reconstruction with VAE**

This notebook demonstrates image reconstruction using the [Bonsai library](https://github.com/jax-ml/bonsai) and the [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse) weights.

+++

## **Set-up**

```{code-cell}
!pip install -q git+https://github.com/jax-ml/bonsai@main
!pip install -q pillow matplotlib requests
!pip install -q scikit-image
```

```{code-cell}
import os
import zipfile

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

print(f"JAX version: {jax.__version__}")
print(f"JAX device: {jax.devices()[0].platform}")
```

## **Download Sample Images**

```{code-cell}
def download_coco_test_set(dest_folder="./coco_val2017"):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    url = "http://images.cocodataset.org/zips/val2017.zip"
    target_path = os.path.join(dest_folder, "val2017.zip")

    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with (
        open(target_path, "wb") as f,
        tqdm(
            desc="Progress",
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

    print("\nExtracting files...")
    with zipfile.ZipFile(target_path, "r") as zip_ref:
        zip_ref.extractall(dest_folder)

    os.remove(target_path)
    print(f"Done! Images are saved in: {os.path.abspath(dest_folder)}")


download_coco_test_set()
```

## **Load VAE Model**

```{code-cell}
from huggingface_hub import snapshot_download

from bonsai.models.vae import modeling, params


def load_vae_model():
    model_name = "stabilityai/sd-vae-ft-mse"
    config = modeling.ModelConfig.stable_diffusion_v1_5()

    print(f"Downloading {model_name}...")
    model_ckpt_path = snapshot_download(model_name)
    print("Download complete!")

    model = params.create_model_from_safe_tensors(file_dir=model_ckpt_path, cfg=config)

    print("VAE model loaded_successfully!")

    return model
```

## **Image Preprocessing**

```{code-cell}
def preprocess(image):
    image = image.convert("RGB").resize((256, 256))

    # normalization: [0, 255] -> [0, 1] -> [-1, 1]
    image = np.array(image).astype(np.float32) / 255.0
    image = (image * 2.0) - 1.0

    # add dimension: (256, 256, 3) -> (1, 256, 256, 3)
    return jnp.array(image[None, ...])
```

## **Image Postprocessing**

```{code-cell}
def postprocess(tensor):
    # restoration
    tensor = jnp.clip(tensor, -1.0, 1.0)
    tensor = (tensor + 1.0) / 2.0
    tensor = (tensor * 255).astype(np.uint8)

    # (1, 256, 256, 3) -> (256, 256, 3)
    return Image.fromarray(np.array(tensor[0]))
```

## **Run Reconstruct on Sample Images**

```{code-cell}
vae = load_vae_model()

dest_folder = "./coco_val2017"
image_dir = os.path.join(dest_folder, "val2017")

if not os.path.exists(image_dir):
    raise FileNotFoundError(f"Could not find images folder: {image_dir}")

image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".JPEG"))][:5]

if not image_files:
    raise Exception("There are no image files in the folder.")

psnr_scores = []
ssim_scores = []

fig, axes = plt.subplots(5, 2, figsize=(10, 25))
plt.subplots_adjust(hspace=0.3)

for i, file_name in enumerate(image_files):
    img_path = os.path.join(image_dir, file_name)
    raw_img = Image.open(img_path).convert("RGB")

    input_tensor = preprocess(raw_img)
    reconstructed_tensor = vae(input_tensor)
    reconstructed_img = postprocess(reconstructed_tensor)

    original_resized = raw_img.resize((256, 256))

    # convert unit8 to numpy array
    orig_np = np.array(original_resized)
    recon_np = np.array(reconstructed_img)

    # PSNR, SSIM calculation
    p_score = psnr(orig_np, recon_np, data_range=255)
    s_score = ssim(orig_np, recon_np, channel_axis=2, data_range=255)

    psnr_scores.append(p_score)
    ssim_scores.append(s_score)

    # visualization
    axes[i, 0].imshow(original_resized)
    axes[i, 0].set_title(f"Original: {file_name}")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(reconstructed_img)
    axes[i, 1].set_title(f"Reconstructed\nPSNR: {p_score:.2f}, SSIM: {s_score:.4f}")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.show()

print(f"\n{'=' * 40}")
print("--- Final Reconstruction Quality Report (N=5) ---")
print(f"Average PSNR: {np.mean(psnr_scores):.2f} dB")
print(f"Average SSIM: {np.mean(ssim_scores):.4f}")
print(f"{'=' * 40}")
```

## **Batch Processing**

```{code-cell}
def batch_reconstruct_vae(vae, image_paths):
    # 1. Preprocessing and batch stacking
    input_tensors = []
    original_images_resized = []

    for path in image_paths:
        raw_img = Image.open(path).convert("RGB")
        original_resized = raw_img.resize((256, 256))
        original_images_resized.append(original_resized)

        tensor = preprocess(raw_img)
        # Assuming the result is in the form [B, H, W, C]
        input_tensors.append(tensor[0])

    batch_tensor = jnp.stack(input_tensors)

    # 2. Inference
    recon_batch = vae(batch_tensor)

    # 3. Results processing and indicator calculator
    batch_results = []

    for i in range(len(image_paths)):
        recon_img = postprocess(recon_batch[i : i + 1])

        orig_np = np.array(original_images_resized[i])
        recon_np = np.array(recon_img)

        p_val = psnr(orig_np, recon_np, data_range=255)
        s_val = ssim(orig_np, recon_np, channel_axis=2, data_range=255)

        batch_results.append(
            {
                "name": os.path.basename(image_paths[i]),
                "recon_img": recon_img,
                "orig_img": original_images_resized[i],
                "psnr": p_val,
                "ssim": s_val,
            }
        )

    return batch_results


print("\n" + "=" * 50)
print("VAE BATCH RECONSTRUCTION RESULTS")
print("=" * 50)

target_paths = [os.path.join(image_dir, f) for f in image_files[:5]]
results = batch_reconstruct_vae(vae, target_paths)

all_psnr = []
all_ssim = []

for i, res in enumerate(results):
    print(f"[{i + 1}] {res['name']}: PSNR={res['psnr']:.2f}dB, SSIM={res['ssim']:.4f}")
    all_psnr.append(res["psnr"])
    all_ssim.append(res["ssim"])

print("-" * 50)
print(f"Batch Average PSNR: {np.mean(all_psnr):.2f} dB")
print(f"Batch Average SSIM: {np.mean(all_ssim):.4f}")
```
