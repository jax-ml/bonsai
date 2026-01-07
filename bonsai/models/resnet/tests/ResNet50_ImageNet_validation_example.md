---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: venv (3.12.8)
  language: python
  name: python3
---

<a href="https://colab.research.google.com/github/jax-ml/bonsai/blob/main/bonsai/models/resnet50/tests/ResNet50_ImageNet_validation_example.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

+++

Suggested runtime: GPU (A100/H100) or TPU v2-8

+++

# **ImageNet Classification with ResNet-50**

This notebook demonstrates how to use the ResNet-50 model from the Bonsai library to perform ImageNet classification on real images. The model is pre-trained on ImageNet-1K and can classify images into 1000 different categories.

*This colab demonstrates the ResNet-50 implementation from the [Bonsai library](https://github.com/jax-ml/bonsai).*

+++

## **Set-up**

```{code-cell} ipython3
!pip install -q git+https://github.com/jax-ml/bonsai@main
!pip install -q pillow matplotlib requests
```

```{code-cell} ipython3
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import requests
from huggingface_hub import snapshot_download
from PIL import Image, ImageOps

print(f"JAX version: {jax.__version__}")
print(f"JAX device: {jax.devices()[0].platform}")
```

## **Download Sample Images**

Let's download some sample images to test our ResNet-50 model. We'll use images that are commonly used for testing image classification models.

```{code-cell} ipython3
# Create images directory
os.makedirs("./images", exist_ok=True)

# Download sample images
!wget -q -P ./images/ tench.jpg "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01440764_tench.JPEG"
!wget -q -P ./images/ goldfish.jpg "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01443537_goldfish.JPEG"
!wget -q -P ./images/ tiger_shark.jpg "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01491361_tiger_shark.JPEG"
!wget -q -P ./images/ hammerhead.jpg "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01494475_hammerhead.JPEG"
!wget -q -P ./images/ electric_ray.jpg "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01496331_electric_ray.JPEG"

" ".join(os.listdir("./images"))
```

## **Load ImageNet Class Names**

We need to load the ImageNet class names to interpret the model's predictions.

```{code-cell} ipython3
def load_imagenet_classes():
    """Load ImageNet class names from a common source."""
    # Download ImageNet class names
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(url)
    response.raise_for_status()  # Raise HTTPError for bad responses (4xx, 5xx)
    classes = response.text.strip().split("\n")
    return classes


try:
    imagenet_classes = load_imagenet_classes()
    print(f"Loaded {len(imagenet_classes)} ImageNet classes")
    print("Sample classes:", imagenet_classes[:5])
except requests.exceptions.RequestException:
    # Fallback to a subset of common classes
    print("Could not load ImageNet classes, using fallback")
    imagenet_classes = [f"class_{i}" for i in range(1000)]
```

## **Load ResNet-50 Model**

Now let's load the pre-trained ResNet-50 model from the Bonsai library.

```{code-cell} ipython3
from bonsai.models.resnet import modeling as model_lib
from bonsai.models.resnet import params

model_ckpt_path = snapshot_download("microsoft/resnet-50")
config = model_lib.ModelConfig.resnet50()
model = params.create_resnet_from_pretrained(model_ckpt_path, config)
```

## **Image Preprocessing Functions**

ResNet-50 expects images to be preprocessed in a specific way: resized to 224x224, normalized with ImageNet statistics, and converted to the right format.

```{code-cell} ipython3
def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for ResNet-50 inference."""
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Resize image
    image = ImageOps.fit(image, target_size, method=Image.Resampling.LANCZOS)

    # Convert to numpy array and normalize to [0, 1]
    image_array = np.array(image).astype(np.float32) / 255.0

    # ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Normalize
    image_array = (image_array - mean) / std

    # Add batch dimension and ensure correct shape (batch, height, width, channels)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array, image


def show_image_with_predictions(image, predictions, top_k=5):
    """Display image with top-k predictions."""
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Show image
    ax1.imshow(image)
    ax1.set_title("Input Image", fontsize=14)
    ax1.axis("off")

    # Show predictions
    y_pos = np.arange(len(predictions))
    ax2.barh(y_pos, [p[1] for p in predictions])
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([p[0] for p in predictions])
    ax2.set_xlabel("Confidence Score")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([p[0] for p in predictions])
    ax2.set_xlabel("Confidence Score")
    ax2.set_title(f"Top-{top_k} Predictions", fontsize=14)
    ax2.invert_yaxis()  # Top prediction at the top

    plt.tight_layout()
    plt.show()
```

## **Run Inference on Sample Images**

Now let's test our ResNet-50 model on the downloaded images!

```{code-cell} ipython3
def classify_image(model, image_path, top_k=5):
    """Classify a single image and return top-k predictions."""
    # Preprocess image
    input_tensor, original_image = preprocess_image(image_path)

    # Convert to JAX array
    input_tensor = jnp.array(input_tensor)

    # Run inference
    logits = model_lib.forward(model, input_tensor)

    # Apply softmax to get probabilities
    probabilities = jax.nn.softmax(logits[0])

    # Get top-k predictions
    top_indices = jnp.argsort(probabilities)[-top_k:][::-1]
    top_probs = probabilities[top_indices]

    # Format results
    predictions = []
    for idx, prob in zip(top_indices, top_probs):
        class_name = imagenet_classes[idx] if idx < len(imagenet_classes) else f"class_{idx}"
        predictions.append((class_name, float(prob)))

    return predictions, original_image


# Test on all images
image_files = [f for f in os.listdir("./images") if f.endswith((".jpg", ".jpeg", ".png", ".JPEG"))]

for image_file in image_files:
    print(f"\n{'=' * 60}")
    print(f"Classifying: {image_file}")
    print(f"{'=' * 60}")

    try:
        predictions, image = classify_image(model, f"./images/{image_file}")

        print("\nTop-5 predictions:")
        for i, (class_name, confidence) in enumerate(predictions, 1):
            print(f"{i}. {class_name}: {confidence:.4f} ({confidence * 100:.2f}%)")

        # Show visualization
        show_image_with_predictions(image, predictions)

    except Exception as e:
        print(f"Error processing {image_file}: {e}")
```

## **Batch Processing**

Let's also demonstrate batch processing for multiple images at once, which is more efficient.

```{code-cell} ipython3
def batch_classify_images(model, image_paths, top_k=5):
    """Classify multiple images in a single batch."""
    # Preprocess all images
    input_tensors = []
    original_images = []

    for image_path in image_paths:
        input_tensor, original_image = preprocess_image(image_path)
        input_tensors.append(input_tensor[0])  # Remove batch dimension
        original_images.append(original_image)

    # Stack into batch
    batch_tensor = jnp.stack(input_tensors)

    # Run batch inference
    logits = model_lib.forward(model, batch_tensor)

    # Process results
    all_predictions = []
    for i in range(len(image_paths)):
        probabilities = jax.nn.softmax(logits[i])
        top_indices = jnp.argsort(probabilities)[-top_k:][::-1]
        top_probs = probabilities[top_indices]

        predictions = []
        for idx, prob in zip(top_indices, top_probs):
            class_name = imagenet_classes[idx] if idx < len(imagenet_classes) else f"class_{idx}"
            predictions.append((class_name, float(prob)))

        all_predictions.append(predictions)

    return all_predictions, original_images


# Test batch processing
print("\n" + "=" * 80)
print("BATCH PROCESSING RESULTS")
print("=" * 80)

image_paths = [f"./images/{f}" for f in image_files[:3]]  # Process first 3 images
batch_predictions, batch_images = batch_classify_images(model, image_paths)

for i, (image_path, predictions, image) in enumerate(zip(image_paths, batch_predictions, batch_images)):
    print(f"\nImage {i + 1}: {os.path.basename(image_path)}")
    print("-" * 40)
    for j, (class_name, confidence) in enumerate(predictions, 1):
        print(f"{j}. {class_name}: {confidence:.4f} ({confidence * 100:.2f}%)")

    # Show individual image with predictions
    show_image_with_predictions(image, predictions)
```

## **Conclusion**

This notebook demonstrates how to use the ResNet-50 model from the Bonsai library to perform ImageNet classification on real images. The model successfully:

1. **Loads pre-trained weights** from Microsoft's ResNet-50 model
2. **Preprocesses images** according to ImageNet standards
3. **Performs inference** on individual and batched images
4. **Provides confidence scores** for top-k predictions

The implementation shows that the Bonsai ResNet-50 model works correctly for real-world image classification tasks
