# UNETR: UNEt TRansformers

This directory contains a Flax NNX implementation of UNETR, a Vision Transformer-based architecture for semantic image segmentation that combines a ViT encoder with a U-Net-style decoder.

## Architecture

UNETR consists of two main components:

1. **Vision Transformer Encoder**: Processes images as sequences of patches through multi-head self-attention layers, extracting hierarchical features at different depths.

2. **U-Net Decoder**: Progressively upsamples features using transposed convolutions while incorporating skip connections from intermediate transformer layers to preserve spatial information.

The architecture enables effective semantic segmentation by leveraging the long-range modeling capabilities of transformers while maintaining fine-grained spatial detail through skip connections.

## Model Components

### Core Modules

- **PatchEmbeddingBlock**: Converts image patches to token embeddings with learnable positional encodings
- **ViTEncoderBlock**: Multi-head self-attention with feedforward network and layer normalization
- **UnetrBasicBlock**: Residual convolution block for initial skip connection processing
- **UnetrPrUpBlock**: Progressive upsampling block with transposed convolutions and residual connections
- **UnetrUpBlock**: Decoder block that combines upsampled features with skip connections via concatenation

### Architecture Details

The model extracts features from transformer layers at depths 3, 6, and 9 (for a 12-layer ViT) to create multi-scale skip connections. These are progressively upsampled and combined through the decoder pathway to produce the final segmentation map.

## Training

- Data loading with the Grain library for efficient batching and multi-worker loading
- Augmentation strategies using Albumentations (rotations, crops, flips, brightness adjustments)
- Loss functions: Combined cross-entropy and Jaccard (IoU) loss for direct optimization of segmentation metrics
- Optimizer setup: Adam with linear learning rate decay from 0.003 to 0
- Evaluation metrics: Per-class IoU, mean IoU, and pixel accuracy using confusion matrices

## References

This implementation is based on:
- "UNETR: Transformers for 3D Medical Image Segmentation" (Hatamizadeh et al., 2022)
- "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Dosovitskiy et al., 2021)

For more details on utilizing JAX libraries and best practices in the JAX ecosystem, refer to the [JAX AI Stack documentation](https://docs.jaxstack.ai/).
