"""Debug vision encoder - compare layer-by-layer forward outputs."""

import jax.numpy as jnp
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image

from bonsai.models.qwen3_vl import params


MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"


def cos_sim(a, b):
    a_flat, b_flat = a.flatten(), b.flatten()
    return np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-8)


def compare(name, pt_arr, flax_arr):
    if pt_arr.shape != flax_arr.shape:
        print(f"{name}: SHAPE MISMATCH! PT {pt_arr.shape} vs Flax {flax_arr.shape}")
        return False
    diff = np.abs(pt_arr - flax_arr)
    cs = cos_sim(pt_arr, flax_arr)
    status = "✓" if cs > 0.99 else ("~" if cs > 0.9 else "✗")
    print(f"{name}: diff max={diff.max():.4f}, mean={diff.mean():.4f}, cos_sim={cs:.4f} {status}")
    return cs > 0.9


def main():
    print("Loading models...")
    model_path = snapshot_download(MODEL_ID)
    pt_model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float32).eval()
    flax_config = params.get_pretrained_config("2b")
    flax_model = params.create_model_from_safe_tensors(model_path, flax_config)
    processor = AutoProcessor.from_pretrained(model_path)

    # Load test image
    image = Image.open("/home/LinuxGod/opensource/bonsai/bonsai/models/qwen3_vl/image.jpg").resize((256, 256))
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "What?"}]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )

    pixel_values_pt = inputs["pixel_values"]
    grid_thw_pt = inputs["image_grid_thw"]
    pixel_values_jax = jnp.array(pixel_values_pt.numpy())
    grid_thw_jax = jnp.array(grid_thw_pt.numpy())

    print(f"pixel_values: {pixel_values_pt.shape}, grid_thw: {grid_thw_pt}")

    # =========================================================================
    # Step 1: Patch Embedding
    # =========================================================================
    print("\n=== Patch Embedding ===")
    with torch.no_grad():
        pt_patch = pt_model.model.visual.patch_embed(pixel_values_pt).numpy()
    flax_patch = np.array(flax_model.model.visual.patch_embed(pixel_values_jax))
    compare("Patch embed", pt_patch, flax_patch)

    # =========================================================================
    # Step 2: Position Embedding (using model methods)
    # =========================================================================
    print("\n=== Position Embedding (using model methods) ===")
    with torch.no_grad():
        pt_pos_embeds = pt_model.model.visual.fast_pos_embed_interpolate(grid_thw_pt).numpy()

    flax_pos_embeds = np.array(flax_model.model.visual._fast_pos_embed_interpolate(grid_thw_jax))

    print(f"PyTorch pos_embeds shape: {pt_pos_embeds.shape}")
    print(f"Flax pos_embeds shape: {flax_pos_embeds.shape}")
    compare("Position embeds", pt_pos_embeds, flax_pos_embeds)

    # =========================================================================
    # Step 3: RoPE (using model methods)
    # =========================================================================
    print("\n=== RoPE Embeddings (using model methods) ===")
    with torch.no_grad():
        pt_rope = pt_model.model.visual.rot_pos_emb(grid_thw_pt)
        # PyTorch returns raw frequencies, then applies cos/sin
        pt_emb = torch.cat([pt_rope, pt_rope], dim=-1)  # double for complex rotation
        pt_cos = pt_emb.cos().numpy()
        pt_sin = pt_emb.sin().numpy()

    flax_cos, flax_sin = flax_model.model.visual._rot_pos_emb(grid_thw_jax)
    flax_cos = np.array(flax_cos)
    flax_sin = np.array(flax_sin)

    print(f"PyTorch RoPE cos shape: {pt_cos.shape}")
    print(f"Flax RoPE cos shape: {flax_cos.shape}")
    if pt_cos.shape == flax_cos.shape:
        compare("RoPE cos", pt_cos, flax_cos)
        compare("RoPE sin", pt_sin, flax_sin)

    # =========================================================================
    # Step 4: Full Vision Output
    # =========================================================================
    print("\n=== Final Vision Output ===")
    with torch.no_grad():
        pt_vision = pt_model.model.visual(pixel_values_pt, grid_thw_pt)[0].numpy()
    flax_vision, _ = flax_model.model.visual(pixel_values_jax, grid_thw_jax)
    compare("Final vision output", pt_vision, np.array(flax_vision))


if __name__ == "__main__":
    main()
