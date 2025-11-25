# ITA-CLIP — CLIP-style model (JAX / Flax)

This directory contains a compact CLIP-like implementation (ITA-CLIP) in JAX/Flax,
intended for zero-shot image classification, prompt-guided heatmaps, and image-text embedding experiments.

## Paper (reference)

- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (OpenAI CLIP)  
  Local copy used during development: `/mnt/data/2103.00020v1.pdf`

---

## Tested on

| Model Name | Config | CPU | GPU (single) | GPU (multi) | TPU |
| :--- | :---: | :---: | :---: | :---: | :---: |
| ITA-CLIP (TinyViT + TinyText) | ✅ Compact research config | ✅ Runs (CPU) | ❔ Needs check (CUDA JAX) | ❔ Needs check | ❔ Needs check |

> Notes: This implementation uses a compact TinyViT and small text-transformer to make local testing and CI-friendly smoke tests possible. For large-scale ViT-B/32 or ViT-L/14 variants, add config presets and provide pretrained weights.

---

### Running this model (quick smoke test)

Run a forward pass / smoke test:

```bash
python3 -m bonsai.models.clip.tests.run_model
