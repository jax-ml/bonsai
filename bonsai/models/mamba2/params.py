# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parameter utilities for Mamba2 models."""

import re
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from bonsai.models.mamba2 import modeling


def create_random_model(cfg: modeling.ModelConfig, seed: int = 0) -> modeling.Mamba2ForCausalLM:
    """Create a randomly initialized Mamba2ForCausalLM.

    Args:
        cfg: ModelConfig for the model.
        seed: Random seed for initialization.

    Returns:
        Randomly initialized Mamba2ForCausalLM.
    """
    return modeling.Mamba2ForCausalLM(cfg, rngs=nnx.Rngs(seed))


def create_random_forecaster(
    input_dim: int,
    d_model: int = 768,
    n_layers: int = 4,
    output_dim: int = 1,
    forecast_horizon: int = 24,
    seed: int = 0,
    **kwargs,
) -> modeling.Mamba2Forecaster:
    """Create a randomly initialized Mamba2Forecaster.

    Args:
        input_dim: Number of input features per timestep.
        d_model: Hidden dimension of the model.
        n_layers: Number of Mamba2 layers.
        output_dim: Number of output features per timestep.
        forecast_horizon: Number of future timesteps to predict.
        seed: Random seed for initialization.
        **kwargs: Additional arguments passed to Mamba2Forecaster.

    Returns:
        Randomly initialized Mamba2Forecaster.
    """
    return modeling.Mamba2Forecaster(
        input_dim=input_dim,
        d_model=d_model,
        n_layers=n_layers,
        output_dim=output_dim,
        forecast_horizon=forecast_horizon,
        rngs=nnx.Rngs(seed),
        **kwargs,
    )


def count_parameters(model: nnx.Module) -> int:
    """Count the total number of trainable parameters in a model.

    Args:
        model: NNX module to count parameters for.

    Returns:
        Total number of parameters.
    """
    _graphdef, state = nnx.split(model)
    params = state.filter(nnx.Param)
    return sum(p.size for p in jax.tree.leaves(params))


def _get_key_mapping() -> list[tuple[re.Pattern, str, str]]:
    """Get mapping from PyTorch state-spaces/mamba2 keys to JAX parameter paths.

    Based on the official state-spaces/mamba repository checkpoint format.
    Keys follow the pattern: backbone.layers.{idx}.mixer.{param}

    Returns list of (pattern, replacement, transform_type) tuples.
    Transform types: LINEAR, CONV1D, EMBED, SCALE, NONE
    """
    return [
        # Embedding - state-spaces uses "embedding" (singular)
        (re.compile(r"^backbone\.embedding\.weight$"), "backbone.embedder.embedding", "EMBED"),
        # Layer pre-norm (RMSNorm before mixer)
        (re.compile(r"^backbone\.layers\.(\d+)\.norm\.weight$"), r"backbone.layers.\1.norm.weight", "SCALE"),
        # Mixer input projection
        (
            re.compile(r"^backbone\.layers\.(\d+)\.mixer\.in_proj\.weight$"),
            r"backbone.layers.\1.mixer.in_proj.kernel",
            "LINEAR",
        ),
        # Mixer conv1d (DepthwiseConv1d wraps nnx.Conv as self.conv)
        (
            re.compile(r"^backbone\.layers\.(\d+)\.mixer\.conv1d\.weight$"),
            r"backbone.layers.\1.mixer.conv1d.conv.kernel",
            "CONV1D",
        ),
        (
            re.compile(r"^backbone\.layers\.(\d+)\.mixer\.conv1d\.bias$"),
            r"backbone.layers.\1.mixer.conv1d.conv.bias",
            "NONE",
        ),
        # SSM parameters (A_log, D, dt_bias)
        (re.compile(r"^backbone\.layers\.(\d+)\.mixer\.A_log$"), r"backbone.layers.\1.mixer.A_log", "NONE"),
        (re.compile(r"^backbone\.layers\.(\d+)\.mixer\.D$"), r"backbone.layers.\1.mixer.D", "NONE"),
        (re.compile(r"^backbone\.layers\.(\d+)\.mixer\.dt_bias$"), r"backbone.layers.\1.mixer.dt_bias", "NONE"),
        # Mixer internal norm (RMSNorm with residual gate)
        (
            re.compile(r"^backbone\.layers\.(\d+)\.mixer\.norm\.weight$"),
            r"backbone.layers.\1.mixer.norm.weight",
            "SCALE",
        ),
        # Mixer output projection
        (
            re.compile(r"^backbone\.layers\.(\d+)\.mixer\.out_proj\.weight$"),
            r"backbone.layers.\1.mixer.out_proj.kernel",
            "LINEAR",
        ),
        (
            re.compile(r"^backbone\.layers\.(\d+)\.mixer\.out_proj\.bias$"),
            r"backbone.layers.\1.mixer.out_proj.bias",
            "NONE",
        ),
        # Final norm
        (re.compile(r"^backbone\.norm_f\.weight$"), "backbone.final_norm.weight", "SCALE"),
        # LM head (may be tied to embeddings)
        (re.compile(r"^lm_head\.weight$"), "lm_head.kernel", "LINEAR"),
    ]


def _transform_tensor(tensor: jnp.ndarray, transform_type: str) -> jnp.ndarray:
    """Apply transformation to convert PyTorch tensor to JAX format."""
    if transform_type == "LINEAR":
        return tensor.T
    elif transform_type == "CONV1D":
        # PyTorch conv1d: (out_channels, in_channels/groups, kernel_size)
        # JAX conv: (kernel_size, in_channels/groups, out_channels)
        return jnp.transpose(tensor, (2, 1, 0))
    elif transform_type == "EMBED":
        return tensor
    elif transform_type == "SCALE":
        return tensor
    elif transform_type == "NONE":
        return tensor
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")


def _set_nested_attr(obj: Any, path: str, value: Any) -> None:
    """Set a nested attribute on an object using dot-separated path."""
    parts = path.split(".")
    for part in parts[:-1]:
        if obj is None:
            raise AttributeError(f"Encountered None while traversing path '{path}' at '{part}'")
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)

    final_part = parts[-1]
    if obj is None:
        raise AttributeError(f"Encountered None while setting '{path}'")

    if final_part.isdigit():
        obj[int(final_part)] = value
        return

    if not hasattr(obj, final_part):
        raise AttributeError(f"Object of type {type(obj).__name__} has no attribute '{final_part}' (path='{path}')")

    attr = getattr(obj, final_part)
    if isinstance(attr, nnx.Param):
        attr[...] = value
    else:
        setattr(obj, final_part, value)


def load_pytorch_weights(
    model: modeling.Mamba2ForCausalLM,
    state_dict: Mapping[str, Any],
    dtype: jnp.dtype = jnp.float32,
    strict: bool = False,
) -> tuple[modeling.Mamba2ForCausalLM, list[str], list[str]]:
    key_mapping = _get_key_mapping()
    loaded_keys: list[str] = []
    skipped_keys: list[str] = []

    tie = getattr(model.cfg, "tie_word_embeddings", False)
    embedding_loaded = False

    for pt_key, pt_tensor in state_dict.items():
        matched_rule = False

        for pattern, replacement, transform_type in key_mapping:
            if not pattern.match(pt_key):
                continue

            matched_rule = True
            jax_path = pattern.sub(replacement, pt_key)

            # Track embedding load
            if pt_key == "backbone.embedding.weight":
                embedding_loaded = True

            # Special-case tied head:
            # - never overwrite embedding with transposed lm_head
            if pt_key == "lm_head.weight" and tie and getattr(model, "lm_head", None) is None:
                if embedding_loaded:
                    # Redundant in tied models (and HF mamba2 checkpoints contain both).
                    loaded_keys.append(f"{pt_key} (skipped: tied embeddings)")
                else:
                    # Fallback: if embedding.weight is absent, use lm_head.weight *without transpose*
                    tensor = jnp.array(pt_tensor, dtype=dtype)  # NO _transform_tensor here
                    try:
                        _set_nested_attr(model, "backbone.embedder.embedding", tensor)
                        loaded_keys.append(f"{pt_key} (used as embedding; no transpose)")
                        embedding_loaded = True
                    except (AttributeError, IndexError, KeyError, TypeError) as e:
                        if strict:
                            raise ValueError(f"Failed to set tied embedding from {pt_key}: {e}") from e
                        skipped_keys.append(f"{pt_key} (tied embedding set failed: {e})")
                break

            # Normal path
            tensor = jnp.array(pt_tensor, dtype=dtype)
            tensor = _transform_tensor(tensor, transform_type)

            try:
                _set_nested_attr(model, jax_path, tensor)
                loaded_keys.append(pt_key)
            except (AttributeError, IndexError, KeyError, TypeError) as e:
                if strict:
                    raise ValueError(f"Failed to set {jax_path} from {pt_key}: {e}") from e
                skipped_keys.append(f"{pt_key} (error: {e})")

            break  # only first matching rule

        if not matched_rule:
            skipped_keys.append(pt_key)

    if strict and skipped_keys:
        raise ValueError(f"Unexpected/unloaded keys in state_dict (first 20): {skipped_keys[:20]}")

    return model, loaded_keys, skipped_keys


def create_model_from_torch_checkpoint(
    checkpoint_path: str,
    cfg: modeling.ModelConfig | None = None,
    dtype: jnp.dtype = jnp.float32,
    seed: int = 0,
) -> modeling.Mamba2ForCausalLM:
    """Create model from PyTorch checkpoint file.

    Args:
        checkpoint_path: Path to .pt/.bin checkpoint or directory with model files.
        cfg: Model config. Required for now.
        dtype: Target dtype.
        seed: Random seed for initialization (weights will be overwritten).

    Returns:
        Mamba2ForCausalLM with loaded weights.
    """
    import os

    if os.path.isdir(checkpoint_path):
        safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
        pytorch_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if os.path.exists(safetensors_path):
            checkpoint_path = safetensors_path
        elif os.path.exists(pytorch_path):
            checkpoint_path = pytorch_path
        else:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}")

    if checkpoint_path.endswith(".safetensors"):
        try:
            from safetensors import safe_open
        except ImportError as e:
            raise ImportError("safetensors required: pip install safetensors") from e

        state_dict = {}
        with safe_open(checkpoint_path, framework="numpy") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    else:
        try:
            import torch
        except ImportError as e:
            raise ImportError("torch required: pip install torch") from e

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = {k: v.numpy() for k, v in checkpoint.items()}

    if cfg is None:
        raise ValueError("Config inference not yet implemented. Please provide cfg.")

    model = create_random_model(cfg, seed=seed)
    model, loaded, skipped = load_pytorch_weights(model, state_dict, dtype=dtype)

    print(f"Loaded {len(loaded)} parameters")
    if skipped:
        print(f"Skipped {len(skipped)} keys: {skipped[:5]}...")

    return model


def create_model_from_huggingface(
    model_id: str,
    cfg: modeling.ModelConfig | None = None,
    dtype: jnp.dtype = jnp.float32,
    seed: int = 0,
    revision: str = "main",
) -> modeling.Mamba2ForCausalLM:
    """Create model from HuggingFace Hub.

    Args:
        model_id: HuggingFace model ID (e.g., "state-spaces/mamba2-130m").
        cfg: Model config. If None, will try to infer from config.json.
        dtype: Target dtype.
        seed: Random seed for initialization.
        revision: Git revision to use.

    Returns:
        Mamba2ForCausalLM with loaded weights.

    Example:
        >>> cfg = modeling.ModelConfig(
        ...     vocab_size=50280, hidden_size=768,
        ...     state_size=128, num_hidden_layers=24, head_dim=64
        ... )
        >>> model = create_model_from_huggingface("state-spaces/mamba2-130m", cfg=cfg)
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError("huggingface_hub required: pip install huggingface_hub") from e

    # Try safetensors first, fall back to pytorch_model.bin
    try:
        checkpoint_path = hf_hub_download(model_id, "model.safetensors", revision=revision)
    except Exception:
        try:
            checkpoint_path = hf_hub_download(model_id, "pytorch_model.bin", revision=revision)
        except Exception as e:
            raise FileNotFoundError(f"Could not find model.safetensors or pytorch_model.bin in {model_id}") from e

    if cfg is None:
        import json

        config_path = hf_hub_download(model_id, "config.json", revision=revision)
        with open(config_path) as f:
            hf_config = json.load(f)

        cfg = modeling.ModelConfig(
            vocab_size=hf_config.get("vocab_size", 50280),
            hidden_size=hf_config.get("d_model", hf_config.get("hidden_size", 768)),
            state_size=hf_config.get("d_state", hf_config.get("state_size", 128)),
            num_hidden_layers=hf_config.get("n_layer", hf_config.get("num_hidden_layers", 24)),
            expand=hf_config.get("expand", 2),
            conv_kernel=hf_config.get("d_conv", hf_config.get("conv_kernel", 4)),
            head_dim=hf_config.get("headdim", hf_config.get("head_dim", 64)),
        )

    return create_model_from_torch_checkpoint(checkpoint_path, cfg=cfg, dtype=dtype, seed=seed)


def print_checkpoint_keys(checkpoint_path: str) -> None:
    """Print keys from a checkpoint file for debugging.

    Args:
        checkpoint_path: Path to checkpoint file.
    """
    if checkpoint_path.endswith(".safetensors"):
        from safetensors import safe_open

        with safe_open(checkpoint_path, framework="numpy") as f:
            print(f"Keys in {checkpoint_path}:")
            for key in sorted(f.keys()):
                shape = f.get_tensor(key).shape
                print(f"  {key}: {shape}")
    else:
        import torch

        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        print(f"Keys in {checkpoint_path}:")
        for key in sorted(state_dict.keys()):
            shape = tuple(state_dict[key].shape)
            print(f"  {key}: {shape}")
