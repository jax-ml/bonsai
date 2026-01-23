import gc
import re
from enum import Enum
from pathlib import Path

import jax
import safetensors
from flax import nnx

from bonsai.models.qwen3_vl import modeling as model_lib


class Transform(Enum):
    # (Permute_axes, reshape, reshape_first)
    DEFAULT = (None, None, False)
    BIAS = (None, None, False)
    LINEAR = ((1, 0), None, False)  # PyTorch (out, in) -> JAX (in, out)
    CONV3D = ((2, 3, 4, 1, 0), None, False)  # (out, in, T, H, W) -> (T, H, W, in, out)
    EMBED = (None, None, False)
    ATTN_Q = None  # Special handling needed
    ATTN_KV = None  # Special handling needed


def _get_vision_key_mapping():
    """Mapping for vision encoder weights."""
    return {
        # Patch embedding - nnx.Conv uses (D, H, W, in_channels, out_channels) kernel
        # PyTorch Conv3d: (out_channels, in_channels, D, H, W)
        r"^model\.visual\.patch_embed\.proj\.weight$": (
            "model.visual.patch_embed.proj.kernel",
            Transform.CONV3D,
        ),
        r"^model\.visual\.patch_embed\.proj\.bias$": (
            "model.visual.patch_embed.proj.bias",
            Transform.BIAS,
        ),
        # Position embedding
        r"^model\.visual\.pos_embed\.weight$": (
            "model.visual.pos_embed.embedding",
            Transform.EMBED,
        ),
        # Vision blocks
        r"^model\.visual\.blocks\.(\d+)\.norm1\.weight$": (
            r"model.visual.blocks.\1.norm1.scale",
            Transform.DEFAULT,
        ),
        r"^model\.visual\.blocks\.(\d+)\.norm1\.bias$": (
            r"model.visual.blocks.\1.norm1.bias",
            Transform.BIAS,
        ),
        r"^model\.visual\.blocks\.(\d+)\.norm2\.weight$": (
            r"model.visual.blocks.\1.norm2.scale",
            Transform.DEFAULT,
        ),
        r"^model\.visual\.blocks\.(\d+)\.norm2\.bias$": (
            r"model.visual.blocks.\1.norm2.bias",
            Transform.BIAS,
        ),
        # Vision attention - fused QKV
        r"^model\.visual\.blocks\.(\d+)\.attn\.qkv\.weight$": (
            r"model.visual.blocks.\1.attn.qkv.kernel",
            Transform.LINEAR,
        ),
        r"^model\.visual\.blocks\.(\d+)\.attn\.qkv\.bias$": (
            r"model.visual.blocks.\1.attn.qkv.bias",
            Transform.BIAS,
        ),
        r"^model\.visual\.blocks\.(\d+)\.attn\.proj\.weight$": (
            r"model.visual.blocks.\1.attn.proj.kernel",
            Transform.LINEAR,
        ),
        r"^model\.visual\.blocks\.(\d+)\.attn\.proj\.bias$": (
            r"model.visual.blocks.\1.attn.proj.bias",
            Transform.BIAS,
        ),
        # Vision MLP
        r"^model\.visual\.blocks\.(\d+)\.mlp\.linear_fc1\.weight$": (
            r"model.visual.blocks.\1.mlp.linear_fc1.kernel",
            Transform.LINEAR,
        ),
        r"^model\.visual\.blocks\.(\d+)\.mlp\.linear_fc1\.bias$": (
            r"model.visual.blocks.\1.mlp.linear_fc1.bias",
            Transform.BIAS,
        ),
        r"^model\.visual\.blocks\.(\d+)\.mlp\.linear_fc2\.weight$": (
            r"model.visual.blocks.\1.mlp.linear_fc2.kernel",
            Transform.LINEAR,
        ),
        r"^model\.visual\.blocks\.(\d+)\.mlp\.linear_fc2\.bias$": (
            r"model.visual.blocks.\1.mlp.linear_fc2.bias",
            Transform.BIAS,
        ),
        # Merger
        r"^model\.visual\.merger\.norm\.weight$": (
            "model.visual.merger.norm.scale",
            Transform.DEFAULT,
        ),
        r"^model\.visual\.merger\.norm\.bias$": (
            "model.visual.merger.norm.bias",
            Transform.BIAS,
        ),
        r"^model\.visual\.merger\.linear_fc1\.weight$": (
            "model.visual.merger.linear_fc1.kernel",
            Transform.LINEAR,
        ),
        r"^model\.visual\.merger\.linear_fc1\.bias$": (
            "model.visual.merger.linear_fc1.bias",
            Transform.BIAS,
        ),
        r"^model\.visual\.merger\.linear_fc2\.weight$": (
            "model.visual.merger.linear_fc2.kernel",
            Transform.LINEAR,
        ),
        r"^model\.visual\.merger\.linear_fc2\.bias$": (
            "model.visual.merger.linear_fc2.bias",
            Transform.BIAS,
        ),
        # Deepstack mergers
        r"^model\.visual\.deepstack_merger_list\.(\d+)\.norm\.weight$": (
            r"model.visual.deepstack_merger_list.\1.norm.scale",
            Transform.DEFAULT,
        ),
        r"^model\.visual\.deepstack_merger_list\.(\d+)\.norm\.bias$": (
            r"model.visual.deepstack_merger_list.\1.norm.bias",
            Transform.BIAS,
        ),
        r"^model\.visual\.deepstack_merger_list\.(\d+)\.linear_fc1\.weight$": (
            r"model.visual.deepstack_merger_list.\1.linear_fc1.kernel",
            Transform.LINEAR,
        ),
        r"^model\.visual\.deepstack_merger_list\.(\d+)\.linear_fc1\.bias$": (
            r"model.visual.deepstack_merger_list.\1.linear_fc1.bias",
            Transform.BIAS,
        ),
        r"^model\.visual\.deepstack_merger_list\.(\d+)\.linear_fc2\.weight$": (
            r"model.visual.deepstack_merger_list.\1.linear_fc2.kernel",
            Transform.LINEAR,
        ),
        r"^model\.visual\.deepstack_merger_list\.(\d+)\.linear_fc2\.bias$": (
            r"model.visual.deepstack_merger_list.\1.linear_fc2.bias",
            Transform.BIAS,
        ),
    }


def _get_text_key_mapping():
    """Mapping for text decoder weights."""
    return {
        # Token embedding (note: PyTorch key is model.language_model.embed_tokens)
        r"^model\.language_model\.embed_tokens\.weight$": (
            "model.language_model.embed_tokens.embedding",
            Transform.EMBED,
        ),
        # Decoder layers - attention
        r"^model\.language_model\.layers\.(\d+)\.self_attn\.q_proj\.weight$": (
            r"model.language_model.layers.\1.self_attn.q_proj.kernel",
            Transform.LINEAR,
        ),
        r"^model\.language_model\.layers\.(\d+)\.self_attn\.k_proj\.weight$": (
            r"model.language_model.layers.\1.self_attn.k_proj.kernel",
            Transform.LINEAR,
        ),
        r"^model\.language_model\.layers\.(\d+)\.self_attn\.v_proj\.weight$": (
            r"model.language_model.layers.\1.self_attn.v_proj.kernel",
            Transform.LINEAR,
        ),
        r"^model\.language_model\.layers\.(\d+)\.self_attn\.o_proj\.weight$": (
            r"model.language_model.layers.\1.self_attn.o_proj.kernel",
            Transform.LINEAR,
        ),
        # Q/K norms
        r"^model\.language_model\.layers\.(\d+)\.self_attn\.q_norm\.weight$": (
            r"model.language_model.layers.\1.self_attn.q_norm.weight",
            Transform.DEFAULT,
        ),
        r"^model\.language_model\.layers\.(\d+)\.self_attn\.k_norm\.weight$": (
            r"model.language_model.layers.\1.self_attn.k_norm.weight",
            Transform.DEFAULT,
        ),
        # Decoder layers - MLP
        r"^model\.language_model\.layers\.(\d+)\.mlp\.gate_proj\.weight$": (
            r"model.language_model.layers.\1.mlp.gate_proj.kernel",
            Transform.LINEAR,
        ),
        r"^model\.language_model\.layers\.(\d+)\.mlp\.up_proj\.weight$": (
            r"model.language_model.layers.\1.mlp.up_proj.kernel",
            Transform.LINEAR,
        ),
        r"^model\.language_model\.layers\.(\d+)\.mlp\.down_proj\.weight$": (
            r"model.language_model.layers.\1.mlp.down_proj.kernel",
            Transform.LINEAR,
        ),
        # Decoder layers - norms
        r"^model\.language_model\.layers\.(\d+)\.input_layernorm\.weight$": (
            r"model.language_model.layers.\1.input_layernorm.weight",
            Transform.DEFAULT,
        ),
        r"^model\.language_model\.layers\.(\d+)\.post_attention_layernorm\.weight$": (
            r"model.language_model.layers.\1.post_attention_layernorm.weight",
            Transform.DEFAULT,
        ),
        # Final norm
        r"^model\.language_model\.norm\.weight$": (
            "model.language_model.norm.weight",
            Transform.DEFAULT,
        ),
        # For tied embeddings models, lm_head.weight is saved but embed_tokens is not
        # Map lm_head.weight to embed_tokens.embedding (they share the same weight)
        r"^lm_head\.weight$": (
            "model.language_model.embed_tokens.embedding",
            Transform.EMBED,
        ),
    }


def _get_key_and_transform_mapping():
    """Combined key mapping for all model components."""
    mapping = {}
    mapping.update(_get_vision_key_mapping())
    mapping.update(_get_text_key_mapping())
    return mapping


def _torch_key_to_jax_key(mapping: dict, source_key: str) -> tuple[str | None, Transform | None]:
    """Map a PyTorch key to JAX key with transform specification."""
    matches = [
        (re.sub(pat, repl, source_key), transform)
        for pat, (repl, transform) in mapping.items()
        if re.match(pat, source_key)
    ]
    if not matches:
        return None, None
    if len(matches) > 1:
        raise ValueError(f"Multiple mappings found for {source_key}: {[m[0] for m in matches]}")
    return matches[0]


def _stoi(s: str) -> int | str:
    """Convert string to int if possible."""
    try:
        return int(s)
    except ValueError:
        return s


def _apply_transform(tensor, transform: Transform | None):
    """Apply weight transformation to tensor."""
    if transform is None or transform.value is None:
        return tensor

    permute, reshape, reshape_first = transform.value

    if reshape_first and reshape is not None:
        tensor = tensor.reshape(reshape)
    if permute is not None:
        tensor = tensor.transpose(permute)
    if not reshape_first and reshape is not None:
        tensor = tensor.reshape(reshape)

    return tensor


def _assign_weights(
    keys: list,
    tensor,
    state_dict: dict,
    torch_key: str,
    transform: Transform | None,
    sharding_dict: dict | None,
):
    """Recursively assign transformed weight to state dict."""
    key, *rest = keys

    if not rest:
        # Apply transformation
        tensor = _apply_transform(tensor, transform)

        # Validate shape
        if tensor.shape != state_dict[key].shape:
            raise ValueError(f"Shape mismatch for {torch_key}: got {tensor.shape}, expected {state_dict[key].shape}")

        # Place on device with optional sharding
        if sharding_dict is not None:
            state_dict[key] = jax.device_put(tensor, sharding_dict[key])
        else:
            state_dict[key] = jax.device_put(tensor)
    else:
        next_sharding = sharding_dict[key] if sharding_dict is not None else None
        _assign_weights(rest, tensor, state_dict[key], torch_key, transform, next_sharding)


def create_model_from_safe_tensors(
    file_dir: str,
    config: model_lib.Qwen3VLConfig,
    mesh: jax.sharding.Mesh | None = None,
    model_filename: str | None = None,
) -> model_lib.Qwen3VLForConditionalGeneration:
    """Load pretrained weights from safetensors and create model.

    Args:
        file_dir: Path to directory containing .safetensors files.
        config: Model configuration.
        mesh: Optional JAX mesh for sharding.
        model_filename: Optional specific filename to load (e.g., "model.safetensors").

    Returns:
        Qwen3VLForConditionalGeneration model with loaded weights.

    Raises:
        ValueError: If no safetensors files found or weight conversion fails.
    """
    path = Path(file_dir).expanduser()
    if model_filename:
        files = [path / model_filename]
        if not files[0].exists():
            raise ValueError(f"Specified file not found: {files[0]}")
    else:
        files = list(path.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors files found in {file_dir}")

    # Create model with abstract state (no actual arrays)
    model = nnx.eval_shape(lambda: model_lib.Qwen3VLForConditionalGeneration(config, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(model)
    state_dict = abs_state.to_pure_dict()

    # Get sharding if mesh provided
    sharding = nnx.get_named_sharding(abs_state, mesh).to_pure_dict() if mesh is not None else None

    # Key mapping
    key_mapping = _get_key_and_transform_mapping()

    conversion_errors = []
    loaded_keys = set()

    for f in files:
        with safetensors.safe_open(f, framework="numpy") as sf:
            for torch_key in sf.keys():
                tensor = sf.get_tensor(torch_key)

                jax_key, transform = _torch_key_to_jax_key(key_mapping, torch_key)
                if jax_key is None:
                    # Skip unmapped keys (e.g., rotary buffers)
                    continue

                keys = [_stoi(k) for k in jax_key.split(".")]
                try:
                    _assign_weights(keys, tensor, state_dict, torch_key, transform, sharding)
                    loaded_keys.add(torch_key)
                except Exception as e:
                    full_jax_key = ".".join(str(k) for k in keys)
                    conversion_errors.append(
                        f"Failed to assign '{torch_key}' to '{full_jax_key}': {type(e).__name__}: {e}"
                    )

        gc.collect()

    if conversion_errors:
        error_log = "\n".join(conversion_errors)
        raise RuntimeError(f"Encountered {len(conversion_errors)} weight conversion errors:\n{error_log}")

    # Handle tied embeddings
    if config.text_config.tie_word_embeddings:
        # lm_head shares weights with embed_tokens
        # The model implementation handles this at inference time
        pass

    gc.collect()
    return nnx.merge(graph_def, state_dict)
