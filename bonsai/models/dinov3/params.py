import gc
import re
from enum import Enum

import jax
import safetensors
from etils import epath
from flax import nnx

from bonsai.models.dinov3.modeling import DINOv3ViTFlaxConfig, Dinov3ViTModel


def _get_key_and_transform_mapping():
    class Transform(Enum):
        BIAS = (None, None, False)
        LINEAR = ((1, 0), None, False)
        CONV2D = ((2, 3, 1, 0), None, False)
        DEFAULT = (None, None, False)

    # Mapping st_keys -> (nnx_keys, (permute_rule, reshape_rule, reshape_first))
    return {
        # Embeddings
        r"embeddings\.cls_token$": ("embeddings.cls_token", Transform.DEFAULT),
        r"embeddings\.mask_token$": ("embeddings.mask_token", Transform.DEFAULT),
        r"embeddings\.register_tokens$": ("embeddings.register_tokens", Transform.DEFAULT),
        r"embeddings\.patch_embeddings\.weight$": ("embeddings.patch_embeddings.kernel", Transform.CONV2D),
        r"embeddings\.patch_embeddings\.bias$": ("embeddings.patch_embeddings.bias", Transform.BIAS),
        # Attention weights and biases
        r"layer\.([0-9]+)\.attention\.q_proj\.weight$": (r"layer.\1.attention.q_proj.kernel", Transform.LINEAR),
        r"layer\.([0-9]+)\.attention\.k_proj\.weight$": (r"layer.\1.attention.k_proj.kernel", Transform.LINEAR),
        r"layer\.([0-9]+)\.attention\.v_proj\.weight$": (r"layer.\1.attention.v_proj.kernel", Transform.LINEAR),
        r"layer\.([0-9]+)\.attention\.o_proj\.weight$": (r"layer.\1.attention.o_proj.kernel", Transform.LINEAR),
        r"layer\.([0-9]+)\.attention\.q_proj\.bias$": (r"layer.\1.attention.q_proj.bias", Transform.BIAS),
        r"layer\.([0-9]+)\.attention\.k_proj\.bias$": (r"layer.\1.attention.k_proj.bias", Transform.BIAS),
        r"layer\.([0-9]+)\.attention\.v_proj\.bias$": (r"layer.\1.attention.v_proj.bias", Transform.BIAS),
        r"layer\.([0-9]+)\.attention\.o_proj\.bias$": (r"layer.\1.attention.o_proj.bias", Transform.BIAS),
        # MLP (gated or not)
        r"layer\.([0-9]+)\.mlp\.gate_proj\.weight$": (r"layer.\1.mlp.gate_proj.kernel", Transform.LINEAR),
        r"layer\.([0-9]+)\.mlp\.up_proj\.weight$": (r"layer.\1.mlp.up_proj.kernel", Transform.LINEAR),
        r"layer\.([0-9]+)\.mlp\.down_proj\.weight$": (r"layer.\1.mlp.down_proj.kernel", Transform.LINEAR),
        r"layer\.([0-9]+)\.mlp\.gate_proj\.bias$": (r"layer.\1.mlp.gate_proj.bias", Transform.BIAS),
        r"layer\.([0-9]+)\.mlp\.up_proj\.bias$": (r"layer.\1.mlp.up_proj.bias", Transform.BIAS),
        r"layer\.([0-9]+)\.mlp\.down_proj\.bias$": (r"layer.\1.mlp.down_proj.bias", Transform.BIAS),
        # layer_scale1 / layer_scale2 keys
        r"layer\.([0-9]+)\.layer_scale1\.lambda1$": (r"layer.\1.layer_scale1.lambda1", Transform.DEFAULT),
        r"layer\.([0-9]+)\.layer_scale2\.lambda1$": (r"layer.\1.layer_scale2.lambda1", Transform.DEFAULT),
        # norm1 / norm2 mapping
        r"layer\.([0-9]+)\.norm1\.weight$": (r"layer.\1.norm1.scale", Transform.DEFAULT),
        r"layer\.([0-9]+)\.norm1\.bias$": (r"layer.\1.norm1.bias", Transform.DEFAULT),
        r"layer\.([0-9]+)\.norm2\.weight$": (r"layer.\1.norm2.scale", Transform.DEFAULT),
        r"layer\.([0-9]+)\.norm2\.bias$": (r"layer.\1.norm2.bias", Transform.DEFAULT),
        # final model norm
        r"norm\.weight": ("norm.scale", Transform.DEFAULT),
        r"norm\.bias": ("norm.bias", Transform.DEFAULT),
    }


def _torch_key_to_jax_key(mapping, source_key):
    subs = [
        (re.sub(pat, repl, source_key), reshape)
        for pat, (repl, reshape) in mapping.items()
        if re.match(pat, source_key)
    ]
    if len(subs) != 1:
        raise ValueError(f"Only one key should be found: {subs[0]}")
    return subs[0]


def _assign_weights(keys, tensor, state_dict, st_key, transform, sharding_dict):
    """Recursively descend into state_dict and assign the (possibly permuted/reshaped) tensor."""
    key, *rest = keys
    if not rest:
        if transform is not None:
            permute, reshape, reshape_first = transform
            if reshape_first and reshape is not None:
                tensor = tensor.reshape(reshape)
            if permute:
                tensor = tensor.transpose(permute)
            if not reshape_first and reshape is not None:
                tensor = tensor.reshape(reshape)
        if tensor.shape != state_dict[key].shape:
            raise ValueError(f"Shape mismatch for {st_key}: {tensor.shape} vs {state_dict[key].shape}")
        # Only apply sharding if sharding_dict is provided
        if sharding_dict is not None:
            state_dict[key] = jax.device_put(tensor, sharding_dict[key])
        else:
            state_dict[key] = jax.device_put(tensor)
    else:
        next_sharding = sharding_dict[key] if sharding_dict is not None else None
        _assign_weights(rest, tensor, state_dict[key], st_key, transform, next_sharding)


def _stoi(s):
    try:
        return int(s)
    except ValueError:
        return s


def create_model_from_safe_tensors(
    file_dir: str,
    cfg: DINOv3ViTFlaxConfig,
    mesh: jax.sharding.Mesh | None = None,
) -> Dinov3ViTModel:
    """Load tensors from the safetensors file and create a Dinov3 model (memory-optimized)."""
    files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors found in {file_dir}")

    dinov3 = nnx.eval_shape(lambda: Dinov3ViTModel(cfg, rngs=nnx.Rngs(0)))
    graph_def, abs_state = nnx.split(dinov3)
    state_dict = nnx.to_pure_dict(abs_state)
    # Only use sharding if mesh is provided
    sharding = nnx.to_pure_dict(nnx.get_named_sharding(abs_state, mesh)) if mesh is not None else None

    key_mapping = _get_key_and_transform_mapping()

    conversion_errors = []
    for f in files:
        with safetensors.safe_open(f, framework="numpy") as sf:
            for torch_key in sf.keys():
                tensor = sf.get_tensor(torch_key)

                jax_key, transform = _torch_key_to_jax_key(key_mapping, torch_key)
                if jax_key is None:
                    continue
                keys = [_stoi(k) for k in jax_key.split(".")]
                try:
                    _assign_weights(keys, tensor, state_dict, torch_key, transform.value, sharding)
                except Exception as e:
                    full_jax_key = ".".join([str(k) for k in keys])
                    conversion_errors.append(
                        f"Failed to assign '{torch_key}' to '{full_jax_key}': {type(e).__name__}: {e}"
                    )

        gc.collect()

    if conversion_errors:
        full_error_log = "\n".join(conversion_errors)
        raise RuntimeError(f"Encountered {len(conversion_errors)} weight conversion errors. Log: \n{full_error_log}")

    m = nnx.merge(graph_def, state_dict)
    m.eval()
    return m
