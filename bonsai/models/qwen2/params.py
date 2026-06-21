import gc
import re
from dataclasses import dataclass
from typing import Any

import jax
import safetensors
from etils import epath
from flax import nnx

from bonsai.models.qwen2 import modeling as model_lib


@dataclass(frozen=True)
class Transform:
    permute: tuple[int, ...] | None = None
    reshape: tuple[int, ...] | None = None
    reshape_first: bool = False


TRANSFORM_LINEAR = Transform(permute=(1, 0))
TRANSFORM_NONE = Transform()


def _get_key_and_transform_mapping(cfg: model_lib.ModelConfig) -> dict[str, tuple[str | None, Transform | None]]:
    return {
        r"model\.embed_tokens\.weight": ("embedder.embedding", TRANSFORM_NONE),
        r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (r"layers.\1.attn.q_proj.kernel", TRANSFORM_LINEAR),
        r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (r"layers.\1.attn.k_proj.kernel", TRANSFORM_LINEAR),
        r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (r"layers.\1.attn.v_proj.kernel", TRANSFORM_LINEAR),
        r"model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": (r"layers.\1.attn.o_proj.kernel", TRANSFORM_LINEAR),
        r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.bias": (r"layers.\1.attn.q_proj.bias", TRANSFORM_NONE),
        r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.bias": (r"layers.\1.attn.k_proj.bias", TRANSFORM_NONE),
        r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.bias": (r"layers.\1.attn.v_proj.bias", TRANSFORM_NONE),
        r"model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": (r"layers.\1.mlp.gate_proj.kernel", TRANSFORM_LINEAR),
        r"model\.layers\.([0-9]+)\.mlp\.up_proj\.weight": (r"layers.\1.mlp.up_proj.kernel", TRANSFORM_LINEAR),
        r"model\.layers\.([0-9]+)\.mlp\.down_proj\.weight": (r"layers.\1.mlp.down_proj.kernel", TRANSFORM_LINEAR),
        r"model\.norm\.weight": ("final_norm.scale", TRANSFORM_NONE),
        r"model\.layers\.([0-9]+)\.input_layernorm\.weight": (r"layers.\1.input_layernorm.scale", TRANSFORM_NONE),
        r"model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": (
            r"layers.\1.post_attention_layernorm.scale",
            TRANSFORM_NONE,
        ),
        r"lm_head\.weight": ("lm_head.kernel", TRANSFORM_LINEAR),
    }


def _get_jax_key(
    mapping: dict[str, tuple[str | None, Transform | None]], source_key: str
) -> tuple[str | None, Transform | None]:
    for pat, (repl, transform) in mapping.items():
        if re.match(pat, source_key):
            if repl is None:
                return None, None
            return re.sub(pat, repl, source_key), transform

    print(f"Warning: No mapping found for key '{source_key}', skipping...")
    return None, None


def _assign_weights(
    keys: list[str | int],
    tensor: Any,
    state_dict: dict,
    st_key: str,
    transform: Transform | None,
    sharding_dict: dict | None,
) -> None:
    key, *rest = keys
    if not rest:
        if transform is not None:
            if transform.reshape_first and transform.reshape is not None:
                tensor = tensor.reshape(transform.reshape)
            if transform.permute is not None:
                tensor = tensor.transpose(transform.permute)
            if not transform.reshape_first and transform.reshape is not None:
                tensor = tensor.reshape(transform.reshape)

        if tensor.shape != state_dict[key].shape:
            raise ValueError(f"Shape mismatch for {st_key}: {tensor.shape} vs {state_dict[key].shape}")

        state_dict[key] = jax.device_put(tensor, sharding_dict[key] if sharding_dict else None)
    else:
        next_sharding = sharding_dict[key] if sharding_dict is not None else None
        _assign_weights(rest, tensor, state_dict[key], st_key, transform, next_sharding)


def _stoi(s: str) -> str | int:
    try:
        return int(s)
    except ValueError:
        return s


def create_model_from_safe_tensors(
    file_dir: str, cfg: model_lib.ModelConfig, mesh: jax.sharding.Mesh | None = None
) -> model_lib.Qwen2:
    files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors found in {file_dir}")

    qwen2 = nnx.eval_shape(lambda: model_lib.Qwen2(cfg, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(qwen2)
    state_dict = abs_state.to_pure_dict()
    sharding = nnx.get_named_sharding(abs_state, mesh).to_pure_dict() if mesh is not None else None

    key_mapping = _get_key_and_transform_mapping(cfg)
    conversion_errors = []

    for f in files:
        with safetensors.safe_open(f, framework="numpy") as sf:
            for torch_key in sf.keys():
                jax_key, transform = _get_jax_key(key_mapping, torch_key)
                if jax_key is None:
                    continue

                keys = [_stoi(k) for k in jax_key.split(".")]
                try:
                    tensor = sf.get_tensor(torch_key)
                    _assign_weights(keys, tensor, state_dict, torch_key, transform, sharding)
                except Exception as e:
                    full_jax_key = ".".join([str(k) for k in keys])
                    conversion_errors.append(
                        f"Failed to assign '{torch_key}' to '{full_jax_key}': {type(e).__name__}: {e}"
                    )
        gc.collect()

    if conversion_errors:
        raise RuntimeError(
            f"Encountered {len(conversion_errors)} weight conversion errors:\n" + "\n".join(conversion_errors)
        )

    if cfg.tie_word_embeddings:
        state_dict["lm_head"]["kernel"] = state_dict["embedder"]["embedding"].T

    model = nnx.merge(graph_def, state_dict)
    gc.collect()

    return model
