import re
from enum import Enum
import jax.numpy as jnp
import safetensors.flax as safetensors
from etils import epath
from flax import nnx

from bonsai.models.gpt_oss import modeling as model_lib


def _get_key_and_transform_mapping(cfg: model_lib.GptOssConfig):
    class Transform(Enum):
        DEFAULT = None
        BIAS = None
        LINEAR = ((1, 0), None)
        SCALE = None

    return {
        r"^model.embed_tokens.weight$": (r"model.embed_tokens.embedding", Transform.DEFAULT),
        r"^model.norm.weight$": (r"model.norm.weight", Transform.DEFAULT),
        r"^lm_head.weight$": (r"lm_head.kernel", Transform.LINEAR),
        # Layers
        r"^model.layers.([0-9]+).input_layernorm.weight$": (
            r"model.layers.\1.input_layernorm.weight",
            Transform.DEFAULT,
        ),
        r"^model.layers.([0-9]+).post_attention_layernorm.weight$": (
            r"model.layers.\1.post_attention_layernorm.weight",
            Transform.DEFAULT,
        ),
        # Attention projections
        r"^model.layers.([0-9]+).self_attn.q_proj.weight$": (
            r"model.layers.\1.self_attn.q_proj.kernel",
            Transform.LINEAR,
        ),
        r"^model.layers.([0-9]+).self_attn.k_proj.weight$": (
            r"model.layers.\1.self_attn.k_proj.kernel",
            Transform.LINEAR,
        ),
        r"^model.layers.([0-9]+).self_attn.v_proj.weight$": (
            r"model.layers.\1.self_attn.v_proj.kernel",
            Transform.LINEAR,
        ),
        r"^model.layers.([0-9]+).self_attn.o_proj.weight$": (
            r"model.layers.\1.self_attn.o_proj.kernel",
            Transform.LINEAR,
        ),
        # Attention biases
        r"^model.layers.([0-9]+).self_attn.q_proj.bias$": (r"model.layers.\1.self_attn.q_proj.bias", Transform.BIAS),
        r"^model.layers.([0-9]+).self_attn.k_proj.bias$": (r"model.layers.\1.self_attn.k_proj.bias", Transform.BIAS),
        r"^model.layers.([0-9]+).self_attn.v_proj.bias$": (r"model.layers.\1.self_attn.v_proj.bias", Transform.BIAS),
        r"^model.layers.([0-9]+).self_attn.o_proj.bias$": (r"model.layers.\1.self_attn.o_proj.bias", Transform.BIAS),
        # Attention sinks
        r"^model.layers.([0-9]+).self_attn.sinks$": (r"model.layers.\1.self_attn.sinks", Transform.DEFAULT),
        # MoE Router
        r"^model.layers.([0-9]+).mlp.router.weight$": (r"model.layers.\1.mlp.router.linear.kernel", Transform.LINEAR),
        r"^model.layers.([0-9]+).mlp.router.bias$": (r"model.layers.\1.mlp.router.linear.bias", Transform.BIAS),
        # MoE Experts
        r"^model.layers.([0-9]+).mlp.experts.gate_up_proj$": (
            r"model.layers.\1.mlp.experts.gate_up_proj",
            Transform.DEFAULT,
        ),
        r"^model.layers.([0-9]+).mlp.experts.gate_up_proj_bias$": (
            r"model.layers.\1.mlp.experts.gate_up_proj_bias",
            Transform.DEFAULT,
        ),
        r"^model.layers.([0-9]+).mlp.experts.down_proj$": (r"model.layers.\1.mlp.experts.down_proj", Transform.DEFAULT),
        r"^model.layers.([0-9]+).mlp.experts.down_proj_bias$": (
            r"model.layers.\1.mlp.experts.down_proj_bias",
            Transform.DEFAULT,
        ),
    }


def _st_key_to_jax_key(mapping, source_key):
    subs = []
    for pat, (repl, transform) in mapping.items():
        if re.match(pat, source_key):
            target_key = re.sub(pat, repl, source_key)
            subs.append((target_key, transform))

    if not subs:
        return None, None

    if len(subs) > 1:
        keys = [s for s, _ in subs]
        raise ValueError(f"Multiple mappings found for {source_key!r}: {keys}")

    return subs[0]


def _assign_weights(keys, tensor, state_dict, st_key, transform):
    key = keys[0]
    rest = keys[1:]

    if not rest:
        if transform is not None:
            if hasattr(transform, "value"):
                val = transform.value
            else:
                val = transform

            permute = None
            reshape = None
            if val is not None:
                permute, reshape = val
            if permute:
                tensor = tensor.transpose(permute)
            if reshape:
                tensor = tensor.reshape(reshape)

        if key not in state_dict:
            available = list(state_dict.keys())
            raise ValueError(
                f"Target key '{key}' not found in state dict at this level. Available: {available}. Source: {st_key}"
            )

        target_shape = state_dict[key].shape
        if tensor.shape != target_shape:
            raise ValueError(f"Shape mismatch for {st_key} -> {key}: Source {tensor.shape} vs Target {target_shape}")

        state_dict[key] = jnp.array(tensor)
    else:
        if key not in state_dict:
            available = list(state_dict.keys())
            raise ValueError(f"Intermediate key '{key}' not found. Available: {available}. Source: {st_key}")

        _assign_weights(rest, tensor, state_dict[key], st_key, transform)


def _stoi(s):
    try:
        return int(s)
    except ValueError:
        return s


def create_gpt_oss_from_pretrained(file_dir: str, config: model_lib.GptOssConfig):
    files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors found in {file_dir}")

    tensor_dict = {}
    for f in files:
        tensor_dict.update(safetensors.load_file(f))

    model = model_lib.GptOssForCausalLM(config, rngs=nnx.Rngs(0))
    graph_def, abs_state = nnx.split(model)
    jax_state = abs_state.to_pure_dict()

    mapping = _get_key_and_transform_mapping(config)
    conversion_errors = []

    for st_key, tensor in tensor_dict.items():
        jax_key, transform = _st_key_to_jax_key(mapping, st_key)

        if jax_key is None:
            continue

        keys = [_stoi(k) for k in jax_key.split(".")]

        try:
            _assign_weights(keys, tensor, jax_state, st_key, transform)
        except Exception as e:
            full_jax_key = ".".join([str(k) for k in keys])
            conversion_errors.append(f"Failed to assign '{st_key}' to '{full_jax_key}': {type(e).__name__}: {e}")

    if conversion_errors:
        full_error_log = "\n".join(conversion_errors)
        raise RuntimeError(f"Encountered {len(conversion_errors)} weight conversion errors:\n{full_error_log}")

    return nnx.merge(graph_def, jax_state)
