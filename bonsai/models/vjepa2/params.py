import gc
import re
from enum import Enum

import jax
import safetensors
from etils import epath
from flax import nnx

from bonsai.models.vjepa2.modeling import (
    VJEPA2Config,
    VJEPA2ForVideoClassification,
    VJEPA2Model,
)


def _get_key_and_transform_mapping(classifier: bool):
    class Transform(Enum):
        BIAS = (None, None, False)
        LINEAR = ((1, 0), None, False)
        CONV3D = ((2, 3, 4, 1, 0), None, False)
        DEFAULT = (None, None, False)

    # Mapping st_keys -> (nnx_keys, (permute_rule, reshape_rule, reshape_first))
    key_mapping = {
        # Encoder Embeddings
        r"encoder\.embeddings\.patch_embeddings\.proj\.weight": (
            r"encoder.embeddings.patch_embeddings.proj.kernel",
            Transform.CONV3D,
        ),
        r"encoder\.embeddings\.patch_embeddings\.proj\.bias": (
            r"encoder.embeddings.patch_embeddings.proj.bias",
            Transform.BIAS,
        ),
        # Encoder Attention
        r"encoder\.layer\.([0-9]+)\.attention\.query\.weight$": (
            r"encoder.layer.\1.attention.query.kernel",
            Transform.LINEAR,
        ),
        r"encoder\.layer\.([0-9]+)\.attention\.key\.weight$": (
            r"encoder.layer.\1.attention.key.kernel",
            Transform.LINEAR,
        ),
        r"encoder\.layer\.([0-9]+)\.attention\.value\.weight$": (
            r"encoder.layer.\1.attention.value.kernel",
            Transform.LINEAR,
        ),
        r"encoder\.layer\.([0-9]+)\.attention\.query\.bias$": (
            r"encoder.layer.\1.attention.query.bias",
            Transform.BIAS,
        ),
        r"encoder\.layer\.([0-9]+)\.attention\.key\.bias$": (
            r"encoder.layer.\1.attention.key.bias",
            Transform.BIAS,
        ),
        r"encoder\.layer\.([0-9]+)\.attention\.value\.bias$": (
            r"encoder.layer.\1.attention.value.bias",
            Transform.BIAS,
        ),
        # Encoder Attention Output Projection
        r"encoder\.layer\.([0-9]+)\.attention\.proj\.weight$": (
            r"encoder.layer.\1.attention.proj.kernel",
            Transform.LINEAR,
        ),
        r"encoder\.layer\.([0-9]+)\.attention\.proj\.bias$": (
            r"encoder.layer.\1.attention.proj.bias",
            Transform.BIAS,
        ),
        # Encoder Norms
        r"encoder\.layer\.([0-9]+)\.norm1\.weight$": (r"encoder.layer.\1.norm1.scale", Transform.DEFAULT),
        r"encoder\.layer\.([0-9]+)\.norm1\.bias$": (r"encoder.layer.\1.norm1.bias", Transform.BIAS),
        r"encoder\.layer\.([0-9]+)\.norm2\.weight$": (r"encoder.layer.\1.norm2.scale", Transform.DEFAULT),
        r"encoder\.layer\.([0-9]+)\.norm2\.bias$": (r"encoder.layer.\1.norm2.bias", Transform.BIAS),
        # Encoder Final LayerNorm
        r"encoder\.layernorm\.weight$": (r"encoder.layernorm.scale", Transform.DEFAULT),
        r"encoder\.layernorm\.bias$": (r"encoder.layernorm.bias", Transform.BIAS),
        # Encoder MLP
        r"encoder\.layer\.([0-9]+)\.mlp\.fc1\.weight$": (r"encoder.layer.\1.mlp.fc1.kernel", Transform.LINEAR),
        r"encoder\.layer\.([0-9]+)\.mlp\.fc1\.bias$": (r"encoder.layer.\1.mlp.fc1.bias", Transform.BIAS),
        r"encoder\.layer\.([0-9]+)\.mlp\.fc2\.weight$": (r"encoder.layer.\1.mlp.fc2.kernel", Transform.LINEAR),
        r"encoder\.layer\.([0-9]+)\.mlp\.fc2\.bias$": (r"encoder.layer.\1.mlp.fc2.bias", Transform.BIAS),
        # Predictor embeddings
        r"predictor\.embeddings\.mask_tokens$": (r"predictor.embeddings.mask_tokens", Transform.DEFAULT),
        r"predictor\.embeddings\.predictor_embeddings\.weight$": (
            r"predictor.embeddings.predictor_embeddings.kernel",
            Transform.LINEAR,
        ),
        r"predictor\.embeddings\.predictor_embeddings\.bias$": (
            r"predictor.embeddings.predictor_embeddings.bias",
            Transform.BIAS,
        ),
        # Predictor Attention
        r"predictor\.layer\.([0-9]+)\.attention\.query\.weight$": (
            r"predictor.layer.\1.attention.query.kernel",
            Transform.LINEAR,
        ),
        r"predictor\.layer\.([0-9]+)\.attention\.key\.weight$": (
            r"predictor.layer.\1.attention.key.kernel",
            Transform.LINEAR,
        ),
        r"predictor\.layer\.([0-9]+)\.attention\.value\.weight$": (
            r"predictor.layer.\1.attention.value.kernel",
            Transform.LINEAR,
        ),
        r"predictor\.layer\.([0-9]+)\.attention\.query\.bias$": (
            r"predictor.layer.\1.attention.query.bias",
            Transform.BIAS,
        ),
        r"predictor\.layer\.([0-9]+)\.attention\.key\.bias$": (
            r"predictor.layer.\1.attention.key.bias",
            Transform.BIAS,
        ),
        r"predictor\.layer\.([0-9]+)\.attention\.value\.bias$": (
            r"predictor.layer.\1.attention.value.bias",
            Transform.BIAS,
        ),
        # Predictor Attention Output Projection
        r"predictor\.layer\.([0-9]+)\.attention\.proj\.weight$": (
            r"predictor.layer.\1.attention.proj.kernel",
            Transform.LINEAR,
        ),
        r"predictor\.layer\.([0-9]+)\.attention\.proj\.bias$": (
            r"predictor.layer.\1.attention.proj.bias",
            Transform.BIAS,
        ),
        # Predictor norms
        r"predictor\.layer\.([0-9]+)\.norm1\.weight$": (r"predictor.layer.\1.norm1.scale", Transform.DEFAULT),
        r"predictor\.layer\.([0-9]+)\.norm1\.bias$": (r"predictor.layer.\1.norm1.bias", Transform.BIAS),
        r"predictor\.layer\.([0-9]+)\.norm2\.weight$": (r"predictor.layer.\1.norm2.scale", Transform.DEFAULT),
        r"predictor\.layer\.([0-9]+)\.norm2\.bias$": (r"predictor.layer.\1.norm2.bias", Transform.BIAS),
        # Predictor Final LayerNorm
        r"predictor\.layernorm\.weight$": (r"predictor.layernorm.scale", Transform.DEFAULT),
        r"predictor\.layernorm\.bias$": (r"predictor.layernorm.bias", Transform.BIAS),
        # Predictor mlp
        r"predictor\.layer\.([0-9]+)\.mlp\.fc1\.weight$": (r"predictor.layer.\1.mlp.fc1.kernel", Transform.LINEAR),
        r"predictor\.layer\.([0-9]+)\.mlp\.fc1\.bias$": (r"predictor.layer.\1.mlp.fc1.bias", Transform.BIAS),
        r"predictor\.layer\.([0-9]+)\.mlp\.fc2\.weight$": (r"predictor.layer.\1.mlp.fc2.kernel", Transform.LINEAR),
        r"predictor\.layer\.([0-9]+)\.mlp\.fc2\.bias$": (r"predictor.layer.\1.mlp.fc2.bias", Transform.BIAS),
        # Predictor projection
        r"predictor\.proj\.weight$": (r"predictor.proj.kernel", Transform.LINEAR),
        r"predictor\.proj\.bias$": (r"predictor.proj.bias", Transform.BIAS),
    }
    if not classifier:
        return key_mapping
    else:
        key_mapping = {r"vjepa2\." + k: (r"vjepa2." + v[0], v[1]) for k, v in key_mapping.items()}
        classifier_keys = {
            # Pooler query tokens
            r"pooler\.query_tokens$": (r"pooler.query_tokens", Transform.DEFAULT),
            # Pooler Cross Attention
            r"pooler\.cross_attention_layer\.cross_attn\.q_proj\.weight$": (
                r"pooler.cross_attention_layer.cross_attn.q_proj.kernel",
                Transform.LINEAR,
            ),
            r"pooler\.cross_attention_layer\.cross_attn\.k_proj\.weight$": (
                r"pooler.cross_attention_layer.cross_attn.k_proj.kernel",
                Transform.LINEAR,
            ),
            r"pooler\.cross_attention_layer\.cross_attn\.v_proj\.weight$": (
                r"pooler.cross_attention_layer.cross_attn.v_proj.kernel",
                Transform.LINEAR,
            ),
            r"pooler\.cross_attention_layer\.cross_attn\.q_proj\.bias$": (
                r"pooler.cross_attention_layer.cross_attn.q_proj.bias",
                Transform.BIAS,
            ),
            r"pooler\.cross_attention_layer\.cross_attn\.k_proj\.bias$": (
                r"pooler.cross_attention_layer.cross_attn.k_proj.bias",
                Transform.BIAS,
            ),
            r"pooler\.cross_attention_layer\.cross_attn\.v_proj\.bias$": (
                r"pooler.cross_attention_layer.cross_attn.v_proj.bias",
                Transform.BIAS,
            ),
            # Pooler cross attention layer norm
            r"pooler\.cross_attention_layer\.layer_norm1\.weight$": (
                r"pooler.cross_attention_layer.layer_norm1.scale",
                Transform.DEFAULT,
            ),
            r"pooler\.cross_attention_layer\.layer_norm1\.bias$": (
                r"pooler.cross_attention_layer.layer_norm1.bias",
                Transform.BIAS,
            ),
            r"pooler\.cross_attention_layer\.layer_norm2\.weight$": (
                r"pooler.cross_attention_layer.layer_norm2.scale",
                Transform.DEFAULT,
            ),
            r"pooler\.cross_attention_layer\.layer_norm2\.bias$": (
                r"pooler.cross_attention_layer.layer_norm2.bias",
                Transform.BIAS,
            ),
            # Pooler cross attention mlp
            r"pooler\.cross_attention_layer\.mlp\.fc1\.weight$": (
                r"pooler.cross_attention_layer.mlp.fc1.kernel",
                Transform.LINEAR,
            ),
            r"pooler\.cross_attention_layer\.mlp\.fc1\.bias$": (
                r"pooler.cross_attention_layer.mlp.fc1.bias",
                Transform.BIAS,
            ),
            r"pooler\.cross_attention_layer\.mlp\.fc2\.weight$": (
                r"pooler.cross_attention_layer.mlp.fc2.kernel",
                Transform.LINEAR,
            ),
            r"pooler\.cross_attention_layer\.mlp\.fc2\.bias$": (
                r"pooler.cross_attention_layer.mlp.fc2.bias",
                Transform.BIAS,
            ),
            # Pooler Self Attention
            r"pooler\.self_attention_layers\.([0-9]+)\.self_attn\.q_proj\.weight$": (
                r"pooler.self_attention_layers.\1.self_attn.q_proj.kernel",
                Transform.LINEAR,
            ),
            r"pooler\.self_attention_layers\.([0-9]+)\.self_attn\.k_proj\.weight$": (
                r"pooler.self_attention_layers.\1.self_attn.k_proj.kernel",
                Transform.LINEAR,
            ),
            r"pooler\.self_attention_layers\.([0-9]+)\.self_attn\.v_proj\.weight$": (
                r"pooler.self_attention_layers.\1.self_attn.v_proj.kernel",
                Transform.LINEAR,
            ),
            r"pooler\.self_attention_layers\.([0-9]+)\.self_attn\.q_proj\.bias$": (
                r"pooler.self_attention_layers.\1.self_attn.q_proj.bias",
                Transform.BIAS,
            ),
            r"pooler\.self_attention_layers\.([0-9]+)\.self_attn\.k_proj\.bias$": (
                r"pooler.self_attention_layers.\1.self_attn.k_proj.bias",
                Transform.BIAS,
            ),
            r"pooler\.self_attention_layers\.([0-9]+)\.self_attn\.v_proj\.bias$": (
                r"pooler.self_attention_layers.\1.self_attn.v_proj.bias",
                Transform.BIAS,
            ),
            # Pooler Self Attention output projection
            r"pooler\.self_attention_layers\.([0-9]+)\.self_attn\.out_proj\.weight$": (
                r"pooler.self_attention_layers.\1.self_attn.out_proj.kernel",
                Transform.LINEAR,
            ),
            r"pooler\.self_attention_layers\.([0-9]+)\.self_attn\.out_proj\.bias$": (
                r"pooler.self_attention_layers.\1.self_attn.out_proj.bias",
                Transform.BIAS,
            ),
            # Pooler Self Attention layer norm
            r"pooler\.self_attention_layers\.([0-9]+)\.layer_norm1\.weight$": (
                r"pooler.self_attention_layers.\1.layer_norm1.scale",
                Transform.DEFAULT,
            ),
            r"pooler\.self_attention_layers\.([0-9]+)\.layer_norm1\.bias$": (
                r"pooler.self_attention_layers.\1.layer_norm1.bias",
                Transform.BIAS,
            ),
            r"pooler\.self_attention_layers\.([0-9]+)\.layer_norm2\.weight$": (
                r"pooler.self_attention_layers.\1.layer_norm2.scale",
                Transform.DEFAULT,
            ),
            r"pooler\.self_attention_layers\.([0-9]+)\.layer_norm2\.bias$": (
                r"pooler.self_attention_layers.\1.layer_norm2.bias",
                Transform.BIAS,
            ),
            # Pooler Self Attention mlp
            r"pooler\.self_attention_layers\.([0-9]+)\.mlp\.fc1\.weight$": (
                r"pooler.self_attention_layers.\1.mlp.fc1.kernel",
                Transform.LINEAR,
            ),
            r"pooler\.self_attention_layers\.([0-9]+)\.mlp\.fc1\.bias$": (
                r"pooler.self_attention_layers.\1.mlp.fc1.bias",
                Transform.BIAS,
            ),
            r"pooler\.self_attention_layers\.([0-9]+)\.mlp\.fc2\.weight$": (
                r"pooler.self_attention_layers.\1.mlp.fc2.kernel",
                Transform.LINEAR,
            ),
            r"pooler\.self_attention_layers\.([0-9]+)\.mlp\.fc2\.bias$": (
                r"pooler.self_attention_layers.\1.mlp.fc2.bias",
                Transform.BIAS,
            ),
            # Classifier
            r"classifier.weight": (r"classifier.kernel", Transform.LINEAR),
            r"classifier.bias": (r"classifier.bias", Transform.BIAS),
        }
        key_mapping.update(classifier_keys)
        return key_mapping


def _torch_key_to_jax_key(mapping, source_key):
    subs = [
        (re.sub(pat, repl, source_key), reshape)
        for pat, (repl, reshape) in mapping.items()
        if re.match(pat, source_key)
    ]
    if len(subs) == 0:
        return (None, None)
    if len(subs) != 1:
        raise ValueError(f"Multiple keys found for {source_key}: {subs}")
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
    file_dir: str, cfg: VJEPA2Config, mesh: jax.sharding.Mesh | None = None, classifier: bool = False
) -> VJEPA2Model | VJEPA2ForVideoClassification:
    """Load tensors from the safetensors file and create a Dinov3 model (memory-optimized)."""
    files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors found in {file_dir}")

    if classifier:
        vjepa2 = nnx.eval_shape(lambda: VJEPA2ForVideoClassification(cfg, rngs=nnx.Rngs(0)))
    else:
        vjepa2 = nnx.eval_shape(lambda: VJEPA2Model(cfg, rngs=nnx.Rngs(0)))
    graph_def, abs_state = nnx.split(vjepa2)
    state_dict = nnx.to_pure_dict(abs_state)
    # Only use sharding if mesh is provided
    sharding = nnx.to_pure_dict(nnx.get_named_sharding(abs_state, mesh)) if mesh is not None else None

    key_mapping = _get_key_and_transform_mapping(classifier)

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
