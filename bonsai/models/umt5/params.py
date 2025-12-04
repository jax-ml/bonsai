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

import gc
import json
import re
from enum import Enum

import jax
import jax.numpy as jnp
import safetensors
from etils import epath
from flax import nnx

from bonsai.models.umt5 import modeling as model_lib


def _get_key_and_transform_mapping(cfg: model_lib.UMT5Config):
    """Define mapping from HuggingFace UMT5 keys to JAX UMT5 keys."""

    class Transform(Enum):
        """Transformations for UMT5 parameters"""

        NONE = None
        # For linear layers: (out, in) -> (in, out)
        TRANSPOSE = ((1, 0), None, False)

    # T5/UMT5 uses standard HuggingFace naming
    mapping = {
        # Shared token embeddings
        r"shared\.weight": ("encoder.embed_tokens.embedding", Transform.NONE),
        # Encoder blocks - Self attention
        r"encoder\.block\.([0-9]+)\.layer\.0\.SelfAttention\.q\.weight": (
            r"encoder.block.\1.layer.0.SelfAttention.q.kernel",
            Transform.TRANSPOSE,
        ),
        r"encoder\.block\.([0-9]+)\.layer\.0\.SelfAttention\.k\.weight": (
            r"encoder.block.\1.layer.0.SelfAttention.k.kernel",
            Transform.TRANSPOSE,
        ),
        r"encoder\.block\.([0-9]+)\.layer\.0\.SelfAttention\.v\.weight": (
            r"encoder.block.\1.layer.0.SelfAttention.v.kernel",
            Transform.TRANSPOSE,
        ),
        r"encoder\.block\.([0-9]+)\.layer\.0\.SelfAttention\.o\.weight": (
            r"encoder.block.\1.layer.0.SelfAttention.o.kernel",
            Transform.TRANSPOSE,
        ),
        r"encoder\.block\.([0-9]+)\.layer\.0\.SelfAttention\.relative_attention_bias\.weight": (
            r"encoder.block.\1.layer.0.SelfAttention.relative_attention_bias.embedding",
            Transform.NONE,
        ),
        r"encoder\.block\.([0-9]+)\.layer\.0\.layer_norm\.weight": (
            r"encoder.block.\1.layer.0.layer_norm.scale",
            Transform.NONE,
        ),
        # Encoder blocks - Feed forward
        r"encoder\.block\.([0-9]+)\.layer\.1\.DenseReluDense\.wi_0\.weight": (
            r"encoder.block.\1.layer.1.DenseReluDense.wi_0.kernel",
            Transform.TRANSPOSE,
        ),
        r"encoder\.block\.([0-9]+)\.layer\.1\.DenseReluDense\.wi_1\.weight": (
            r"encoder.block.\1.layer.1.DenseReluDense.wi_1.kernel",
            Transform.TRANSPOSE,
        ),
        r"encoder\.block\.([0-9]+)\.layer\.1\.DenseReluDense\.wo\.weight": (
            r"encoder.block.\1.layer.1.DenseReluDense.wo.kernel",
            Transform.TRANSPOSE,
        ),
        r"encoder\.block\.([0-9]+)\.layer\.1\.layer_norm\.weight": (
            r"encoder.block.\1.layer.1.layer_norm.scale",
            Transform.NONE,
        ),
        # Final layer norm
        r"encoder\.final_layer_norm\.weight": ("encoder.final_layer_norm.scale", Transform.NONE),
    }

    return mapping


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
    cfg: model_lib.UMT5Config,
    param_dtype: jnp.dtype | None = jnp.float32,
    mesh: jax.sharding.Mesh | None = None,
) -> model_lib.UMT5EncoderModel:
    """Load tensors from the safetensors file and create a UMT5 model (memory-optimized)."""
    files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors found in {file_dir}")

    umt5 = nnx.eval_shape(
        lambda: model_lib.UMT5EncoderModel(cfg, param_dtype=param_dtype, rngs=nnx.Rngs(params=0, dropout=0))
    )
    graph_def, abs_state = nnx.split(umt5)
    state_dict = abs_state.to_pure_dict()
    # Only use sharding if mesh is provided
    sharding = nnx.get_named_sharding(abs_state, mesh).to_pure_dict() if mesh is not None else None

    key_mapping = _get_key_and_transform_mapping(cfg)
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
        raise RuntimeError(f"Encountered {len(conversion_errors)} weight conversion errors. Log:\n{full_error_log}")

    gc.collect()
    m = nnx.merge(graph_def, state_dict)
    m.eval()
    return m


def load_model_config(model_path: str) -> model_lib.UMT5Config:
    """Load the model config from the model path."""
    model_dir = epath.Path(model_path).expanduser()
    config_path = model_dir / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with config_path.open("r") as f:
        config_dict = json.load(f)

    return model_lib.UMT5Config(**config_dict)
