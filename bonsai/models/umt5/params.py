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

import dataclasses
import gc
import json
import logging
import re
from enum import Enum, auto

import jax
import jax.numpy as jnp
import safetensors
import torch
from etils import epath
from flax import nnx

from bonsai.models.umt5 import modeling as model_lib


def _get_key_and_transform_mapping(cls, cfg: model_lib.UMT5Config):
    """Define mapping from HuggingFace UMT5 keys to JAX UMT5 keys."""

    class Transform(Enum):
        """Transformations for UMT5 parameters"""

        NONE = None
        # For linear layers: (out, in) -> (in, out)
        TRANSPOSE = ((1, 0), None, False)

    # T5/UMT5 uses standard HuggingFace naming
    encoder_mapping = {
        r"encoder\.embed_tokens\.weight": ("encoder.embed_tokens.embedding", Transform.NONE),
        # Shared token embeddings
        r"shared\.weight": ("encoder.embed_tokens.embedding", Transform.NONE),
        # Encoder
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
        # Encoder Final layer norm
        r"encoder\.final_layer_norm\.weight": ("encoder.final_layer_norm.scale", Transform.NONE),
    }
    decoder_mapping = {
        # Decoder
        # Decoder embedding
        r"decoder\.embed_tokens\.weight": ("decoder.embed_tokens.embedding", Transform.NONE),
        # Decoder blocks - Self attention
        r"decoder\.block\.([0-9]+)\.layer\.0\.SelfAttention\.q\.weight": (
            r"decoder.block.\1.layer.0.SelfAttention.q.kernel",
            Transform.TRANSPOSE,
        ),
        r"decoder\.block\.([0-9]+)\.layer\.0\.SelfAttention\.k\.weight": (
            r"decoder.block.\1.layer.0.SelfAttention.k.kernel",
            Transform.TRANSPOSE,
        ),
        r"decoder\.block\.([0-9]+)\.layer\.0\.SelfAttention\.v\.weight": (
            r"decoder.block.\1.layer.0.SelfAttention.v.kernel",
            Transform.TRANSPOSE,
        ),
        r"decoder\.block\.([0-9]+)\.layer\.0\.SelfAttention\.o\.weight": (
            r"decoder.block.\1.layer.0.SelfAttention.o.kernel",
            Transform.TRANSPOSE,
        ),
        r"decoder\.block\.([0-9]+)\.layer\.0\.SelfAttention\.relative_attention_bias\.weight": (
            r"decoder.block.\1.layer.0.SelfAttention.relative_attention_bias.embedding",
            Transform.NONE,
        ),
        r"decoder\.block\.([0-9]+)\.layer\.0\.layer_norm\.weight": (
            r"decoder.block.\1.layer.0.layer_norm.scale",
            Transform.NONE,
        ),
        # Decoder blocks - Cross attention
        r"decoder\.block\.([0-9]+)\.layer\.1\.EncDecAttention\.q\.weight": (
            r"decoder.block.\1.layer.1.EncDecAttention.q.kernel",
            Transform.TRANSPOSE,
        ),
        r"decoder\.block\.([0-9]+)\.layer\.1\.EncDecAttention\.k\.weight": (
            r"decoder.block.\1.layer.1.EncDecAttention.k.kernel",
            Transform.TRANSPOSE,
        ),
        r"decoder\.block\.([0-9]+)\.layer\.1\.EncDecAttention\.v\.weight": (
            r"decoder.block.\1.layer.1.EncDecAttention.v.kernel",
            Transform.TRANSPOSE,
        ),
        r"decoder\.block\.([0-9]+)\.layer\.1\.EncDecAttention\.o\.weight": (
            r"decoder.block.\1.layer.1.EncDecAttention.o.kernel",
            Transform.TRANSPOSE,
        ),
        r"decoder\.block\.([0-9]+)\.layer\.1\.layer_norm\.weight": (
            r"decoder.block.\1.layer.1.layer_norm.scale",
            Transform.NONE,
        ),
        # Decoder blocks - Feed forward
        r"decoder\.block\.([0-9]+)\.layer\.2\.DenseReluDense\.wi_0\.weight": (
            r"decoder.block.\1.layer.2.DenseReluDense.wi_0.kernel",
            Transform.TRANSPOSE,
        ),
        r"decoder\.block\.([0-9]+)\.layer\.2\.DenseReluDense\.wi_1\.weight": (
            r"decoder.block.\1.layer.2.DenseReluDense.wi_1.kernel",
            Transform.TRANSPOSE,
        ),
        r"decoder\.block\.([0-9]+)\.layer\.2\.DenseReluDense\.wo\.weight": (
            r"decoder.block.\1.layer.2.DenseReluDense.wo.kernel",
            Transform.TRANSPOSE,
        ),
        r"decoder\.block\.([0-9]+)\.layer\.2\.layer_norm\.weight": (
            r"decoder.block.\1.layer.2.layer_norm.scale",
            Transform.NONE,
        ),
        # Decoder Final layer norm
        r"decoder\.final_layer_norm\.weight": ("decoder.final_layer_norm.scale", Transform.NONE),
        # lm head
        r"lm_head\.weight": ("lm_head.kernel", Transform.TRANSPOSE),
    }

    if cls == model_lib.UMT5EncoderModel:
        return encoder_mapping

    full_mapping = encoder_mapping.copy()
    full_mapping.update(decoder_mapping)
    return full_mapping


def _torch_key_to_jax_key(mapping, source_key):
    subs = [
        (re.sub(pat, repl, source_key), reshape)
        for pat, (repl, reshape) in mapping.items()
        if re.match(pat, source_key)
    ]
    if len(subs) > 1:
        raise ValueError(f"Only one key should be found: {subs[0]}")
    if len(subs) == 0:
        return (None, None)
    return subs[0]


def _assign_weights(keys, tensor, state_dict, st_key, transform, sharding_dict, dtype):
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
            state_dict[key] = jax.device_put(tensor, sharding_dict[key]).astype(dtype)
        else:
            state_dict[key] = jax.device_put(tensor).astype(dtype)
    else:
        next_sharding = sharding_dict[key] if sharding_dict is not None else None
        _assign_weights(rest, tensor, state_dict[key], st_key, transform, next_sharding, dtype)


def _stoi(s):
    try:
        return int(s)
    except ValueError:
        return s


class WeightFileType(Enum):
    ST = auto()  # safetensor
    BIN = auto()
    PT = auto()
    PTH = auto()


def search_available_weight_file(file_path):
    p = epath.Path(file_path).expanduser()
    if p.is_file():
        # if file_path is a file, return [p], type
        file_ext = p.suffix.lower()
        if file_ext == ".safetensors":
            print(f"Using {p} to load weight")
            return [p], WeightFileType.ST
        elif file_ext == ".bin":
            print(f"Using {p} to load weight")
            return [p], WeightFileType.BIN
        elif file_ext == ".pt":
            print(f"Using {p} to load weight")
            return [p], WeightFileType.PT
        elif file_ext == ".pth":
            print(f"Using {p} to load weight")
            return [p], WeightFileType.PTH
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")
    else:
        # if file_path is dir, return all matching files
        files = list(p.glob("*.safetensors"))
        if not files:
            logging.warning(f"No *.safetensors found in {file_path}, try to search others")
        else:
            print("Using *.safetensors to load weight")
            return files, WeightFileType.ST

        files = list(p.glob("*.bin"))
        if not files:
            logging.warning(f"No *.bin found in {file_path}, try to search others")
        else:
            print("Using *.bin to load weight")
            return files, WeightFileType.BIN

        files = list(p.glob("*.pth"))
        if not files:
            logging.warning(f"No *.pth found in {file_path}, try to search others")
        else:
            print("Using *.pth to load weight")
            return files, WeightFileType.PTH

    raise ValueError(f"No weight file found in {file_path}")


def open_weight_file(f, file_type):
    if file_type in [WeightFileType.BIN, WeightFileType.PTH]:
        sf = torch.load(f, map_location="cpu")
    elif file_type == WeightFileType.ST:
        sf = safetensors.safe_open(f, framework="numpy")
    else:
        raise ValueError(f"invalid file type: {file_type}")
    return sf


def get_tensor(sf, torch_key, file_type):
    if file_type in [WeightFileType.BIN, WeightFileType.PTH]:
        tensor = sf[torch_key]
    elif file_type == WeightFileType.ST:
        tensor = sf.get_tensor(torch_key)
    else:
        raise ValueError(f"invalid file type: {file_type}")

    return tensor


def create_model(
    cls,
    file_dir: str,
    cfg: model_lib.UMT5Config,
    key_mapping=None,
    param_dtype: jnp.dtype | None = jnp.float32,
    mesh: jax.sharding.Mesh | None = None,
) -> model_lib.UMT5Model | model_lib.UMT5EncoderModel:
    """Load weight and create a UMT5Encoder model (memory-optimized).

    Args:
        cls: model class. UMT5Model and UMT5EncoderModel is available.
        file_dir: model weight path.
        cfg: model config. Use 'load_model_config' to get it.
        key_mapping: model weight key map. Used in unofficial umt5 model, such as Wan2.1
        param_dtype: model weight dtype.
        mesh: model weight mesh.
    Returns:
        The instance of model defined in cls.
    """
    files, file_type = search_available_weight_file(file_dir)

    umt5 = nnx.eval_shape(lambda: cls(cfg, param_dtype=param_dtype, rngs=nnx.Rngs(params=0, dropout=0)))
    graph_def, abs_state = nnx.split(umt5)
    state_dict = nnx.to_pure_dict(abs_state)
    # Only use sharding if mesh is provided
    sharding = nnx.to_pure_dict(nnx.get_named_sharding(abs_state, mesh)) if mesh is not None else None

    if not key_mapping:
        key_mapping = _get_key_and_transform_mapping(cls, cfg)
    conversion_errors = []

    print(f"Loading Weight: {cfg.dtype=}")
    for f in files:
        sf = open_weight_file(f, file_type)

        for torch_key in sf.keys():
            ts = get_tensor(sf, torch_key, file_type)
            if isinstance(ts, torch.Tensor):
                npy = ts.numpy() if ts.dtype != torch.bfloat16 else ts.to(dtype=torch.float32).numpy()

            jax_key, transform = _torch_key_to_jax_key(key_mapping, torch_key)
            if jax_key is None:
                continue

            keys = [_stoi(k) for k in jax_key.split(".")]
            try:
                _assign_weights(keys, npy, state_dict, torch_key, transform.value, sharding, cfg.dtype)
            except Exception as e:
                full_jax_key = ".".join([str(k) for k in keys])
                conversion_errors.append(f"Failed to assign '{torch_key}' to '{full_jax_key}': {type(e).__name__}: {e}")
        gc.collect()

    if conversion_errors:
        full_error_log = "\n".join(conversion_errors)
        raise RuntimeError(f"Encountered {len(conversion_errors)} weight conversion errors. Log:\n{full_error_log}")

    if cls == model_lib.UMT5Model:
        state_dict["decoder"]["embed_tokens"]["embedding"] = state_dict["encoder"]["embed_tokens"]["embedding"]

    gc.collect()
    m = nnx.merge(graph_def, state_dict)
    m.eval()
    return m


def get_weight_dtype_from_config(conf_dict):
    def get_dtype(dtype_str):
        if dtype_str in ["float32", "fp32"]:
            return jnp.float32
        elif dtype_str in ["bloat16", "bf16"]:
            return jnp.bfloat16
        elif dtype_str in ["float16", "fp16"]:
            return jnp.float16
        else:
            logging.warning(f"Unrecognized dtype: {dtype_str}")
            return jnp.float32

    if "dtype" in conf_dict:
        return get_dtype(conf_dict["dtype"])
    elif "torch_dtype" in conf_dict:
        return get_dtype(conf_dict["torch_dtype"])
    else:
        logging.warning("No 'dtype' config found in config file")
        return jnp.float32


def load_model_config(model_path: str) -> model_lib.UMT5Config:
    """Load the model config from the model path."""
    model_dir = epath.Path(model_path).expanduser()
    config_path = model_dir / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with config_path.open("r") as f:
        config_dict = json.load(f)

    dtype = get_weight_dtype_from_config(config_dict)

    # Filter config_dict to only include fields defined in UMT5Config
    config_fields = {f.name for f in dataclasses.fields(model_lib.UMT5Config)}
    filtered_config = {k: v for k, v in config_dict.items() if k in config_fields}

    return model_lib.UMT5Config(**filtered_config, dtype=dtype)
