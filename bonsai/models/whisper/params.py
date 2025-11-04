# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import re
from enum import Enum

import jax
import safetensors.flax as safetensors
from etils import epath
from flax import nnx

from bonsai.models.whisper import modeling as model_lib


def _get_key_and_transform_mapping(config: model_lib.ModelCfg):
    nheads = config.decoder_attention_heads
    head_dim = config.d_model // nheads

    class Transform(Enum):
        """Transformations for model parameters"""

        BIAS = None
        LINEAR = ((1, 0), None, False)
        CONV1D = ((2, 1, 0), None, False)
        EMBED = None
        ATTN_KERNEL = ((2, 0, 1), (nheads, head_dim, config.d_model), True)
        ATTN_BIAS = (None, (nheads, head_dim), False)
        OUT_KERNEL = ((1, 2, 0), (config.d_model, nheads, head_dim), True)
        OUT_BIAS = None
        LN_SCALE = None

    # Mapping st_keys -> (nnx_keys, (permute_rule, reshape_rule, reshape_first)).
    return {
        r"^model\.decoder\.embed_positions\.weight$": (r"decoder\.embed_positions\.embedding", Transform.EMBED),
        r"^model\.decoder\.embed_tokens\.weight$": (r"decoder\.embed_tokens\.embedding", Transform.EMBED),
        r"^model\.decoder\.layer_norm\.bias$": (r"decoder\.layer_norm\.bias", Transform.BIAS),
        r"^model\.decoder\.layer_norm\.weight$": (r"decoder\.layer_norm\.scale", Transform.LN_SCALE),
        r"^model\.decoder\.layers\.(\d+)\.encoder_attn\.k_proj\.weight$": (
            r"decoder\.layers\.\1\.encoder_attn\.key\.kernel",
            Transform.ATTN_KERNEL,
        ),
        r"^model\.decoder\.layers\.(\d+)\.encoder_attn\.out_proj\.bias$": (
            r"decoder\.layers\.\1\.encoder_attn\.out\.bias",
            Transform.OUT_BIAS,
        ),
        r"^model\.decoder\.layers\.(\d+)\.encoder_attn\.out_proj\.weight$": (
            r"decoder\.layers\.\1\.encoder_attn\.out\.kernel",
            Transform.OUT_KERNEL,
        ),
        r"^model\.decoder\.layers\.(\d+)\.encoder_attn\.q_proj\.bias$": (
            r"decoder\.layers\.\1\.encoder_attn\.query\.bias",
            Transform.ATTN_BIAS,
        ),
        r"^model\.decoder\.layers\.(\d+)\.encoder_attn\.q_proj\.weight$": (
            r"decoder\.layers\.\1\.encoder_attn\.query\.kernel",
            Transform.ATTN_KERNEL,
        ),
        r"^model\.decoder\.layers\.(\d+)\.encoder_attn\.v_proj\.bias$": (
            r"decoder\.layers\.\1\.encoder_attn\.value\.bias",
            Transform.ATTN_BIAS,
        ),
        r"^model\.decoder\.layers\.(\d+)\.encoder_attn\.v_proj\.weight$": (
            r"decoder\.layers\.\1\.encoder_attn\.value\.kernel",
            Transform.ATTN_KERNEL,
        ),
        r"^model\.decoder\.layers\.(\d+)\.encoder_attn_layer_norm\.bias$": (
            r"decoder\.layers\.\1\.encoder_attn_layer_norm\.bias",
            Transform.BIAS,
        ),
        r"^model\.decoder\.layers\.(\d+)\.encoder_attn_layer_norm\.weight$": (
            r"decoder\.layers\.\1\.encoder_attn_layer_norm\.scale",
            Transform.LN_SCALE,
        ),
        r"^model\.decoder\.layers\.(\d+)\.fc(\d+)\.bias$": (r"decoder\.layers\.\1\.fc\2\.bias", Transform.BIAS),
        r"^model\.decoder\.layers\.(\d+)\.fc(\d+)\.weight$": (
            r"decoder\.layers\.\1\.fc\2\.kernel",
            Transform.LINEAR,
        ),
        r"^model\.decoder\.layers\.(\d+)\.final_layer_norm\.bias$": (
            r"decoder\.layers\.\1\.final_layer_norm\.bias",
            Transform.BIAS,
        ),
        r"^model\.decoder\.layers\.(\d+)\.final_layer_norm\.weight$": (
            r"decoder\.layers\.\1\.final_layer_norm\.scale",
            Transform.LN_SCALE,
        ),
        r"^model\.decoder\.layers\.(\d+)\.self_attn\.k_proj\.weight$": (
            r"decoder\.layers\.\1\.self_attn\.key\.kernel",
            Transform.ATTN_KERNEL,
        ),
        r"^model\.decoder\.layers\.(\d+)\.self_attn\.out_proj\.bias$": (
            r"decoder\.layers\.\1\.self_attn\.out\.bias",
            Transform.OUT_BIAS,
        ),
        r"^model\.decoder\.layers\.(\d+)\.self_attn\.out_proj\.weight$": (
            r"decoder\.layers\.\1\.self_attn\.out\.kernel",
            Transform.OUT_KERNEL,
        ),
        r"^model\.decoder\.layers\.(\d+)\.self_attn\.q_proj\.bias$": (
            r"decoder\.layers\.\1\.self_attn\.query\.bias",
            Transform.ATTN_BIAS,
        ),
        r"^model\.decoder\.layers\.(\d+)\.self_attn\.q_proj\.weight$": (
            r"decoder\.layers\.\1\.self_attn\.query\.kernel",
            Transform.ATTN_KERNEL,
        ),
        r"^model\.decoder\.layers\.(\d+)\.self_attn\.v_proj\.bias$": (
            r"decoder\.layers\.\1\.self_attn\.value\.bias",
            Transform.ATTN_BIAS,
        ),
        r"^model\.decoder\.layers\.(\d+)\.self_attn\.v_proj\.weight$": (
            r"decoder\.layers\.\1\.self_attn\.value\.kernel",
            Transform.ATTN_KERNEL,
        ),
        r"^model\.decoder\.layers\.(\d+)\.self_attn_layer_norm\.bias$": (
            r"decoder\.layers\.\1\.self_attn_layer_norm\.bias",
            Transform.BIAS,
        ),
        r"^model\.decoder\.layers\.(\d+)\.self_attn_layer_norm\.weight$": (
            r"decoder\.layers\.\1\.self_attn_layer_norm\.scale",
            Transform.LN_SCALE,
        ),
        r"^model\.encoder\.conv(\d+)\.bias$": (r"encoder\.conv\1\.bias", Transform.BIAS),
        r"^model\.encoder\.conv(\d+)\.weight$": (r"encoder\.conv\1\.kernel", Transform.CONV1D),
        r"^model\.encoder\.embed_positions\.weight$": (
            r"encoder\.embed_positions\.embedding",
            Transform.EMBED,
        ),
        r"^model\.encoder\.layer_norm\.bias$": (r"encoder\.layer_norm\.bias", Transform.BIAS),
        r"^model\.encoder\.layer_norm\.weight$": (r"encoder\.layer_norm\.scale", Transform.LN_SCALE),
        r"^model\.encoder\.layers\.(\d+)\.fc(\d+)\.bias$": (r"encoder\.layers\.\1\.fc\2\.bias", Transform.BIAS),
        r"^model\.encoder\.layers\.(\d+)\.fc(\d+)\.weight$": (
            r"encoder\.layers\.\1\.fc\2\.kernel",
            Transform.LINEAR,
        ),
        r"^model\.encoder\.layers\.(\d+)\.final_layer_norm\.bias$": (
            r"encoder\.layers\.\1\.final_layer_norm\.bias",
            Transform.BIAS,
        ),
        r"^model\.encoder\.layers\.(\d+)\.final_layer_norm\.weight$": (
            r"encoder\.layers\.\1\.final_layer_norm\.scale",
            Transform.LN_SCALE,
        ),
        r"^model\.encoder\.layers\.(\d+)\.self_attn\.k_proj\.weight$": (
            r"encoder\.layers\.\1\.self_attn\.key\.kernel",
            Transform.ATTN_KERNEL,
        ),
        r"^model\.encoder\.layers\.(\d+)\.self_attn\.out_proj\.bias$": (
            r"encoder\.layers\.\1\.self_attn\.out\.bias",
            Transform.OUT_BIAS,
        ),
        r"^model\.encoder\.layers\.(\d+)\.self_attn\.out_proj\.weight$": (
            r"encoder\.layers\.\1\.self_attn\.out\.kernel",
            Transform.OUT_KERNEL,
        ),
        r"^model\.encoder\.layers\.(\d+)\.self_attn\.q_proj\.bias$": (
            r"encoder\.layers\.\1\.self_attn\.query\.bias",
            Transform.ATTN_BIAS,
        ),
        r"^model\.encoder\.layers\.(\d+)\.self_attn\.q_proj\.weight$": (
            r"encoder\.layers\.\1\.self_attn\.query\.kernel",
            Transform.ATTN_KERNEL,
        ),
        r"^model\.encoder\.layers\.(\d+)\.self_attn\.v_proj\.bias$": (
            r"encoder\.layers\.\1\.self_attn\.value\.bias",
            Transform.ATTN_BIAS,
        ),
        r"^model\.encoder\.layers\.(\d+)\.self_attn\.v_proj\.weight$": (
            r"encoder\.layers\.\1\.self_attn\.value\.kernel",
            Transform.ATTN_KERNEL,
        ),
        r"^model\.encoder\.layers\.(\d+)\.self_attn_layer_norm\.bias$": (
            r"encoder\.layers\.\1\.self_attn_layer_norm\.bias",
            Transform.BIAS,
        ),
        r"^model\.encoder\.layers\.(\d+)\.self_attn_layer_norm\.weight$": (
            r"encoder\.layers\.\1\.self_attn_layer_norm\.scale",
            Transform.LN_SCALE,
        ),
    }


def _st_key_to_jax_key(mapping, source_key):
    """Map a safetensors key to exactly one JAX key & transform, else warn/error."""
    subs = [
        (re.sub(pat, repl, source_key), transform)
        for pat, (repl, transform) in mapping.items()
        if re.match(pat, source_key)
    ]
    if not subs:
        logging.warning(f"No mapping found for key: {source_key!r}")
        return None, None
    if len(subs) > 1:
        keys = [s for s, _ in subs]
        raise ValueError(f"Multiple mappings found for {source_key!r}: {keys}")
    return subs[0]


def _assign_weights(keys, tensor, state_dict, st_key, transform, dtype):
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
        if dtype is not None:
            tensor = tensor.astype(dtype)
        state_dict[key] = tensor
    else:
        _assign_weights(rest, tensor, state_dict[key], st_key, transform, dtype)


def _stoi(s):
    try:
        return int(s)
    except ValueError:
        return s


def create_whisper_from_pretrained(
    file_dir: str,
    num_classes: int = 1000,
    *,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jax.typing.DTypeLike | None = None,
):
    """
    Load safetensor weights from a file, then convert & merge into a flax.nnx ViT model.

    Returns:
      A flax.nnx.Model instance with loaded parameters.
    """
    files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors found in {file_dir}")

    tensor_dict = {}
    for f in files:
        tensor_dict |= safetensors.load_file(f)

    config = model_lib.ModelCfg.whisper_tiny()
    whisper = model_lib.Whisper(config, rngs=nnx.Rngs(0))
    graph_def, abs_state = nnx.split(whisper)
    jax_state = abs_state.to_pure_dict()

    mapping = _get_key_and_transform_mapping(config)
    error_count = 0
    for st_key, tensor in tensor_dict.items():
        jax_key, transform = _st_key_to_jax_key(mapping, st_key)
        if jax_key is None:
            continue
        keys = [_stoi(k) for k in jax_key.split(r"\.")]
        try:
            _assign_weights(keys, tensor, jax_state, st_key, transform.value, dtype)
        except KeyError as e:
            error_count += 1
            print(f"Key error: {keys} at {e}")
        except ValueError as e:
            error_count += 1
            print(e)
        except Exception as e:
            error_count += 1
            print(keys)
            raise e

    if error_count > 0:
        raise RuntimeError("Encountered errors in loading weights. Refer to previous messages")

    return nnx.merge(graph_def, jax_state)


if __name__ == "__main__":
    from huggingface_hub import snapshot_download

    name = "openai/whisper-tiny"
    model_ckpt_path = snapshot_download(name)

    model = create_whisper_from_pretrained(model_ckpt_path)
