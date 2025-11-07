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

import logging
import re

import jax
import safetensors.flax as safetensors
from etils import epath
from flax import nnx

from bonsai.models.vae import modeling as model_lib

TO_JAX_CONV_2D_KERNEL = (2, 3, 1, 0)  # (C_out, C_in, kH, kW) -> (kH, kW, C_in, C_out)
TO_JAX_LINEAR_KERNEL = (1, 0)


def _get_key_and_transform_mapping():
    return {
        # encoder
        ## conv in
        r"^encoder.conv_in.weight$": (r"encoder.conv_in.kernel", (TO_JAX_CONV_2D_KERNEL, None)),
        r"^encoder.conv_in.bias$": (r"encoder.conv_in.bias", None),
        ## down blocks
        r"^encoder.down_blocks.([0-3]).resnets.([0-1]).norm([1-2]).weight$": (
            r"encoder.down_blocks.\1.resnets.\2.norm\3.scale",
            None,
        ),
        r"^encoder.down_blocks.([0-3]).resnets.([0-1]).norm([1-2]).bias$": (
            r"encoder.down_blocks.\1.resnets.\2.norm\3.bias",
            None,
        ),
        r"^encoder.down_blocks.([0-3]).resnets.([0-1]).conv([1-2]).weight$": (
            r"encoder.down_blocks.\1.resnets.\2.conv\3.kernel",
            (TO_JAX_CONV_2D_KERNEL, None),
        ),
        r"^encoder.down_blocks.([0-3]).resnets.([0-1]).conv([1-2]).bias$": (
            r"encoder.down_blocks.\1.resnets.\2.conv\3.bias",
            None,
        ),
        r"^encoder.down_blocks.([1-2]).resnets.0.conv_shortcut.weight$": (
            r"encoder.down_blocks.\1.resnets.0.conv_shortcut.kernel",
            (TO_JAX_CONV_2D_KERNEL, None),
        ),
        r"^encoder.down_blocks.([1-2]).resnets.0.conv_shortcut.bias$": (
            r"encoder.down_blocks.\1.resnets.0.conv_shortcut.bias",
            None,
        ),
        r"^encoder.down_blocks.([0-2]).downsamplers.0.conv.weight$": (
            r"encoder.down_blocks.\1.downsamplers.kernel",
            (TO_JAX_CONV_2D_KERNEL, None),
        ),
        r"^encoder.down_blocks.([0-2]).downsamplers.0.conv.bias$": (r"encoder.down_blocks.\1.downsamplers.bias", None),
        ## mid block
        r"^encoder.mid_block.attentions.0.group_norm.weight$": (
            r"encoder.mid_block.attentions.0.group_norm.scale",
            None,
        ),
        r"^encoder.mid_block.attentions.0.group_norm.bias$": (r"encoder.mid_block.attentions.0.group_norm.bias", None),
        r"^encoder.mid_block.attentions.0.query.weight$": (
            r"encoder.mid_block.attentions.0.to_q.kernel",
            (TO_JAX_LINEAR_KERNEL, None),
        ),
        r"^encoder.mid_block.attentions.0.query.bias$": (r"encoder.mid_block.attentions.0.to_q.bias", None),
        r"^encoder.mid_block.attentions.0.key.weight$": (
            r"encoder.mid_block.attentions.0.to_k.kernel",
            (TO_JAX_LINEAR_KERNEL, None),
        ),
        r"^encoder.mid_block.attentions.0.key.bias$": (r"encoder.mid_block.attentions.0.to_k.bias", None),
        r"^encoder.mid_block.attentions.0.value.weight$": (
            r"encoder.mid_block.attentions.0.to_v.kernel",
            (TO_JAX_LINEAR_KERNEL, None),
        ),
        r"^encoder.mid_block.attentions.0.value.bias$": (r"encoder.mid_block.attentions.0.to_v.bias", None),
        r"^encoder.mid_block.attentions.0.proj_attn.weight$": (
            r"encoder.mid_block.attentions.0.to_out.kernel",
            (TO_JAX_LINEAR_KERNEL, None),
        ),
        r"^encoder.mid_block.attentions.0.proj_attn.bias$": (r"encoder.mid_block.attentions.0.to_out.bias", None),
        r"^encoder.mid_block.resnets.([0-1]).conv([1-2]).weight$": (
            r"encoder.mid_block.resnets.\1.conv\2.kernel",
            (TO_JAX_CONV_2D_KERNEL, None),
        ),
        r"^encoder.mid_block.resnets.([0-1]).conv([1-2]).bias$": (r"encoder.mid_block.resnets.\1.conv\2.bias", None),
        r"^encoder.mid_block.resnets.([0-1]).norm([1-2]).weight$": (r"encoder.mid_block.resnets.\1.norm\2.scale", None),
        r"^encoder.mid_block.resnets.([0-1]).norm([1-2]).bias$": (r"encoder.mid_block.resnets.\1.norm\2.bias", None),
        ## conv norm out
        r"^encoder.conv_norm_out.weight$": (r"encoder.conv_norm_out.scale", None),
        r"^encoder.conv_norm_out.bias$": (r"encoder.conv_norm_out.bias", None),
        ## conv out
        r"^encoder.conv_out.weight$": (r"encoder.conv_out.kernel", (TO_JAX_CONV_2D_KERNEL, None)),
        r"^encoder.conv_out.bias": (r"encoder.conv_out.bias", None),
        # latent space
        ## quant_conv
        r"^quant_conv.weight$": (r"quant_conv.kernel", (TO_JAX_CONV_2D_KERNEL, None)),
        r"^quant_conv.bias$": (r"quant_conv.bias", None),
        ## post_quant_conv
        r"^post_quant_conv.weight$": (r"post_quant_conv.kernel", (TO_JAX_CONV_2D_KERNEL, None)),
        r"^post_quant_conv.bias$": (r"post_quant_conv.bias", None),
        # decoder
        ## conv in
        r"^decoder.conv_in.weight$": (r"decoder.conv_in.kernel", (TO_JAX_CONV_2D_KERNEL, None)),
        r"^decoder.conv_in.bias$": (r"decoder.conv_in.bias", None),
        ## mid block
        r"^decoder.mid_block.attentions.0.group_norm.weight$": (
            r"decoder.mid_block.attentions.0.group_norm.scale",
            None,
        ),
        r"^decoder.mid_block.attentions.0.group_norm.bias$": (r"decoder.mid_block.attentions.0.group_norm.bias", None),
        r"^decoder.mid_block.attentions.0.query.weight$": (
            r"decoder.mid_block.attentions.0.to_q.kernel",
            (TO_JAX_LINEAR_KERNEL, None),
        ),
        r"^decoder.mid_block.attentions.0.query.bias$": (r"decoder.mid_block.attentions.0.to_q.bias", None),
        r"^decoder.mid_block.attentions.0.key.weight$": (
            r"decoder.mid_block.attentions.0.to_k.kernel",
            (TO_JAX_LINEAR_KERNEL, None),
        ),
        r"^decoder.mid_block.attentions.0.key.bias$": (r"decoder.mid_block.attentions.0.to_k.bias", None),
        r"^decoder.mid_block.attentions.0.value.weight$": (
            r"decoder.mid_block.attentions.0.to_v.kernel",
            (TO_JAX_LINEAR_KERNEL, None),
        ),
        r"^decoder.mid_block.attentions.0.value.bias$": (r"decoder.mid_block.attentions.0.to_v.bias", None),
        r"^decoder.mid_block.attentions.0.proj_attn.weight$": (
            r"decoder.mid_block.attentions.0.to_out.kernel",
            (TO_JAX_LINEAR_KERNEL, None),
        ),
        r"^decoder.mid_block.attentions.0.proj_attn.bias$": (r"decoder.mid_block.attentions.0.to_out.bias", None),
        r"^decoder.mid_block.resnets.([0-1]).norm([1-2]).weight$": (r"decoder.mid_block.resnets.\1.norm\2.scale", None),
        r"^decoder.mid_block.resnets.([0-1]).norm([1-2]).bias$": (r"decoder.mid_block.resnets.\1.norm\2.bias", None),
        r"^decoder.mid_block.resnets.([0-1]).conv([1-2]).weight$": (
            r"decoder.mid_block.resnets.\1.conv\2.kernel",
            (TO_JAX_CONV_2D_KERNEL, None),
        ),
        r"^decoder.mid_block.resnets.([0-1]).conv([1-2]).bias": (r"decoder.mid_block.resnets.\1.conv\2.bias", None),
        ## up blocks
        r"^decoder.up_blocks.([0-3]).resnets.([0-2]).norm([1-2]).weight$": (
            r"decoder.up_blocks.\1.resnets.\2.norm\3.scale",
            None,
        ),
        r"^decoder.up_blocks.([0-3]).resnets.([0-2]).norm([1-2]).bias$": (
            r"decoder.up_blocks.\1.resnets.\2.norm\3.bias",
            None,
        ),
        r"^decoder.up_blocks.([0-3]).resnets.([0-2]).conv([1-2]).weight$": (
            r"decoder.up_blocks.\1.resnets.\2.conv\3.kernel",
            (TO_JAX_CONV_2D_KERNEL, None),
        ),
        r"^decoder.up_blocks.([0-3]).resnets.([0-2]).conv([1-2]).bias$": (
            r"decoder.up_blocks.\1.resnets.\2.conv\3.bias",
            None,
        ),
        r"^decoder.up_blocks.([2-3]).resnets.0.conv_shortcut.weight$": (
            r"decoder.up_blocks.\1.resnets.0.conv_shortcut.kernel",
            (TO_JAX_CONV_2D_KERNEL, None),
        ),
        r"^decoder.up_blocks.([2-3]).resnets.0.conv_shortcut.bias$": (
            r"decoder.up_blocks.\1.resnets.0.conv_shortcut.bias",
            None,
        ),
        r"^decoder.up_blocks.([0-2]).upsamplers.0.conv.weight$": (
            r"decoder.up_blocks.\1.upsamplers.conv.kernel",
            (TO_JAX_CONV_2D_KERNEL, None),
        ),
        r"^decoder.up_blocks.([0-2]).upsamplers.0.conv.bias$": (r"decoder.up_blocks.\1.upsamplers.conv.bias", None),
        ## conv norm out
        r"^decoder.conv_norm_out.weight$": (r"decoder.conv_norm_out.scale", None),
        r"^decoder.conv_norm_out.bias$": (r"decoder.conv_norm_out.bias", None),
        ## conv out
        r"^decoder.conv_out.weight$": (r"decoder.conv_out.kernel", (TO_JAX_CONV_2D_KERNEL, None)),
        r"^decoder.conv_out.bias$": (r"decoder.conv_out.bias", None),
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


def _assign_weights(keys, tensor, state_dict, st_key, transform):
    """Recursively descend into state_dict and assign the (possibly permuted/reshaped) tensor."""
    key, *rest = keys
    if not rest:
        if transform is not None:
            permute, reshape = transform
            if permute:
                tensor = tensor.transpose(permute)
            if reshape:
                tensor = tensor.reshape(reshape)
        if tensor.shape != state_dict[key].shape:
            raise ValueError(f"Shape mismatch for {st_key}: {tensor.shape} vs {state_dict[key].shape}")
        state_dict[key] = tensor
    else:
        _assign_weights(rest, tensor, state_dict[key], st_key, transform)


def _stoi(s):
    try:
        return int(s)
    except ValueError:
        return s


def create_model_from_safe_tensors(
    file_dir: str,
    mesh: jax.sharding.Mesh | None = None,
) -> model_lib.VAE:
    """Load tensors from the safetensors file and create a VAE model."""
    files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors found in {file_dir}")

    tensor_dict = {}
    for f in files:
        tensor_dict |= safetensors.load_file(f)

    vae = nnx.eval_shape(lambda: model_lib.VAE(rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(vae)
    jax_state = abs_state.to_pure_dict()

    mapping = _get_key_and_transform_mapping()

    for st_key, tensor in tensor_dict.items():
        jax_key, transform = _st_key_to_jax_key(mapping, st_key)
        if jax_key is None:
            continue
        keys = [_stoi(k) for k in jax_key.split(".")]
        _assign_weights(keys, tensor, jax_state, st_key, transform)

    if mesh is not None:
        sharding = nnx.get_named_sharding(abs_state, mesh).to_pure_dict()
        state_dict = jax.device_put(jax_state, sharding)
    else:
        state_dict = jax.device_put(jax_state, jax.devices()[0])

    return nnx.merge(graph_def, state_dict)
