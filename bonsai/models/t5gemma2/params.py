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
import re
from enum import Enum

import jax
import safetensors
from etils import epath
from flax import nnx

from bonsai.models.t5gemma2 import modeling as model_lib
from bonsai.models.t5gemma2.modeling import T5Gemma2Config


def _get_key_and_transform_mapping(cfg: T5Gemma2Config):
    """Returns mapping from checkpoint keys to model keys with transformations."""
    enc = cfg.encoder.text_config
    dec = cfg.decoder
    vis = cfg.encoder.vision_config

    class Transform(Enum):
        """Transformations for model parameters.

        Format: (permute_axes, reshape_shape, reshape_first)
        - If reshape_first=True: reshape then permute
        - If reshape_first=False: permute then reshape
        """

        NONE = None
        SCALE = None
        EMBED = None

        # Encoder attention transforms
        # q_proj: (num_q_heads * head_dim, embed_dim) -> (num_q_heads, embed_dim, head_dim)
        # reshape to (num_q_heads, head_dim, embed_dim), permute (0, 2, 1)
        ENC_ATTN_Q = (
            (0, 2, 1),
            (enc.num_attention_heads, enc.head_dim, enc.embed_dim),
            True,
        )
        # k/v_proj: (num_kv_heads * head_dim, embed_dim) -> (num_kv_heads, embed_dim, head_dim)
        ENC_ATTN_KV = (
            (0, 2, 1),
            (enc.num_key_value_heads, enc.head_dim, enc.embed_dim),
            True,
        )
        # o_proj: (embed_dim, num_q_heads * head_dim) -> (num_q_heads, head_dim, embed_dim)
        # reshape to (embed_dim, num_q_heads, head_dim), permute (1, 2, 0)
        ENC_ATTN_OUT = (
            (1, 2, 0),
            (enc.embed_dim, enc.num_attention_heads, enc.head_dim),
            True,
        )

        # Decoder attention transforms (same structure as encoder, separate k/v)
        DEC_ATTN_Q = (
            (0, 2, 1),
            (dec.num_attention_heads, dec.head_dim, dec.embed_dim),
            True,
        )
        DEC_ATTN_KV = (
            (0, 2, 1),
            (dec.num_key_value_heads, dec.head_dim, dec.embed_dim),
            True,
        )
        DEC_ATTN_OUT = (
            (1, 2, 0),
            (dec.embed_dim, dec.num_attention_heads, dec.head_dim),
            True,
        )

        # MLP transforms - gating uses no transform (transpose_gating_einsum=True)
        MLP_GATE = None  # checkpoint is (hidden, embed), model is (2, hidden, embed) after stack
        MLP_DOWN = ((1, 0), None, False)  # transpose

        # Vision transforms
        VIT_LINEAR = ((1, 0), None, False)
        VIT_BIAS = None
        VIT_CONV = ((2, 3, 1, 0), None, False)  # OIHW -> HWIO
        VIT_POS_EMBED = None  # Will handle separately

    # Maping of checkpoint_keys -> (model_keys, Transform)
    return {
        r"model\.encoder\.embed_tokens\.weight": (
            "decoder.embedder.embedding",
            Transform.EMBED,
        ),
        r"model\.encoder\.embed_tokens\.eoi_embedding": (
            "decoder.embedder.eoi_embedding",
            Transform.EMBED,
        ),
        # === Encoder Layers ===
        r"model\.encoder\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (
            r"encoder.blocks.\1.attn.q_einsum.w",
            Transform.ENC_ATTN_Q,
        ),
        r"model\.encoder\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (
            r"encoder.blocks.\1.attn.kv_einsum.w.__stacked_k",
            Transform.ENC_ATTN_KV,
        ),
        r"model\.encoder\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (
            r"encoder.blocks.\1.attn.kv_einsum.w.__stacked_v",
            Transform.ENC_ATTN_KV,
        ),
        r"model\.encoder\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": (
            r"encoder.blocks.\1.attn.attn_vec_einsum.w",
            Transform.ENC_ATTN_OUT,
        ),
        r"model\.encoder\.layers\.([0-9]+)\.self_attn\.q_norm\.weight": (
            r"encoder.blocks.\1.attn._query_norm.scale",
            Transform.SCALE,
        ),
        r"model\.encoder\.layers\.([0-9]+)\.self_attn\.k_norm\.weight": (
            r"encoder.blocks.\1.attn._key_norm.scale",
            Transform.SCALE,
        ),
        r"model\.encoder\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": (
            r"encoder.blocks.\1.mlp.gating_einsum.w.__stacked_gate",
            Transform.MLP_GATE,
        ),
        r"model\.encoder\.layers\.([0-9]+)\.mlp\.up_proj\.weight": (
            r"encoder.blocks.\1.mlp.gating_einsum.w.__stacked_up",
            Transform.MLP_GATE,
        ),
        r"model\.encoder\.layers\.([0-9]+)\.mlp\.down_proj\.weight": (
            r"encoder.blocks.\1.mlp.linear.w",
            Transform.MLP_DOWN,
        ),
        r"model\.encoder\.layers\.([0-9]+)\.pre_self_attn_layernorm\.weight": (
            r"encoder.blocks.\1.pre_attention_norm.scale",
            Transform.SCALE,
        ),
        r"model\.encoder\.layers\.([0-9]+)\.post_self_attn_layernorm\.weight": (
            r"encoder.blocks.\1.post_attention_norm.scale",
            Transform.SCALE,
        ),
        r"model\.encoder\.layers\.([0-9]+)\.pre_feedforward_layernorm\.weight": (
            r"encoder.blocks.\1.pre_ffw_norm.scale",
            Transform.SCALE,
        ),
        r"model\.encoder\.layers\.([0-9]+)\.post_feedforward_layernorm\.weight": (
            r"encoder.blocks.\1.post_ffw_norm.scale",
            Transform.SCALE,
        ),
        r"model\.encoder\.norm\.weight": ("encoder.norm.scale", Transform.SCALE),
        # === Decoder Layers (uses separate einsum modules for k, v) ===
        # Note: Decoder embeddings are tied with encoder (shared embedder)
        r"model\.decoder\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (
            r"decoder.blocks.\1.attn.q_proj.w",
            Transform.DEC_ATTN_Q,
        ),
        r"model\.decoder\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (
            r"decoder.blocks.\1.attn.k_proj.w",
            Transform.DEC_ATTN_KV,
        ),
        r"model\.decoder\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (
            r"decoder.blocks.\1.attn.v_proj.w",
            Transform.DEC_ATTN_KV,
        ),
        r"model\.decoder\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": (
            r"decoder.blocks.\1.attn.o_proj.w",
            Transform.DEC_ATTN_OUT,
        ),
        r"model\.decoder\.layers\.([0-9]+)\.self_attn\.q_norm\.weight": (
            r"decoder.blocks.\1.attn.q_norm.scale",
            Transform.SCALE,
        ),
        r"model\.decoder\.layers\.([0-9]+)\.self_attn\.k_norm\.weight": (
            r"decoder.blocks.\1.attn.k_norm.scale",
            Transform.SCALE,
        ),
        r"model\.decoder\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": (
            r"decoder.blocks.\1.mlp.gating_einsum.w.__stacked_gate",
            Transform.MLP_GATE,
        ),
        r"model\.decoder\.layers\.([0-9]+)\.mlp\.up_proj\.weight": (
            r"decoder.blocks.\1.mlp.gating_einsum.w.__stacked_up",
            Transform.MLP_GATE,
        ),
        r"model\.decoder\.layers\.([0-9]+)\.mlp\.down_proj\.weight": (
            r"decoder.blocks.\1.mlp.linear.w",
            Transform.MLP_DOWN,
        ),
        r"model\.decoder\.layers\.([0-9]+)\.pre_self_attn_layernorm\.weight": (
            r"decoder.blocks.\1.pre_attention_norm.scale",
            Transform.SCALE,
        ),
        r"model\.decoder\.layers\.([0-9]+)\.post_self_attn_layernorm\.weight": (
            r"decoder.blocks.\1.post_attention_norm.scale",
            Transform.SCALE,
        ),
        r"model\.decoder\.layers\.([0-9]+)\.pre_feedforward_layernorm\.weight": (
            r"decoder.blocks.\1.pre_ffw_norm.scale",
            Transform.SCALE,
        ),
        r"model\.decoder\.layers\.([0-9]+)\.post_feedforward_layernorm\.weight": (
            r"decoder.blocks.\1.post_ffw_norm.scale",
            Transform.SCALE,
        ),
        r"model\.decoder\.norm\.weight": ("decoder.norm.scale", Transform.SCALE),
        # === Vision Tower ===
        r"model\.encoder\.vision_tower\.vision_model\.embeddings\.patch_embedding\.weight": (
            "encoder.vision_embedder.soft_tokenizer.embedding.kernel",
            Transform.VIT_CONV,
        ),
        r"model\.encoder\.vision_tower\.vision_model\.embeddings\.patch_embedding\.bias": (
            "encoder.vision_embedder.soft_tokenizer.embedding.bias",
            Transform.VIT_BIAS,
        ),
        r"model\.encoder\.vision_tower\.vision_model\.embeddings\.position_embedding\.weight": (
            "encoder.vision_embedder.soft_tokenizer.pos_embedding",
            Transform.VIT_POS_EMBED,
        ),
        r"model\.encoder\.vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.layer_norm1\.weight": (
            r"encoder.vision_embedder.soft_tokenizer.transformer.blocks.\1.layer_norm1.scale",
            Transform.SCALE,
        ),
        r"model\.encoder\.vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.layer_norm1\.bias": (
            r"encoder.vision_embedder.soft_tokenizer.transformer.blocks.\1.layer_norm1.bias",
            Transform.VIT_BIAS,
        ),
        r"model\.encoder\.vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.layer_norm2\.weight": (
            r"encoder.vision_embedder.soft_tokenizer.transformer.blocks.\1.layer_norm2.scale",
            Transform.SCALE,
        ),
        r"model\.encoder\.vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.layer_norm2\.bias": (
            r"encoder.vision_embedder.soft_tokenizer.transformer.blocks.\1.layer_norm2.bias",
            Transform.VIT_BIAS,
        ),
        r"model\.encoder\.vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (
            r"encoder.vision_embedder.soft_tokenizer.transformer.blocks.\1.mha.query.kernel",
            Transform.VIT_LINEAR,
        ),
        r"model\.encoder\.vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.self_attn\.q_proj\.bias": (
            r"encoder.vision_embedder.soft_tokenizer.transformer.blocks.\1.mha.query.bias",
            Transform.VIT_BIAS,
        ),
        r"model\.encoder\.vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (
            r"encoder.vision_embedder.soft_tokenizer.transformer.blocks.\1.mha.key.kernel",
            Transform.VIT_LINEAR,
        ),
        r"model\.encoder\.vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.self_attn\.k_proj\.bias": (
            r"encoder.vision_embedder.soft_tokenizer.transformer.blocks.\1.mha.key.bias",
            Transform.VIT_BIAS,
        ),
        r"model\.encoder\.vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (
            r"encoder.vision_embedder.soft_tokenizer.transformer.blocks.\1.mha.value.kernel",
            Transform.VIT_LINEAR,
        ),
        r"model\.encoder\.vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.self_attn\.v_proj\.bias": (
            r"encoder.vision_embedder.soft_tokenizer.transformer.blocks.\1.mha.value.bias",
            Transform.VIT_BIAS,
        ),
        r"model\.encoder\.vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.self_attn\.out_proj\.weight": (
            r"encoder.vision_embedder.soft_tokenizer.transformer.blocks.\1.mha.out.kernel",
            Transform.VIT_LINEAR,
        ),
        r"model\.encoder\.vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.self_attn\.out_proj\.bias": (
            r"encoder.vision_embedder.soft_tokenizer.transformer.blocks.\1.mha.out.bias",
            Transform.VIT_BIAS,
        ),
        r"model\.encoder\.vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.mlp\.fc1\.weight": (
            r"encoder.vision_embedder.soft_tokenizer.transformer.blocks.\1.mlp.dense1.kernel",
            Transform.VIT_LINEAR,
        ),
        r"model\.encoder\.vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.mlp\.fc1\.bias": (
            r"encoder.vision_embedder.soft_tokenizer.transformer.blocks.\1.mlp.dense1.bias",
            Transform.VIT_BIAS,
        ),
        r"model\.encoder\.vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.mlp\.fc2\.weight": (
            r"encoder.vision_embedder.soft_tokenizer.transformer.blocks.\1.mlp.dense2.kernel",
            Transform.VIT_LINEAR,
        ),
        r"model\.encoder\.vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.mlp\.fc2\.bias": (
            r"encoder.vision_embedder.soft_tokenizer.transformer.blocks.\1.mlp.dense2.bias",
            Transform.VIT_BIAS,
        ),
        r"model\.encoder\.vision_tower\.vision_model\.post_layernorm\.weight": (
            "encoder.vision_embedder.soft_tokenizer.transformer.encoder_norm.scale",
            Transform.SCALE,
        ),
        r"model\.encoder\.vision_tower\.vision_model\.post_layernorm\.bias": (
            "encoder.vision_embedder.soft_tokenizer.transformer.encoder_norm.bias",
            Transform.VIT_BIAS,
        ),
        # === Multi-modal Projector ===
        r"model\.encoder\.multi_modal_projector\.mm_input_projection_weight": (
            "encoder.vision_embedder.soft_tokens_embedder.mm_input_projection.w",
            Transform.NONE,
        ),
        r"model\.encoder\.multi_modal_projector\.mm_soft_emb_norm\.weight": (
            "encoder.vision_embedder.soft_tokens_embedder.mm_soft_embedding_norm.scale",
            Transform.SCALE,
        ),
    }, vis


def _torch_key_to_jax_key(mapping, source_key):
    """Convert checkpoint key to model key using regex substitution."""
    subs = [
        (re.sub(pat, repl, source_key), transform)
        for pat, (repl, transform) in mapping.items()
        if re.match(pat, source_key)
    ]
    if len(subs) != 1:
        if len(subs) == 0:
            return None, None
        raise ValueError(f"Multiple matches for key: {source_key}")
    return subs[0]


def _assign_weights(keys, tensor, state_dict, st_key, transform, sharding_dict):
    """Recursively descend into state_dict and assign the (possibly transformed) tensor."""
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
        if sharding_dict is not None:
            state_dict[key] = jax.device_put(tensor, sharding_dict[key])
        else:
            state_dict[key] = jax.device_put(tensor)
    else:
        next_sharding = sharding_dict[key] if sharding_dict is not None else None
        _assign_weights(rest, tensor, state_dict[key], st_key, transform, next_sharding)


def _stoi(s):
    """String to int if possible."""
    try:
        return int(s)
    except ValueError:
        return s


def create_model_from_safe_tensors(
    file_dir: str, cfg: T5Gemma2Config, mesh: jax.sharding.Mesh | None = None
) -> model_lib.T5Gemma2:
    """Load tensors from the safetensors file and create a T5Gemma2 model."""
    files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors found in {file_dir}")

    # Create actual model (not eval_shape) to preserve non-parameter arrays
    model = model_lib.T5Gemma2(cfg, rngs=nnx.Rngs(params=0, dropout=0))
    graph_def, state = nnx.split(model)
    state_dict = state.to_pure_dict()
    sharding = nnx.get_named_sharding(state, mesh).to_pure_dict() if mesh is not None else None

    key_mapping, vis_cfg = _get_key_and_transform_mapping(cfg)
    conversion_errors = []

    # Buffer for stacked parameters (k+v for encoder, gate+up for mlp)
    stacked_buffers = {}

    # Vision config for reshaping
    vit_num_heads = vis_cfg.num_heads if vis_cfg else 16
    vit_head_dim = vis_cfg.width // vis_cfg.num_heads if vis_cfg else 72

    for f in files:
        with safetensors.safe_open(f, framework="numpy") as sf:
            for torch_key in sf.keys():
                tensor = sf.get_tensor(torch_key)

                jax_key, transform = _torch_key_to_jax_key(key_mapping, torch_key)
                if jax_key is None:
                    conversion_errors.append(f"No mapping found for: {torch_key}")
                    continue

                # Skip vision weights if vision is not enabled
                if "vision_embedder" in jax_key and cfg.encoder.vision_config is None:
                    continue

                # Handle stacked parameters (use __stacked_ prefix to avoid conflicts)
                if ".__stacked_k" in jax_key or ".__stacked_v" in jax_key:
                    if ".__stacked_k" in jax_key:
                        base_key = jax_key.replace(".__stacked_k", "")
                        idx = 0
                    else:
                        base_key = jax_key.replace(".__stacked_v", "")
                        idx = 1

                    # Apply transform
                    if transform.value is not None:
                        permute, reshape, reshape_first = transform.value
                        if reshape_first and reshape is not None:
                            tensor = tensor.reshape(reshape)
                        if permute:
                            tensor = tensor.transpose(permute)
                        if not reshape_first and reshape is not None:
                            tensor = tensor.reshape(reshape)

                    if base_key not in stacked_buffers:
                        stacked_buffers[base_key] = {}
                    stacked_buffers[base_key][idx] = tensor
                    continue

                if ".__stacked_gate" in jax_key or ".__stacked_up" in jax_key:
                    if ".__stacked_gate" in jax_key:
                        base_key = jax_key.replace(".__stacked_gate", "")
                        idx = 0
                    else:
                        base_key = jax_key.replace(".__stacked_up", "")
                        idx = 1

                    # No transform for MLP gating (already correct shape)
                    if base_key not in stacked_buffers:
                        stacked_buffers[base_key] = {}
                    stacked_buffers[base_key][idx] = tensor
                    continue

                # Special handling for vision attention weights
                if "mha.query.kernel" in jax_key or "mha.key.kernel" in jax_key or "mha.value.kernel" in jax_key:
                    # q/k/v: (num_heads * head_dim, in_features) -> (in_features, num_heads, head_dim)
                    tensor = tensor.T  # transpose
                    in_features = tensor.shape[0]
                    tensor = tensor.reshape(in_features, vit_num_heads, vit_head_dim)
                    keys = [_stoi(k) for k in jax_key.split(".")]
                    try:
                        _assign_weights(keys, tensor, state_dict, torch_key, None, sharding)
                    except Exception as e:
                        full_jax_key = ".".join([str(k) for k in keys])
                        conversion_errors.append(f"Failed '{torch_key}' -> '{full_jax_key}': {type(e).__name__}: {e}")
                    continue

                if "mha.out.kernel" in jax_key:
                    # out: (out_features, num_heads * head_dim) -> (num_heads, head_dim, out_features)
                    tensor = tensor.T  # transpose to (num_heads * head_dim, out_features)
                    out_features = tensor.shape[1]
                    tensor = tensor.reshape(vit_num_heads, vit_head_dim, out_features)
                    keys = [_stoi(k) for k in jax_key.split(".")]
                    try:
                        _assign_weights(keys, tensor, state_dict, torch_key, None, sharding)
                    except Exception as e:
                        full_jax_key = ".".join([str(k) for k in keys])
                        conversion_errors.append(f"Failed '{torch_key}' -> '{full_jax_key}': {type(e).__name__}: {e}")
                    continue

                if "mha.query.bias" in jax_key or "mha.key.bias" in jax_key or "mha.value.bias" in jax_key:
                    # bias: (num_heads * head_dim,) -> (num_heads, head_dim)
                    tensor = tensor.reshape(vit_num_heads, vit_head_dim)
                    keys = [_stoi(k) for k in jax_key.split(".")]
                    try:
                        _assign_weights(keys, tensor, state_dict, torch_key, None, sharding)
                    except Exception as e:
                        full_jax_key = ".".join([str(k) for k in keys])
                        conversion_errors.append(f"Failed '{torch_key}' -> '{full_jax_key}': {type(e).__name__}: {e}")
                    continue

                # Special handling for position embedding (add batch dim)
                if "pos_embedding" in jax_key and "position_embedding" in torch_key:
                    tensor = tensor[None, :, :]  # (4096, 1152) -> (1, 4096, 1152)
                    keys = [_stoi(k) for k in jax_key.split(".")]
                    try:
                        _assign_weights(keys, tensor, state_dict, torch_key, None, sharding)
                    except Exception as e:
                        full_jax_key = ".".join([str(k) for k in keys])
                        conversion_errors.append(f"Failed '{torch_key}' -> '{full_jax_key}': {type(e).__name__}: {e}")
                    continue

                keys = [_stoi(k) for k in jax_key.split(".")]
                try:
                    _assign_weights(keys, tensor, state_dict, torch_key, transform.value, sharding)
                except Exception as e:
                    full_jax_key = ".".join([str(k) for k in keys])
                    conversion_errors.append(f"Failed '{torch_key}' -> '{full_jax_key}': {type(e).__name__}: {e}")
        gc.collect()

    # Now assign stacked parameters
    for base_key, indices in stacked_buffers.items():
        if 0 not in indices or 1 not in indices:
            conversion_errors.append(f"Incomplete stacked parameter: {base_key}")
            continue

        # Stack the tensors
        stacked = jax.numpy.stack([indices[0], indices[1]], axis=0)

        keys = [_stoi(k) for k in base_key.split(".")]
        try:
            _assign_weights(keys, stacked, state_dict, f"stacked:{base_key}", None, sharding)
        except Exception as e:
            full_jax_key = ".".join([str(k) for k in keys])
            conversion_errors.append(f"Failed stacked '{base_key}' -> '{full_jax_key}': {type(e).__name__}: {e}")

    if conversion_errors:
        full_error_log = "\n".join(conversion_errors)
        raise RuntimeError(f"Encountered {len(conversion_errors)} weight conversion errors:\n{full_error_log}")

    gc.collect()
    return nnx.merge(graph_def, state_dict)
