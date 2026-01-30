import logging
import re

import h5py
import jax
import jax.numpy as jnp
from etils import epath
from flax import nnx

from bonsai.models.convnext import modeling as model_lib


def _get_key_and_transform_mapping():
    """
    Creates the mapping from the TF/Keras .h5 keys to the JAX/NNX keys.
    The transform is `None` because TF and JAX have the same weight shapes.
    """
    prefix = r"^convnext/tf_conv_next_for_image_classification/convnext/"
    classifier_prefix = r"^classifier/tf_conv_next_for_image_classification/classifier/"

    mapping = {
        # embedding
        prefix + r"embeddings/patch_embeddings/kernel:0$": ("embedding_layer.layers.0.kernel", None),
        prefix + r"embeddings/patch_embeddings/bias:0$": ("embedding_layer.layers.0.bias", None),
        prefix + r"embeddings/layernorm/beta:0$": ("embedding_layer.layers.1.bias", None),
        prefix + r"embeddings/layernorm/gamma:0$": ("embedding_layer.layers.1.scale", None),
        # stages
        prefix + r"encoder/stages\.([1-3])/downsampling_layer\.0/beta:0$": (
            r"stages.\1.downsample_layers.0.bias",
            None,
        ),
        prefix + r"encoder/stages\.([1-3])/downsampling_layer\.0/gamma:0$": (
            r"stages.\1.downsample_layers.0.scale",
            None,
        ),
        prefix + r"encoder/stages\.([1-3])/downsampling_layer\.1/kernel:0$": (
            r"stages.\1.downsample_layers.1.kernel",
            None,
        ),
        prefix + r"encoder/stages\.([1-3])/downsampling_layer\.1/bias:0$": (
            r"stages.\1.downsample_layers.1.bias",
            None,
        ),
        prefix + r"encoder/stages\.([0-3])/layers\.([0-9]+)/dwconv/kernel:0$": (
            r"stages.\1.layers.\2.dwconv.kernel",
            None,
        ),
        prefix + r"encoder/stages\.([0-3])/layers\.([0-9]+)/dwconv/bias:0$": (
            r"stages.\1.layers.\2.dwconv.bias",
            None,
        ),
        prefix + r"encoder/stages\.([0-3])/layers\.([0-9]+)/layernorm/beta:0$": (
            r"stages.\1.layers.\2.norm.bias",
            None,
        ),
        prefix + r"encoder/stages\.([0-3])/layers\.([0-9]+)/layernorm/gamma:0$": (
            r"stages.\1.layers.\2.norm.scale",
            None,
        ),
        prefix + r"encoder/stages\.([0-3])/layers\.([0-9]+)/pwconv1/kernel:0$": (
            r"stages.\1.layers.\2.pwconv1.kernel",
            None,
        ),
        prefix + r"encoder/stages\.([0-3])/layers\.([0-9]+)/pwconv1/bias:0$": (
            r"stages.\1.layers.\2.pwconv1.bias",
            None,
        ),
        prefix + r"encoder/stages\.([0-3])/layers\.([0-9]+)/pwconv2/kernel:0$": (
            r"stages.\1.layers.\2.pwconv2.kernel",
            None,
        ),
        prefix + r"encoder/stages\.([0-3])/layers\.([0-9]+)/pwconv2/bias:0$": (
            r"stages.\1.layers.\2.pwconv2.bias",
            None,
        ),
        prefix + r"encoder/stages\.([0-3])/layers\.([0-9]+)/layer_scale_parameter:0$": (
            r"stages.\1.layers.\2.gamma",
            None,
        ),
        # head
        prefix + r"layernorm/beta:0$": ("norm.bias", None),
        prefix + r"layernorm/gamma:0$": ("norm.scale", None),
        classifier_prefix + r"kernel:0$": ("head.kernel", None),  # done
        classifier_prefix + r"bias:0$": ("head.bias", None),  # done
    }
    return mapping


def _h5_key_to_jax_key(mapping, source_key):
    """Map a h5 key to exactly one JAX key & transform, else warn/error."""
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
        state_dict[key] = jnp.array(tensor)
    else:
        _assign_weights(rest, tensor, state_dict[key], st_key, transform)


def _stoi(s):
    try:
        return int(s)
    except ValueError:
        return s


def create_convnext_from_pretrained(
    file_dir: str,
    cfg: model_lib.ModelConfig,
    *,
    mesh: jax.sharding.Mesh | None = None,
):
    """
    Load h5 weights from a file, then convert & merge into a flax.nnx ResNet model.
    Returns:
      A flax.nnx.Model instance with loaded parameters.
    """
    files = list(epath.Path(file_dir).expanduser().glob("*.h5"))
    if not files:
        raise ValueError(f"No .h5 files found in {file_dir}")

    state_dict = {}
    for f in files:
        with h5py.File(f, "r") as hf:
            # Recursively visit all objects (groups and datasets)
            hf.visititems(
                lambda name, obj: (
                    # If it's a Dataset (a tensor), read it (obj[()]) and add to dict
                    state_dict.update({name: obj[()]}) if isinstance(obj, h5py.Dataset) else None
                )
            )

    model = model_lib.ConvNeXt(cfg=cfg, rngs=nnx.Rngs(params=0))
    graph_def, abs_state = nnx.split(model)
    jax_state = nnx.to_pure_dict(abs_state)

    mapping = _get_key_and_transform_mapping()

    conversion_errors = []
    for h5_key, tensor in state_dict.items():
        jax_key, transform = _h5_key_to_jax_key(mapping, h5_key)
        if jax_key is None:
            continue

        keys = [_stoi(k) for k in jax_key.split(".")]
        try:
            _assign_weights(keys, tensor, jax_state, h5_key, transform)
        except Exception as e:
            full_jax_key = ".".join([str(k) for k in keys])
            conversion_errors.append(f"Failed to assign '{h5_key}' to '{full_jax_key}': {type(e).__name__}: {e}")

    if conversion_errors:
        full_error_log = "\n".join(conversion_errors)
        raise RuntimeError(f"Encountered {len(conversion_errors)} weight conversion errors. Log:\n{full_error_log}")

    if mesh is not None:
        sharding = nnx.to_pure_dict(nnx.get_named_sharding(abs_state, mesh))
        jax_state = jax.device_put(jax_state, sharding)
    else:
        jax_state = jax.device_put(jax_state, jax.devices()[0])

    return nnx.merge(graph_def, jax_state)
