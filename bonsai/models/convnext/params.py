import h5py
from etils import epath
from flax import nnx

from bonsai.models.convnext import modeling as model_lib
from bonsai.utils.params import stoi, safetensors_key_to_bonsai_key, assign_weights_from_eval_shape


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


def create_convnext_from_pretrained(
    file_dir: str,
    cfg: model_lib.ModelConfig,
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

    model = nnx.eval_shape(lambda: model_lib.ConvNeXt(cfg=cfg, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(model)
    jax_state = nnx.to_pure_dict(abs_state)

    mapping = _get_key_and_transform_mapping()

    conversion_errors = []
    for h5_key, tensor in state_dict.items():
        jax_key, transform = safetensors_key_to_bonsai_key(mapping, h5_key)
        if jax_key is None:
            continue

        keys = [stoi(k) for k in jax_key.split(".")]
        try:
            assign_weights_from_eval_shape(keys, tensor, jax_state, h5_key, transform)
        except Exception as e:
            full_jax_key = ".".join([str(k) for k in keys])
            conversion_errors.append(f"Failed to assign '{h5_key}' to '{full_jax_key}': {type(e).__name__}: {e}")

    if conversion_errors:
        full_error_log = "\n".join(conversion_errors)
        raise RuntimeError(f"Encountered {len(conversion_errors)} weight conversion errors. Log:\n{full_error_log}")

    return nnx.merge(graph_def, jax_state)
