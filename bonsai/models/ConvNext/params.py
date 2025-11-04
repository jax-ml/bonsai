import jax 
from flax import nnx 
import logging
from bonsai.models.ConvNext import modeling as model_lib
import re
from typing import Callable
from etils import epath
import h5py 

def _get_key_and_transform_mapping():
    """
    Creates the mapping from the TF/Keras .h5 keys to the JAX/NNX keys.
    The transform is `None` because TF and JAX have the same weight shapes.
    """
    # Prefix for the main 'convnext' model parts
    convnext_prefix = r"^convnext/tf_conv_next_for_image_classification/convnext/"
    
    # A separate prefix for the final 'classifier' head
    classifier_prefix = r"^classifier/tf_conv_next_for_image_classification/classifier/"
    
    mapping = {
        # --- Stem (downsample_layers.0) ---
        r"" + convnext_prefix + r"embeddings/patch_embeddings/kernel:0$": (
            "downsample_layers.0.layers.0.kernel", None
        ),
        r"" + convnext_prefix + r"embeddings/patch_embeddings/bias:0$": (
            "downsample_layers.0.layers.0.bias", None
        ),
        # This is the 'layernorm' right after patch_embeddings
        r"" + convnext_prefix + r"embeddings/layernorm/beta:0$": (
            "downsample_layers.0.layers.1.bias", None  # Keras 'beta' is 'bias'
        ),
        r"" + convnext_prefix + r"embeddings/layernorm/gamma:0$": (
            "downsample_layers.0.layers.1.scale", None # Keras 'gamma' is 'scale'
        ),

        # --- Downsampling Layers (Stages 1, 2, 3) ---
        r"" + convnext_prefix + r"encoder/stages\.([1-3])/downsampling_layer\.0/beta:0$": (
            r"downsample_layers.\1.layers.0.bias", None
        ),
        r"" + convnext_prefix + r"encoder/stages\.([1-3])/downsampling_layer\.0/gamma:0$": (
            r"downsample_layers.\1.layers.0.scale", None
        ),
        r"" + convnext_prefix + r"encoder/stages\.([1-3])/downsampling_layer\.1/kernel:0$": (
            r"downsample_layers.\1.layers.1.kernel", None
        ),
        r"" + convnext_prefix + r"encoder/stages\.([1-3])/downsampling_layer\.1/bias:0$": (
            r"downsample_layers.\1.layers.1.bias", None
        ),

        # --- Main Blocks (All Stages 0-3, All Layers 0-N) ---
        r"" + convnext_prefix + r"encoder/stages\.([0-3])/layers\.([0-9]+)/dwconv/kernel:0$": (
            r"stages.\1.\2.dwconv.kernel", None
        ),
        r"" + convnext_prefix + r"encoder/stages\.([0-3])/layers\.([0-9]+)/dwconv/bias:0$": (
            r"stages.\1.\2.dwconv.bias", None
        ),
        r"" + convnext_prefix + r"encoder/stages\.([0-3])/layers\.([0-9]+)/layernorm/beta:0$": (
            r"stages.\1.\2.norm.bias", None
        ),
        r"" + convnext_prefix + r"encoder/stages\.([0-3])/layers\.([0-9]+)/layernorm/gamma:0$": (
            r"stages.\1.\2.norm.scale", None
        ),
        r"" + convnext_prefix + r"encoder/stages\.([0-3])/layers\.([0-9]+)/pwconv1/kernel:0$": (
            r"stages.\1.\2.pwconv1.kernel", None
        ),
        r"" + convnext_prefix + r"encoder/stages\.([0-3])/layers\.([0-9]+)/pwconv1/bias:0$": (
            r"stages.\1.\2.pwconv1.bias", None
        ),
        r"" + convnext_prefix + r"encoder/stages\.([0-3])/layers\.([0-9]+)/pwconv2/kernel:0$": (
            r"stages.\1.\2.pwconv2.kernel", None
        ),
        r"" + convnext_prefix + r"encoder/stages\.([0-3])/layers\.([0-9]+)/pwconv2/bias:0$": (
            r"stages.\1.\2.pwconv2.bias", None
        ),
        # This is the 'gamma' param in your nnx.Block
        r"" + convnext_prefix + r"encoder/stages\.([0-3])/layers\.([0-9]+)/layer_scale_parameter:0$": (
            r"stages.\1.\2.gamma", None
        ),
        
        # --- Head (Final Norm and Linear Layer) ---
        # Final LayerNorm before the classifier
        r"" + convnext_prefix + r"layernorm/beta:0$": ("norm.bias", None),
        r"" + convnext_prefix + r"layernorm/gamma:0$": ("norm.scale", None),
        
        # Final Linear 'head' layer (note the different prefix)
        r"" + classifier_prefix + r"kernel:0$": ("head.kernel", None),
        r"" + classifier_prefix + r"bias:0$": ("head.bias", None),
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

def _assign_weights(keys, tensor, state_dict, h5_key, transform):
    """Recursively descend into state_dict and assign the (possibly permuted/reshaped) tensor."""
    key, *rest = keys
    if not rest:
        if transform is not None:
            permute, reshape = transform
            if permute:
                tensor = tensor.transpose(permute)
            if reshape:
                tensor = tensor.reshape(reshape)
        
        # Ensure shapes match before assigning
        if key not in state_dict:
            raise KeyError(f"JAX key {key} (from {h5_key}) not found in model state.")
        if tensor.shape != state_dict[key].shape:
            raise ValueError(
                f"Shape mismatch for {h5_key} -> {'.'.join(map(str, keys))}:\n"
                f"H5 shape: {tensor.shape} vs Model shape: {state_dict[key].shape}"
            )
        
        state_dict[key] = tensor
    else:
        # Recurse into the nested dictionary/list
        _assign_weights(rest, tensor, state_dict[key], h5_key, transform)


def _stoi(s):
    try:
        return int(s)
    except ValueError:
        return s
    

def _create_convnext_from_pretrained(
        model_cls: Callable[...,model_lib.ConvNeXt],
        file_dir: str, 
        num_classes: int = 1000, 
        model_name: str = None,
        *,
        mesh: jax.sharding.Mesh | None = None
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
        with h5py.File(f, 'r') as hf:
            # Recursively visit all objects (groups and datasets)
            hf.visititems(
                lambda name, obj: (
                    # If it's a Dataset (a tensor), read it (obj[()]) and add to dict
                    state_dict.update({name: obj[()]}) if isinstance(obj, h5py.Dataset) else None
                )
            )

    model =model_cls(num_classes=num_classes,rngs=nnx.Rngs(params=0))
    graph_def , abs_state = nnx.split(model)
    jax_state = abs_state.to_pure_dict()

    mapping=_get_key_and_transform_mapping()
    
    # Keep track of unassigned JAX keys to warn the user
    assigned_jax_keys = set()

    for h5_key, tensor in state_dict.items():
        jax_key,transform =_h5_key_to_jax_key(mapping, h5_key)
        if jax_key is None:
            continue

        keys = [_stoi(k) for k in jax_key.split(".")]
        try:
            _assign_weights(keys,tensor,jax_state,h5_key,transform)
            assigned_jax_keys.add(jax_key)
        except (KeyError, ValueError) as e:
            logging.error(f"Failed to assign weight for {h5_key}:\n{e}")

    if mesh is not None:
        sharding = nnx.get_named_sharding(abs_state, mesh).to_pure_dict()
        jax_state = jax.device_put(jax_state, sharding)
    else:
        jax_state = jax.device_put(jax_state, jax.devices()[0])

    return nnx.merge(graph_def, jax_state)