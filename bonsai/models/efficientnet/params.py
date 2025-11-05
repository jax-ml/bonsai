import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from bonsai.models.efficientnet import modeling as model_lib


def create_model(
    cfg: model_lib.ModelCfg,
    block_configs: model_lib.BlockConfigs,
    rngs: nnx.Rngs,
    mesh: jax.sharding.Mesh | None = None,
) -> model_lib.EfficientNet:
    """Generic EfficientNet creator."""
    model = model_lib.EfficientNet(cfg, block_configs=block_configs, rngs=rngs)
    if mesh is not None:
        graph_def, state = nnx.split(model)
        sharding = nnx.get_named_sharding(model, mesh)
        state = jax.device_put(state, sharding)
        return nnx.merge(graph_def, state)
    else:
        return model


def EfficientNetB0(num_classes: int, rngs: nnx.Rngs, mesh: jax.sharding.Mesh | None = None):
    return create_model(model_lib.ModelCfg.b0(num_classes), model_lib.BlockConfigs.default_block_config(), rngs, mesh)


def EfficientNetB1(num_classes: int, rngs: nnx.Rngs, mesh: jax.sharding.Mesh | None = None):
    return create_model(model_lib.ModelCfg.b1(num_classes), model_lib.BlockConfigs.default_block_config(), rngs, mesh)


def EfficientNetB2(num_classes: int, rngs: nnx.Rngs, mesh: jax.sharding.Mesh | None = None):
    return create_model(model_lib.ModelCfg.b2(num_classes), model_lib.BlockConfigs.default_block_config(), rngs, mesh)


def EfficientNetB3(num_classes: int, rngs: nnx.Rngs, mesh: jax.sharding.Mesh | None = None):
    return create_model(model_lib.ModelCfg.b3(num_classes), model_lib.BlockConfigs.default_block_config(), rngs, mesh)


def EfficientNetB4(num_classes: int, rngs: nnx.Rngs, mesh: jax.sharding.Mesh | None = None):
    return create_model(model_lib.ModelCfg.b4(num_classes), model_lib.BlockConfigs.default_block_config(), rngs, mesh)


def EfficientNetB5(num_classes: int, rngs: nnx.Rngs, mesh: jax.sharding.Mesh | None = None):
    return create_model(model_lib.ModelCfg.b5(num_classes), model_lib.BlockConfigs.tf_block_config(), rngs, mesh)


def EfficientNetB6(num_classes: int, rngs: nnx.Rngs, mesh: jax.sharding.Mesh | None = None):
    return create_model(model_lib.ModelCfg.b6(num_classes), model_lib.BlockConfigs.tf_block_config(), rngs, mesh)


def EfficientNetB7(num_classes: int, rngs: nnx.Rngs, mesh: jax.sharding.Mesh | None = None):
    return create_model(model_lib.ModelCfg.b7(num_classes), model_lib.BlockConfigs.tf_block_config(), rngs, mesh)


def get_timm_pretrained_weights(model_name: str = "efficientnet_b0"):
    """
    Downloads and returns pre-trained EfficientNet weights from the 'timm' library.

    This requires PyTorch and timm to be installed:
    !pip install -q torch timm

    Returns:
      A dictionary mapping pre-trained layer names to NumPy arrays.
    """
    import timm
    import torch

    # Map to correct timm model names. Some larger models use specific checkpoints.
    timm_name_map = {
        "efficientnet_b0": "efficientnet_b0",
        "efficientnet_b1": "efficientnet_b1",
        "efficientnet_b2": "efficientnet_b2",
        "efficientnet_b3": "efficientnet_b3",
        "efficientnet_b4": "efficientnet_b4",
        "efficientnet_b5": "tf_efficientnet_b5_ap",  # AdvProp
        "efficientnet_b6": "tf_efficientnet_b6_ap",  # AdvProp
        "efficientnet_b7": "tf_efficientnet_b7_ap",  # AdvProp
    }
    timm_model_name = timm_name_map.get(model_name)
    if not timm_model_name:
        raise ValueError(f"No timm mapping for '{model_name}'. Available models are: {list(timm_name_map.keys())}")

    m = timm.create_model(timm_model_name, pretrained=True)
    m.eval()

    # Convert weights to a dictionary of numpy arrays
    return {k: v.numpy() for k, v in m.state_dict().items()}


def create_name_map(cfg: model_lib.ModelCfg):
    """
    Creates a mapping from the JAX model's parameter names to the timm model's names.
    This version correctly handles the different architectures of the MBConv blocks.
    """
    bn_map = {
        "scale": "weight",
        "bias": "bias",
        "mean": "running_mean",
        "var": "running_var",
    }
    name_map = {}

    # 1. Stem
    name_map["stem_conv"] = {"kernel": "conv_stem.weight"}
    name_map["stem_bn"] = {jax_n: f"bn1.{timm_n}" for jax_n, timm_n in bn_map.items()}

    # 2. Blocks
    block_configs = model_lib.BlockConfigs.default_block_config().items
    total_jax_block_idx = 0
    for i, bc in enumerate(block_configs):
        num_repeat = model_lib.round_repeats(bc.num_repeat, cfg.depth_coefficient)
        for j in range(num_repeat):
            jax_base = f"blocks.{total_jax_block_idx}"
            timm_base = f"blocks.{i}.{j}"

            if bc.expand_ratio != 1:
                name_map[f"{jax_base}.expand_conv"] = {"kernel": f"{timm_base}.conv_pw.weight"}
                name_map[f"{jax_base}.bn0"] = {jax_n: f"{timm_base}.bn1.{timm_n}" for jax_n, timm_n in bn_map.items()}
                name_map[f"{jax_base}.depthwise_conv"] = {"kernel": f"{timm_base}.conv_dw.weight"}
                name_map[f"{jax_base}.bn1"] = {jax_n: f"{timm_base}.bn2.{timm_n}" for jax_n, timm_n in bn_map.items()}
                name_map[f"{jax_base}.project_conv"] = {"kernel": f"{timm_base}.conv_pwl.weight"}
                name_map[f"{jax_base}.bn2"] = {jax_n: f"{timm_base}.bn3.{timm_n}" for jax_n, timm_n in bn_map.items()}
            else:  # This block handles the first MBConv layer where expand_ratio = 1
                name_map[f"{jax_base}.depthwise_conv"] = {"kernel": f"{timm_base}.conv_dw.weight"}
                name_map[f"{jax_base}.bn1"] = {jax_n: f"{timm_base}.bn1.{timm_n}" for jax_n, timm_n in bn_map.items()}
                name_map[f"{jax_base}.project_conv"] = {
                    "kernel": f"{timm_base}.conv_pw.weight"  # <--- THIS IS THE CORRECTED LINE
                }
                name_map[f"{jax_base}.bn2"] = {jax_n: f"{timm_base}.bn2.{timm_n}" for jax_n, timm_n in bn_map.items()}

            # Squeeze-and-Excitation is the same for both block types
            name_map[f"{jax_base}.se.conv1"] = {
                "kernel": f"{timm_base}.se.conv_reduce.weight",
                "bias": f"{timm_base}.se.conv_reduce.bias",
            }
            name_map[f"{jax_base}.se.conv2"] = {
                "kernel": f"{timm_base}.se.conv_expand.weight",
                "bias": f"{timm_base}.se.conv_expand.bias",
            }

            total_jax_block_idx += 1

    # 3. Head
    name_map["head_conv"] = {"kernel": "conv_head.weight"}
    name_map["head_bn"] = {jax_n: f"bn2.{timm_n}" for jax_n, timm_n in bn_map.items()}
    name_map["classifier"] = {
        "kernel": "classifier.weight",
        "bias": "classifier.bias",
    }

    return name_map


def load_pretrained_weights(model: model_lib.EfficientNet, pretrained_weights: dict):
    """
    Loads pre-trained weights by directly modifying the JAX model's attributes in-place.
    """
    name_map = create_name_map(model.cfg)

    timm_to_jax_map = {}
    for jax_module_path, params_map in name_map.items():
        for jax_param_name, timm_param_name in params_map.items():
            path_parts = jax_module_path.split(".")
            path_tuple = (*path_parts, jax_param_name)
            timm_to_jax_map[timm_param_name] = path_tuple

    for timm_name, weight in pretrained_weights.items():
        if timm_name not in timm_to_jax_map:
            continue

        path = timm_to_jax_map[timm_name]
        weight_np = weight
        param_name = path[-1]

        if param_name == "kernel" and len(weight_np.shape) == 4:
            weight_np = np.transpose(weight_np, (2, 3, 1, 0))
        if param_name == "kernel" and len(weight_np.shape) == 2:
            weight_np = np.transpose(weight_np, (1, 0))

        target_module = model
        for part in path[:-1]:
            if part.isdigit():
                target_module = target_module[int(part)]
            else:
                target_module = getattr(target_module, part)

        param_to_update = getattr(target_module, param_name)
        if param_to_update.shape != weight_np.shape:
            raise ValueError(
                f"Shape mismatch for '{'.'.join(path)}': "
                f"JAX model has {param_to_update.shape}, "
                f"pre-trained weight has {weight_np.shape}."
            )

        param_to_update.value = jnp.array(weight_np)

    return model
