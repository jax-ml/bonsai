import jax.numpy as jnp
import numpy as np
from flax import nnx

from . import modeling as model_lib
from flax.traverse_util import flatten_dict, unflatten_dict

def get_timm_pretrained_weights(model_name: str = "efficientnet_b0"):
    """
    Downloads and returns pre-trained EfficientNet weights from the 'timm' library.

    This requires PyTorch and timm to be installed:
    !pip install -q torch timm

    Returns:
      A dictionary mapping pre-trained layer names to NumPy arrays.
    """
    import torch
    import timm

    print(f"Fetching '{model_name}' weights from timm...")
    m = timm.create_model(model_name, pretrained=True)
    m.eval()
    
    # Convert weights to a dictionary of numpy arrays
    return {k: v.numpy() for k, v in m.state_dict().items()}


# --- Core Name-Mapping Logic ---

def create_name_map(cfg: model_lib.ModelCfg):
    """
    Creates a mapping from the JAX model's parameter names to the timm model's names.
    This version correctly handles the different architectures of the MBConv blocks.
    """
    
    bn_map = {"scale": "weight", "bias": "bias", "mean": "running_mean", "var": "running_var"}
    name_map = {}
    
    # 1. Stem
    name_map["stem_conv"] = {"kernel": "conv_stem.weight"}
    name_map["stem_bn"] = {jax_n: f"bn1.{timm_n}" for jax_n, timm_n in bn_map.items()}

    # 2. Blocks (with special handling for the first block type)
    block_configs = [
        model_lib.BlockConfig(32, 16, 3, 1, 1, 1, 0.25),
        model_lib.BlockConfig(16, 24, 3, 2, 6, 2, 0.25),
        model_lib.BlockConfig(24, 40, 5, 2, 6, 2, 0.25),
        model_lib.BlockConfig(40, 80, 3, 3, 6, 2, 0.25),
        model_lib.BlockConfig(80, 112, 5, 3, 6, 1, 0.25),
        model_lib.BlockConfig(112, 192, 5, 4, 6, 2, 0.25),
        model_lib.BlockConfig(192, 320, 3, 1, 6, 1, 0.25),
    ]
    
    total_jax_block_idx = 0
    for i, bc in enumerate(block_configs):
        num_repeat = model_lib.round_repeats(bc.num_repeat, cfg.depth_coefficient)
        for j in range(num_repeat):
            jax_base = f"blocks.{total_jax_block_idx}"
            timm_base = f"blocks.{i}.{j}"
            
            if bc.expand_ratio != 1:
                # --- Mapping for STANDARD blocks (with expansion) ---
                # (This part is correct and remains unchanged)
                name_map[f"{jax_base}.expand_conv"] = {"kernel": f"{timm_base}.conv_pw.weight"}
                name_map[f"{jax_base}.bn0"] = {jax_n: f"{timm_base}.bn1.{timm_n}" for jax_n, timm_n in bn_map.items()}
                name_map[f"{jax_base}.depthwise_conv"] = {"kernel": f"{timm_base}.conv_dw.weight"}
                name_map[f"{jax_base}.bn1"] = {jax_n: f"{timm_base}.bn2.{timm_n}" for jax_n, timm_n in bn_map.items()}
                name_map[f"{jax_base}.project_conv"] = {"kernel": f"{timm_base}.conv_pwl.weight"}
                name_map[f"{jax_base}.bn2"] = {jax_n: f"{timm_base}.bn3.{timm_n}" for jax_n, timm_n in bn_map.items()}
            else:
                # --- CORRECTED Mapping for the FIRST block type (no expansion) ---
                # JAX layers: depthwise_conv, bn1, se, project_conv, bn2
                # timm EdgeResidual layers: conv_dw, bn1, se, conv_pwl, bn2
                name_map[f"{jax_base}.depthwise_conv"] = {"kernel": f"{timm_base}.conv_dw.weight"}
                name_map[f"{jax_base}.bn1"] = {jax_n: f"{timm_base}.bn1.{timm_n}" for jax_n, timm_n in bn_map.items()}
                name_map[f"{jax_base}.project_conv"] = {"kernel": f"{timm_base}.conv_pwl.weight"}
                name_map[f"{jax_base}.bn2"] = {jax_n: f"{timm_base}.bn2.{timm_n}" for jax_n, timm_n in bn_map.items()}

            # Squeeze-and-Excitation is the same for both block types
            name_map[f"{jax_base}.se.fc1"] = {"kernel": f"{timm_base}.se.conv_reduce.weight", "bias": f"{timm_base}.se.conv_reduce.bias"}
            name_map[f"{jax_base}.se.fc2"] = {"kernel": f"{timm_base}.se.conv_expand.weight", "bias": f"{timm_base}.se.conv_expand.bias"}
            
            total_jax_block_idx += 1
            
    # 3. Head
    name_map["head_conv"] = {"kernel": "conv_head.weight"}
    name_map["head_bn"] = {jax_n: f"bn2.{timm_n}" for jax_n, timm_n in bn_map.items()}
    name_map["classifier"] = {"kernel": "classifier.weight", "bias": "classifier.bias"}

    return name_map


def load_pretrained_weights(model: model_lib.EfficientNet, pretrained_weights: dict):
    """
    Loads pre-trained weights by directly modifying the JAX model's attributes in-place.
    """
    print("Loading pre-trained weights directly into the model...")
    name_map = create_name_map(model.cfg)

    # Invert the name map for easier lookup from timm names to JAX paths
    timm_to_jax_map = {}
    for jax_module_path, params_map in name_map.items():
        for jax_param_name, timm_param_name in params_map.items():
            # Create a path tuple, e.g., ('blocks', '0', 'expand_conv', 'kernel')
            path_parts = jax_module_path.split('.')
            path_tuple = (*path_parts, jax_param_name)
            timm_to_jax_map[timm_param_name] = path_tuple

    # Iterate through the pre-trained weights and place them in the model
    for timm_name, weight in pretrained_weights.items():
        if timm_name not in timm_to_jax_map:
            continue  # Skip unneeded params like 'num_batches_tracked'

        path = timm_to_jax_map[timm_name]
        
        # --- Transpose weights to match JAX's expected format ---
        weight_np = weight
        param_name = path[-1]
        module_path_str = ".".join(path[:-1])

        if param_name == 'kernel' and len(weight_np.shape) == 4:
            # PyTorch (O, I, H, W) -> JAX (H, W, I, O)
            weight_np = np.transpose(weight_np, (2, 3, 1, 0))
        
        if param_name == 'kernel' and len(weight_np.shape) == 2:
            # PyTorch (O, I) -> JAX (I, O)
            weight_np = np.transpose(weight_np, (1, 0))

        target_module = model
        # Navigate through modules like 'blocks.0.expand_conv'
        for part in path[:-1]:
            if part.isdigit():
                target_module = target_module[int(part)]
            else:
                target_module = getattr(target_module, part)
        
        # Get the actual parameter object to check shape
        param_to_update = getattr(target_module, param_name)

        if param_to_update.shape != weight_np.shape:
             raise ValueError(
                f"Shape mismatch for '{'.'.join(path)}': "
                f"JAX model has {param_to_update.shape}, "
                f"pre-trained weight has {weight_np.shape}."
            )
        
        # Set the updated weight on the final module
        param_to_update = getattr(target_module, param_name)
        param_to_update.value = jnp.asarray(weight_np)

    print("Successfully loaded pre-trained weights into the model.")
    return model 