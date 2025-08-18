import pprint
from etils import epath
import jax
import numpy as np
from flax import nnx

from bonsai.models.whisper import modeling as model_lib
from bonsai.models.whisper import params as P


def flatten_leaves(tree, prefix=""):
    items = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            items |= flatten_leaves(v, f"{prefix}.{k}" if prefix else k)
    else:
        items[prefix] = tree
    return items


def analyze_model_structure():
    """Analyze the model structure to understand parameter counts."""
    cfg = model_lib.WhisperConfig.whisper_tiny()
    
    # Create model and get structure
    m = nnx.eval_shape(lambda: model_lib.WhisperModel(cfg, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(m)
    jax_state = abs_state.to_pure_dict()
    
    print("=== Model Structure Analysis ===")
    print(f"Config: {cfg}")
    
    # Count parameters by component
    encoder_params = 0
    decoder_params = 0
    
    if "encoder" in jax_state:
        encoder_flat = flatten_leaves(jax_state["encoder"])
        encoder_params = len(encoder_flat)
        print(f"Encoder parameters: {encoder_params}")
        print("Encoder structure:")
        for k in sorted(encoder_flat.keys())[:10]:
            print(f"  {k}")
        if len(encoder_flat) > 10:
            print(f"  ... and {len(encoder_flat) - 10} more")
    
    if "decoder" in jax_state:
        decoder_flat = flatten_leaves(jax_state["decoder"])
        decoder_params = len(decoder_flat)
        print(f"Decoder parameters: {decoder_params}")
        print("Decoder structure:")
        for k in sorted(decoder_flat.keys())[:10]:
            print(f"  {k}")
        if len(decoder_flat) > 10:
            print(f"  ... and {len(decoder_flat) - 10} more")
    
    total_params = encoder_params + decoder_params
    print(f"Total parameters: {total_params}")
    
    return jax_state, graph_def


def main(model_dir: str = "/tmp/models-bonsai/whisper-tiny"):
    # First analyze the model structure
    jax_state, graph_def = analyze_model_structure()
    
    # Load HF weights
    model_path = epath.Path(model_dir)
    files = list(model_path.glob("*.safetensors"))
    assert files, "No safetensors in model_dir"
    hf = P.safetensors.load_file(str(files[0]))

    print(f"\n=== HF Weights Analysis ===")
    print(f"HF weights count: {len(hf)}")
    
    # Count HF weights by component
    hf_encoder = [k for k in hf.keys() if "encoder" in k]
    hf_decoder = [k for k in hf.keys() if "decoder" in k]
    print(f"HF encoder weights: {len(hf_encoder)}")
    print(f"HF decoder weights: {len(hf_decoder)}")
    
    mapping = P._get_key_and_transform_mapping(model_lib.WhisperConfig.whisper_tiny())

    # Assign into a COPY so we don't mutate original
    target = jax_state.copy()
    assigned = 0
    skipped = []
    errors = []
    for k, v in hf.items():
        try:
            jax_key, transform = P._torch_key_to_jax_key(mapping, k)
            keys = [P._stoi(s) for s in jax_key.split(".")]
            P._assign_weights(keys, v, target, k, transform)
            assigned += 1
        except Exception as e:
            skipped.append(k)
            continue

    flat_abs = flatten_leaves(jax_state)
    flat_target = flatten_leaves(target)
    num_abs = len(flat_abs)
    num_target = len(flat_target)
    num_arrays = sum(isinstance(v, (np.ndarray, jax.Array)) for v in flat_target.values())

    print(f"\n=== State Analysis ===")
    print(f"Original abs_state leaves: {num_abs}")
    print(f"Modified target leaves: {num_target}")
    print(f"Arrays in target: {num_arrays}")
    
    missing = P._find_non_array_keys(target)
    print(f"missing leaves (non-arrays): {len(missing)}")
    for s in missing[:20]:
        print("  -", s)
    print(f"assigned tensors: {assigned}")
    print(f"skipped tensors: {len(skipped)} (showing first 20)")
    for s in skipped[:20]:
        print("  -", s)

    # Try tying output projection
    try:
        tok = target["decoder"]["token_embedding"]["embedding"]
        out = target["decoder"]["output_projection"]["kernel"]
        if not isinstance(out, (np.ndarray, jax.Array)) and isinstance(tok, (np.ndarray, jax.Array)):
            target["decoder"]["output_projection"]["kernel"] = tok.T
            print("Tied output projection to token embeddings")
    except Exception as e:
        print("Tie error:", e)

    # Final check
    missing2 = P._find_non_array_keys(target)
    print(f"missing after tie: {len(missing2)}")
    for s in missing2[:20]:
        print("  -", s)

    # Show structure differences
    abs_keys = set(flat_abs.keys())
    target_keys = set(flat_target.keys())
    extra = target_keys - abs_keys
    missing_keys = abs_keys - target_keys
    
    print(f"\nExtra keys in target: {len(extra)}")
    for k in sorted(extra)[:10]:
        print(f"  + {k}")
    print(f"Missing keys in target: {len(missing_keys)}")
    for k in sorted(missing_keys)[:10]:
        print(f"  - {k}")

    # If clean, attempt merge
    if not missing2 and len(extra) == 0 and len(missing_keys) == 0:
        try:
            merged = nnx.merge(graph_def, target)
            print("merge ok")
        except Exception as e:
            print("merge error:", e)
    else:
        print("Cannot merge - structure mismatch")


if __name__ == "__main__":
    main()
