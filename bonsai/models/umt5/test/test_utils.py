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

"""Utility function for comparing JAX and PyTorch arrays."""

import os
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import torch


def dump_array(
    array: Union[jnp.ndarray, torch.Tensor, np.ndarray],
    filepath: Union[str, Path],
    metadata: Dict[str, Any] | None = None,
) -> None:
    """Dump JAX array or PyTorch tensor to a file.

    Args:
        array: JAX array, PyTorch tensor, or numpy array to save
        filepath: Path to save the array (will be saved as .npz format)
        metadata: Optional metadata dictionary to save with the array

    Examples:
        >>> import jax.numpy as jnp
        >>> arr = jnp.array([1.0, 2.0, 3.0])
        >>> dump_array(arr, "test.npz", metadata={"source": "jax"})
    """
    debug_mode = os.getenv("DEBUG", "0")
    print(f"{debug_mode=} {debug_mode == '0'} {debug_mode == '1'} {debug_mode == 0} {debug_mode == 1}")
    if debug_mode == "0":
        return

    filepath = Path(filepath)

    # Ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert to numpy array
    if isinstance(array, torch.Tensor):
        # Handle bfloat16
        if array.dtype == torch.bfloat16:
            array = array.float()
        np_array = array.detach().cpu().numpy()
        array_type = "torch"
    elif isinstance(array, (jnp.ndarray, jax.Array)):
        # Handle bfloat16
        if array.dtype == jnp.bfloat16:
            array = array.astype(jnp.float32)
        np_array = np.array(array)
        array_type = "jax"
    elif isinstance(array, np.ndarray):
        np_array = array
        array_type = "numpy"
    else:
        raise TypeError(f"Unsupported array type: {type(array)}")

    # Prepare metadata
    save_dict = {"array": np_array, "dtype": str(np_array.dtype), "shape": np_array.shape, "array_type": array_type}

    if metadata is not None:
        save_dict.update({f"meta_{k}": v for k, v in metadata.items()})

    # Save to npz file
    np.savez(filepath, **save_dict)
    print(f"✅ Saved {array_type} array with shape {np_array.shape} to {filepath}")


def compare_arrays(jax_input, torch_input, rtol: float = 1e-5, atol: float = 1e-7, verbose: bool = True):
    """Compare JAX array and PyTorch tensor for numerical equivalence.

    Args:
        jax_input: JAX array to compare
        torch_input: PyTorch tensor to compare

    Returns:
        bool: True if arrays are numerically equivalent, False otherwise

    Examples:
        >>> import jax.numpy as jnp
        >>> import torch
        >>> jax_arr = jnp.array([1.0, 2.0, 3.0])
        >>> torch_arr = torch.tensor([1.0, 2.0, 3.0])
        >>> compare_arrays(jax_arr, torch_arr)
        True
    """
    # Convert JAX bfloat16 to float32 (numpy doesn't support bfloat16)
    if hasattr(jax_input, "dtype") and jax_input.dtype == jnp.bfloat16:
        jax_input = jax_input.astype(jnp.float32)

    # Convert PyTorch bfloat16 to float32 for accurate comparison
    if isinstance(torch_input, torch.Tensor) and torch_input.dtype == torch.bfloat16:
        torch_input = torch_input.float()

    # Convert both to numpy arrays
    if isinstance(torch_input, torch.Tensor):
        torch_np = torch_input.detach().cpu().numpy()
    else:
        torch_np = np.array(torch_input)

    jax_np = np.array(jax_input)

    # Check shapes match
    if jax_np.shape != torch_np.shape:
        print(f"❌ Shape mismatch: JAX {jax_np.shape} vs PyTorch {torch_np.shape}")
        return False

    # Check numerical equivalence
    is_close = np.allclose(jax_np, torch_np, rtol=rtol, atol=atol)

    if is_close:
        print(f"✅ Arrays match! Shape: {jax_np.shape}")
    else:
        abs_diff = np.abs(jax_np - torch_np)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        print("❌ Arrays do NOT match!")
        print(f"   Shape: jax->{jax_np.shape} torch->{torch_np.shape}")
        print(f"   Max absolute difference: {max_diff:.2e}")
        print(f"   Mean absolute difference: {mean_diff:.2e}")

    return is_close


def compare_directory_arrays(
    directory: Union[str, Path], rtol: float = 1e-5, atol: float = 1e-7, verbose: bool = True
) -> Dict[str, bool]:
    """Compare JAX and PyTorch arrays stored in a directory.

    Files should be named with prefixes:
    - jax.<method_name> for JAX arrays
    - torch.<method_name> for PyTorch tensors
    - np.<method_name> for numpy arrays

    For example:
    - jax.t5encode.embedding contains JAX array from t5.embedding method
    - torch.t5encode.embedding contains PyTorch tensor from t5.embedding method

    Args:
        directory: Directory containing the array files
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
        verbose: Whether to print comparison results

    Returns:
        Dictionary mapping method names to comparison results (True if match)

    Examples:
        >>> results = compare_directory_arrays("outputs/")
        >>> # Will compare jax.t5encode.embedding with torch.t5encode.embedding
    """
    directory = Path(directory)

    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")

    # Group files by method name
    files_by_method = {}

    for filepath in directory.iterdir():
        if not filepath.is_file():
            continue

        filename = filepath.name

        # Parse filename: prefix.method_name
        if filename.startswith("jax."):
            prefix = "jax"
            method_name = filename[4:]  # Remove "jax."
        elif filename.startswith("torch."):
            prefix = "torch"
            method_name = filename[6:]  # Remove "torch."
        elif filename.startswith("np."):
            prefix = "np"
            method_name = filename[3:]  # Remove "np."
        else:
            continue  # Skip files that don't match the pattern

        # Remove common file extensions if present
        for ext in [".npz", ".npy"]:
            if method_name.endswith(ext):
                method_name = method_name[: -len(ext)]
                break

        if method_name not in files_by_method:
            files_by_method[method_name] = {}

        files_by_method[method_name][prefix] = filepath

    # Compare JAX and PyTorch arrays for each method
    results = {}

    for method_name, files in sorted(files_by_method.items()):
        if "jax" in files and "torch" in files:
            if verbose:
                print(f"\n{'=' * 60}")
                print(f"🔍 Comparing method: {method_name}")
                print(f"   JAX file: {files['jax'].name}")
                print(f"   Torch file: {files['torch'].name}")

            # Load JAX array
            jax_data = np.load(files["jax"])
            jax_array = jax_data["array"]

            # Load PyTorch array
            torch_data = np.load(files["torch"])
            torch_array = torch_data["array"]

            # Compare using the existing compare_arrays function
            is_match = compare_arrays(jax_array, torch_array, rtol=rtol, atol=atol, verbose=verbose)
            results[method_name] = is_match
        else:
            if verbose:
                if "jax" not in files:
                    print(f"\n⚠️  Skipping {method_name}: missing jax file")
                if "torch" not in files:
                    print(f"\n⚠️  Skipping {method_name}: missing torch file")

    # Print summary
    if verbose and results:
        print(f"\n{'=' * 60}")
        print("📊 Summary:")
        total = len(results)
        passed = sum(results.values())
        failed = total - passed
        print(f"   Total comparisons: {total}")
        print(f"   ✅ Passed: {passed}")
        print(f"   ❌ Failed: {failed}")
        if passed > 0:
            print("\n   Passed methods:")
            for method, result in results.items():
                if result:
                    print(f"✅      - {method}")
        if failed > 0:
            print("\n   Failed methods:")
            for method, result in results.items():
                if not result:
                    print(f"❌      - {method}")
        print(f"{'=' * 60}")

    return results
