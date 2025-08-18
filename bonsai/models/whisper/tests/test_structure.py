#!/usr/bin/env python3
"""
Simple test script to validate Whisper model structure.
This script doesn't require JAX and can be used to verify the implementation.
"""

import os
import sys
import importlib

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    # Test basic Python modules
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False
    
    try:
        import transformers
        print("✓ transformers imported successfully")
    except ImportError as e:
        print(f"✗ transformers import failed: {e}")
        return False
    
    # Test JAX-related modules (optional)
    try:
        import jax
        print("✓ jax imported successfully")
    except ImportError as e:
        print(f"⚠ jax not available: {e}")
        print("  This is expected if JAX is not installed")
    
    try:
        import flax
        print("✓ flax imported successfully")
    except ImportError as e:
        print(f"⚠ flax not available: {e}")
        print("  This is expected if Flax is not installed")
    
    return True

def test_file_structure():
    """Test if all required files exist."""
    print("\nTesting file structure...")
    
    required_files = [
        "bonsai/models/whisper/__init__.py",
        "bonsai/models/whisper/modeling.py",
        "bonsai/models/whisper/params.py",
        "bonsai/models/whisper/README.md",
        "bonsai/models/whisper/tests/__init__.py",
        "bonsai/models/whisper/tests/run_model.py",
        "bonsai/models/whisper/tests/test_structure.py",
        "bonsai/models/whisper/tests/Whisper_speech_recognition_example.ipynb",
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            all_exist = False
    
    return all_exist

def test_modeling_structure():
    """Test the modeling.py file structure."""
    print("\nTesting modeling.py structure...")
    
    try:
        # Try to read the file and check for key components
        with open("bonsai/models/whisper/modeling.py", "r") as f:
            content = f.read()
        
        # Check for key classes and functions
        key_components = [
            "class WhisperConfig",
            "class MultiHeadAttention",
            "class ResidualAttentionBlock",
            "class AudioEncoder",
            "class TextDecoder",
            "class WhisperModel",
            "def forward",
            "def generate",
        ]
        
        for component in key_components:
            if component in content:
                print(f"✓ {component}")
            else:
                print(f"✗ {component} - MISSING")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error reading modeling.py: {e}")
        return False

def test_params_structure():
    """Test the params.py file structure."""
    print("\nTesting params.py structure...")
    
    try:
        with open("bonsai/models/whisper/params.py", "r") as f:
            content = f.read()
        
        key_components = [
            "def convert_hf_whisper_to_nnx",
            "def create_model_from_safe_tensors",
            "def load_whisper_model",
        ]
        
        for component in key_components:
            if component in content:
                print(f"✓ {component}")
            else:
                print(f"✗ {component} - MISSING")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error reading params.py: {e}")
        return False

def test_readme_structure():
    """Test the README.md file structure."""
    print("\nTesting README.md structure...")
    
    try:
        with open("bonsai/models/whisper/README.md", "r") as f:
            content = f.read()
        
        key_sections = [
            "# Whisper in JAX",
            "## Tested on",
            "## Model Configurations",
            "## Usage Examples",
            "## How to contribute",
        ]
        
        for section in key_sections:
            if section in content:
                print(f"✓ {section}")
            else:
                print(f"✗ {section} - MISSING")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error reading README.md: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Whisper Model Structure Validation")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("File Structure", test_file_structure),
        ("Modeling Structure", test_modeling_structure),
        ("Params Structure", test_params_structure),
        ("README Structure", test_readme_structure),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The Whisper model structure is valid.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
