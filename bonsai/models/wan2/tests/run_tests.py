#!/usr/bin/env python3
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

"""Simple CLI to run individual tests or all tests."""

import sys

from test_outputs import run_all_tests, test_dit_transformer, test_t5_encoder, test_vae_decoder


def main():
    """Run tests based on command line arguments."""
    if len(sys.argv) < 2:
        print("Usage: python run_tests.py [all|t5|dit|vae]")
        print("\nExamples:")
        print("  python run_tests.py all   # Run all tests")
        print("  python run_tests.py t5    # Test T5 encoder only")
        print("  python run_tests.py dit   # Test DiT transformer only")
        print("  python run_tests.py vae   # Test VAE decoder only")
        sys.exit(1)

    test_name = sys.argv[1].lower()

    if test_name == "all":
        success = run_all_tests()
    elif test_name == "t5":
        success = test_t5_encoder()
    elif test_name == "dit":
        success = test_dit_transformer()
    elif test_name == "vae":
        success = test_vae_decoder()
    else:
        print(f"Unknown test: {test_name}")
        print("Valid options: all, t5, dit, vae")
        sys.exit(1)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
