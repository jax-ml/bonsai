import os
import subprocess
import sys
import tomllib
from collections import defaultdict
from pathlib import Path


def get_changed_files():
    base_ref = os.environ.get("GITHUB_BASE_REF") or "main"

    if os.environ.get("GITHUB_EVENT_NAME") == "pull_request":
        cmd = ["git", "diff", "--name-only", f"origin/{base_ref}...HEAD"]
    else:
        base_sha = os.environ.get("GITHUB_BEFORE") or "HEAD~1"
        cmd = ["git", "diff", "--name-only", base_sha, "HEAD"]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return set(result.stdout.strip().splitlines())


def find_tests_in_directory(directory: Path) -> list[Path]:
    """Recursively find all test_*.py and *_test.py files in a directory."""
    if not directory.exists():
        return []

    test_files = []
    test_files.extend(directory.glob("**/test_*.py"))
    return sorted(set(test_files))


def get_model_extras(model_name: str, optional_deps: dict) -> set:
    """Get extras for a model if it exists in optional-dependencies."""
    extras = {"testing"}
    if model_name in optional_deps:
        extras.add(model_name)
    return extras


def main():
    changed_files = get_changed_files()
    print(f"Changed files detected: {changed_files}")

    if not changed_files:
        print("No changes detected.")
        sys.exit(0)

    with open("pyproject.toml", "rb") as f:
        config = tomllib.load(f)

    optional_deps = config.get("project", {}).get("optional-dependencies", {})

    # Map: test_directory -> set of extras required
    test_targets = defaultdict(set)
    models_with_changes = set()

    for f in changed_files:
        path = Path(f)
        if not path.exists():  # Skip deleted/renamed paths
            continue

        parts = path.parts

        # Handle bonsai/models/<model_name>
        if len(parts) >= 3 and parts[0] == "bonsai" and parts[1] == "models":
            model_name = parts[2]
            models_with_changes.add(model_name)
            model_dir = Path("bonsai/models") / model_name
            target_dir = str(model_dir)

            print(f"[TRIGGER] File '{f}' triggered tests for model: {model_name}")

            # Get extras for this model (if defined in pyproject.toml)
            extras = get_model_extras(model_name, optional_deps)
            test_targets[target_dir].update(extras)

        # Handle bonsai/utils
        elif len(parts) >= 2 and parts[0] == "bonsai" and parts[1] == "utils":
            print(f"[TRIGGER] File '{f}' triggered tests for: bonsai/utils")
            test_targets["bonsai/utils"].add("testing")

        else:
            print(f"[IGNORE] File '{f}' is a root/config change or outside source scope.")

    if not test_targets:
        print("No relevant source changes for testing (bonsai/models or bonsai/utils).")
        sys.exit(0)

    print(f"Detected model changes: {models_with_changes}")

    # Validate that tests exist for models with changes
    test_count = 0
    for target_dir in test_targets.keys():
        test_files = find_tests_in_directory(Path(target_dir))
        if not test_files:
            print(f"Warning: no tests found in '{target_dir}'")
        else:
            test_count += len(test_files)
            print(f"Found {len(test_files)} test file(s) in '{target_dir}':")
            for test_file in test_files:
                print(f"- {test_file}")

    if test_count == 0:
        print("No test files found in changed model directories.")
        sys.exit(0)

    # Create venv once
    print("Creating virtual environment")
    subprocess.run(["uv", "venv", "--clear"], check=True)

    for target, extras in test_targets.items():
        extras_str = ",".join(sorted(extras))
        pkg_spec = f".[{extras_str}]"

        print(f"Running tests for target: {target}")
        print(f"Installing with extras: {extras_str}")

        # Install dependencies
        subprocess.run(["uv", "pip", "install", pkg_spec], check=True)

        # Run tests
        test_files = find_tests_in_directory(Path(target))
        if test_files:
            subprocess.run(["uv", "run", "pytest", target, "-v"], check=True)
        else:
            print(f"No tests to run for {target}")


if __name__ == "__main__":
    main()
