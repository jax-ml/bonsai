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


def main():
    changed_files = get_changed_files()
    print(f"Changed files detected: {changed_files}")
    print("-" * 40)

    if not changed_files:
        print("No changes detected.")
        sys.exit(0)

    with open("pyproject.toml", "rb") as f:
        config = tomllib.load(f)

    optional_deps = config.get("project", {}).get("optional-dependencies", {})
    valid_extras = set(optional_deps.keys())

    # Map: test_directory -> set of extras required
    test_targets = defaultdict(set)

    for f in changed_files:
        path = Path(f)
        parts = path.parts

        # Handle bonsai/models/<model_name>
        if len(parts) >= 3 and parts[0] == "bonsai" and parts[1] == "models":
            model_name = parts[2]
            target_dir = str(Path("bonsai/models") / model_name)

            print(f"[TRIGGER] File '{f}' triggered tests for model: {model_name}")

            if model_name in valid_extras:
                test_targets[target_dir].add(model_name)
            else:
                test_targets[target_dir].add(None)

        # Handle bonsai/utils
        elif len(parts) >= 2 and parts[0] == "bonsai" and parts[1] == "utils":
            print(f"[TRIGGER] File '{f}' triggered tests for: bonsai/utils")
            test_targets["bonsai/utils"].add(None)

        else:
            print(f"[IGNORE]  File '{f}' is a root/config change or outside source scope.")

    print("-" * 40)

    if not test_targets:
        print("No relevant source changes for testing (bonsai/models or bonsai/utils).")
        sys.exit(0)

    # Create venv once
    print("Creating virtual environment...")
    subprocess.run(["uv", "venv"], check=True)

    for target, extras in test_targets.items():
        active_extras = {"testing"}
        active_extras.update(e for e in extras if e is not None)

        extras_str = ",".join(active_extras)
        pkg_spec = f".[{extras_str}]"

        print(f"Running tests for target: {target} (Extras: {extras_str})")

        # Install dependencies
        subprocess.run(["uv", "pip", "install", pkg_spec], check=True)

        # Run tests
        subprocess.run(["uv", "run", "pytest", target], check=True)


if __name__ == "__main__":
    main()
