#!/usr/bin/env python3

import argparse
import importlib
import importlib.util
import os
import sys
import tempfile

from bonsai import __description__, __version__


def get_module(name, package=None):
    absolute_name = importlib.util.resolve_name(name, package)
    if absolute_name in sys.modules:
        return sys.modules[absolute_name]

    path, parent_module, child_name, path, spec = None, None, None, None, None
    if "." in absolute_name:
        parent_name, _, child_name = absolute_name.rpartition(".")
        parent_module = get_module(parent_name)
        path = parent_module.__spec__.submodule_search_locations
    for finder in sys.meta_path:
        spec = finder.find_spec(absolute_name, path)
        if spec is not None:
            break
    if spec is None:
        raise ModuleNotFoundError(
            f"No module named {absolute_name!r}", name=absolute_name
        )
    module = importlib.util.module_from_spec(spec)
    if path is not None:
        setattr(parent_module, child_name, module)
    return module


def filepath_from_module(module):
    return os.path.dirname(get_module(module).__file__)


def _build_parser():
    """
    Parser builder

    :return: instanceof argparse.ArgumentParser
    :rtype: ```argparse.ArgumentParser```
    """
    parser = argparse.ArgumentParser(
        prog="python3 -m bonsai",
        description=__description__,
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {__version__}".format(__version__=__version__),
    )
    parser.add_argument(
        "-s",
        "--search",
        action="append",
        default=[os.path.join(os.path.dirname(__file__), "models")],
        help="Alternative filepath(s) or fully-qualified name (FQN) to use models from.",
    )

    subparsers: argparse._SubParsersAction[argparse.ArgumentParser] = (
        parser.add_subparsers()
    )
    subparsers.required = True
    subparsers.dest = "command"

    ######
    # ls #
    ######
    ls_parser: argparse.ArgumentParser = subparsers.add_parser(
        "ls",
        help="List installed models",
    )

    #######
    # run #
    #######
    run_parser: argparse.ArgumentParser = subparsers.add_parser(
        "run",
        help="Run specified model",
    )
    run_parser.add_argument(
        "-n",
        "--model-name",
        help="Model name",
    )
    run_parser.add_argument(
        "-p",
        "--path-root",
        help="Path root (for model.safetensors directory &etc.)."
        'Defaults to "/tmp/models-bonsai/${model_name}"',
    )

    return parser


def main(cli_argv=None, return_args=False):
    """
    Run the CLI parser

    :param cli_argv: CLI arguments. If None uses `sys.argv`.
    :type cli_argv: ```None | list[str]```

    :param return_args: Primarily use is for tests. Returns the args rather than executing anything.
    :type return_args: ```bool```

    :return: the args if `return_args`, else None
    :rtype: ```None | Namespace```
    """
    _parser: argparse.ArgumentParser = _build_parser()
    args: argparse.Namespace = _parser.parse_args(args=cli_argv)
    if return_args:
        return args

    if args.command == "ls":
        print(
            "\n".join(
                sorted(
                    f"- {d}"
                    for search_path in frozenset(args.search)
                    for d in (
                        os.listdir(search_path)
                        if os.path.isdir(search_path)
                        else filepath_from_module(search_path)
                    )
                    if os.path.isdir(os.path.join(search_path, d))
                    and d not in frozenset(("__pycache__",))
                )
            )
        )
        return None
    elif args.command == "run":
        search_paths = tuple(
            sorted(
                (
                    search_path
                    if os.path.isdir(search_path)
                    else filepath_from_module(search_path)
                )
                for search_path in frozenset(args.search)
            )
        )
        for search_path in search_paths:
            og_search_path = search_path
            if os.path.isdir(search_path):
                prev = None
                while not os.path.isfile(
                    os.path.join(search_path, "setup.py")
                ) and not os.path.isfile(os.path.join(search_path, "pyproject.toml")):
                    search_path = os.path.dirname(search_path)
                    if search_path == prev:
                        raise ModuleNotFoundError("Could not find project root")
                    prev = search_path
                search_path = str(
                    os.path.join(
                        og_search_path[len(search_path) + 1 :],
                        args.model_name,
                        "tests",
                        "run_model",
                    ).replace(os.path.sep, ".")
                )
            module = importlib.import_module(search_path)
            module.run_model(args.path_root)
            return None
        raise ImportError(f"Could not find `run_model` function in {search_paths!r}")
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
