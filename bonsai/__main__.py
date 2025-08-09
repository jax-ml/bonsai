#!/usr/bin/env python

import argparse
import importlib
import os
import tempfile

from bonsai import __description__, __version__


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
        default=os.path.join(os.path.dirname(__file__), "models"),
        help="An alternative filepath or fully-qualified name (FQN) to use models from.",
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
    ls_parser.add_argument(
        "--additional-search-path",
        help="Additional alternative filepath or fully-qualified name (FQN) to use models from.",
        action="append",
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
        default=os.path.join(tempfile.gettempdir(), "models-bonsai"),
        help="--model-name",
    )

    return parser


def main(cli_argv=None, return_args=False):
    """
    Run the CLI parser

    :param cli_argv: CLI arguments. If None uses `sys.argv`.
    :type cli_argv: ```Optional[List[str]]```

    :param return_args: Primarily use is for tests. Returns the args rather than executing anything.
    :type return_args: ```bool```

    :return: the args if `return_args`, else None
    :rtype: ```Optional[Namespace]```
    """
    _parser: argparse.ArgumentParser = _build_parser()
    args: argparse.Namespace = _parser.parse_args(args=cli_argv)
    if return_args:
        return args
    if args.command == "ls":
        if args.additional_search_path is None:
            args.additional_search_path = []
        args.additional_search_path.append(args.search)
        print(
            "\n".join(
                sorted(
                    f"- {d}"
                    for search_path in frozenset(args.additional_search_path)
                    for d in os.listdir(search_path)  # TODO: filepath from module name
                    if os.path.isdir(os.path.join(args.search, d))
                    and d not in frozenset(("__pycache__",))
                )
            )
        )
        return None
    elif args.command == "run":
        root = args.search
        prev = None
        while not os.path.isfile(os.path.join(root, "setup.py")) and not os.path.isfile(
            os.path.join(root, "pyproject.toml")
        ):
            root = os.path.dirname(root)
            if root == prev:
                raise ModuleNotFoundError("Could not find project root")
            prev = root
        module = importlib.import_module(
            str(
                os.path.join(
                    args.search[len(root) + 1 :], args.model_name, "tests", "run_model"
                ).replace(os.path.sep, ".")
            )
        )
        module.run_model(args.path_root)
        return None
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
