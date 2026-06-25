"""The unified ``eztrack`` command line interface.

One entrypoint with subcommand groups::

    eztrack notebooks list
    eztrack notebooks copy individual
"""

import argparse

from . import notebooks

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level ``eztrack`` parser."""
    parser = argparse.ArgumentParser(
        prog="eztrack",
        description="ezTrack command line tools: copy out the bundled analysis notebooks.",
    )
    sub = parser.add_subparsers(title="subcommands", dest="group", required=True)
    notebooks.add_subparser(sub)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
