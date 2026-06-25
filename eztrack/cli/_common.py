"""Small shared helpers for the ``eztrack`` CLI subcommands."""

import argparse
from collections.abc import Iterable


def add_select_args(parser: argparse.ArgumentParser, kind: str) -> None:
    """Add the shared ``<name>`` positional + ``--all`` flag to a subparser."""
    parser.add_argument("name", nargs="?", help=f"{kind} name (see `list`)")
    parser.add_argument("--all", action="store_true", help=f"select every {kind}")


def selected(args: argparse.Namespace, all_names: Iterable[str], kind: str) -> list[str]:
    """Resolve the ``<name>`` / ``--all`` args added by :func:`add_select_args`."""
    if args.all:
        return list(all_names)
    if args.name:
        return [args.name]
    raise SystemExit(f"Give a {kind} name or --all (see `list`).")


def print_table(rows: list[tuple]) -> None:
    """Print rows as left-aligned columns; the last column is free-width."""
    rows = [tuple(str(c) for c in row) for row in rows]
    if not rows:
        return
    ncol = len(rows[0])
    widths = [max(len(r[i]) for r in rows) for i in range(ncol - 1)]
    for row in rows:
        head = "  ".join(f"{row[i]:<{widths[i]}}" for i in range(ncol - 1))
        print(f"{head}  {row[-1]}" if head else row[-1])
