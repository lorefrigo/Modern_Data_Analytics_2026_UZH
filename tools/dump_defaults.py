"""Print every argparse default of a script without running it, one sorted `dest=default` line each.

Patches argparse.ArgumentParser.parse_args so that the script stops at its first parse_args() call and prints
the defaults of that parser (sub-commands as `<name>.dest`), then runs the script as __main__ with its own
folder on sys.path. Used by tools/verify_golden.sh to show that a refactoring changed no command-line default.

Usage (from the repository root):
  uv run python tools/dump_defaults.py scripts/run_plans.py
"""
import argparse
import os
import runpy
import sys


def defaults(parser: argparse.ArgumentParser, prefix: str = "") -> list[str]:
    out = []
    for act in parser._actions:
        if isinstance(act, argparse._SubParsersAction):
            for name, sub in act.choices.items():
                out += defaults(sub, f"{prefix}{name}.")
        elif act.dest != "help":
            out.append(f"{prefix}{act.dest}={act.default!r}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Print the argparse defaults of a script.")
    ap.add_argument("script", help="path of the script to inspect")
    path = os.path.abspath(ap.parse_args().script)

    def report(self: argparse.ArgumentParser, *args: object, **kwargs: object) -> None:
        print("\n".join(sorted(defaults(self))))
        sys.exit(0)

    argparse.ArgumentParser.parse_args = report  # type: ignore[method-assign]
    sys.path.insert(0, os.path.dirname(path))
    sys.argv = [path]
    runpy.run_path(path, run_name="__main__")
    sys.exit(f"{path}: no parse_args() call reached")


if __name__ == "__main__":
    main()
