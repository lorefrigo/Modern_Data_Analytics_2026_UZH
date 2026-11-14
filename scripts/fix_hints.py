"""E1: write a fixed copy of the pinned candidate-plan SQL, with every NestedLoop( replaced by NestLoop(.

pg_hint_plan does not know the keyword NestedLoop: it reports a syntax error and ignores every hint after it,
including Leading(...). The replacement is done on the bytes of each file, so everything else stays unchanged
(same output as sed 's/NestedLoop(/NestLoop(/g'). Prints the changed lines per file and in total
(771 lines in 20 of the 58 files of the authors' pinned SQL).

Usage (from the repository root):
  uv run python scripts/fix_hints.py
  uv run python scripts/fix_hints.py --out_dir /tmp/sql_fixed
Reads every *.sql file of --sql_dir (default: the authors' join_order_full_pg_10_hinted) and writes it to --out_dir
(default: results/sql_pinned_fixed). An --out_dir that already holds .sql files is not overwritten unless
--overwrite is given.
"""
import argparse
import glob
import os
import re
import sys

from lcm_recheck.config import SQL_PINNED_DIR, SQL_PINNED_FIXED_DIR
from lcm_recheck.hints import BROKEN_HINT, FIXED_HINT, fix_nestloop


def query_order(path: str) -> tuple[int, str]:
    """Sort key: the number in the file name (job_light_2 before job_light_10), then the name."""
    name = os.path.basename(path)
    m = re.search(r"(\d+)\.sql$", name)
    return (int(m.group(1)) if m else -1, name)


def main() -> None:
    ap = argparse.ArgumentParser(description="Replace NestedLoop( by NestLoop( in the pinned candidate-plan SQL.")
    ap.add_argument("--sql_dir", default=SQL_PINNED_DIR, help="folder with the original *.sql files (only read)")
    ap.add_argument("--out_dir", default=SQL_PINNED_FIXED_DIR, help="folder for the fixed copies (created if missing)")
    ap.add_argument("--overwrite", action="store_true", help="replace the .sql files of an existing --out_dir")
    a = ap.parse_args()
    src, out = os.path.expanduser(a.sql_dir), os.path.expanduser(a.out_dir)

    files = sorted(glob.glob(os.path.join(src, "*.sql")), key=query_order)
    if not files:
        sys.exit(f"no .sql files in {src}")
    if os.path.abspath(src) == os.path.abspath(out):
        sys.exit("--out_dir must differ from --sql_dir (never overwrite an input)")
    if glob.glob(os.path.join(out, "*.sql")) and not a.overwrite:
        sys.exit(f"{out} already holds .sql files; pass --overwrite to replace them")
    os.makedirs(out, exist_ok=True)

    tot_lines = tot_repl = n_changed = 0
    for path in files:
        with open(path, "rb") as f:
            fixed, n_lines, n_repl = fix_nestloop(f.read())
        with open(os.path.join(out, os.path.basename(path)), "wb") as f:
            f.write(fixed)
        if n_lines:
            n_changed += 1
            print(f"{os.path.basename(path)}: {n_lines} lines")
        tot_lines, tot_repl = tot_lines + n_lines, tot_repl + n_repl

    print(f"{tot_lines} lines changed in {n_changed} of {len(files)} files ({tot_repl} replacements of "
          f"{BROKEN_HINT.decode()} by {FIXED_HINT.decode()}); written to {out}")


if __name__ == "__main__":
    main()
