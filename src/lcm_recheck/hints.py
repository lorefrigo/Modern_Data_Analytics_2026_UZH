"""pg_hint_plan hints in the candidate-plan SQL.

Usage:
  from lcm_recheck.hints import attach_sql, fix_nestloop, leading, sql_hash, sql_lines
  leading("/*+ HashJoin(t mi) Leading((t mi)) */ SELECT ...")   # -> "(t mi)"
  fixed, n_lines, n_repl = fix_nestloop(open(path, "rb").read())   # NestedLoop( -> NestLoop(
  sqls = sql_lines("results/sql_pinned_fixed", "job_light_0")      # the plans of one query, as run_plans.py runs them
  df = attach_sql(pd.read_csv("results/pg18_explain_fixed.csv"), "results/sql_pinned_fixed")   # adds the sql column

The committed PG18 files (extract_runs.py --no_sql) hold no SQL, only its hash (`sql_sha1`): the SQL is the authors'
and is read from the pinned workload files, line `plan_idx` of `<sql_dir>/<query>.sql`.
"""
import hashlib
import os

import pandas as pd


def leading(s: str) -> str | None:
    """Content of the Leading(...) hint, with nested parentheses; None if absent."""
    i = s.find("Leading(")
    if i < 0:
        return None
    depth, j = 0, i + len("Leading")
    for k in range(j, len(s)):
        depth += {"(": 1, ")": -1}.get(s[k], 0)
        if depth == 0:
            return " ".join(s[j + 1:k].split())
    return None


# The pinned SQL uses "NestedLoop(", which pg_hint_plan does not know: it reports a syntax error and ignores every
# hint after it, including Leading(...). The correct keyword is "NestLoop(".
BROKEN_HINT = b"NestedLoop("
FIXED_HINT = b"NestLoop("


def fix_nestloop(data: bytes) -> tuple[bytes, int, int]:
    """Replace every NestedLoop( by NestLoop( in the bytes of a SQL file.

    Returns (fixed bytes, number of changed lines, number of replacements). Works on bytes, so line endings
    and everything else stay exactly as they were (same output as sed 's/NestedLoop(/NestLoop(/g').
    """
    n_lines = sum(BROKEN_HINT in line for line in data.splitlines())
    return data.replace(BROKEN_HINT, FIXED_HINT), n_lines, data.count(BROKEN_HINT)


def sql_lines(sql_dir: str, query: str) -> list[str]:
    """The SQL statements of `<sql_dir>/<query>.sql`: one per non-empty line, stripped (what run_plans.py runs)."""
    with open(os.path.join(sql_dir, f"{query}.sql")) as f:
        return [line.strip() for line in f if line.strip()]


def sql_hash(sql: str) -> str:
    """Short fingerprint of one SQL statement (first 16 hex digits of its SHA-1)."""
    return hashlib.sha1(sql.encode("utf-8")).hexdigest()[:16]


def attach_sql(df: pd.DataFrame, sql_dir: str) -> pd.DataFrame:
    """Return `df` with a `sql` column: line `plan_idx` of `<sql_dir>/<query>.sql`, checked against `sql_sha1`.

    A frame that already has `sql` is returned unchanged. Raises ValueError if a query has a different number of
    lines than rows, or if any line does not match its hash (e.g. the unfixed SQL for a run on the fixed SQL).
    """
    if "sql" in df.columns:
        return df
    out = []
    for q, g in df.groupby("query", sort=False):
        sqls = sql_lines(sql_dir, q)
        if len(sqls) != len(g) or not g.plan_idx.between(0, len(sqls) - 1).all():
            raise ValueError(f"{q}: {len(g)} rows but {len(sqls)} SQL lines in {sql_dir}")
        sql = g.plan_idx.map(lambda i, sqls=sqls: sqls[i])
        bad = int((sql.map(sql_hash) != g.sql_sha1).sum())
        if bad:
            raise ValueError(f"{q}: {bad} of {len(g)} lines in {sql_dir} do not match sql_sha1; wrong --sql_dir? "
                             f"(the fixed SQL is in results/sql_pinned_fixed, the authors' in config.SQL_PINNED_DIR)")
        out.append(sql)
    return df.assign(sql=pd.concat(out))
