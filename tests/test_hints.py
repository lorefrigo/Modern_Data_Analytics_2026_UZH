"""lcm_recheck.hints: leading(), fix_nestloop(), and the SQL of the pinned files (sql_lines, sql_hash, attach_sql).

All SQL here is invented (no line of the authors' workloads)."""
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from lcm_recheck.hints import attach_sql, fix_nestloop, leading, sql_hash, sql_lines


def test_nested_parentheses() -> None:
    s = "/*+ NestLoop(t mi) HashJoin(t mi mc) Leading(((t mi) mc)) */ SELECT 1;"
    assert leading(s) == "((t mi) mc)"


def test_whitespace_normalised() -> None:
    assert leading("/*+ Leading(( t\n\t mi )  mc) */") == "( t mi ) mc"


def test_absent() -> None:
    assert leading("/*+ HashJoin(t mi) */ SELECT 1;") is None


def test_unbalanced() -> None:
    assert leading("/*+ Leading((t mi) */") is None


def test_fix_nestloop_counts_lines_and_replacements() -> None:
    data = b"/*+ NestedLoop(t mi) NestedLoop(t mi mc) */ SELECT 1;\n/*+ HashJoin(t mi) */ SELECT 2;\n" \
           b"/*+ NestedLoop(mc t) */ SELECT 3;\n"
    fixed, n_lines, n_repl = fix_nestloop(data)
    assert fixed == data.replace(b"NestedLoop(", b"NestLoop(")
    assert (n_lines, n_repl) == (2, 3)


def test_fix_nestloop_leaves_other_text_unchanged() -> None:
    data = b"/*+ NestLoop(t mi) */ -- NestedLoop without parenthesis\r\nSELECT 1;"   # CRLF, no final newline
    assert fix_nestloop(data) == (data, 0, 0)
    fixed, n_lines, _ = fix_nestloop(b"/*+ NestedLoop(t mi) */\r\nSELECT 1;")
    assert fixed == b"/*+ NestLoop(t mi) */\r\nSELECT 1;" and n_lines == 1


# ----------------------------------------------------------------------------- SQL from the pinned files
SQL = {"job_light_7": ["/*+ NestedLoop(a b) Leading((a b)) */ SELECT 1;",
                       "/*+ HashJoin(a b) Leading((b a)) */ SELECT 2;"],
       "job_light_9": ["/*+ SeqScan(a) Leading((a c)) */ SELECT 3;"]}


@pytest.fixture
def sql_dir(tmp_path: Path) -> Path:
    for q, lines in SQL.items():          # blank lines and surrounding whitespace, as sql_lines() must handle
        (tmp_path / f"{q}.sql").write_text("\n".join(f"  {s}  \n" for s in lines) + "\n\n")
    return tmp_path


def frame() -> pd.DataFrame:
    rows = [{"query": q, "plan_idx": i, "sql_sha1": sql_hash(s), "est_cost": 10.0 * i}
            for q, lines in SQL.items() for i, s in enumerate(lines)]
    return pd.DataFrame(rows).iloc[::-1]          # row order must not matter


def test_sql_lines_strip_and_skip_blank_lines(sql_dir: Path) -> None:
    assert sql_lines(str(sql_dir), "job_light_7") == SQL["job_light_7"]


def test_sql_hash_is_sha1_prefix() -> None:
    s = "SELECT 1;"
    assert sql_hash(s) == hashlib.sha1(s.encode()).hexdigest()[:16]


def test_attach_sql_by_plan_idx(sql_dir: Path) -> None:
    df = attach_sql(frame(), str(sql_dir))
    for r in df.itertuples():
        assert r.sql == SQL[r.query][r.plan_idx]
    assert list(df.columns) == ["query", "plan_idx", "sql_sha1", "est_cost", "sql"]


def test_attach_sql_keeps_existing_sql(sql_dir: Path) -> None:
    df = pd.DataFrame({"query": ["job_light_7"], "plan_idx": [0], "sql": ["kept"]})
    assert attach_sql(df, str(sql_dir)) is df


def test_attach_sql_wrong_folder(sql_dir: Path) -> None:
    fixed = [s.replace("NestedLoop(", "NestLoop(") for s in SQL["job_light_7"]]
    (sql_dir / "job_light_7.sql").write_text("\n".join(fixed))
    with pytest.raises(ValueError, match="job_light_7: 1 of 2 lines .* do not match sql_sha1"):
        attach_sql(frame(), str(sql_dir))


def test_attach_sql_line_count(sql_dir: Path) -> None:
    with pytest.raises(ValueError, match="job_light_9: 2 rows but 1 SQL lines"):
        attach_sql(pd.concat([frame(), frame().head(1)]), str(sql_dir))
