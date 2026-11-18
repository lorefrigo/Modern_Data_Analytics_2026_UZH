"""scripts/extract_runs.py on an invented run file: the sql column, or only its hash with --no_sql."""
import json
import runpy
import sys
from pathlib import Path

import pandas as pd
import pytest

from lcm_recheck.hints import sql_hash

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "extract_runs.py"
PLAN = [["Seq Scan on t  (cost=0.00..42.50 rows=10 width=4)"]]
QUERIES = [{"sql": "/*+ Leading((t u)) */ SELECT 1;", "verbose_plan": PLAN, "analyze_plans": [], "timeout": False},
           {"sql": "/*+ Leading((u t)) */ SELECT 2;", "verbose_plan": PLAN, "analyze_plans": [], "timeout": False}]


def run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *extra: str) -> pd.DataFrame:
    runs = tmp_path / "runs"
    runs.mkdir(exist_ok=True)
    (runs / "job_light_3.json").write_text(json.dumps({"query_list": QUERIES}))
    out = tmp_path / "out.csv"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--runs", str(runs), "--version", "v", "--out", str(out), *extra])
    runpy.run_path(str(SCRIPT), run_name="not_main")["main"]()
    return pd.read_csv(out)


def test_sql_column(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    df = run(monkeypatch, tmp_path)
    assert list(df.columns[:4]) == ["version", "query", "plan_idx", "sql"]
    assert list(df.sql) == [q["sql"] for q in QUERIES]
    assert list(df.est_cost) == [42.5, 42.5]


def test_no_sql_writes_only_the_hash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    df = run(monkeypatch, tmp_path, "--no_sql")
    assert list(df.columns[:4]) == ["version", "query", "plan_idx", "sql_sha1"]
    assert "sql" not in df.columns
    assert list(df.sql_sha1) == [sql_hash(q["sql"]) for q in QUERIES]
    assert not (tmp_path / "out.csv").read_text().count("SELECT")
