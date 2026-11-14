"""lcm_recheck.runs: load_run() on small inline run files (all three reading modes), and the plan parsers
lines(), exec_ms(), top_cost(), plan_shape(), plan_nodes() on small raw-mode and JSON-mode plans."""
import json
import math
from pathlib import Path

import ijson
import pytest

from lcm_recheck import runs
from lcm_recheck.runs import exec_ms, lines, load_run, plan_nodes, plan_shape, top_cost

RAW_ANALYZE = [["Seq Scan on title t  (cost=0.00..10.00 rows=1 width=4) (actual time=0.01..0.02 rows=1 loops=1)"],
               ["Planning Time: 0.100 ms"],
               ["Execution Time: 12.345 ms"]]

QUERIES = [{"sql": f"SELECT {i};", "verbose_plan": None, "analyze_plans": [], "timeout": False} for i in range(3)]
STATS = {"table_stats": [{"relname": "title", "reltuples": 2528312.0}]}
MODES = ["stream", "json_load", "no_ijson"]


@pytest.fixture(params=MODES)
def read(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """load_run() in one of three modes: streaming with ijson, stream=False, and stream=True without ijson."""
    if request.param == "no_ijson":
        monkeypatch.setattr(runs, "ijson", None)
    stream = request.param != "json_load"

    def call(path: Path, max_queries: int | None = None) -> tuple[list, dict | None]:
        it, stats = load_run(str(path), max_queries, stream=stream)
        return list(it), stats
    call.mode = request.param
    return call


def write(tmp_path: Path, obj: object, name: str = "run.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return p


# ----------------------------------------------------------------------------- load_run
def test_run_file(read, tmp_path: Path) -> None:
    queries, stats = read(write(tmp_path, {"database_stats": STATS, "query_list": QUERIES}))
    assert queries == QUERIES
    assert stats == STATS


def test_missing_database_stats_is_none(read, tmp_path: Path) -> None:
    queries, stats = read(write(tmp_path, {"query_list": QUERIES}))
    assert queries == QUERIES and stats is None


@pytest.mark.parametrize("max_queries, n", [(2, 2), (0, 3), (None, 3), (10, 3)])
def test_max_queries(read, tmp_path: Path, max_queries: int | None, n: int) -> None:
    queries, _ = read(write(tmp_path, {"query_list": QUERIES}), max_queries)
    assert queries == QUERIES[:n]


def test_empty_query_list_is_no_error(read, tmp_path: Path) -> None:
    queries, _ = read(write(tmp_path, {"query_list": [], "database_stats": STATS}))
    assert queries == []


def test_no_query_list_raises(read, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match=r"is not a run file \(no 'query_list'\)"):
        read(write(tmp_path, {"database_stats": STATS, "something_else": [1, 2]}))


def test_top_level_list_raises(read, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="is not a run file"):
        read(write(tmp_path, [{"query_list": QUERIES}]))


def test_parsed_plans_in_head_raises(read, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="is a parsed_plans file"):
        read(write(tmp_path, {"parsed_plans": [], "database_stats": STATS}))


def test_parsed_plans_after_head_raises(read, tmp_path: Path) -> None:
    """The marker only after the first 4096 bytes: found by the full check, not by the head sniff."""
    p = write(tmp_path, {"padding": "x" * 5000, "parsed_plans": []})
    with pytest.raises(SystemExit, match="is a parsed_plans file"):
        read(p)


def test_nan(read, tmp_path: Path) -> None:
    """Documented behaviour: the streaming reader rejects NaN, json.load reads it."""
    p = tmp_path / "nan.json"
    p.write_text('{"query_list": [{"sql": "x", "total": NaN}]}')
    if read.mode == "stream":
        with pytest.raises(ijson.IncompleteJSONError):
            read(p)
    else:
        queries, _ = read(p)
        assert math.isnan(queries[0]["total"])


# ----------------------------------------------------------------------------- lines, exec_ms
def test_lines_raw_rows_and_strings() -> None:
    assert lines([["a"], ("b",), "c", 5]) == ["a", "b", "c", "5"]


def test_exec_ms_raw() -> None:
    assert exec_ms(RAW_ANALYZE) == 12.345


def test_exec_ms_case_insensitive() -> None:
    assert exec_ms([["execution time: 7 ms"]]) == 7.0


def test_exec_ms_json() -> None:
    assert exec_ms({"Plan": {}, "Execution Time": 3.5}) == 3.5


def test_exec_ms_missing_is_nan() -> None:
    assert math.isnan(exec_ms([["Planning Time: 0.1 ms"]]))


# ----------------------------------------------------------------------------- top_cost, plan_shape
RAW_VERBOSE = [["Hash Join  (cost=10.50..250.75 rows=100 width=8)"],
               ["  Output: t.id"],
               ["  ->  Seq Scan on public.title t  (cost=0.00..120.00 rows=1000 width=4)"]]


def test_top_cost_raw_first_line() -> None:
    assert top_cost(RAW_VERBOSE) == 250.75


def test_top_cost_raw_without_cost_is_nan() -> None:
    assert math.isnan(top_cost([["Result"], ["  (cost=0.00..1.00 rows=1 width=4)"]]))


def test_top_cost_json() -> None:
    assert top_cost({"Plan": {"Node Type": "Seq Scan", "Total Cost": 42.5}}) == 42.5


def test_plan_shape_ignores_numbers() -> None:
    other = [[r[0].replace("250.75", "999.99").replace("rows=100 ", "rows=7 ")] for r in RAW_VERBOSE]
    assert plan_shape(other) == plan_shape(RAW_VERBOSE)
    assert len(plan_shape(RAW_VERBOSE)) == 12


def test_plan_shape_differs_by_operator() -> None:
    other = [[RAW_VERBOSE[0][0].replace("Hash Join", "Merge Join")]] + RAW_VERBOSE[1:]
    assert plan_shape(other) != plan_shape(RAW_VERBOSE)


def test_plan_shape_json() -> None:
    a = {"Plan": {"Node Type": "Seq Scan", "Total Cost": 1.0}}
    b = {"Plan": {"Total Cost": 2.5, "Node Type": "Seq Scan"}}
    assert plan_shape(a) == plan_shape(b)


# ----------------------------------------------------------------------------- plan_nodes
RAW_ANALYZE_TREE = [
    ["Gather  (cost=1000.00..2000.00 rows=10 width=4) (actual time=1.0..2.0 rows=12 loops=1)"],
    ["  ->  Nested Loop  (cost=0.43..900.00 rows=4 width=4) (actual time=0.5..1.5 rows=5 loops=3)"],
    [("        ->  Parallel Seq Scan on public.title t  (cost=0.00..500.00 rows=4 width=4) "
      "(actual time=0.1..0.9 rows=6 loops=3)")],
    ["              Filter: (t.production_year > 2000)"],
    ["              Rows Removed by Filter: 123"],
    [("        ->  Index Scan using movie_id_idx on movie_info mi  (cost=0.43..0.50 rows=1 width=4) "
      "(actual time=0.01..0.02 rows=1 loops=18)")],
    ["              Index Cond: (mi.movie_id = t.id)"],
    ["Planning Time: 0.1 ms"],
    ["Execution Time: 2.5 ms"],
]


def test_plan_nodes_raw() -> None:
    nodes = plan_nodes(RAW_ANALYZE_TREE)
    assert [n["op"] for n in nodes] == ["Gather", "Nested Loop", "Seq Scan", "Index Scan"]
    gather, nl, seq, idx = nodes
    assert gather == {"op": "Gather", "kind": None, "parallel": False, "rel": None, "rows": 12, "loops": 1,
                      "removed": 0, "cond": None}
    assert nl["rows"] == 5 and nl["loops"] == 3
    assert seq == {"op": "Seq Scan", "kind": "seq", "parallel": True, "rel": "title", "rows": 6, "loops": 3,
                   "removed": 123, "cond": None}
    assert idx == {"op": "Index Scan", "kind": "index", "parallel": False, "rel": "movie_info", "rows": 1,
                   "loops": 18, "removed": 0, "cond": "(mi.movie_id = t.id)"}


def test_plan_nodes_raw_recheck_cond_and_no_actuals() -> None:
    nodes = plan_nodes([["Bitmap Heap Scan on title t  (cost=5.00..50.00 rows=10 width=4)"],
                        ["  Recheck Cond: (t.kind_id = 1)"]])
    assert nodes == [{"op": "Bitmap Heap Scan", "kind": "bitmap", "parallel": False, "rel": "title",
                      "rows": None, "loops": None, "removed": 0, "cond": "(t.kind_id = 1)"}]


JSON_PLAN = {"Plan": {"Node Type": "Hash Join", "Actual Rows": 3, "Actual Loops": 1, "Plans": [
    {"Node Type": "Seq Scan", "Relation Name": "title", "Parallel Aware": True, "Actual Rows": 10,
     "Actual Loops": 2, "Rows Removed by Filter": 4},
    {"Node Type": "Index Only Scan", "Relation Name": "cast_info", "Actual Rows": 1, "Actual Loops": 10,
     "Index Cond": "(movie_id = t.id)"}]}}
JSON_NODES = [
    {"op": "Hash Join", "kind": None, "parallel": False, "rel": None, "rows": 3, "loops": 1, "removed": 0,
     "cond": None},
    {"op": "Seq Scan", "kind": "seq", "parallel": True, "rel": "title", "rows": 10, "loops": 2, "removed": 4,
     "cond": None},
    {"op": "Index Only Scan", "kind": "index", "parallel": False, "rel": "cast_info", "rows": 1, "loops": 10,
     "removed": 0, "cond": "(movie_id = t.id)"},
]


@pytest.mark.parametrize("plan", [JSON_PLAN, [JSON_PLAN], JSON_PLAN["Plan"]], ids=["dict", "list", "bare_node"])
def test_plan_nodes_json(plan: object) -> None:
    assert plan_nodes(plan) == JSON_NODES
