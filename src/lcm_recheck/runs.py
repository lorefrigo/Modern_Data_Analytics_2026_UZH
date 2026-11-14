"""Parsing of lcm-eval run files (raw or JSON EXPLAIN mode).

A run file is a JSON object with a 'query_list' (sql, verbose_plan, analyze_plans and timeout per query) and,
in the authors' files, 'database_stats'. Contents:
  load_run()              read a run file: (iterator over query entries, database_stats)
  lines(), exec_ms()      raw-mode rows as strings; execution time of one EXPLAIN ANALYZE output
  top_cost(), plan_shape()  top-node cost and plan-shape hash of a verbose plan (extract_runs.py)
  plan_nodes()            all operator nodes of one EXPLAIN output (bias_data.py)

Usage:
  from lcm_recheck.runs import exec_ms, load_run
  queries, stats = load_run("job_light_0.json", stream=False)
  ms = exec_ms(next(queries)["analyze_plans"][0])
"""
import hashlib
import json
import re
from collections.abc import Iterator
from typing import Any

try:
    import ijson
except ImportError:
    ijson = None

EXEC_RE = re.compile(r"execution time:\s*([\d.]+)\s*ms", re.IGNORECASE)
COST_RE = re.compile(r"cost=([\d.]+)\.\.([\d.]+)")
NUM_RE = re.compile(r"[\d.]+")

SCAN_KIND = {"Seq Scan": "seq", "Index Scan": "index", "Index Only Scan": "index", "Bitmap Heap Scan": "bitmap"}
NODE_RE = re.compile(r"^(?P<op>[A-Z][A-Za-z ]*?)(?: Backward)?(?: using (?P<idx>\S+))?"
                     r"(?: on (?P<rel>\S+)(?: (?P<alias>\S+))?)?\s+\(cost=")
ACT_RE = re.compile(r"\(actual time=[\d.]+\.\.[\d.]+ rows=(\d+) loops=(\d+)\)")


# ----------------------------------------------------------------------------- reading run files
def _not_a_run_file(path: str, parsed_plans: bool) -> SystemExit:
    if parsed_plans:
        return SystemExit(f"{path} is a parsed_plans file. Use the run file of the same workload from runs/raw "
                          f"(or runs/json) instead.")
    return SystemExit(f"{path} is not a run file (no 'query_list'); use a run file from runs/raw or runs/json")


def _top_level_keys(path: str) -> set[str] | None:
    """Keys of the top-level JSON object (streamed), or None if the top level is not an object."""
    keys: set[str] = set()
    with open(path, "rb") as f:
        for i, (prefix, event, value) in enumerate(ijson.parse(f, use_float=True)):
            if i == 0 and event != "start_map":
                return None
            if prefix == "" and event == "map_key":
                keys.add(value)
    return keys


def load_run(path: str, max_queries: int | None = None,
             stream: bool = True) -> tuple[Iterator[dict[str, Any]], dict[str, Any] | None]:
    """Returns (iterator over query entries, database_stats or None).

    max_queries: read at most this many entries (None or 0: all).
    stream=True reads the file with ijson, one entry at a time (large training files); it falls back to json.load
    if ijson is not installed. stream=False always loads the whole file with json.load.
    A parsed_plans file, or any file without a top-level 'query_list', raises SystemExit in both modes (when
    streaming, only once the iterator finds no entries). An empty 'query_list' gives no entries and no error.
    NaN/Infinity: the streaming reader raises ijson.IncompleteJSONError on NaN or Infinity and on integers outside
    64 bits; json.load accepts them (as float('nan'), float('inf') and Python ints).
    """
    with open(path, "rb") as f:
        head = f.read(4096).decode("utf-8", "ignore")
    if '"parsed_plans"' in head and '"query_list"' not in head:
        raise _not_a_run_file(path, parsed_plans=True)
    if stream and ijson is not None:
        with open(path, "rb") as f:
            stats = next(ijson.items(f, "database_stats", use_float=True), None)

        def gen() -> Iterator[dict[str, Any]]:
            n = 0
            with open(path, "rb") as f:
                for i, q in enumerate(ijson.items(f, "query_list.item", use_float=True)):
                    if max_queries and i >= max_queries:
                        break
                    n += 1
                    yield q
            if n == 0:                                  # empty query_list, or none at all?
                keys = _top_level_keys(path)
                if keys is None or "query_list" not in keys:
                    raise _not_a_run_file(path, parsed_plans=keys is not None and "parsed_plans" in keys)
        return gen(), stats
    with open(path) as f:
        run = json.load(f)
    if not isinstance(run, dict) or "query_list" not in run:
        raise _not_a_run_file(path, parsed_plans=isinstance(run, dict) and "parsed_plans" in run)
    ql = run["query_list"][:max_queries] if max_queries else run["query_list"]
    return iter(ql), run.get("database_stats")


# ----------------------------------------------------------------------------- plans
def lines(plan: list[Any]) -> list[str]:
    """Raw-mode plans are lists of 1-element rows; return them as a list of strings."""
    return [r[0] if isinstance(r, (list, tuple)) else str(r) for r in plan]


def exec_ms(analyze_plan: dict[str, Any] | list[Any]) -> float:
    if isinstance(analyze_plan, dict):
        return float(analyze_plan["Execution Time"])
    for line in lines(analyze_plan):
        m = EXEC_RE.search(line)
        if m:
            return float(m.group(1))
    return float("nan")


def top_cost(verbose_plan: dict[str, Any] | list[Any]) -> float:
    if isinstance(verbose_plan, dict):                       # JSON mode
        return float(verbose_plan["Plan"]["Total Cost"])
    m = COST_RE.search(lines(verbose_plan)[0])                # raw mode: top node is first line
    return float(m.group(2)) if m else float("nan")


def plan_shape(verbose_plan: dict[str, Any] | list[Any]) -> str:
    """Hash of the plan with all numbers removed: equal hashes = same physical plan."""
    txt = json.dumps(verbose_plan["Plan"], sort_keys=True) if isinstance(verbose_plan, dict) \
        else "\n".join(lines(verbose_plan))
    return hashlib.md5(NUM_RE.sub("#", txt).encode()).hexdigest()[:12]


def clean_rel(rel: str | None) -> str | None:
    return rel.split(".")[-1].strip('"') if rel else rel


def plan_nodes(plan: dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
    """All operator nodes of one EXPLAIN output (raw text rows or JSON), with the fields used here."""
    if isinstance(plan, list) and plan and isinstance(plan[0], dict):
        plan = plan[0]
    if isinstance(plan, dict):
        out = []

        def walk(n: dict[str, Any]) -> None:
            op, par = n.get("Node Type", ""), bool(n.get("Parallel Aware"))
            out.append({"op": op, "kind": SCAN_KIND.get(op), "parallel": par,
                        "rel": clean_rel(n.get("Relation Name")),
                        "rows": n.get("Actual Rows"), "loops": n.get("Actual Loops"),
                        "removed": n.get("Rows Removed by Filter", 0) or 0,
                        "cond": n.get("Index Cond") or n.get("Recheck Cond")})
            for c in n.get("Plans", []):
                walk(c)
        walk(plan.get("Plan", plan))
        return out
    out, cur = [], None
    for line in lines(plan):
        t = line.strip()
        if t.startswith("->"):
            t = t[2:].strip()
        if "(cost=" in t:
            m, act = NODE_RE.match(t), ACT_RE.search(t)
            op = m.group("op").strip() if m else ""
            par = op.startswith("Parallel ")
            op = op[len("Parallel "):] if par else op
            cur = {"op": op, "kind": SCAN_KIND.get(op), "parallel": par,
                   "rel": clean_rel(m.group("rel") if m else None),
                   "rows": int(act.group(1)) if act else None, "loops": int(act.group(2)) if act else None,
                   "removed": 0, "cond": None}
            out.append(cur)
        elif cur is not None:
            if t.startswith("Rows Removed by Filter:"):
                cur["removed"] = int(t.split(":")[1])
            elif t.startswith(("Index Cond:", "Recheck Cond:")):
                cur["cond"] = t.split(":", 1)[1].strip()
    return out
