"""Real-data check of the JSON-mode plan parsing in lcm_recheck.runs against raw mode.

The authors ran the PG10 JOB-light workload twice, once with text EXPLAIN (runs/raw) and once with JSON EXPLAIN
(runs/json, a separate execution: its verbose plans come from EXPLAIN ANALYZE and its timeouts differ). For every
plan that has a verbose plan in both runs, top_cost() must agree exactly, and plan_shape() must group the plans of
each query the same way (the hashes themselves differ between formats). Also reports exec_ms() on the JSON analyze
plans and how plan_nodes() compares (operator names of aggregates differ between formats).

Reads the JSON run files from --json_dir if it exists, otherwise from --zip in memory (never extracts).
Prints counts only, deterministic. Exit status 1 if top_cost or the shape grouping disagree on any plan; exit 0 with
a note if the data is missing.

Usage (from the repository root):
  uv run python tools/check_json_vs_raw.py
  uv run python tools/check_json_vs_raw.py --json_dir <extracted runs/json/imdb/join_order_full>
"""
import argparse
import collections
import json
import math
import os
import re
import sys
import zipfile
from collections.abc import Iterator
from typing import Any

from lcm_recheck.config import DATA_ROOT, RUNS_JSON_PG10_DIR, RUNS_PG10_DIR, RUNS_ZIP
from lcm_recheck.runs import exec_ms, plan_nodes, plan_shape, top_cost

AGG_RE = re.compile(r"^(?:Finalize |Partial )")


def json_files(json_dir: str, zip_path: str) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """(file name, query_list) of every job_light_*.json, from the folder or else from the zip archive."""
    if os.path.isdir(json_dir):
        for name in sorted(n for n in os.listdir(json_dir) if n.startswith("job_light_") and n.endswith(".json")):
            with open(os.path.join(json_dir, name)) as f:
                yield name, json.load(f)["query_list"]
        return
    prefix = os.path.relpath(json_dir, DATA_ROOT).replace(os.sep, "/") + "/"
    with zipfile.ZipFile(zip_path) as z:
        for member in sorted(m for m in z.namelist() if m.startswith(prefix + "job_light_") and m.endswith(".json")):
            yield os.path.basename(member), json.loads(z.read(member))["query_list"]


def grouping(shapes: dict[int, str]) -> list[list[int]]:
    """Plan indices grouped by equal hash, in a canonical order."""
    groups = collections.defaultdict(list)
    for i, s in shapes.items():
        groups[s].append(i)
    return sorted(groups.values())


def node_key(nodes: list[dict[str, Any]], agg_names: bool) -> list[tuple]:
    return [(n["op"] if agg_names else AGG_RE.sub("", n["op"]), n["kind"], n["parallel"], n["rel"]) for n in nodes]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw_dir", default=RUNS_PG10_DIR, help="raw-mode run files (default: %(default)s)")
    ap.add_argument("--json_dir", default=RUNS_JSON_PG10_DIR,
                    help="JSON-mode run files; read from --zip if this folder does not exist (default: %(default)s)")
    ap.add_argument("--zip", default=RUNS_ZIP, help="OSF archive holding runs/json (default: %(default)s)")
    a = ap.parse_args()

    if not os.path.isdir(a.raw_dir) or not (os.path.isdir(a.json_dir) or os.path.isfile(a.zip)):
        print(f"skipped: needs {a.raw_dir} and {a.json_dir} (or {a.zip})")
        return 0

    c = collections.Counter()
    for name, jq in json_files(a.json_dir, a.zip):
        raw_path = os.path.join(a.raw_dir, name)
        if not os.path.exists(raw_path):
            c["JSON files without a raw file"] += 1
            continue
        with open(raw_path) as f:
            rq = json.load(f)["query_list"]
        c["files paired"] += 1
        c["candidate plans"] += len(jq)
        c["files with a different plan count"] += len(jq) != len(rq)
        shapes_j, shapes_r = {}, {}
        for i, (qj, qr) in enumerate(zip(jq, rq)):
            c["plans with the same SQL"] += qj.get("sql") == qr.get("sql")
            vj, vr = qj.get("verbose_plan"), qr.get("verbose_plan")
            if vj and vr:
                c["plans with a verbose plan in both"] += 1
                c[f"JSON verbose_plan type {type(vj).__name__}"] += 1
                cj, cr = top_cost(vj), top_cost(vr)
                c["top_cost equal"] += cj == cr
                c["top_cost differs"] += cj != cr
                shapes_j[i], shapes_r[i] = plan_shape(vj), plan_shape(vr)
                nj, nr = plan_nodes(vj), plan_nodes(vr)
                c["verbose plan_nodes equal"] += node_key(nj, True) == node_key(nr, True)
                c["verbose plan_nodes equal without Finalize/Partial"] += node_key(nj, False) == node_key(nr, False)
            for p in qj.get("analyze_plans") or []:
                c["JSON analyze plans"] += 1
                c["JSON analyze plans with a finite exec_ms"] += math.isfinite(exec_ms(p))
            aj, ar = qj.get("analyze_plans") or [], qr.get("analyze_plans") or []
            if aj and ar:
                c["plans executed in both"] += 1
                nj, nr = plan_nodes(aj[0]), plan_nodes(ar[0])
                c["analyze plan_nodes equal"] += node_key(nj, True) == node_key(nr, True)
                c["analyze plan_nodes equal without Finalize/Partial"] += node_key(nj, False) == node_key(nr, False)
        same = grouping(shapes_j) == grouping(shapes_r)
        c["files with the same plan_shape grouping"] += same
        c["files with a different plan_shape grouping"] += not same
        c["distinct raw shapes (summed over files)"] += len(set(shapes_r.values()))

    for k in sorted(c):
        print(f"{k}: {c[k]}")
    ok = c["files paired"] > 0 and c["top_cost differs"] == 0 and c["files with a different plan_shape grouping"] == 0
    print("result:", "OK" if ok else "DISAGREEMENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
