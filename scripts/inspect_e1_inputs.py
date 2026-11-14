"""E1 input check: which SQL did the authors run, and do plan positions line up across folders?

Usage (from MDA/):  uv run python scripts/inspect_e1_inputs.py
"""
import glob
import json
import os

from lcm_recheck.config import JOB_LIGHT_IDS as IDS
from lcm_recheck.config import RUNS_PG10_DIR, RUNS_PG16_DIR, SQL_FULL_DIR, SQL_PINNED_DIR
from lcm_recheck.hints import leading

SQL_DIRS = {"full": SQL_FULL_DIR, "pinned": SQL_PINNED_DIR}
RUN_DIRS = {"pg10": RUNS_PG10_DIR, "pg16": RUNS_PG16_DIR}


def norm(s):
    return " ".join(s.split())


def read_sql(d, q):
    p = os.path.join(d, f"job_light_{q}.sql")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return [norm(l) for l in f if l.strip()]


def read_run(d, q):
    p = os.path.join(d, f"job_light_{q}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)["query_list"]


def main():
    sql = {k: {q: read_sql(d, q) for q in IDS} for k, d in SQL_DIRS.items()}
    runs = {k: {q: read_run(d, q) for q in IDS} for k, d in RUN_DIRS.items()}

    print("== Files for the 58 paper queries")
    for k, d in {**SQL_DIRS, **RUN_DIRS}.items():
        src = sql.get(k) or runs.get(k)
        have = [q for q in IDS if src[q] is not None]
        n = sum(len(src[q]) for q in have)
        extra = len(glob.glob(os.path.join(d, "job_light_*"))) - len(have)
        print(f"  {k:<7} {len(have):2d}/58 queries, {n:6d} plans/lines, {extra} other files   ({d})")
        if len(have) < 58:
            print(f"          missing: {[q for q in IDS if src[q] is None]}")

    print("\n== Example hints (job_light_3, first line)")
    for k in SQL_DIRS:
        if sql[k][3]:
            s = sql[k][3][0]
            print(f"  {k}: {s[:s.find('*/') + 2] if '*/' in s else s[:300]}")

    print("\n== Pinned line i has the same Leading(...) as full line i")
    same = tot = lencheck = 0
    for q in IDS:
        a, b = sql["full"][q], sql["pinned"][q]
        if a and b:
            lencheck += len(a) == len(b)
            for x, y in zip(a, b):
                tot += 1; same += leading(x) == leading(y)
    print(f"  {same}/{tot} lines equal; {lencheck} queries have equal line counts")

    print("\n== Which SQL did each run use, and are run entries in SQL line order?")
    for rk in RUN_DIRS:
        for sk in SQL_DIRS:
            found = in_order = tot = 0
            for q in IDS:
                r, s = runs[rk][q], sql[sk][q]
                if not r or not s:
                    continue
                pos = {x: i for i, x in enumerate(s)}
                for i, e in enumerate(r):
                    tot += 1
                    j = pos.get(norm(e["sql"]))
                    found += j is not None
                    in_order += j == i
            print(f"  {rk} runs vs {sk:<6} SQL: {found}/{tot} run SQLs found, {in_order} at the same position")
        n_to = sum(sum(bool(e.get("timeout")) for e in runs[rk][q]) for q in IDS if runs[rk][q])
        hints = {bool(e.get("hint")) for q in IDS if runs[rk][q] for e in runs[rk][q]}
        print(f"  {rk}: {n_to} timed-out entries; separate 'hint' field used: {hints}")


if __name__ == "__main__":
    main()
