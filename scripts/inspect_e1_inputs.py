"""E1 input check: which SQL did the authors run, and do plan positions line up across folders?

Usage (from MDA/):  uv run python scripts/inspect_e1_inputs.py
"""
import glob, json, os, re

W = "lcm-eval/data/workloads/evaluation/imdb"
R = "lcm-eval/data/runs/raw/imdb"
SQL_DIRS = {"full": f"{W}/join_order_full", "pinned": f"{W}/join_order_full_pg_10_hinted"}
RUN_DIRS = {"pg10": f"{R}/join_order_full", "pg16": f"{R}/join_order_full_v16"}
IDS = [int(x) for x in ("0 1 2 3 4 5 6 8 9 10 11 12 13 14 15 16 17 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 "
                        "35 36 37 38 41 43 44 45 46 47 49 50 51 52 53 54 55 61 62 63 64 65 66 67 68").split()]


def norm(s):
    return " ".join(s.split())


def leading(s):
    """Content of the Leading(...) hint, with nested parentheses; None if absent."""
    i = s.find("Leading(")
    if i < 0:
        return None
    depth, j = 0, i + len("Leading")
    for k in range(j, len(s)):
        depth += {"(": 1, ")": -1}.get(s[k], 0)
        if depth == 0:
            return norm(s[j + 1:k])
    return None


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