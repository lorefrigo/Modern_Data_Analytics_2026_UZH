"""Was the authors' Sc. PG16 hit by the same hint bug ("NestedLoop" is not a pg_hint_plan keyword)?

A pinned line with that keyword loses all hints after it, including Leading(...), so the planner picks its
own plan: lines with the same hint prefix get the SAME plan and therefore EXACTLY tied costs. If PG16 ran
the same broken SQL, the authors' Sc. PG16 file must show tied predictions on exactly the broken rows.

Rows of the Sc. PG16 / Sc. PG18 files = pinned lines with a PG10 runtime, in pinned order (verified 58/58).

Usage (from MDA/):
  uv run python scripts/check_pg16_ties.py --pg10 results/pg10_paper.csv --pg18 results/pg18_explain.csv \
         --status results/pg18_hint_status.csv --eval_root lcm-eval/data/evaluation
"""
import argparse, os
import numpy as np, pandas as pd


def leading(s):
    i = s.find("Leading(")
    if i < 0:
        return None
    depth, j = 0, i + len("Leading")
    for k in range(j, len(s)):
        depth += {"(": 1, ")": -1}.get(s[k], 0)
        if depth == 0:
            return " ".join(s[j + 1:k].split())
    return None


ap = argparse.ArgumentParser()
ap.add_argument("--pg10", required=True); ap.add_argument("--pg18", required=True)
ap.add_argument("--status", required=True); ap.add_argument("--eval_root", required=True)
ap.add_argument("--models", nargs="+", default=["postgres_16", "postgres_18"])
a = ap.parse_args()

pg10, pg18 = pd.read_csv(a.pg10), pd.read_csv(a.pg18)
pg10["ok"] = ~pg10.timeout & pg10.runtime_s.notna()
status = pd.read_csv(a.status)[["query", "plan_idx", "leading_used"]]
pg18 = pg18.merge(status, on=["query", "plan_idx"])

tab = {m: [] for m in a.models}
per_q = []
for q in sorted(pg18["query"].unique(), key=lambda s: int(s.split("_")[-1])):
    d10 = pg10[pg10["query"] == q].set_index("plan_idx")
    lead2idx = {leading(s): i for i, s in d10.sql.items()}
    d = pg18[pg18["query"] == q].sort_values("plan_idx")
    d = d[d.sql.map(lambda s: bool(d10.loc[lead2idx[leading(s)], "ok"]))]      # rows of the prediction files
    broken = ~d.leading_used.values
    row = dict(query=q, rows=len(d), broken=int(broken.sum()))
    for m in a.models:
        f = pd.read_csv(os.path.join(a.eval_root, m, "imdb", "join_order_full", f"{q}_0_test_pred.csv"))
        if len(f) != len(d):
            row[f"{m}_note"] = "row count differs"; continue
        p = pd.Series(np.round(f.prediction.values, 9))
        tied = p.duplicated(keep=False).values
        tab[m].append(pd.DataFrame(dict(broken=broken, tied=tied)))
        sel = int(np.argmin(f.prediction.values))
        row[f"{m}_tied_broken"] = int((tied & broken).sum())
        row[f"{m}_tied_clean"] = int((tied & ~broken).sum())
        row[f"{m}_selects_broken"] = bool(broken[sel])
    per_q.append(row)

for m, parts in tab.items():
    t = pd.concat(parts)
    print(f"\n== {m}: rows tied with another row (exactly equal prediction) vs broken hint line")
    print(pd.crosstab(t.broken.map({True: "broken line", False: "clean line"}),
                      t.tied.map({True: "tied", False: "not tied"})).to_string())
o = pd.DataFrame(per_q)
print("\nQueries with broken lines:")
print(o[o.broken > 0].to_string(index=False))
for m in a.models:
    c = f"{m}_selects_broken"
    if c in o:
        print(f"{m}: selected plan is a broken line in {int(o[c].sum())} of {int((o.broken > 0).sum())} affected queries")