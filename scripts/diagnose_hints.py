"""E1 hint diagnosis: which pinned hints does pg_hint_plan actually apply in PG18?

Re-plans every SQL line of the PG18 explain run with pg_hint_plan's debug output switched on and records,
per line: hint syntax errors (pg_hint_plan ignores all hints AFTER the error, and Leading(...) is last in
every pinned line), whether Leading(...) was used, and the hints reported as not used / duplicate / error.
Then relates this to the duplicate plans and to the queries where Sc. PG18 and Sc. PG16 pick different plans.

Usage (from MDA/):
  uv run python scripts/diagnose_hints.py --pg18 results/pg18_explain.csv --port 5418 \
         --eval_root lcm-eval/data/evaluation --out results/pg18_hint_status.csv
Sc. PG18 is read from --extra_root (default results/predictions) if it has a postgres_18 folder, else from --eval_root.
"""
import argparse
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

from lcm_recheck.config import (
    JOIN_ORDER_FOLDER,
    PG16,
    PG18,
    PORTS,
    PRED_ROOT,
    model_root,
)
from lcm_recheck.db import add_db_args, connect

DUMP = re.compile(r"\{used hints:(.*?)\}, \{not used hints:(.*?)\}, \{duplicate hints:(.*?)\}, \{error hints:(.*?)\}",
                  re.DOTALL)
KW = re.compile(r"([A-Za-z]+)\(")


def keywords(s):
    return [] if s.strip() in ("", "(none)") else KW.findall(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg18", required=True,
                    help="extract_runs.py output of the PG18 explain-only run whose SQL lines are re-planned")
    ap.add_argument("--eval_root", required=True, help="the authors' evaluation folder (predictions downloaded from OSF)")
    ap.add_argument("--out", default="results/pg18_hint_status.csv",
                    help="output CSV, one row per SQL line with its hint status")
    ap.add_argument("--extra_root", default=PRED_ROOT,
                    help="generated prediction folders (make_predcsv.py); used for a model that has a folder here")
    add_db_args(ap, port=PORTS[PG18])
    a = ap.parse_args()

    df = pd.read_csv(a.pg18).sort_values(["query", "plan_idx"])
    df["dup"] = df.groupby("query")["shape"].transform(lambda s: s.duplicated(keep=False))

    conn = connect(a)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SET pg_hint_plan.debug_print = 'detailed'; SET client_min_messages = 'log'; "
                "SET pg_hint_plan.parse_messages = 'info'")

    rows = []
    for i, r in enumerate(df.itertuples()):
        conn.notices.clear()
        cur.execute("EXPLAIN (COSTS OFF) " + r.sql)
        cur.fetchall()
        msgs = "\n".join(conn.notices)
        err = re.search(r"hint syntax error at or near \"(.*?)\"\s*DETAIL:\s*(.*)", msgs)
        m = DUMP.findall(msgs)
        used, not_used, dup_h, err_h = m[-1] if m else ("", "", "", "")
        rows.append({"query": r.query, "plan_idx": r.plan_idx, "dup_plan": r.dup,
                     "syntax_error": err.group(2).strip() if err else "",
                     "error_near": err.group(1)[:60] if err else "",
                     "leading_used": "Leading(" in used, "used": used.strip(), "not_used": not_used.strip(),
                     "duplicate_hints": dup_h.strip(), "error_hints": err_h.strip(), "dump_found": bool(m)})
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(df)} lines re-planned", flush=True)
    conn.close()

    t = pd.DataFrame(rows)
    t.to_csv(a.out, index=False)
    n = len(t)
    print(f"\n{n} lines; debug output found for {t.dump_found.sum()}; written to {a.out}")
    print(f"Leading(...) NOT used: {(~t.leading_used).sum()} lines in {t[~t.leading_used]['query'].nunique()} queries")
    print(f"hint syntax errors:    {(t.syntax_error != '').sum()} lines")
    for msg, c in Counter(t.syntax_error[t.syntax_error != ""]).most_common(8):
        print(f"   {c:5d}  {msg}")
    for col in ["not_used", "duplicate_hints", "error_hints"]:
        kw = Counter(k for s in t[col] for k in keywords(s))
        print(f"hint keywords in '{col}': {dict(kw.most_common(8)) or 'none'}")

    print("\nDuplicate plans vs Leading used (rows: duplicate plan, columns: Leading used)")
    print(pd.crosstab(t.dup_plan, t.leading_used).to_string())

    ex = t[~t.leading_used].head(3)
    if len(ex):
        print("\nExamples of lines where Leading(...) was not used:")
        for e in ex.itertuples():
            s = df[(df["query"] == e.query) & (df.plan_idx == e.plan_idx)].sql.iloc[0]
            print(f"  {e.query} line {e.plan_idx}: syntax error: {e.syntax_error or '-'}; near: {e.error_near or '-'}")
            print(f"    hint: {s[:s.find('*/') + 2] if '*/' in s else s[:300]}")
            print(f"    used: {e.used or '-'} | not used: {e.not_used or '-'}")

    print("\nPer query: lines without Leading, and the plan Sc. PG16 / Sc. PG18 select (PG10 runtime, s)")
    out = []
    for q, g in t.groupby("query"):
        row = {"query": q, "lines": len(g), "no_leading": int((~g.leading_used).sum()), "dup": int(g.dup_plan.sum())}
        for model in [PG16, PG18]:
            p = os.path.join(model_root(a.eval_root, a.extra_root, model), model, "imdb", JOIN_ORDER_FOLDER,
                             f"{q}_0_test_pred.csv")
            if os.path.exists(p):
                f = pd.read_csv(p)
                row[model] = f.label.values[np.argmin(f.prediction.values)]
        out.append(row)
    o = pd.DataFrame(out)
    o["diff"] = o.get(PG18, np.nan) - o.get(PG16, np.nan)
    o["k"] = o["query"].str.split("_").str[-1].astype(int)
    show = o[(o.no_leading > 0) | (o.dup > 0) | (o["diff"].abs() > 1e-6)].sort_values("k").drop(columns="k")
    print(show.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\ntotal selected: Sc. PG16 {o[PG16].sum():.1f} s, Sc. PG18 {o[PG18].sum():.1f} s; "
          f"difference from queries without any missing Leading: "
          f"{o[o.no_leading == 0]['diff'].sum():.1f} s, from queries with missing Leading: {o[o.no_leading > 0]['diff'].sum():.1f} s")


if __name__ == "__main__":
    main()
