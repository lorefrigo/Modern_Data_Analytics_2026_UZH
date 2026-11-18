"""E1: build prediction files for another PostgreSQL version exactly the way the authors built Sc. PG16.

Reconstructed rule (checked with check_pinned.py, 58/58 queries): the candidates of a query are the plans of
join_order_full_pg_10_hinted (PG10's physical plans, pinned by hints) that have a runtime in the published PG10
run; rows are in pinned-file order; label = PG10 runtime; prediction = the version's estimated cost.

The version's plans are linked to PG10 plans through their Leading(...) hint (the join order), not by line number:
pinned line i is generally NOT PG10 plan i. Raw costs are used: a positive linear rescaling ("Scaled PG")
never changes which plan has the lowest estimate.

Usage (from the repository root):
  uv run python scripts/make_predcsv.py --label postgres_18 --pg10 results/pg10_paper.csv \
         --costs results/pg18_explain_fixed.csv --sql_dir results/sql_pinned_fixed --eval_root lcm-eval/data/evaluation
  uv run python scripts/make_predcsv.py --label postgres_18_asis --pg10 results/pg10_paper.csv \
         --costs results/pg18_explain.csv --eval_root lcm-eval/data/evaluation
Writes <out_root>/<label>/imdb/join_order_full/job_light_<k>_0_test_pred.csv (out_root default: results/predictions),
the layout sigtests.py paper --extra_root reads. eval_root is only read (the authors' Sc. PG10 and Sc. PG16, for the
comparison printed at the end). An existing label folder is not overwritten unless --overwrite is given.
A --costs CSV without SQL (extract_runs.py --no_sql) gets it from --sql_dir (hints.attach_sql, checked against its
hash).
If --costs comes from the Leading-only SQL instead, the same code yields "all PG10-executed plans" as candidates.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from lcm_recheck.config import JOB_LIGHT_IDS, JOIN_ORDER_FOLDER, PG16, PRED_ROOT, REF, SQL_PINNED_DIR
from lcm_recheck.hints import attach_sql, leading


def ref_file(root, model, q):
    p = os.path.join(root, model, "imdb", JOIN_ORDER_FOLDER, f"{q}_0_test_pred.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True,
                    help="prediction folder name, e.g. postgres_18 (sigtests.py uses the labels in config.PG_VERSIONS)")
    ap.add_argument("--pg10", required=True, help="extract_runs.py output of the authors' PG10 runs")
    ap.add_argument("--costs", required=True, help="extract_runs.py output of the version's explain-only runs")
    ap.add_argument("--sql_dir", default=SQL_PINNED_DIR,
                    help="folder with the job_light_<k>.sql the costs were computed on, read if the CSV has no SQL "
                         "(extract_runs.py --no_sql); results/sql_pinned_fixed for the fixed SQL")
    ap.add_argument("--eval_root", required=True,
                    help="authors' evaluation root (read only: Sc. PG10 and Sc. PG16 for the comparison)")
    ap.add_argument("--out_root", default=PRED_ROOT, help="output root; files go to <out_root>/<label>/imdb/join_order_full/")
    ap.add_argument("--overwrite", action="store_true", help="replace the files of an existing label folder")
    a = ap.parse_args()
    root = os.path.expanduser(a.eval_root)

    out = os.path.join(os.path.expanduser(a.out_root), a.label, "imdb", JOIN_ORDER_FOLDER)
    if glob.glob(os.path.join(out, "*_test_pred.csv")) and not a.overwrite:
        sys.exit(f"{out} already holds prediction files; pass --overwrite to replace them")
    pg10, costs = pd.read_csv(a.pg10), attach_sql(pd.read_csv(a.costs), a.sql_dir)
    pg10["ok"] = ~pg10.timeout & pg10.runtime_s.notna()
    os.makedirs(out, exist_ok=True)

    stats, problems = [], []
    for k in JOB_LIGHT_IDS:
        q = f"job_light_{k}"
        d10 = pg10[pg10["query"] == q].set_index("plan_idx")
        dv = costs[costs["query"] == q].sort_values("plan_idx")
        if dv.empty:
            problems.append(f"{q}: no {a.label} plans"); continue
        lead2idx = {leading(s): i for i, s in d10.sql.items()}
        dv = dv.assign(pg10_idx=dv.sql.map(lambda s, lead2idx=lead2idx: lead2idx.get(leading(s))))
        n_unmapped, n_nocost = int(dv.pg10_idx.isna().sum()), int(dv.est_cost.isna().sum())
        if n_unmapped or n_nocost:
            problems.append(f"{q}: {n_unmapped} {a.label} plans without a PG10 join order, {n_nocost} without a cost")
        dv = dv[dv.pg10_idx.notna() & dv.est_cost.notna()]
        dv = dv[dv.pg10_idx.map(lambda i, d10=d10: bool(d10.loc[int(i), "ok"]))]          # keep plans with a PG10 runtime
        idx = dv.pg10_idx.astype(int).values
        label, pred = d10.loc[idx, "runtime_s"].values, dv.est_cost.values

        pd.DataFrame({"query_index": range(len(idx)), "label": label, "prediction": pred, "qerror": np.nan}) \
            .to_csv(os.path.join(out, f"{q}_0_test_pred.csv"), index=False)

        s = {"query": q, "n": len(idx), "selected": label[np.argmin(pred)], "optimal": label.min()}
        for model in [PG16, REF]:
            r = ref_file(root, model, q)
            s[f"same_as_{model}"] = r is not None and len(r) == len(idx) and \
                np.allclose(r.label.values, label, rtol=1e-3)
            if model == PG16 and s[f"same_as_{model}"] and len(idx) >= 3:
                s["rho_pg16"] = spearmanr(pred, r.prediction.values)[0]
        if len(idx) >= 3:
            s["rho_pg10"] = spearmanr(pred, d10.loc[idx, "est_cost"].values)[0]
        stats.append(s)

    t = pd.DataFrame(stats)
    print(f"written {len(t)} files to {out}")
    for p in problems:
        print("  WARNING", p)
    print(f"candidates and labels identical to Sc. PG16 (row by row): {t.same_as_postgres_16.sum()}/58   "
          f"(expected 58 if the costs come from the pinned SQL)")
    print(f"candidates and labels identical to Sc. PG10 (row by row): {t.same_as_postgres.sum()}/58   "
          f"(expected 44: same as Sc. PG16 vs Sc. PG10)")
    print(f"{a.label} totals: selected {t.selected.sum():.1f} s, optimal {t.optimal.sum():.1f} s "
          f"(Sc. PG16: 562.9 s and 445.8 s)")
    for col, name in [("rho_pg16", "Sc. PG16 predictions"), ("rho_pg10", "PG10 costs")]:
        if col in t:
            r = t.dropna(subset=[col]).sort_values(col)
            print(f"Spearman rank correlation of {a.label} costs with {name}: median {r[col].median():.3f} over {len(r)} "
                  f"queries; lowest: " + ", ".join(f"{x['query']} {x[col]:.2f} (n={x['n']})" for _, x in r.head(4).iterrows()))


if __name__ == "__main__":
    main()
