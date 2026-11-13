"""E1: build Sc. PG18 prediction files exactly the way the authors built Sc. PG16.

Reconstructed rule (checked with check_pinned.py, 58/58 queries): the candidates of a query are the plans of
join_order_full_pg_10_hinted (PG10's physical plans, pinned by hints) that have a runtime in the published PG10
run; rows are in pinned-file order; label = PG10 runtime; prediction = the version's estimated cost.

PG18 plans are linked to PG10 plans through their Leading(...) hint (the join order), not by line number:
pinned line i is generally NOT PG10 plan i. Raw PG18 costs are used: a positive linear rescaling ("Scaled PG")
never changes which plan has the lowest estimate.

Usage (from MDA/):
  uv run python scripts/make_pg18_predcsv.py --pg10 results/pg10_paper.csv --pg18 results/pg18_explain.csv \
         --eval_root lcm-eval/data/evaluation
Writes <eval_root>/postgres_18/imdb/join_order_full/job_light_<k>_0_test_pred.csv.
If --pg18 comes from the Leading-only SQL instead, the same code yields "all PG10-executed plans" as candidates.
"""
import argparse, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr

from lcm_recheck.config import JOB_LIGHT_IDS, REF, PG16, PG18, JOIN_ORDER_FOLDER


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


def ref_file(root, model, q):
    p = os.path.join(root, model, "imdb", JOIN_ORDER_FOLDER, f"{q}_0_test_pred.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


ap = argparse.ArgumentParser()
ap.add_argument("--pg10", required=True, help="extract_runs.py output of the authors' PG10 runs")
ap.add_argument("--pg18", required=True, help="extract_runs.py output of the PG18 explain-only runs")
ap.add_argument("--eval_root", required=True)
ap.add_argument("--model", default=PG18, help="output folder name under eval_root")
a = ap.parse_args()
root = os.path.expanduser(a.eval_root)

pg10, pg18 = pd.read_csv(a.pg10), pd.read_csv(a.pg18)
pg10["ok"] = ~pg10.timeout & pg10.runtime_s.notna()
out = os.path.join(root, a.model, "imdb", JOIN_ORDER_FOLDER)
os.makedirs(out, exist_ok=True)

stats, problems = [], []
for k in JOB_LIGHT_IDS:
    q = f"job_light_{k}"
    d10 = pg10[pg10["query"] == q].set_index("plan_idx")
    d18 = pg18[pg18["query"] == q].sort_values("plan_idx")
    if d18.empty:
        problems.append(f"{q}: no PG18 plans"); continue
    lead2idx = {leading(s): i for i, s in d10.sql.items()}
    d18 = d18.assign(pg10_idx=d18.sql.map(lambda s: lead2idx.get(leading(s))))
    n_unmapped, n_nocost = int(d18.pg10_idx.isna().sum()), int(d18.est_cost.isna().sum())
    if n_unmapped or n_nocost:
        problems.append(f"{q}: {n_unmapped} PG18 plans without a PG10 join order, {n_nocost} without a cost")
    d18 = d18[d18.pg10_idx.notna() & d18.est_cost.notna()]
    d18 = d18[d18.pg10_idx.map(lambda i: bool(d10.loc[int(i), "ok"]))]          # keep plans with a PG10 runtime
    idx = d18.pg10_idx.astype(int).values
    label, pred = d10.loc[idx, "runtime_s"].values, d18.est_cost.values

    pd.DataFrame(dict(query_index=range(len(idx)), label=label, prediction=pred, qerror=np.nan)) \
        .to_csv(os.path.join(out, f"{q}_0_test_pred.csv"), index=False)

    s = dict(query=q, n=len(idx), selected=label[np.argmin(pred)], optimal=label.min())
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
      f"(expected 58 if PG18 ran the pinned SQL)")
print(f"candidates and labels identical to Sc. PG10 (row by row): {t.same_as_postgres.sum()}/58   "
      f"(expected 44: same as Sc. PG16 vs Sc. PG10)")
print(f"Sc. PG18 totals: selected {t.selected.sum():.1f} s, optimal {t.optimal.sum():.1f} s "
      f"(Sc. PG16: 562.9 s and 445.8 s)")
for col, name in [("rho_pg16", "Sc. PG16 predictions"), ("rho_pg10", "PG10 costs")]:
    if col in t:
        r = t.dropna(subset=[col]).sort_values(col)
        print(f"Spearman rank correlation of PG18 costs with {name}: median {r[col].median():.3f} over {len(r)} "
              f"queries; lowest: " + ", ".join(f"{x['query']} {x[col]:.2f} (n={x['n']})" for _, x in r.head(4).iterrows()))