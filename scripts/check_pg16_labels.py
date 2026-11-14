"""Which runtimes and candidate set did the authors use for Sc. PG10 and Sc. PG16 (Task 1, join order)?

Rebuilds each query's candidate set from the extracted raw runs under four hypotheses and compares the
sorted labels with the authors' prediction files. It also checks that the authors' predictions are a
linear function of PG16's (or PG10's) estimated costs.

Usage (from MDA/):
  uv run python scripts/check_pg16_labels.py --pg10 results/pg10_paper.csv --pg16 results/pg16_paper.csv \
         --eval_root lcm-eval/data/evaluation
"""
import argparse
import os

import numpy as np
import pandas as pd

from lcm_recheck.config import JOIN_ORDER_FOLDER, PG16, REF

HYP = {  # name: (runtimes used as labels, which plans are candidates)
    "PG10 runtimes, PG10 executed": ("pg10", "ok10"),
    "PG10 runtimes, executed in both": ("pg10", "ok_both"),
    "PG16 runtimes, PG16 executed": ("pg16", "ok16"),
    "PG16 runtimes, executed in both": ("pg16", "ok_both"),
}


def candidates(both, runs, q, labels_from, keep):
    idx = both[(both["query"] == q) & both[keep]].plan_idx
    d = runs[labels_from]
    return d[(d["query"] == q) & d.plan_idx.isin(idx)].sort_values("plan_idx")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg10", required=True); ap.add_argument("--pg16", required=True)
    ap.add_argument("--eval_root", required=True)
    a = ap.parse_args()

    runs = {"pg10": pd.read_csv(a.pg10), "pg16": pd.read_csv(a.pg16)}
    for d in runs.values():
        d["ok"] = ~d.timeout & d.runtime_s.notna()
    both = runs["pg10"][["query", "plan_idx", "ok"]].merge(runs["pg16"][["query", "plan_idx", "ok"]],
                                                           on=["query", "plan_idx"], suffixes=("10", "16"))
    both["ok_both"] = both.ok10 & both.ok16

    for model in [REF, PG16]:
        folder = os.path.join(a.eval_root, model, "imdb", JOIN_ORDER_FOLDER)
        queries = sorted(runs["pg10"]["query"].unique(), key=lambda s: int(s.split("_")[-1]))
        theirs = {q: pd.read_csv(p) for q in queries
                  if os.path.exists(p := os.path.join(folder, f"{q}_0_test_pred.csv"))}
        print(f"\n== {model}: {len(theirs)} prediction files")
        if not theirs:
            continue
        t_sel = sum(t.label.values[np.argmin(t.prediction.values)] for t in theirs.values())
        t_opt = sum(t.label.min() for t in theirs.values())
        print(f"  authors' file: selected total {t_sel:.1f} s, optimal total {t_opt:.1f} s")

        best = None
        for name, (src, keep) in HYP.items():
            same_n = same_labels = 0
            for q, t in theirs.items():
                c = candidates(both, runs, q, src, keep)
                if len(c) == len(t):
                    same_n += 1
                    same_labels += np.allclose(np.sort(c.runtime_s.values), np.sort(t.label.values), rtol=1e-3)
            print(f"  {name:<33} same candidate count {same_n:2d}/{len(theirs)}, same labels {same_labels:2d}/{len(theirs)}")
            if best is None or same_labels > best[1]:
                best = (name, same_labels)

        # predictions vs estimated costs, pairing rows in plan order under the best hypothesis
        src, keep = HYP[best[0]]
        corr = {"pg10": [], "pg16": []}
        for q, t in theirs.items():
            c = candidates(both, runs, q, src, keep)
            if len(c) != len(t) or len(t) < 3:
                continue
            for v, vals in corr.items():
                cost = runs[v].set_index(["query", "plan_idx"]).loc[[(q, i) for i in c.plan_idx], "est_cost"].values
                vals.append(np.corrcoef(cost, t.prediction.values)[0, 1])
        print(f"  best: {best[0]}; median correlation of predictions with est. cost (plan order): "
              + ", ".join(f"{v} {np.nanmedian(r):.4f}" for v, r in corr.items() if r))


if __name__ == "__main__":
    main()
