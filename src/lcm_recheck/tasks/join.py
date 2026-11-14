"""sigtests.py paper, Task 1: join ordering on JOB-light (Fig. 1, Fig. 4).

H1 selected runtime (LCMs and PG versions vs Sc. PG10), H2 median Q-error, per-seed totals,
H1b LCMs trained with actual cardinalities. Bootstrap over queries; each family resamples with its own
index matrix (stats.resample_idx), shared by all its models. Family ids: config.FAMILY_IDS.

Usage:
  from lcm_recheck.tasks.join import task_join
  task_join(argparse.Namespace(eval_root="lcm-eval/data/evaluation"))
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

from lcm_recheck.config import ACT_CARD_LCMS, JOIN_ORDER_FOLDER, LCMS, PG_VERSIONS, REF
from lcm_recheck.report import section
from lcm_recheck.stats import boot, ci_p, holm, resample_idx
from lcm_recheck.tasks.common import compare, finish_family, record, show


def load_preds(eval_root: str, model: str) -> pd.DataFrame | None:
    """Task 1: all JOB-light prediction files of one model (all seeds)."""
    files = glob.glob(os.path.join(eval_root, model, "imdb", JOIN_ORDER_FOLDER, "job_light_*_*_test_pred.csv"))
    out = []
    for f in files:
        q, seed = re.match(r"(job_light_\d+)_(\d+)_test_pred\.csv", os.path.basename(f)).groups()
        d = pd.read_csv(f)
        d["query"], d["seed"] = q, int(seed)
        out.append(d)
    return pd.concat(out) if out else None


def selected(df: pd.DataFrame) -> pd.DataFrame:
    """Task 1, per query: runtime of the plan with the lowest prediction, and the optimum."""
    g = df.groupby("query")
    return pd.DataFrame({"r": g.apply(lambda d: d.loc[d.prediction.idxmin(), "label"]),
                         "opt": g.label.min()})


def task_join(a: argparse.Namespace) -> None:
    section("TASK 1: JOIN ORDERING (JOB-light, IMDB) - Fig. 1, Fig. 4")
    preds = {m: load_preds(a.eval_root, m) for m in [REF] + LCMS + PG_VERSIONS + ACT_CARD_LCMS}
    preds = {m: d for m, d in preds.items() if d is not None}
    if REF not in preds:
        print(f"skipped: no Sc. PG10 predictions under {a.eval_root}/postgres/imdb/join_order_full/")
        return
    print("models found:", ", ".join(preds))
    avg = {m: d.groupby(["query", "query_index"], as_index=False)[["label", "prediction"]].mean()
           for m, d in preds.items()}                                     # authors average over seeds
    sel = {m: selected(d) for m, d in avg.items()}
    core = [m for m in sel if m == REF or m in LCMS]                       # extras must not shrink H1/H2
    queries = sorted(set.intersection(*[set(sel[m].index) for m in core]))
    print(f"queries evaluated by all models: {len(queries)} (paper text says JOB-light has 70)")
    opt = sel[REF].loc[queries, "opt"]
    for m, s in sel.items():
        qs = [q for q in queries if q in s.index]
        if not np.allclose(s.loc[qs, "opt"], opt.loc[qs]):
            print(f"  WARNING: {m} sees a different optimum for some queries (different candidate sets)")
            ref_sets = avg[REF].groupby("query").label
            m_sets = avg[m].groupby("query").label
            for q in qs:
                n_ref, n_m = ref_sets.get_group(q).size, m_sets.get_group(q).size
                if not np.isclose(s.loc[q, "opt"], opt.loc[q]) or n_ref != n_m:
                    missing = "PG10's optimum missing" if s.loc[q, "opt"] > opt.loc[q] + 1e-9 else "optimum present"
                    print(f"     {q}: {n_m} plans vs {n_ref} for PG10 ({missing})")

    print("\nTotals over these queries (paper: optimal 446s, Sc.PG10 518s, Zero-Shot 530s, QueryFormer 830s)")
    print(f"  optimal: {opt.sum():.1f}s")
    for m, s in sel.items():
        qs = [q for q in queries if q in s.index]
        print(f"  {m:17s} {s.loc[qs, 'r'].sum():8.1f}s" + ("" if len(qs) == len(queries) else f"  (only {len(qs)} queries)"))

    # H1: selected runtime, each LCM vs Sc. PG10, paired by query
    for fam, members in {"LCM vs PG10": [x for x in LCMS if x in sel],
                         "PG version vs PG10": [x for x in PG_VERSIONS if x in sel]}.items():
        if not members:
            continue
        rows = []
        for m in members:
            qs = [q for q in queries if q in sel[m].index]
            rows.append({"model": m, **compare(sel[m].loc[qs, "r"].values, sel[REF].loc[qs, "r"].values,
                                               len(members), "runtime", family=f"sigtests/join/H1/{fam}")})
        h1 = finish_family(rows)
        show(f"H1 [{fam}] delta = sum_q (r_model - r_PG10) in s; bootstrap over queries", h1)
        if fam == "LCM vs PG10":
            record("1 join order", "LCM vs PG10", "selected runtime", h1)

    # H2: accuracy, median Q-error, cluster bootstrap by query
    def qerr(d: pd.DataFrame) -> pd.Series:
        p = d.prediction.clip(lower=1e-6)
        return np.maximum(p / d.label, d.label / p)

    def stat(ps: list[tuple[np.ndarray, np.ndarray]]) -> float:
        return np.median(np.concatenate([x for x, _ in ps])) - np.median(np.concatenate([y for _, y in ps]))

    rows = []
    ref_q = {q: qerr(g).values for q, g in avg[REF][avg[REF]["query"].isin(queries)].groupby("query")}
    neg = (avg[REF].prediction <= 0).sum()
    if neg:
        print(f"\nnote: {neg} Sc.PG10 predictions <= 0 (linear fit), clipped for Q-error")
    for m in [x for x in LCMS if x in avg]:
        mq = {q: qerr(g).values for q, g in avg[m][avg[m]["query"].isin(queries)].groupby("query")}
        pairs = [(mq[q], ref_q[q]) for q in queries]
        obs, lo, hi, p = ci_p(boot(pairs, stat, idx=resample_idx("sigtests/join/H2/LCM vs PG10", len(pairs))),
                              stat(pairs))
        rows.append({"model": m, "median_qerr_diff": obs, "ci_lo": lo, "ci_hi": hi, "boot_p": p})
    h2 = pd.DataFrame(rows)
    h2["boot_p_holm"] = holm(h2.boot_p)
    show("H2  median Q-error(LCM) - median Q-error(PG10) on the same candidate plans (negative = LCM more accurate)",
         h2, note=f"PG10's own median Q-error on these plans: {np.median(np.concatenate(list(ref_q.values()))):.3f}")

    # seed robustness
    print("\nPer-seed totals (the paper averages predictions over seeds before picking a plan):")
    for m, d in preds.items():
        tot = [selected(g.groupby(["query", "query_index"], as_index=False)[["label", "prediction"]].mean())
               .loc[lambda x: x.index.isin(queries), "r"].sum() for _, g in d.groupby("seed")]
        print(f"  {m:17s} " + "  ".join(f"{t:8.1f}" for t in tot))

    # H1b: LCMs trained with actual cardinalities vs PG10 with estimated cardinalities
    members = [m for m in ACT_CARD_LCMS if m in sel]
    if members:
        rows = []
        for m in members:
            qs = [q for q in queries if q in sel[m].index]
            rows.append({"model": m, **compare(sel[m].loc[qs, "r"].values, sel[REF].loc[qs, "r"].values,
                                               len(members), "runtime",
                                               family="sigtests/join/H1b/LCM (act. card.) vs PG10")})
        h1b = finish_family(rows)
        show("H1b [LCM with actual cardinalities vs PG10 with estimated ones] delta = sum_q (r_model - r_PG10) in s",
             h1b, note="Fig. 4 lighter bars. The PG10/PG16 values with actual cardinalities in Fig. 5 are typed "
                       "into the authors' notebook, not computed from published files, so they cannot be tested.")
        record("1 join order", "LCM (act. card.) vs PG10", "selected runtime", h1b)
