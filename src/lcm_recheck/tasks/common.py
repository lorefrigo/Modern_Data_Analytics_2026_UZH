"""State and helpers shared by the sigtests.py tasks (join, access, physical, versions).

No random generator lives here: compare() and compare_pick() bootstrap with the resample-index matrix of
the test family they are given (stats.resample_idx), so all models of a family share the same resamples.
SUMMARY collects every LCM-vs-PG10 verdict for summary(); only record() appends to it.

Usage:
  from lcm_recheck.tasks.common import compare, finish_family, record, show
  fam = "sigtests/join/H1/LCM vs PG10"
  df = finish_family([{"model": m, **compare(x, ref, k, "runtime", family=fam)} for m, x in ...])
  show("title", df); record("1 join order", "LCM vs PG10", "selected runtime", df)
"""
import os
from typing import Any

import numpy as np
import pandas as pd

from lcm_recheck.config import LABEL_SOURCE, REF, SEEDS
from lcm_recheck.report import section
from lcm_recheck.stats import boot_fast, ci_p, holm, mcnemar, resample_idx, wilcoxon_p

SUMMARY: list[dict[str, Any]] = []   # every LCM-vs-PG10 verdict, collected for the final summary


# ----------------------------------------------------------------------------- paired comparisons
def compare(x: Any, ref: Any, k: int, kind: str, *, family: str) -> dict[str, Any]:
    """Paired comparison of one model with PG10 over units (queries or columns), bootstrapped with the
    resample-index matrix of `family` (a config.FAMILY_IDS entry).

    kind="runtime": statistic = sum of differences (lower is better), Wilcoxon on log ratios.
    kind="score":   statistic = mean of differences (higher is better), Wilcoxon on differences.
    Verdict from the Bonferroni-adjusted (level 1 - 0.05/k) bootstrap interval; none for fewer than 10 units."""
    x, ref = np.asarray(x, float), np.asarray(ref, float)
    d = x - ref
    stat = np.sum if kind == "runtime" else np.mean
    samples = boot_fast(d, stat, idx=resample_idx(family, len(d)))
    obs, lo, hi, _ = ci_p(samples, stat(d))
    _, flo, fhi, _ = ci_p(samples, stat(d), alpha=0.05 / k)
    w = np.log(x / ref) if kind == "runtime" and (x > 0).all() and (ref > 0).all() else d
    p, nz = wilcoxon_p(w)
    if len(d) < 10:
        verdict = "too few units to test"           # bootstrap over < 10 units is unreliable
    elif kind == "runtime":
        verdict = "PG10 better" if flo > 0 else ("LCM better" if fhi < 0 else "no clear difference")
    else:
        verdict = "LCM better" if flo > 0 else ("PG10 better" if fhi < 0 else "no clear difference")
    return {"n": len(d), "delta": obs, "ci95_lo": lo, "ci95_hi": hi, "fwer_ci_lo": flo, "fwer_ci_hi": fhi,
            "units_differ": nz, "wilcoxon_p": p,
            "largest_unit_share": np.abs(d).max() / max(np.abs(d).sum(), 1e-12), "verdict": verdict}


def compare_pick(x: Any, ref: Any, *, family: str) -> dict[str, Any]:
    """Paired comparison of pick rates (picked the optimal plan: yes/no per query); exact McNemar test.
    The interval is bootstrapped with the resample-index matrix of `family`; it carries no verdict."""
    x, ref = np.asarray(x, bool), np.asarray(ref, bool)
    d = x.astype(float) - ref.astype(float)
    obs, lo, hi, _ = ci_p(boot_fast(d, np.mean, idx=resample_idx(family, len(d))), d.mean())
    only_ref, only_x = int((ref & ~x).sum()), int((x & ~ref).sum())
    p = mcnemar(only_x, only_ref)
    return {"n": len(d), "pick_lcm": x.mean(), "pick_pg10": ref.mean(), "delta": obs, "ci95_lo": lo,
            "ci95_hi": hi, "only_lcm_optimal": only_x, "only_pg10_optimal": only_ref, "mcnemar_p": p}


def finish_family(rows: list[dict[str, Any]], pick: bool = False) -> pd.DataFrame:
    """Holm correction within a family; for pick rates the verdict comes from the Holm-adjusted McNemar p."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if pick:
        df["mcnemar_p_holm"] = holm(df.mcnemar_p)
        df["verdict"] = np.where(df.mcnemar_p_holm >= 0.05, "no clear difference",
                                 np.where(df.only_lcm_optimal > df.only_pg10_optimal, "LCM better", "PG10 better"))
    else:
        df["wilcoxon_p_holm"] = holm(df.wilcoxon_p.fillna(1))
        cols = [c for c in df.columns if c != "verdict"] + ["verdict"]
        df = df[cols]
    return df


# ----------------------------------------------------------------------------- output
def show(title: str, df: pd.DataFrame, note: str | None = None) -> None:
    print(f"\n{title}")
    if note:
        print(f"  {note}")
    if df.empty:
        print("  (no data)")
    else:
        print(df.round(3).to_string(index=False))


def record(task: str, comparison: str, metric: str, df: pd.DataFrame) -> None:
    for _, r in df.iterrows():
        SUMMARY.append({"task": task, "comparison": comparison, "dataset": r.get("dataset", ""), "metric": metric,
                        "model": r["model"], "verdict": r["verdict"]})


def summary() -> None:
    section("SUMMARY: every LCM-vs-PG10 comparison (paper's claim: no LCM significantly beats PostgreSQL in any task)")
    if not SUMMARY:
        print("no comparisons were run")
        return
    s = pd.DataFrame(SUMMARY)
    counts = s.groupby(["task", "comparison", "metric"]).verdict.value_counts().unstack(fill_value=0)
    order = ["PG10 better", "no clear difference", "LCM better", "too few units to test"]
    for col in order[:3]:
        if col not in counts:
            counts[col] = 0
    print(counts[[c for c in order if c in counts]].to_string())
    wins = s[s.verdict == "LCM better"]
    print(f"\nComparisons where an LCM is significantly better than PG10 (after family-wise correction): {len(wins)}")
    if len(wins):
        print(wins.to_string(index=False))


# ----------------------------------------------------------------------------- loading (Tasks 2 and 3)
def load_workload(root: str, model: str, db: str, folder: str, wl: str) -> pd.DataFrame | None:
    """Tasks 2 and 3: one workload of one model, predictions averaged over the seeds (as the authors do)."""
    frames = []
    for s in SEEDS:
        f = os.path.join(root, model, db, folder, f"{wl}_{s}_test_pred.csv")
        if os.path.exists(f):
            frames.append(pd.read_csv(f)[["query_index", "label", "prediction"]])
    if not frames:
        return None
    d = pd.concat(frames)
    return d.groupby("query_index").agg(label=("label", "first"), prediction=("prediction", "mean"))


def with_common_labels(preds: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """The authors give QPP-Net the labels of Flat Vector; we use Flat Vector's labels for every model."""
    src = LABEL_SOURCE if LABEL_SOURCE in preds else REF
    common = preds[src]["label"]
    out, deviating = {}, []
    for m, d in preds.items():
        lab = common.reindex(d.index)
        if (np.abs(d.label - lab) > 0.01 * np.abs(lab)).any() and m != "qppnet":
            deviating.append(m)
        out[m] = d.assign(label=lab).dropna()
    return out, deviating
