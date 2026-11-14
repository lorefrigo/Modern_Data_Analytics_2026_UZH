"""sigtests.py paper, Task 2: access path selection, index scan vs sequential scan (Table 3, Fig. 14).

T2a balanced accuracy and T2b selected runtime of each LCM (and PG16) vs PG10, T2c fine-tuned LCMs
(Fig. 14). Bootstrap over columns; each family resamples with its own index matrix (via compare()),
shared by all its models. Family ids: config.FAMILY_IDS.

Usage:
  from lcm_recheck.tasks.access import task_access
  task_access(argparse.Namespace(eval_root=..., retrain_root=...))
"""
import argparse
import os
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from lcm_recheck.config import (
    FINE_TUNED_LCMS,
    LCMS,
    PG16,
    REF,
    SCAN_COLUMNS,
    SCAN_FOLDER,
)
from lcm_recheck.report import section
from lcm_recheck.tasks.common import (
    compare,
    finish_family,
    load_workload,
    record,
    show,
    with_common_labels,
)


def balanced_accuracy(y_true: Any, y_pred: Any) -> float:
    """Same as sklearn.metrics.balanced_accuracy_score: mean recall over the classes present in y_true."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean([np.mean(y_pred[y_true == c] == c) for c in np.unique(y_true)]))


def access_path_scores(root: Callable[[str], str], models: list[str],
                       columns_by_db: dict[str, list[str]]) -> tuple[dict[str, dict[str, dict[str, Any]]], set[str]]:
    """Per model and column: balanced accuracy, selected runtime and optimal runtime over the selectivities."""
    out = {m: {} for m in models}
    deviating = set()
    for db, cols in columns_by_db.items():
        for col in cols:
            seq = {m: load_workload(root(m), m, db, SCAN_FOLDER, f"seq.{col}") for m in models}
            idx = {m: load_workload(root(m), m, db, SCAN_FOLDER, f"index.{col}") for m in models}
            seq = {m: d for m, d in seq.items() if d is not None}
            idx = {m: d for m, d in idx.items() if d is not None}
            if REF not in seq or REF not in idx:
                continue
            seq, dev1 = with_common_labels(seq)
            idx, dev2 = with_common_labels(idx)
            deviating.update(dev1 + dev2)
            for m in models:
                if m not in seq or m not in idx:
                    continue
                d = pd.DataFrame({"ls": seq[m].label, "ps": seq[m].prediction,
                                  "li": idx[m].label, "pi": idx[m].prediction}).dropna()
                if d.empty:
                    continue
                optimal, predicted = d.li < d.ls, d.pi < d.ps
                out[m][f"{db}:{col}"] = {"B": balanced_accuracy(optimal, predicted),
                                         "runtime": np.where(predicted, d.li, d.ls).sum(),
                                         "optimum": np.minimum(d.li, d.ls).sum(), "n": len(d)}
    return out, deviating


def task_access(a: argparse.Namespace) -> None:
    section("TASK 2: ACCESS PATH SELECTION (index scan vs sequential scan) - Table 3, Fig. 14")
    models = [REF, PG16] + LCMS
    scores, deviating = access_path_scores(lambda m: a.eval_root, models, SCAN_COLUMNS)
    if not scores[REF]:
        print(f"skipped: no Sc. PG10 predictions under {a.eval_root}/postgres/<db>/scan_costs_percentiles/")
        return
    core = [m for m in [REF] + LCMS if scores[m]]
    cols = sorted(set.intersection(*[set(scores[m]) for m in core]))
    print(f"columns evaluated by all models: {len(cols)} of 21 (IMDB 6, Baseball 7, TPC-H 8)")
    print(f"selectivities per column: {sorted({scores[REF][c]['n'] for c in cols})}")
    if deviating:
        print(f"note: labels differ by >1% from Flat Vector's for: {', '.join(sorted(deviating))} (Flat Vector's used)")
    print("\nMean balanced accuracy over the columns (paper Table 3: Sc. PG10, Sc. PG16, DACE about 0.64)"
          " and total selected runtime:")
    print(f"  optimal{'':10s} B = 1.000   runtime {sum(scores[REF][c]['optimum'] for c in cols):10.2f}")
    for m in [x for x in models if scores[x]]:
        cs = [c for c in cols if c in scores[m]]
        print(f"  {m:17s} B = {np.mean([scores[m][c]['B'] for c in cs]):.3f}   runtime "
              f"{sum(scores[m][c]['runtime'] for c in cs):10.2f}" + ("" if len(cs) == len(cols) else f"  ({len(cs)} columns)"))

    for metric, kind, key, test, title in [
            ("balanced accuracy", "score", "B", "T2a", "T2a [LCM vs PG10] balanced accuracy: delta = mean over columns of (B_LCM - B_PG10); bootstrap over columns"),
            ("selected runtime", "runtime", "runtime", "T2b", "T2b [LCM vs PG10] selected runtime: delta = sum over columns of (runtime_LCM - runtime_PG10); bootstrap over columns")]:
        members = [m for m in LCMS if scores[m]]
        rows = []
        for m in members:
            cs = [c for c in cols if c in scores[m]]
            rows.append({"model": m, **compare([scores[m][c][key] for c in cs], [scores[REF][c][key] for c in cs],
                                               len(members), kind, family=f"sigtests/access/{test}/LCM vs PG10")})
        df = finish_family(rows)
        show(title, df)
        record("2 access path", "LCM vs PG10", metric, df)
        if scores[PG16]:
            cs = [c for c in cols if c in scores[PG16]]
            show(f"{title.split(']')[0].replace('LCM', 'PG16')}] {metric}",
                 finish_family([{"model": PG16, **compare([scores[PG16][c][key] for c in cs],
                                                          [scores[REF][c][key] for c in cs], 1, kind,
                                                          family=f"sigtests/access/{test}/PG16 vs PG10")}]))

    # Fig. 14: fine-tuned LCMs on the IMDB columns
    if not os.path.isdir(a.retrain_root):
        print(f"\nFig. 14 skipped: {a.retrain_root} not found (fine-tuned predictions not downloaded)")
        return
    ref_root = a.retrain_root if os.path.isdir(os.path.join(a.retrain_root, REF)) else a.eval_root

    def roots(m: str) -> str:
        return ref_root if m == REF else a.retrain_root

    ft, _ = access_path_scores(roots, [REF] + FINE_TUNED_LCMS, {"imdb": SCAN_COLUMNS["imdb"]})
    members = [m for m in FINE_TUNED_LCMS if ft[m]]
    if not ft[REF] or not members:
        print(f"\nFig. 14 skipped: no fine-tuned predictions found under {a.retrain_root}")
        return
    cols_ft = sorted(set.intersection(*[set(ft[m]) for m in [REF] + members]))
    print(f"\nFig. 14: fine-tuned LCMs vs PG10 on {len(cols_ft)} IMDB columns (PG10 read from {ref_root})")
    print("  CAUTION: with only 6 columns, Wilcoxon cannot reach p < 0.05 after Holm correction and bootstrap "
          "intervals over 6 units are unreliable, so no verdict is given; read these rows as descriptive.")
    for m in [REF] + members:
        before = [scores[m][c] for c in cols_ft if c in scores.get(m, {})]
        b_txt = f"   before fine-tuning: B = {np.mean([s['B'] for s in before]):.3f}, runtime {sum(s['runtime'] for s in before):.2f}" \
            if m != REF and len(before) == len(cols_ft) else ""
        print(f"  {m:17s} B = {np.mean([ft[m][c]['B'] for c in cols_ft]):.3f}, runtime "
              f"{sum(ft[m][c]['runtime'] for c in cols_ft):.2f}{b_txt}")
    for metric, kind, key in [("balanced accuracy", "score", "B"), ("selected runtime", "runtime", "runtime")]:
        rows = [{"model": m, **compare([ft[m][c][key] for c in cols_ft], [ft[REF][c][key] for c in cols_ft],
                                       len(members), kind, family=f"sigtests/access/T2c/{metric}")} for m in members]
        df = finish_family(rows)
        show(f"T2c [fine-tuned LCM vs PG10] {metric}", df)
        record("2 access path", "fine-tuned LCM vs PG10 (Fig. 14)", metric, df)
