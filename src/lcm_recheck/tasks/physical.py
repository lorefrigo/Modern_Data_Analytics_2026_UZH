"""sigtests.py paper, Task 3: physical operator selection, hash vs merge vs index nested-loop join
(Fig. 11, Section 6.6).

Selected runtime (bootstrap over queries) and pick rate (exact McNemar, Holm) of each LCM and PG16 vs PG10;
Section 6.6 with indexes on the filter columns. Each comparison, metric and dataset resamples with its own
index matrix (via compare/compare_pick), shared by all its models. Family ids: config.FAMILY_IDS.

Usage:
  from lcm_recheck.tasks.physical import task_physical
  task_physical(argparse.Namespace(eval_root="lcm-eval/data/evaluation"))
"""
import argparse
from typing import Any

import numpy as np

from lcm_recheck.config import INDEX_LCMS, LCMS, PG16, PHYSICAL, PHYSICAL_INDEX, REF
from lcm_recheck.report import section
from lcm_recheck.tasks.common import (
    compare,
    compare_pick,
    finish_family,
    load_workload,
    record,
    show,
    with_common_labels,
)


def physical_scores(root: str, models: list[str], db: str, folder: str,
                    names: list[str]) -> tuple[dict[str, dict[str, dict[str, Any]]], set[str]]:
    """Per model and query: selected runtime, picked optimum (yes/no), optimum, median Q-error."""
    out = {m: {} for m in models}
    deviating = set()
    for wl in names:
        preds = {m: load_workload(root, m, db, folder, wl) for m in models}
        preds = {m: d for m, d in preds.items() if d is not None}
        if REF not in preds:
            continue
        preds, dev = with_common_labels(preds)
        deviating.update(dev)
        for m, d in preds.items():
            if d.empty:
                continue
            p = np.abs(d.prediction.values)
            l = d.label.values
            with np.errstate(divide="ignore", invalid="ignore"):
                q = np.nan_to_num(np.maximum(l / p, p / l), nan=np.inf)
            i_pred, i_opt = int(np.argmin(d.prediction.values)), int(np.argmin(l))
            out[m][wl] = {"runtime": l[i_pred], "pick": i_pred == i_opt, "optimum": l[i_opt],
                          "qerr": np.percentile(q, 50)}
    return out, deviating


def physical_family(scores_by_ds: dict[str, tuple[dict[str, Any], list[str]]], members: list[str],
                    k_runtime: int, comparison: str, task: str) -> None:
    """Runtime and pick-rate comparisons of `members` vs PG10, one family across all datasets."""
    rt, pk = [], []
    for ds, (scores, queries) in scores_by_ds.items():
        for m in members:
            qs = [q for q in queries if q in scores[m]]
            if not qs:
                continue
            fam = f"sigtests/physical/{comparison}"
            rt.append({"dataset": ds, "model": m, **compare([scores[m][q]["runtime"] for q in qs],
                                                            [scores[REF][q]["runtime"] for q in qs], k_runtime,
                                                            "runtime", family=f"{fam}/selected runtime/{ds}")})
            pk.append({"dataset": ds, "model": m, **compare_pick([scores[m][q]["pick"] for q in qs],
                                                                 [scores[REF][q]["pick"] for q in qs],
                                                                 family=f"{fam}/pick rate/{ds}")})
    rt, pk = finish_family(rt), finish_family(pk, pick=True)
    show(f"[{comparison}] selected runtime: delta = sum_q (r_model - r_PG10); bootstrap over queries", rt)
    show(f"[{comparison}] pick rate (share of queries where the optimal join operator was picked); "
         f"exact McNemar test, Holm across the family", pk)
    if "LCM" in comparison:
        record(task, comparison, "selected runtime", rt)
        record(task, comparison, "pick rate", pk)


def task_physical(a: argparse.Namespace) -> None:
    section("TASK 3: PHYSICAL OPERATOR SELECTION (hash vs merge vs index nested-loop join) - Fig. 11, Section 6.6")
    models = [REF, PG16] + LCMS
    by_ds = {}
    for ds, (db, folder, names) in PHYSICAL.items():
        scores, deviating = physical_scores(a.eval_root, models, db, folder, names)
        if not scores[REF]:
            print(f"{ds}: skipped, no Sc. PG10 predictions under {a.eval_root}/postgres/{db}/{folder}/")
            continue
        core = [m for m in [REF] + LCMS if scores[m]]
        queries = sorted(set.intersection(*[set(scores[m]) for m in core]))
        # authors' rule (Fig. 11): drop queries where any model's median Q-error exceeds 1e5
        excluded = [q for q in queries if any(scores[m][q]["qerr"] > 1e5 for m in models if q in scores[m])]
        kept = [q for q in queries if q not in excluded]
        by_ds[ds] = (scores, kept)
        print(f"\n{ds}: {len(queries)} queries evaluated by all models, {len(excluded)} excluded "
              f"(median Q-error > 1e5 for some model, as in the authors' notebook), {len(kept)} used")
        if deviating:
            print(f"  note: labels differ by >1% from Flat Vector's for: {', '.join(sorted(deviating))}")
        print(f"  optimal{'':10s} runtime {sum(scores[REF][q]['optimum'] for q in kept):10.2f}   pick rate 100.0%")
        for m in [x for x in models if scores[x]]:
            qs = [q for q in kept if q in scores[m]]
            print(f"  {m:17s} runtime {sum(scores[m][q]['runtime'] for q in qs):10.2f}   pick rate "
                  f"{100 * np.mean([scores[m][q]['pick'] for q in qs]):5.1f}%")
    if by_ds:
        members = [m for m in LCMS if any(s[0][m] for s in by_ds.values())]
        physical_family(by_ds, members, len(members) * len(by_ds), "Fig. 11: LCM vs PG10", "3 join operator")
        if any(s[0][PG16] for s in by_ds.values()):
            physical_family(by_ds, [PG16], len(by_ds), "Fig. 11: PG16 vs PG10", "3 join operator")

    # Section 6.6: additional indexes on the filter columns (no exclusion in the authors' notebook)
    db, folder, names = PHYSICAL_INDEX
    models = [REF, PG16] + INDEX_LCMS
    scores, _ = physical_scores(a.eval_root, models, db, folder, names)
    if not scores[REF]:
        print(f"\nSection 6.6 skipped: no predictions under {a.eval_root}/postgres/{db}/{folder}/")
        return
    core = [m for m in [REF] + INDEX_LCMS if scores[m]]
    queries = sorted(set.intersection(*[set(scores[m]) for m in core]))
    print(f"\nSection 6.6 (IMDB with indexes on the filter columns): {len(queries)} queries")
    print(f"  optimal{'':10s} runtime {sum(scores[REF][q]['optimum'] for q in queries):10.2f}   pick rate 100.0%")
    for m in [x for x in models if scores[x]]:
        qs = [q for q in queries if q in scores[m]]
        print(f"  {m:17s} runtime {sum(scores[m][q]['runtime'] for q in qs):10.2f}   pick rate "
              f"{100 * np.mean([scores[m][q]['pick'] for q in qs]):5.1f}%")
    idx = {"IMDB+indexes": (scores, queries)}
    members = [m for m in INDEX_LCMS if scores[m]]
    physical_family(idx, members, len(members), "Section 6.6: LCM vs PG10", "3 join operator")
    if scores[PG16]:
        physical_family(idx, [PG16], 1, "Section 6.6: PG16 vs PG10", "3 join operator")
