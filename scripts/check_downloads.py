"""Check which of the authors' prediction files were downloaded, for all three tasks, and print the
exact commands for the next steps. Needs sigtests.py in the same folder (it reuses its workload lists).

Usage (from the MDA folder):  python scripts/check_downloads.py
"""
import glob, os
from sigtests import (LCMS, ACT_CARD_LCMS, FINE_TUNED_LCMS, INDEX_LCMS, REF, PG16, SEEDS,
                      SCAN_COLUMNS, PHYSICAL, PHYSICAL_INDEX)

DATA = "lcm-eval/data"
# the 58 JOB-light queries the authors evaluate (active entries of FullJoinOrder in src/classes/workloads.py)
JOB_LIGHT = {f"job_light_{i}" for i in (
    "0 1 2 3 4 5 6 8 9 10 11 12 13 14 15 16 17 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 "
    "41 43 44 45 46 47 49 50 51 52 53 54 55 61 62 63 64 65 66 67 68").split()}


def has_all_seeds(root, model, db, folder, wl):
    return all(os.path.exists(os.path.join(root, model, db, folder, f"{wl}_{s}_test_pred.csv")) for s in SEEDS)


def line(model, found, expected):
    status = "ok" if found == expected else ("MISSING" if found == 0 else "incomplete")
    return f"  {model:17s} {found:4d} / {expected:<4d} {status}"


# evaluation roots: folders that contain postgres/<database>/...
roots = sorted({os.path.dirname(os.path.dirname(p))
                for p in glob.glob(os.path.join(DATA, "**", "postgres", "*"), recursive=True)
                if os.path.isdir(p) and os.path.basename(p) in ("imdb", "baseball", "tpc_h_pk")
                and "retraining" not in p})
if not roots:
    print("No prediction files found under", DATA, "- check the download steps.")

for root in roots:
    print(f"\nEvaluation root: {root}")

    print("\nTask 1, join ordering (JOB-light queries with all 3 seeds; 58 expected):")
    for m in [REF, PG16] + LCMS + ACT_CARD_LCMS:
        found = {os.path.basename(f).rsplit("_", 3)[0]
                 for f in glob.glob(os.path.join(root, m, "imdb", "join_order_full", "job_light_*_0_test_pred.csv"))
                 if all(os.path.exists(f.replace("_0_test_pred", f"_{s}_test_pred")) for s in SEEDS)}
        extra = sorted(found - JOB_LIGHT, key=lambda q: int(q.rsplit("_", 1)[1]))
        print(line(m, len(found & JOB_LIGHT), 58)
              + (f"   + not among the 58: {', '.join(extra)} (ignored by sigtests.py)" if extra else "")
              + ("   (optional: actual-cardinality variant)" if m in ACT_CARD_LCMS and not found else ""))

    print("\nTask 2, access paths (columns with seq and index files for all 3 seeds; 21 expected):")
    for m in [REF, PG16] + LCMS:
        n = sum(has_all_seeds(root, m, db, "scan_costs_percentiles", f"seq.{c}") and
                has_all_seeds(root, m, db, "scan_costs_percentiles", f"index.{c}")
                for db, cols in SCAN_COLUMNS.items() for c in cols)
        print(line(m, n, 21))

    retrain = os.path.join(os.path.dirname(os.path.normpath(root)), "retraining_evaluation")
    print(f"\nTask 2, Fig. 14 fine-tuned models in {retrain} (IMDB columns; 6 expected):")
    if not os.path.isdir(retrain):
        print("  folder not found - Fig. 14 will be skipped (optional)")
    else:
        for m in [REF] + FINE_TUNED_LCMS:
            n = sum(has_all_seeds(retrain, m, "imdb", "scan_costs_percentiles", f"seq.{c}") and
                    has_all_seeds(retrain, m, "imdb", "scan_costs_percentiles", f"index.{c}")
                    for c in SCAN_COLUMNS["imdb"])
            print(line(m, n, 6) + ("   (PG10 is then read from the normal evaluation folder)" if m == REF and n == 0 else ""))

    for ds, (db, folder, names) in list(PHYSICAL.items()) + [("IMDB with indexes (Section 6.6)", PHYSICAL_INDEX)]:
        models = [REF, PG16] + (INDEX_LCMS if "indexes" in ds else LCMS)
        print(f"\nTask 3, join operators, {ds} (queries with all 3 seeds; 100 expected):")
        for m in models:
            print(line(m, sum(has_all_seeds(root, m, db, folder, wl) for wl in names), 100))

    print("\n  next:")
    print(f'  uv run python scripts/sigtests.py paper --eval_root "{root}" --out results/sigtests_paper.txt')

# raw PG10 run files (optional diagnostics for Task 1)
runs = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(DATA, "**", "join_order_full", "job_light_*.json"),
                                                      recursive=True)
               if not any(x in p for x in ("parsed", "augmented"))})
for r in runs:
    n = len(glob.glob(os.path.join(r, "job_light_*.json")))
    print(f"\nRun files: {r} ({n} files)")
    print(f'  optional: uv run python scripts/extract_runs.py --runs "{r}" --version pg10_paper --out results/pg10_paper.csv')