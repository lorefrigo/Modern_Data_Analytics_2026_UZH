"""Turn lcm-eval run files (raw or JSON mode) into one tidy CSV, one row per candidate plan.

Independent of the authors' parser, so it also works for PostgreSQL 18 output and doubles as a
cross-check of their labels.

Usage:
  python extract_runs.py --runs DIR_WITH_job_light_*.json --version pg10_paper --out pg10_paper.csv
"""
import argparse
import glob
import os

import pandas as pd

from lcm_recheck.runs import exec_ms, load_run, plan_shape, top_cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="folder containing job_light_*.json run files")
    ap.add_argument("--version", required=True, help="label, e.g. pg10_paper, pg10_local, pg18_local")
    ap.add_argument("--out", required=True, help="output CSV, one row per candidate plan")
    a = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(a.runs, "job_light_*.json"))):
        query = os.path.basename(path)[:-5]
        queries, _ = load_run(path, stream=False)
        for i, q in enumerate(queries):
            vp, aps = q.get("verbose_plan"), q.get("analyze_plans") or []
            reps = [exec_ms(p) for p in aps]
            rows.append({
                "version": a.version, "query": query, "plan_idx": i, "sql": q["sql"],
                "timeout": bool(q.get("timeout")),
                "est_cost": top_cost(vp) if vp else float("nan"),
                "shape": plan_shape(vp) if vp else None,
                "n_reps": len(reps),
                "runtime_s": (sum(reps) / len(reps) / 1000) if reps else float("nan"),
                "runtime_reps_ms": ";".join(f"{r:.3f}" for r in reps)})
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)

    # quick diagnostics that matter for validity
    print(f"{len(df)} plans in {df['query'].nunique()} queries; timeouts: {df['timeout'].sum()}")
    print("repetitions per executed plan:", df.loc[~df.timeout, "n_reps"].value_counts().to_dict())
    dup = df.groupby("query")["shape"].agg(lambda s: s.duplicated().sum())
    print(f"plans identical to another candidate of the same query (hint not effective?): {int(dup.sum())}")


if __name__ == "__main__":
    main()
