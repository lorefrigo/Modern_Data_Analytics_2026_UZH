"""Turn lcm-eval run files (raw or JSON mode) into one tidy CSV, one row per candidate plan.

Independent of the authors' parser, so it also works for PostgreSQL 18 output and doubles as a
cross-check of their labels.

Usage:
  python extract_runs.py --runs DIR_WITH_job_light_*.json --version pg10_paper --out pg10_paper.csv
"""
import argparse, glob, hashlib, json, os, re
import pandas as pd

COST_RE = re.compile(r"cost=([\d.]+)\.\.([\d.]+)")
EXEC_RE = re.compile(r"execution time:\s*([\d.]+)\s*ms", re.I)
NUM_RE = re.compile(r"[\d.]+")


def lines(plan):
    """Raw-mode plans are lists of 1-element rows; return them as a list of strings."""
    return [r[0] if isinstance(r, (list, tuple)) else str(r) for r in plan]


def top_cost(verbose_plan):
    if isinstance(verbose_plan, dict):                       # JSON mode
        return float(verbose_plan["Plan"]["Total Cost"])
    m = COST_RE.search(lines(verbose_plan)[0])                # raw mode: top node is first line
    return float(m.group(2)) if m else float("nan")


def exec_ms(analyze_plan):
    if isinstance(analyze_plan, dict):
        return float(analyze_plan["Execution Time"])
    for l in lines(analyze_plan):
        m = EXEC_RE.search(l)
        if m:
            return float(m.group(1))
    return float("nan")


def plan_shape(verbose_plan):
    """Hash of the plan with all numbers removed: equal hashes = same physical plan."""
    txt = json.dumps(verbose_plan["Plan"], sort_keys=True) if isinstance(verbose_plan, dict) \
        else "\n".join(lines(verbose_plan))
    return hashlib.md5(NUM_RE.sub("#", txt).encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="folder containing job_light_*.json run files")
    ap.add_argument("--version", required=True, help="label, e.g. pg10_paper, pg10_local, pg18_local")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(a.runs, "job_light_*.json"))):
        query = os.path.basename(path)[:-5]
        run = json.load(open(path))
        if "query_list" not in run:
            raise SystemExit(f"{path} is not a run file (no 'query_list'); point --runs at the raw or json run folder")
        for i, q in enumerate(run["query_list"]):
            vp, aps = q.get("verbose_plan"), q.get("analyze_plans") or []
            reps = [exec_ms(p) for p in aps]
            rows.append(dict(
                version=a.version, query=query, plan_idx=i, sql=q["sql"],
                timeout=bool(q.get("timeout")),
                est_cost=top_cost(vp) if vp else float("nan"),
                shape=plan_shape(vp) if vp else None,
                n_reps=len(reps),
                runtime_s=(sum(reps) / len(reps) / 1000) if reps else float("nan"),
                runtime_reps_ms=";".join(f"{r:.3f}" for r in reps)))
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)

    # quick diagnostics that matter for validity
    print(f"{len(df)} plans in {df['query'].nunique()} queries; timeouts: {df['timeout'].sum()}")
    print("repetitions per executed plan:", df.loc[~df.timeout, "n_reps"].value_counts().to_dict())
    dup = df.groupby("query")["shape"].agg(lambda s: s.duplicated().sum())
    print(f"plans identical to another candidate of the same query (hint not effective?): {int(dup.sum())}")


if __name__ == "__main__":
    main()
