"""Was Sc. PG16 built from join_order_full_pg_10_hinted (PG10's exact physical plans, costed by PG16)?

Maps every pinned SQL line to its PG10 plan through the Leading(...) hint, then compares, per query,
the pinned plans with the rows of the authors' Sc. PG16 file: count, labels (PG10 runtimes of the mapped
plans) and the correlation of the predictions with PG10's estimated costs of those plans.

Usage (from MDA/):
  uv run python scripts/check_pinned.py --pg10 results/pg10_paper.csv \
         --pinned_dir lcm-eval/data/workloads/evaluation/imdb/join_order_full_pg_10_hinted \
         --eval_root lcm-eval/data/evaluation
"""
import argparse, os
import numpy as np, pandas as pd

from lcm_recheck.config import JOB_LIGHT_IDS, PG16, JOIN_ORDER_FOLDER
from lcm_recheck.hints import leading


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg10", required=True); ap.add_argument("--pinned_dir", required=True)
    ap.add_argument("--eval_root", required=True)
    a = ap.parse_args()
    pg10 = pd.read_csv(a.pg10)
    pg10["ok"] = ~pg10.timeout & pg10.runtime_s.notna()

    rows, corrs = [], []
    for k in JOB_LIGHT_IDS:
        q = f"job_light_{k}"
        d = pg10[pg10["query"] == q].set_index("plan_idx")
        lead = d.sql.map(leading)
        lead2idx = {l: i for i, l in lead.items()}
        with open(os.path.join(a.pinned_dir, f"{q}.sql")) as f:
            pinned = [l for l in f if l.strip()]
        mapped = [lead2idx.get(leading(s)) for s in pinned]
        ok_map = [i for i in mapped if i is not None]
        sc16 = pd.read_csv(os.path.join(a.eval_root, PG16, "imdb", JOIN_ORDER_FOLDER, f"{q}_0_test_pred.csv"))

        kept = [i for i in ok_map if d.loc[i, "ok"]]          # rule: pinned plans that have a PG10 runtime
        labels_ok = len(sc16) == len(kept) and \
            np.allclose(np.sort(d.loc[kept, "runtime_s"].values), np.sort(sc16.label.values), rtol=1e-3)
        in_order = len(sc16) == len(kept) and np.allclose(d.loc[kept, "runtime_s"].values, sc16.label.values, rtol=1e-3)
        if len(sc16) == len(kept) >= 3:
            corrs.append((np.corrcoef(d.loc[kept, "est_cost"].values, sc16.prediction.values)[0, 1], q, len(kept)))
        rows.append(dict(query=q, pinned=len(pinned), unmapped=len(pinned) - len(ok_map),
                         dup_leading=int(lead.duplicated().sum()), pg10_executed=int(d.ok.sum()),
                         pinned_timed_out_in_pg10=int((~d.loc[ok_map, "ok"]).sum()),
                         pg10_executed_not_pinned=int(len(set(d.index[d.ok]) - set(ok_map))),
                         kept=len(kept), sc16_rows=len(sc16), labels_match=labels_ok, same_order=in_order))

    t = pd.DataFrame(rows)
    print(f"pinned lines {t.pinned.sum()}, Sc. PG16 rows {t.sc16_rows.sum()}, PG10 executed plans {t.pg10_executed.sum()}")
    print(f"pinned lines without a matching PG10 plan: {t.unmapped.sum()}; duplicate Leading hints in PG10 SQL: {t.dup_leading.sum()}")
    print(f"queries with pinned count == Sc. PG16 rows: {(t.pinned == t.sc16_rows).sum()}/58")
    print(f"queries with pinned set == PG10 executed set: "
          f"{((t.pinned_timed_out_in_pg10 == 0) & (t.pg10_executed_not_pinned == 0)).sum()}/58")
    print(f"queries where Sc. PG16 rows = pinned plans with a PG10 runtime: count {(t.kept == t.sc16_rows).sum()}/58, "
          f"labels {t.labels_match.sum()}/58, same order {t.same_order.sum()}/58")
    r = np.array([c[0] for c in corrs])
    print(f"correlation of Sc. PG16 predictions with PG10 est. cost of those plans: median {np.nanmedian(r):.4f} "
          f"over {len(r)} queries with >= 3 plans; lowest: "
          + ", ".join(f"{q} {c:.2f} (n={n})" for c, q, n in sorted(corrs)[:5]))
    diff = t[(t.pinned != t.pg10_executed) | (t.kept != t.sc16_rows) | ~t.labels_match]
    if len(diff):
        print("\nQueries where something differs:")
        print(diff.to_string(index=False))


if __name__ == "__main__":
    main()
