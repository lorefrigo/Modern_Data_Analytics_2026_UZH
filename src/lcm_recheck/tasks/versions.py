"""sigtests.py versions: plan selection of two PostgreSQL versions on the same machine (E2).

Each version's plan choice (lowest est_cost) is scored against its own runtimes; bootstrap over queries,
drawn from tasks.common.RNG; Wilcoxon on per-query log slow-downs.

Usage:
  from lcm_recheck.tasks.versions import versions
  versions(argparse.Namespace(a="pg10_local.csv", b="pg18_local.csv", timeout_penalty=None))
"""
import argparse

import numpy as np
import pandas as pd

from lcm_recheck.stats import boot, ci_p, wilcoxon_p
from lcm_recheck.tasks.common import RNG


def per_version(df: pd.DataFrame, penalty: float | None) -> pd.DataFrame:
    df = df.copy()
    if penalty is None:
        df = df[~df.timeout]                                    # paper rule: timed-out plans are dropped
    else:
        df.loc[df.timeout, "runtime_s"] = penalty               # sensitivity: count them at the timeout
    g = df.groupby("query")
    return pd.DataFrame({"r": g.apply(lambda d: d.loc[d.est_cost.idxmin(), "runtime_s"]), "opt": g.runtime_s.min()})


def versions(a: argparse.Namespace) -> None:
    A, Bv = pd.read_csv(a.a), pd.read_csv(a.b)
    na, nb = A.version.iloc[0], Bv.version.iloc[0]
    sa, sb = per_version(A, a.timeout_penalty), per_version(Bv, a.timeout_penalty)
    q = sorted(set(sa.index) & set(sb.index))
    sa, sb = sa.loc[q], sb.loc[q]

    def S(s: pd.DataFrame) -> float:
        return s.r.sum() / s.opt.sum() - 1

    print(f"{len(q)} queries. Total slow-down vs own optimum (paper Fig. 5: PG10 16.23%, PG16 24.9%)")
    print(f"  {na}: {100 * S(sa):.2f}%   {nb}: {100 * S(sb):.2f}%")
    per = list(zip(sa.r, sa.opt, sb.r, sb.opt, strict=False))

    def stat(ps: list[tuple[float, float, float, float]]) -> float:
        return (sum(x[2] for x in ps) / sum(x[3] for x in ps)) - (sum(x[0] for x in ps) / sum(x[1] for x in ps))

    obs, lo, hi, _ = ci_p(boot(per, stat, rng=RNG), stat(per))
    d = np.log(sb.r / sb.opt) - np.log(sa.r / sa.opt)
    p, nz = wilcoxon_p(d)
    print(f"  difference ({nb} - {na}) in total slow-down: {100 * obs:.2f} pp, 95% CI [{100 * lo:.2f}, {100 * hi:.2f}]")
    print(f"  per-query log slow-down, Wilcoxon: p = {p:.4f} ({nz} queries differ)")
    print(f"  queries where each version picks its optimum: {na} {(sa.r == sa.opt).mean():.0%}, {nb} {(sb.r == sb.opt).mean():.0%}")
