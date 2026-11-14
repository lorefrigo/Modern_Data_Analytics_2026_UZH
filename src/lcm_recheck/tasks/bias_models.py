"""bias_models.py: Tests 2 and 3a of the training-data-bias claim (access-path predictions).

Test 2: 21 columns of Table 3, all LCMs, PG10 and PG16 (2a selectivity slope, 2b over-preference for index
scans, 2c false index picks vs PG10). Test 3a: 6 IMDB columns of Fig. 14, 5 fine-tuned LCMs, exact McNemar
before vs after. See scripts/bias_models.py for the full description.

Test 2 bootstraps over columns with the one resample-index matrix of family "bias_models/test2"
(stats.resample_idx), shared across 2a, 2b, 2c and all models. Test 3a has no bootstrap.

Usage:
  from lcm_recheck.tasks.bias_models import test2, test3a
  test2(argparse.Namespace(eval_root="lcm-eval/data/evaluation"))
"""
import argparse
import os

import numpy as np
import pandas as pd

from lcm_recheck.config import (
    FINE_TUNED_LCMS,
    LABEL_SOURCE,
    LCMS,
    PG16,
    REF,
    SCAN_COLUMNS,
    SCAN_FOLDER,
    SEEDS,
    B,
)
from lcm_recheck.report import section
from lcm_recheck.stats import boot_p, ci, holm, mcnemar, resample_idx


# ----------------------------------------------------------------------------- loading
def read_side(root: str, model: str, db: str, side: str, col: str, seed: int) -> pd.DataFrame | None:
    f = os.path.join(root, model, db, SCAN_FOLDER, f"{side}.{col}_{seed}_test_pred.csv")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    return d.groupby("query_index").agg(label=("label", "first"), prediction=("prediction", "mean"))


def read_model(root: str, model: str, db: str, col: str) -> tuple[pd.DataFrame | None, dict[int, pd.DataFrame]]:
    """Returns (seed-averaged frame, {seed: frame}); columns ls, ps, li, pi, indexed by query_index."""
    per_seed = {}
    for s in SEEDS:
        a, b = read_side(root, model, db, "seq", col, s), read_side(root, model, db, "index", col, s)
        if a is None or b is None:
            continue
        per_seed[s] = pd.DataFrame({"ls": a.label, "ps": a.prediction, "li": b.label, "pi": b.prediction}).dropna()
    if not per_seed:
        return None, {}
    avg = pd.concat(per_seed.values()).groupby(level=0).agg(ls=("ls", "first"), ps=("ps", "mean"),
                                                             li=("li", "first"), pi=("pi", "mean"))
    return avg, per_seed


def truth_for(root: str, db: str, col: str) -> pd.DataFrame | None:
    """True runtimes of both plans (Flat Vector's labels, else PG10's) and the nominal selectivity."""
    for src in (LABEL_SOURCE, REF):
        avg, _ = read_model(root, src, db, col)
        if avg is not None:
            t = avg[["ls", "li"]].copy()
            t["s"] = 1 - t.index.to_numpy(float) / 10          # query k: col >= (10k)-th percentile
            t["opt_idx"] = t.li < t.ls
            return t
    return None


def points(root: str, models: list[str], columns_by_db: dict[str, list[str]],
           truth_root: str | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
    """One row per (model, column, query_index): nominal selectivity, optimal and predicted choice."""
    rows, label_dev = [], {}
    for db, cols in columns_by_db.items():
        for col in cols:
            truth = truth_for(truth_root or root, db, col)
            if truth is None:
                continue
            for m in models:
                avg, _ = read_model(root, m, db, col)
                if avg is None:
                    continue
                d = truth.join(avg[["ps", "pi", "ls", "li"]].rename(columns={"ls": "ls_m", "li": "li_m"}), how="inner")
                dev = np.nanmax(np.abs(np.r_[d.ls_m - d.ls, d.li_m - d.li]) / np.r_[d.ls, d.li]) if len(d) else 0
                label_dev[m] = max(label_dev.get(m, 0), dev)
                rows.append(pd.DataFrame({"model": m, "column": f"{db}:{col}", "qi": d.index, "s": d.s.values,
                                          "opt_idx": d.opt_idx.values, "pred_idx": (d.pi < d.ps).values}))
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), label_dev


def common_points(P: pd.DataFrame, members: list[str]) -> pd.DataFrame:
    """Keep the (column, query_index) points that every model in members has."""
    keys = None
    for m in members:
        k = set(map(tuple, P.loc[P.model == m, ["column", "qi"]].values))
        keys = k if keys is None else keys & k
    keep = P[["column", "qi"]].apply(tuple, axis=1).isin(keys)
    return P[keep & P.model.isin(members)].copy()


def show(title: str, df: pd.DataFrame, note: str | None = None) -> None:
    print(f"\n{title}")
    if note:
        for line in note.split("\n"):
            print(f"  {line}")
    print("  (no data)" if df.empty else df.round(4).to_string(index=False))


# ----------------------------------------------------------------------------- Test 2
def test2(a: argparse.Namespace) -> None:
    section("TEST 2: DO THE MODELS SHOW THE BIAS? (access-path selection, 21 columns of Table 3)")
    models = [REF, PG16] + LCMS
    P, label_dev = points(a.eval_root, models, SCAN_COLUMNS)
    if P.empty or REF not in set(P.model):
        print(f"skipped: no predictions under {a.eval_root}/<model>/<db>/{SCAN_FOLDER}/")
        return
    present = [m for m in models if m in set(P.model)]
    core = [m for m in [REF] + LCMS if m in present]
    Pc = common_points(P, core)                            # the points every LCM and PG10 have
    keys = set(map(tuple, Pc.loc[Pc.model == REF, ["column", "qi"]].values))
    for m in [x for x in present if x not in core]:        # PG16 is a reference row; must cover the same points
        k = set(map(tuple, P.loc[P.model == m, ["column", "qi"]].values))
        if keys <= k:
            Pc = pd.concat([Pc, P[(P.model == m) & P[["column", "qi"]].apply(tuple, axis=1).isin(keys)]])
        else:
            print(f"note: {m} lacks some points and is left out")
            present.remove(m)
    P = Pc
    cols = sorted(P.column.unique())
    print(f"models: {', '.join(present)}")
    print(f"points used: {P[P.model == REF].shape[0]} (column, selectivity) pairs in {len(cols)} columns, "
          f"shared by all models")
    big = {m: v for m, v in label_dev.items() if v > 0.01}
    if big:
        print("note: labels in these models' files differ by >1% from Flat Vector's (Flat Vector's are used): "
              + ", ".join(f"{m} ({100 * v:.0f}%)" for m, v in big.items()))
    opt = P[P.model == REF].set_index(["column", "qi"]).opt_idx
    print(f"optimal choice is the index scan on {100 * opt.mean():.1f}% of the points")
    print(f"bootstrap p-values use {B} resamples, so a printed 0 means p < {1 / B:g}")
    print("\nShare of points where each model picks the index scan, by nominal selectivity s "
          "(compare with the 'optimal' row):")
    tab = P.pivot_table(index="model", columns="s", values="pred_idx", aggfunc="mean")
    tab.loc["optimal"] = P[P.model == REF].groupby("s").opt_idx.mean()
    tab = tab.reindex([m for m in ["optimal"] + present if m in tab.index])
    print((100 * tab).round(0).astype("Int64").to_string())

    # column-level sums used by every bootstrap below (resampling unit = column)
    col_idx = {c: i for i, c in enumerate(cols)}
    C = len(cols)
    idx = resample_idx("bias_models/test2", C)

    # ---- 2a: within-column slope of P(pick index) on s
    def sums(m: str, field: str) -> tuple[np.ndarray, np.ndarray]:
        d = P[P.model == m].assign(y=lambda x: x[field].astype(float))
        g = d.groupby("column")
        sc = d.s - g.s.transform("mean")
        yc = d.y - g.y.transform("mean")
        sxy = np.zeros(C)
        sxx = np.zeros(C)
        np.add.at(sxy, d.column.map(col_idx).values, (sc * yc).values)
        np.add.at(sxx, d.column.map(col_idx).values, (sc * sc).values)
        return sxy, sxx

    sxy_opt, sxx = sums(REF, "opt_idx")
    beta_opt, beta_opt_b = sxy_opt.sum() / sxx.sum(), sxy_opt[idx].sum(1) / sxx[idx].sum(1)
    sxy_ref, _ = sums(REF, "pred_idx")
    beta_ref_b = sxy_ref[idx].sum(1) / sxx[idx].sum(1)
    rows = []
    for m in present:
        sxy, _ = sums(m, "pred_idx")
        beta, bb = sxy.sum() / sxx.sum(), sxy[idx].sum(1) / sxx[idx].sum(1)
        rho_b = bb / beta_opt_b
        rows.append({"model": m, "slope": beta, "ci_lo": ci(bb)[0], "ci_hi": ci(bb)[1], "p_slope0": boot_p(bb),
                     "share_of_optimal": beta / beta_opt, "share_ci_lo": ci(rho_b)[0], "share_ci_hi": ci(rho_b)[1],
                     "p_vs_optimal": boot_p(bb - beta_opt_b),
                     "p_vs_PG10": np.nan if m == REF else boot_p(bb - beta_ref_b)})
    df = pd.DataFrame(rows)
    lcm = df.model.isin(LCMS)
    for c in ["p_slope0", "p_vs_optimal", "p_vs_PG10"]:
        df[c + "_holm"] = np.nan
        df.loc[lcm, c + "_holm"] = holm(df.loc[lcm, c].fillna(1).values)
        df.loc[~lcm, c + "_holm"] = df.loc[~lcm, c]          # PG10/PG16 are reference rows, not corrected

    def verdict2a(r: pd.Series) -> str:
        resp, diff = r.p_slope0_holm < 0.05, r.p_vs_optimal_holm < 0.05
        if resp and not diff:
            return "follows selectivity, consistent with optimum"
        if resp and diff:
            return "follows selectivity, but " + ("less" if abs(r.slope) < abs(beta_opt) else "more") + " than optimum"
        if diff:
            return "no detectable response; below optimum"
        return "inconclusive"
    df["verdict"] = df.apply(verdict2a, axis=1)
    show(f"2a  Selectivity sensitivity. slope = change in P(picks index) when s goes from 0 to 1, within columns "
         f"(optimal slope = {beta_opt:.3f}, 95% CI [{ci(beta_opt_b)[0]:.3f}, {ci(beta_opt_b)[1]:.3f}])",
         df, note="share_of_optimal = slope / optimal slope (1 = responds like the optimum, 0 = ignores s).\n"
                  "p-values: cluster bootstrap over columns; Holm across the LCMs (PG rows uncorrected).")

    # ---- 2b: over-preference for index scans (model choice vs optimal choice)
    rows = []
    for m in present:
        d = P[P.model == m]
        b = int((d.pred_idx & ~d.opt_idx).sum())
        c = int((~d.pred_idx & d.opt_idx).sum())
        bc, cc, nc = per_col_counts(d, col_idx, C)
        boot = (bc[idx].sum(1) - cc[idx].sum(1)) / nc[idx].sum(1)
        rows.append({"model": m, "n": len(d), "picks_index": d.pred_idx.mean(), "optimal_index": d.opt_idx.mean(),
                     "over_preference": (b - c) / len(d), "ci_lo": ci(boot)[0], "ci_hi": ci(boot)[1],
                     "idx_when_seq_faster": b, "seq_when_idx_faster": c, "mcnemar_p": mcnemar(b, c)})
    df = pd.DataFrame(rows)
    lcm = df.model.isin(LCMS)
    df["mcnemar_p_holm"] = df.mcnemar_p
    df.loc[lcm, "mcnemar_p_holm"] = holm(df.loc[lcm, "mcnemar_p"].values)
    df["verdict"] = [verdict_dir(r.mcnemar_p_holm, r.ci_lo, r.ci_hi, r.over_preference, "prefers index", "prefers seq")
                     for r in df.itertuples()]
    show("2b  Over-preference = P(picks index) - P(index is optimal). Positive = index scans chosen too often.",
         df, note="mcnemar_p: exact McNemar, model choice vs optimal choice on the same points; Holm across LCMs.\n"
                  "ci: cluster bootstrap over columns. A verdict needs both the Holm p < 0.05 and a CI excluding 0.")

    # ---- 2c: false-index rate vs PG10 on the points where the sequential scan is faster
    ref = P[P.model == REF].set_index(["column", "qi"])
    rows = []
    for m in [x for x in present if x != REF]:
        d = P[P.model == m].set_index(["column", "qi"])
        d = d[~d.opt_idx]
        r = ref.loc[d.index]
        x, y = d.pred_idx.values, r.pred_idx.values
        only_m, only_ref = int((x & ~y).sum()), int((y & ~x).sum())
        dd = d.reset_index()
        diff = np.zeros(C)
        n_c = np.zeros(C)
        np.add.at(diff, dd.column.map(col_idx).values, x.astype(float) - y.astype(float))
        np.add.at(n_c, dd.column.map(col_idx).values, 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = diff[idx].sum(1) / n_c[idx].sum(1)
        boot = boot[np.isfinite(boot)]
        rows.append({"model": m, "n_seq_faster": len(d), "false_index_model": x.mean(), "false_index_PG10": y.mean(),
                     "difference": x.mean() - y.mean(), "ci_lo": ci(boot)[0], "ci_hi": ci(boot)[1],
                     "only_model_wrong": only_m, "only_PG10_wrong": only_ref, "mcnemar_p": mcnemar(only_m, only_ref)})
    df = pd.DataFrame(rows)
    lcm = df.model.isin(LCMS)
    df["mcnemar_p_holm"] = df.mcnemar_p
    df.loc[lcm, "mcnemar_p_holm"] = holm(df.loc[lcm, "mcnemar_p"].values)
    df["verdict"] = [verdict_dir(r.mcnemar_p_holm, r.ci_lo, r.ci_hi, r.difference,
                                 "more false index picks than PG10", "fewer false index picks than PG10")
                     for r in df.itertuples()]
    show("2c  Points where the sequential scan is faster: share where the model still picks the index scan, vs PG10",
         df, note="mcnemar_p: exact McNemar on the points where exactly one of the two is wrong; Holm across LCMs.")


def per_col_counts(d: pd.DataFrame, col_idx: dict[str, int], C: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b = np.zeros(C)
    c = np.zeros(C)
    n = np.zeros(C)
    ci_ = d.column.map(col_idx).values
    np.add.at(b, ci_, (d.pred_idx & ~d.opt_idx).values.astype(float))
    np.add.at(c, ci_, (~d.pred_idx & d.opt_idx).values.astype(float))
    np.add.at(n, ci_, 1.0)
    return b, c, n


def verdict_dir(p_holm: float, lo: float, hi: float, est: float, pos: str, neg: str) -> str:
    if p_holm < 0.05 and (lo > 0 or hi < 0):
        return pos if est > 0 else neg
    if p_holm < 0.05:
        return "McNemar significant, but not robust to column clustering"
    return "no clear difference"


# ----------------------------------------------------------------------------- Test 3a
def test3a(a: argparse.Namespace) -> None:
    section("TEST 3a: DOES FINE-TUNING ON DIVERSIFIED DATA FIX THE CHOICES? (6 IMDB columns, Fig. 14)")
    if not os.path.isdir(a.retrain_root):
        print(f"skipped: {a.retrain_root} not found (fine-tuned predictions not downloaded)")
        return
    cols = {"imdb": SCAN_COLUMNS["imdb"]}
    print(f"before: {a.eval_root}   after: {a.retrain_root}")
    print("Truth for both = the runtimes in the 'before' files (Flat Vector's labels), so before and after are "
          "judged against the same optimum.")
    rows, seed_rows, col_rows = [], [], []
    for m in FINE_TUNED_LCMS:
        before_all, after_all = [], []
        for db, cs in cols.items():
            for col in cs:
                truth = truth_for(a.eval_root, db, col)
                (bavg, bseed), (aavg, aseed) = read_model(a.eval_root, m, db, col), read_model(a.retrain_root, m, db, col)
                if truth is None or bavg is None or aavg is None:
                    continue
                j = truth.join(bavg[["ps", "pi"]], how="inner").join(
                    aavg[["ps", "pi", "ls", "li"]].rename(columns={"ps": "ps_a", "pi": "pi_a", "ls": "ls_a", "li": "li_a"}),
                    how="inner")
                j["column"] = col
                before_all.append(j)
                for s in sorted(set(bseed) & set(aseed)):
                    k = truth.join(bseed[s][["ps", "pi"]], how="inner").join(
                        aseed[s][["ps", "pi"]].rename(columns={"ps": "ps_a", "pi": "pi_a"}), how="inner")
                    after_all.append(k.assign(seed=s, column=col))
        if not before_all:
            print(f"  {m}: no before/after pair found, skipped")
            continue
        d = pd.concat(before_all)
        dev = np.nanmax(np.abs(np.r_[d.ls_a - d.ls, d.li_a - d.li]) / np.r_[d.ls, d.li])
        if dev > 0.01:
            print(f"  note: {m}: runtimes in the fine-tuned files differ from the 'before' ones by up to "
                  f"{100 * dev:.0f}% (the 'before' runtimes are used)")
        cb = (d.pi < d.ps) == d.opt_idx
        ca = (d.pi_a < d.ps_a) == d.opt_idx
        n01, n10 = int((~cb & ca).sum()), int((cb & ~ca).sum())
        seqf = ~d.opt_idx
        f01, f10 = int((~cb & ca & seqf).sum()), int((cb & ~ca & seqf).sum())
        over_b = ((d.pi < d.ps) & seqf).sum() - (~(d.pi < d.ps) & d.opt_idx).sum()
        over_a = ((d.pi_a < d.ps_a) & seqf).sum() - (~(d.pi_a < d.ps_a) & d.opt_idx).sum()
        rows.append({"model": m, "n": len(d), "acc_before": cb.mean(), "acc_after": ca.mean(), "wrong_to_right": n01,
                     "right_to_wrong": n10, "mcnemar_p": mcnemar(n01, n10), "seq_faster_n": int(seqf.sum()),
                     "seq_faster_fixed": f01, "seq_faster_broken": f10, "seq_faster_p": mcnemar(f01, f10),
                     "over_pref_before": over_b / len(d), "over_pref_after": over_a / len(d)})
        for col, g in d.assign(cb=cb, ca=ca).groupby("column"):
            col_rows.append({"model": m, "column": col, "n": len(g), "wrong_to_right": int((~g.cb & g.ca).sum()),
                             "right_to_wrong": int((g.cb & ~g.ca).sum())})
        if after_all:
            e = pd.concat(after_all)
            for s, g in e.groupby("seed"):
                gb = (g.pi < g.ps) == g.opt_idx
                ga = (g.pi_a < g.ps_a) == g.opt_idx
                x01, x10 = int((~gb & ga).sum()), int((gb & ~ga).sum())
                seed_rows.append({"model": m, "seed": s, "n": len(g), "wrong_to_right": x01, "right_to_wrong": x10,
                                  "mcnemar_p": mcnemar(x01, x10)})
    if not rows:
        print("no fine-tuned predictions found")
        return
    df = pd.DataFrame(rows)
    df["mcnemar_p_holm"] = holm(df.mcnemar_p.values)
    df["seq_faster_p_holm"] = holm(df.seq_faster_p.values)
    df["verdict"] = np.where(df.mcnemar_p_holm >= 0.05, "no clear change",
                             np.where(df.wrong_to_right > df.right_to_wrong, "fine-tuning helped", "fine-tuning hurt"))
    show("3a  Per point, before vs after fine-tuning (predictions averaged over seeds)", df,
         note="mcnemar_p: exact McNemar on wrong->right vs right->wrong; seq_faster_*: the same on the points where\n"
              "the sequential scan is faster (fixed = was a false index pick, now correct). Holm across the 5 models.\n"
              "over_pref_*: P(picks index) - P(index optimal), as in 2b, before and after.")
    show("3a  Per column (is the change spread over the columns or concentrated in one or two?)", pd.DataFrame(col_rows))
    if seed_rows:
        show("3a  Per seed (seed s before vs seed s after; consistent direction across seeds = not seed noise)",
             pd.DataFrame(seed_rows))
    print("\n  CAUTION: points within a column are correlated and there are only 6 columns, so these p-values "
          "describe these 6 columns, not columns in general. Run bias_data.py for the leakage check (do the "
          "fine-tuning queries filter on these columns?).")
