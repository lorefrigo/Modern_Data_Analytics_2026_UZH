"""Significance tests for "How Good are Learned Cost Models, Really?" (Heinrich et al., SIGMOD 2025).

paper    : tests the paper's main claim ("in none of the tasks LCMs can significantly beat traditional
           approaches") on the authors' published predictions, for all three tasks:
             Task 1 join ordering   (JOB-light; Fig. 1, Fig. 4)       H1 runtime, H2 Q-error, seeds,
                                                                      H1b LCMs with actual cardinalities
             Task 2 access paths    (Table 3, Fig. 14)                balanced accuracy, selected runtime,
                                                                      fine-tuned models
             Task 3 join operators  (Fig. 11, Section 6.6)            selected runtime, pick rate,
                                                                      with additional indexes
           and ends with a summary of every LCM-vs-PG10 comparison.
versions : PG10 vs PG18 plan selection on the same machine, from two extract_runs.py CSVs.

Usage:
  python sigtests.py paper    --eval_root lcm-eval/data/evaluation [--out results/sigtests_paper.txt]
                              [--retrain_root lcm-eval/data/retraining_evaluation] [--tasks join access physical]
  python sigtests.py versions --a pg10_local.csv --b pg18_local.csv [--timeout_penalty 30] [--out FILE]

--out writes everything that is printed to FILE as well (still shown in the terminal). An existing
file is always overwritten. The file starts with the date, the exact command and the package versions.
"""
import argparse, glob, os, re
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, binomtest

from lcm_recheck.config import (LCMS, ACT_CARD_LCMS, FINE_TUNED_LCMS, INDEX_LCMS, REF, PG16, PG_VERSIONS,
                                LABEL_SOURCE, B, BASE_SEED, SEEDS, SCAN_COLUMNS, PHYSICAL, PHYSICAL_INDEX,
                                JOIN_ORDER_FOLDER, SCAN_FOLDER, default_retrain_root)
from lcm_recheck.report import section, tee_output
from lcm_recheck.stats import holm

RNG = np.random.default_rng(BASE_SEED)

SUMMARY = []   # every LCM-vs-PG10 verdict, collected for the final summary


# ----------------------------------------------------------------------------- statistics helpers
def boot(per_query, stat, b=B):
    """Resample units with replacement; per_query is a list of per-unit objects (used by Task 1)."""
    n = len(per_query)
    return np.array([stat([per_query[j] for j in RNG.integers(0, n, n)]) for _ in range(b)])


def boot_fast(values, stat, b=B):
    """Vectorised version for numeric arrays (used by Tasks 2 and 3)."""
    values = np.asarray(values, float)
    idx = RNG.integers(0, len(values), (b, len(values)))
    return stat(values[idx], axis=1)


def ci_p(samples, observed, alpha=0.05):
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = min(1.0, 2 * min((samples <= 0).mean(), (samples >= 0).mean()))
    return observed, lo, hi, p


def compare(x, ref, k, kind, legacy=False):
    """Paired comparison of one model with PG10 over units (queries or columns).

    kind="runtime": statistic = sum of differences (lower is better), Wilcoxon on log ratios.
    kind="score":   statistic = mean of differences (higher is better), Wilcoxon on differences.
    Verdict from the Bonferroni-adjusted (level 1 - 0.05/k) bootstrap interval; none for fewer than 10 units."""
    x, ref = np.asarray(x, float), np.asarray(ref, float)
    d = x - ref
    stat = np.sum if kind == "runtime" else np.mean
    samples = boot(list(d), stat) if legacy else boot_fast(d, stat)
    obs, lo, hi, _ = ci_p(samples, stat(d))
    _, flo, fhi, _ = ci_p(samples, stat(d), alpha=0.05 / k)
    w = np.log(x / ref) if kind == "runtime" and (x > 0).all() and (ref > 0).all() else d
    nz = int((np.abs(w) > 1e-12).sum())
    p = wilcoxon(w, zero_method="wilcox").pvalue if nz >= 6 else np.nan
    if len(d) < 10:
        verdict = "too few units to test"           # bootstrap over < 10 units is unreliable
    elif kind == "runtime":
        verdict = "PG10 better" if flo > 0 else ("LCM better" if fhi < 0 else "no clear difference")
    else:
        verdict = "LCM better" if flo > 0 else ("PG10 better" if fhi < 0 else "no clear difference")
    return dict(n=len(d), delta=obs, ci95_lo=lo, ci95_hi=hi, fwer_ci_lo=flo, fwer_ci_hi=fhi, units_differ=nz,
                wilcoxon_p=p, largest_unit_share=np.abs(d).max() / max(np.abs(d).sum(), 1e-12), verdict=verdict)


def compare_pick(x, ref):
    """Paired comparison of pick rates (picked the optimal plan: yes/no per query); exact McNemar test."""
    x, ref = np.asarray(x, bool), np.asarray(ref, bool)
    d = x.astype(float) - ref.astype(float)
    obs, lo, hi, _ = ci_p(boot_fast(d, np.mean), d.mean())
    only_ref, only_x = int((ref & ~x).sum()), int((x & ~ref).sum())
    p = binomtest(only_x, only_ref + only_x, 0.5).pvalue if only_ref + only_x else 1.0
    return dict(n=len(d), pick_lcm=x.mean(), pick_pg10=ref.mean(), delta=obs, ci95_lo=lo, ci95_hi=hi,
                only_lcm_optimal=only_x, only_pg10_optimal=only_ref, mcnemar_p=p)


def finish_family(rows, pick=False):
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


def show(title, df, note=None):
    print(f"\n{title}")
    if note:
        print(f"  {note}")
    if df.empty:
        print("  (no data)")
    else:
        print(df.round(3).to_string(index=False))


def record(task, comparison, metric, df):
    for _, r in df.iterrows():
        SUMMARY.append(dict(task=task, comparison=comparison, dataset=r.get("dataset", ""), metric=metric,
                            model=r["model"], verdict=r["verdict"]))


def balanced_accuracy(y_true, y_pred):
    """Same as sklearn.metrics.balanced_accuracy_score: mean recall over the classes present in y_true."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean([np.mean(y_pred[y_true == c] == c) for c in np.unique(y_true)]))


# ----------------------------------------------------------------------------- loading
def load_preds(eval_root, model):
    """Task 1: all JOB-light prediction files of one model (all seeds)."""
    files = glob.glob(os.path.join(eval_root, model, "imdb", JOIN_ORDER_FOLDER, "job_light_*_*_test_pred.csv"))
    out = []
    for f in files:
        q, seed = re.match(r"(job_light_\d+)_(\d+)_test_pred\.csv", os.path.basename(f)).groups()
        d = pd.read_csv(f); d["query"], d["seed"] = q, int(seed); out.append(d)
    return pd.concat(out) if out else None


def load_workload(root, model, db, folder, wl):
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


def with_common_labels(preds):
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


def selected(df):
    """Task 1, per query: runtime of the plan with the lowest prediction, and the optimum."""
    g = df.groupby("query")
    return pd.DataFrame({"r": g.apply(lambda d: d.loc[d.prediction.idxmin(), "label"]),
                         "opt": g.label.min()})


# ----------------------------------------------------------------------------- Task 1: join ordering
def task_join(a):
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

    print(f"\nTotals over these queries (paper: optimal 446s, Sc.PG10 518s, Zero-Shot 530s, QueryFormer 830s)")
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
            rows.append(dict(model=m, **compare(sel[m].loc[qs, "r"].values, sel[REF].loc[qs, "r"].values,
                                                 len(members), "runtime", legacy=True)))
        h1 = finish_family(rows)
        show(f"H1 [{fam}] delta = sum_q (r_model - r_PG10) in s; bootstrap over queries", h1)
        if fam == "LCM vs PG10":
            record("1 join order", "LCM vs PG10", "selected runtime", h1)

    # H2: accuracy, median Q-error, cluster bootstrap by query
    def qerr(d):
        p = d.prediction.clip(lower=1e-6)
        return np.maximum(p / d.label, d.label / p)
    rows = []
    ref_q = {q: qerr(g).values for q, g in avg[REF][avg[REF]["query"].isin(queries)].groupby("query")}
    neg = (avg[REF].prediction <= 0).sum()
    if neg:
        print(f"\nnote: {neg} Sc.PG10 predictions <= 0 (linear fit), clipped for Q-error")
    for m in [x for x in LCMS if x in avg]:
        mq = {q: qerr(g).values for q, g in avg[m][avg[m]["query"].isin(queries)].groupby("query")}
        pairs = [(mq[q], ref_q[q]) for q in queries]
        stat = lambda ps: np.median(np.concatenate([x for x, _ in ps])) - np.median(np.concatenate([y for _, y in ps]))
        obs, lo, hi, p = ci_p(boot(pairs, stat, b=2000), stat(pairs))
        rows.append(dict(model=m, median_qerr_diff=obs, ci_lo=lo, ci_hi=hi, boot_p=p))
    h2 = pd.DataFrame(rows); h2["boot_p_holm"] = holm(h2.boot_p)
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
            rows.append(dict(model=m, **compare(sel[m].loc[qs, "r"].values, sel[REF].loc[qs, "r"].values,
                                                 len(members), "runtime")))
        h1b = finish_family(rows)
        show("H1b [LCM with actual cardinalities vs PG10 with estimated ones] delta = sum_q (r_model - r_PG10) in s",
             h1b, note="Fig. 4 lighter bars. The PG10/PG16 values with actual cardinalities in Fig. 5 are typed "
                       "into the authors' notebook, not computed from published files, so they cannot be tested.")
        record("1 join order", "LCM (act. card.) vs PG10", "selected runtime", h1b)


# ----------------------------------------------------------------------------- Task 2: access paths
def access_path_scores(root, models, columns_by_db):
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
            seq, dev1 = with_common_labels(seq); idx, dev2 = with_common_labels(idx)
            deviating.update(dev1 + dev2)
            for m in models:
                if m not in seq or m not in idx:
                    continue
                d = pd.DataFrame({"ls": seq[m].label, "ps": seq[m].prediction,
                                  "li": idx[m].label, "pi": idx[m].prediction}).dropna()
                if d.empty:
                    continue
                optimal, predicted = d.li < d.ls, d.pi < d.ps
                out[m][f"{db}:{col}"] = dict(B=balanced_accuracy(optimal, predicted),
                                             runtime=np.where(predicted, d.li, d.ls).sum(),
                                             optimum=np.minimum(d.li, d.ls).sum(), n=len(d))
    return out, deviating


def task_access(a):
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

    for metric, kind, key, title in [
            ("balanced accuracy", "score", "B", "T2a [LCM vs PG10] balanced accuracy: delta = mean over columns of (B_LCM - B_PG10); bootstrap over columns"),
            ("selected runtime", "runtime", "runtime", "T2b [LCM vs PG10] selected runtime: delta = sum over columns of (runtime_LCM - runtime_PG10); bootstrap over columns")]:
        members = [m for m in LCMS if scores[m]]
        rows = []
        for m in members:
            cs = [c for c in cols if c in scores[m]]
            rows.append(dict(model=m, **compare([scores[m][c][key] for c in cs], [scores[REF][c][key] for c in cs],
                                                 len(members), kind)))
        df = finish_family(rows)
        show(title, df)
        record("2 access path", "LCM vs PG10", metric, df)
        if scores[PG16]:
            cs = [c for c in cols if c in scores[PG16]]
            show(f"{title.split(']')[0].replace('LCM', 'PG16')}] {metric}",
                 finish_family([dict(model=PG16, **compare([scores[PG16][c][key] for c in cs],
                                                            [scores[REF][c][key] for c in cs], 1, kind))]))

    # Fig. 14: fine-tuned LCMs on the IMDB columns
    if not os.path.isdir(a.retrain_root):
        print(f"\nFig. 14 skipped: {a.retrain_root} not found (fine-tuned predictions not downloaded)")
        return
    ref_root = a.retrain_root if os.path.isdir(os.path.join(a.retrain_root, REF)) else a.eval_root
    roots = lambda m: ref_root if m == REF else a.retrain_root
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
        rows = [dict(model=m, **compare([ft[m][c][key] for c in cols_ft], [ft[REF][c][key] for c in cols_ft],
                                        len(members), kind)) for m in members]
        df = finish_family(rows)
        show(f"T2c [fine-tuned LCM vs PG10] {metric}", df)
        record("2 access path", "fine-tuned LCM vs PG10 (Fig. 14)", metric, df)


# ----------------------------------------------------------------------------- Task 3: join operators
def physical_scores(root, models, db, folder, names):
    """Per model and query: selected runtime, picked optimum (yes/no), optimum, median Q-error."""
    out = {m: {} for m in models}
    deviating = set()
    for wl in names:
        preds = {m: load_workload(root, m, db, folder, wl) for m in models}
        preds = {m: d for m, d in preds.items() if d is not None}
        if REF not in preds:
            continue
        preds, dev = with_common_labels(preds); deviating.update(dev)
        for m, d in preds.items():
            if d.empty:
                continue
            p = np.abs(d.prediction.values); l = d.label.values
            with np.errstate(divide="ignore", invalid="ignore"):
                q = np.nan_to_num(np.maximum(l / p, p / l), nan=np.inf)
            i_pred, i_opt = int(np.argmin(d.prediction.values)), int(np.argmin(l))
            out[m][wl] = dict(runtime=l[i_pred], pick=i_pred == i_opt, optimum=l[i_opt], qerr=np.percentile(q, 50))
    return out, deviating


def physical_family(scores_by_ds, members, k_runtime, comparison, task):
    """Runtime and pick-rate comparisons of `members` vs PG10, one family across all datasets."""
    rt, pk = [], []
    for ds, (scores, queries) in scores_by_ds.items():
        for m in members:
            qs = [q for q in queries if q in scores[m]]
            if not qs:
                continue
            rt.append(dict(dataset=ds, model=m, **compare([scores[m][q]["runtime"] for q in qs],
                                                          [scores[REF][q]["runtime"] for q in qs], k_runtime, "runtime")))
            pk.append(dict(dataset=ds, model=m, **compare_pick([scores[m][q]["pick"] for q in qs],
                                                               [scores[REF][q]["pick"] for q in qs])))
    rt, pk = finish_family(rt), finish_family(pk, pick=True)
    show(f"[{comparison}] selected runtime: delta = sum_q (r_model - r_PG10); bootstrap over queries", rt)
    show(f"[{comparison}] pick rate (share of queries where the optimal join operator was picked); "
         f"exact McNemar test, Holm across the family", pk)
    if "LCM" in comparison:
        record(task, comparison, "selected runtime", rt)
        record(task, comparison, "pick rate", pk)


def task_physical(a):
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


def summary():
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


def paper(a):
    tasks = {"join": task_join, "access": task_access, "physical": task_physical}
    for t in a.tasks:
        tasks[t](a)
    summary()
    print("\nVerdicts: runtime and balanced accuracy use the bootstrap interval at level 1 - 0.05/k "
          "(k = comparisons in the family; none below 10 units); pick rate uses the Holm-adjusted exact McNemar test.")


# -------------------------------------------------------------------------- versions mode
def per_version(df, penalty):
    df = df.copy()
    if penalty is None:
        df = df[~df.timeout]                                    # paper rule: timed-out plans are dropped
    else:
        df.loc[df.timeout, "runtime_s"] = penalty               # sensitivity: count them at the timeout
    g = df.groupby("query")
    return pd.DataFrame({"r": g.apply(lambda d: d.loc[d.est_cost.idxmin(), "runtime_s"]), "opt": g.runtime_s.min()})


def versions(a):
    A, Bv = pd.read_csv(a.a), pd.read_csv(a.b)
    na, nb = A.version.iloc[0], Bv.version.iloc[0]
    sa, sb = per_version(A, a.timeout_penalty), per_version(Bv, a.timeout_penalty)
    q = sorted(set(sa.index) & set(sb.index)); sa, sb = sa.loc[q], sb.loc[q]
    S = lambda s: s.r.sum() / s.opt.sum() - 1
    print(f"{len(q)} queries. Total slow-down vs own optimum (paper Fig. 5: PG10 16.23%, PG16 24.9%)")
    print(f"  {na}: {100 * S(sa):.2f}%   {nb}: {100 * S(sb):.2f}%")
    per = list(zip(sa.r, sa.opt, sb.r, sb.opt))
    stat = lambda ps: (sum(x[2] for x in ps) / sum(x[3] for x in ps)) - (sum(x[0] for x in ps) / sum(x[1] for x in ps))
    obs, lo, hi, _ = ci_p(boot(per, stat), stat(per))
    d = np.log(sb.r / sb.opt) - np.log(sa.r / sa.opt)
    nz = int((np.abs(d) > 1e-12).sum())
    p = wilcoxon(d, zero_method="wilcox").pvalue if nz >= 6 else np.nan
    print(f"  difference ({nb} - {na}) in total slow-down: {100 * obs:.2f} pp, 95% CI [{100 * lo:.2f}, {100 * hi:.2f}]")
    print(f"  per-query log slow-down, Wilcoxon: p = {p:.4f} ({nz} queries differ)")
    print(f"  queries where each version picks its optimum: {na} {(sa.r == sa.opt).mean():.0%}, {nb} {(sb.r == sb.opt).mean():.0%}")


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", help="also write the output to this file (folders are created; an existing file is overwritten)")
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="mode", required=True)
    p1 = sp.add_parser("paper", parents=[common]); p1.add_argument("--eval_root", required=True)
    p1.add_argument("--retrain_root", help="fine-tuned predictions (Fig. 14); default: retraining_evaluation next to eval_root")
    p1.add_argument("--tasks", nargs="+", choices=["join", "access", "physical"], default=["join", "access", "physical"])
    p2 = sp.add_parser("versions", parents=[common]); p2.add_argument("--a", required=True); p2.add_argument("--b", required=True)
    p2.add_argument("--timeout_penalty", type=float, default=None)
    a = ap.parse_args()
    if a.mode == "paper":
        a.eval_root = os.path.expanduser(a.eval_root)
        a.retrain_root = os.path.expanduser(a.retrain_root or default_retrain_root(a.eval_root))

    with tee_output(a.out, header_extra=f"; bootstrap seed {BASE_SEED}",
                    closing=f"Output written to {a.out} (any previous file with this name was overwritten)"):
        paper(a) if a.mode == "paper" else versions(a)


if __name__ == "__main__":
    main()
