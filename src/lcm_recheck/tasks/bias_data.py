"""bias_data.py: Test 1 of the training-data-bias claim (is the training data itself biased?).

1a scan composition, 1b selectivity of index vs sequential scans (KS), 1c missing negative examples
(Clopper-Pearson, Fisher) against the index/seq retraining queries, 1d timeouts (Fisher), 1e unique and
clustered index columns, and the leakage check for Test 3a. Reads the authors' raw run files via
lcm_recheck.runs. No random numbers. See scripts/bias_data.py for the full description.

Usage:
  from lcm_recheck.tasks.bias_data import run
  run(args)   # args as parsed by scripts/bias_data.py
"""
import argparse
import os
import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta, fisher_exact, ks_2samp

from lcm_recheck.config import SCAN_COLUMNS
from lcm_recheck.report import section
from lcm_recheck.runs import SCAN_KIND, exec_ms, ijson, load_run, plan_nodes
from lcm_recheck.stats import cp_upper, wilson

PARAM_RE = re.compile(r"(?:=|<|>)\s*\(?\s*(?:\$\d|\"?[A-Za-z_]\w*\"?\.\"?[A-Za-z_])")   # col = other.col / $1
COND_COL_RE = re.compile(r"\(*\s*(?:\"?\w+\"?\.)?\"?(\w+)\"?\)?(?:::[\w ]+?)?\s*(?:>=|<=|<>|!=|=|<|>|~~)")
SQL_COL_RE = re.compile(r"\"?(\w+)\"?\.\"?(\w+)\"?\s*(?:>=|<=|<>|!=|=|<|>|\bNOT LIKE\b|\bLIKE\b|\bIS\b|\bIN\b)", re.IGNORECASE)
HINT_RE = re.compile(r"/\*\+.*?\*/", re.DOTALL)


# ----------------------------------------------------------------------------- reading run files
def table_sizes(stats: dict[str, Any] | None) -> dict[str, float]:
    if not stats:
        return {}
    return {t["relname"]: float(t["reltuples"]) for t in stats.get("table_stats", []) if t.get("reltuples")}


def column_stats(stats: dict[str, Any] | None) -> dict[tuple[str, str], tuple[Any, Any]]:
    """{(table, column), lower case: (n_distinct, correlation)} from pg_stats."""
    if not stats:
        return {}
    return {(str(c["tablename"]).lower(), str(c["attname"]).lower()): (c.get("n_distinct"), c.get("correlation"))
            for c in stats.get("column_stats", []) if c.get("tablename") and c.get("attname")}


def scan_records(plan: Any, sizes: dict[str, float]) -> list[dict[str, Any]]:
    """Executed scan nodes with their selectivity.

    filter_scan: a scan that applies a filter to a base table once (loops = 1, or a parallel scan), so its
    selectivity is a filter selectivity; index scans driven by a join (Index Cond = other.col) are excluded.
    s_out:  rows returned / table rows.
    s_cond: rows matched by the index condition / table rows (rows returned + rows removed by a later filter);
            this is what decides the cost of an index scan. Equal to s_out for sequential scans."""
    recs = []
    for n in plan_nodes(plan):
        if n["kind"] is None or n["rows"] is None or not n["loops"]:
            continue
        f = n["loops"] if n["parallel"] else 1                 # parallel nodes report per-process averages
        rows, removed = n["rows"] * f, n["removed"] * f
        size = sizes.get(n["rel"])
        const = n["cond"] is None or not PARAM_RE.search(n["cond"])
        filt = (n["loops"] == 1 or n["parallel"]) and (n["kind"] == "seq" or (n["cond"] is not None and const))
        s_out = min(rows / size, 1.0) if size else np.nan
        s_cond = min((rows + removed) / size, 1.0) if size else np.nan
        m = COND_COL_RE.search(n["cond"]) if n["cond"] else None
        recs.append({"kind": n["kind"], "parallel": n["parallel"], "rel": n["rel"], "filter_scan": filt,
                         "s_out": s_out, "s_cond": s_cond if n["kind"] != "seq" else s_out,
                         "col": m.group(1) if m else None})
    return recs


def first_executed_plan(q: dict[str, Any]) -> Any:
    aps = q.get("analyze_plans") or []
    return aps[0] if aps else None


def mean_exec_ms(q: dict[str, Any]) -> float:
    t = [exec_ms(p) for p in (q.get("analyze_plans") or [])]
    t = [x for x in t if np.isfinite(x)]
    return float(np.mean(t)) if t else np.nan


def normalize_sql(sql: str | None) -> str:
    return re.sub(r"\s+", " ", HINT_RE.sub("", sql or "")).strip().rstrip(";").strip().lower()
# ----------------------------------------------------------------------------- crossover rule
def threshold(s: Any, y: Any) -> tuple[float, float]:
    """Best rule 'index faster iff s <= tau' on these points. Returns (tau, accuracy)."""
    o = np.argsort(s)
    s, y = np.asarray(s)[o], np.asarray(y, bool)[o]
    n = len(s)
    err = np.r_[0, np.cumsum(~y)] + (y.sum() - np.r_[0, np.cumsum(y)])   # predict index for the first j points
    valid = np.ones(n + 1, bool)
    valid[1:n] = s[:-1] < s[1:]                                         # only cut between distinct values
    errv = np.where(valid, err, np.inf)
    best = np.flatnonzero(errv == errv.min())
    j = best[len(best) // 2]
    if j == 0:
        tau = 0.0
    elif j == n:
        tau = 1.0
    else:
        lo, hi = s[j - 1], s[j]
        tau = float(np.sqrt(lo * hi)) if lo > 0 else float((lo + hi) / 2)
    return tau, 1 - err[j] / n


# ----------------------------------------------------------------------------- training data
def file_label(path: str) -> str:
    """'<database folder>/<file name>': the training files of all databases share the same file name."""
    return os.path.join(os.path.basename(os.path.dirname(os.path.abspath(path))), os.path.basename(path))


def read_training(paths: list[str], max_queries: int | None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Per file: scan records of the first executed plan of every completed query, and per-query flags."""
    scan_frames, query_frames, db_stats = [], [], {}
    for p in paths:
        it, stats = load_run(p, max_queries)
        sizes = table_sizes(stats)
        name = file_label(p)
        db_stats[name] = (sizes, column_stats(stats))
        scans, queries = [], []
        n = 0
        for i, q in enumerate(it):
            n += 1
            to = bool(q.get("timeout"))
            vp = q.get("verbose_plan")
            ops = [x["op"] for x in plan_nodes(vp)] if vp else None
            queries.append({"file": name, "q": i, "timeout": to, "has_verbose": ops is not None,
                                "has_nl": ("Nested Loop" in ops) if ops else None,
                                "has_index": any(SCAN_KIND.get(o) == "index" for o in ops) if ops else None})
            ap = None if to else first_executed_plan(q)
            if ap:
                for r in scan_records(ap, sizes):
                    scans.append(dict(file=name, q=i, **r))
        print(f"  {name}: {n} queries read, {len(sizes)} tables with sizes"
              + ("" if sizes else "  (no database_stats: selectivities unavailable)"))
        scan_frames.append(compact(pd.DataFrame(scans)))     # one compact frame per file keeps memory low
        query_frames.append(compact(pd.DataFrame(queries)))
    return (pd.concat(scan_frames, ignore_index=True) if scan_frames else pd.DataFrame(),
            pd.concat(query_frames, ignore_index=True) if query_frames else pd.DataFrame(), db_stats)


def compact(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["file", "kind", "rel", "col"]:
        if c in df:
            df[c] = df[c].astype("category")
    for c in ["s_out", "s_cond"]:
        if c in df:
            df[c] = df[c].astype("float32")
    return df


def test1a(scans: pd.DataFrame) -> None:
    print("\n1a  Composition of the scans in the training plans")
    print("    all = every executed scan node; filter = scans that filter a base table once (see 1b/1c).")
    rows = []
    for (f, sub), d in [((f, "all"), g) for f, g in scans.groupby("file", observed=True)] + \
                       [((f, "filter"), g[g.filter_scan]) for f, g in scans.groupby("file", observed=True)] + \
                       ([(("ALL FILES", "all"), scans), (("ALL FILES", "filter"), scans[scans.filter_scan])]
                        if scans.file.nunique() > 1 else []):
        n = len(d)
        r = {"file": f, "scans": sub, "n": n}
        for k in ["seq", "index", "bitmap"]:
            c = int((d.kind == k).sum())
            lo, hi = wilson(c, n)
            r[f"{k}_%"] = 100 * c / n if n else np.nan
            r[f"{k}_ci95"] = f"[{100 * lo:.1f}, {100 * hi:.1f}]"
        rows.append(r)
    print(pd.DataFrame(rows).round(1).to_string(index=False))


def test1b(scans: pd.DataFrame) -> None:
    print("\n1b  Selectivity (rows returned / table rows) of filter scans: index vs sequential")
    print("    H0: both come from the same distribution (two-sample Kolmogorov-Smirnov).")
    d = scans[scans.filter_scan & scans.s_out.notna()]
    rows = []
    for f, g in list(d.groupby("file", observed=True)) + ([("ALL FILES", d)] if d.file.nunique() > 1 else []):
        a, b = g.loc[g.kind == "index", "s_out"].values, g.loc[g.kind == "seq", "s_out"].values
        r = {"file": f, "n_index": len(a), "n_seq": len(b)}
        for name, v in [("index", a), ("seq", b)]:
            q = np.percentile(v, [10, 50, 90]) if len(v) else [np.nan] * 3
            r[f"{name}_p10/p50/p90"] = "/".join(f"{x:.4f}" for x in q)
        r["KS_D"], r["KS_p"] = ks_2samp(a, b) if len(a) and len(b) else (np.nan, np.nan)
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))


def unique_col(nd: Any, rows: float | None) -> float:
    """n_distinct < 0 is minus the distinct fraction of rows; > 0 is a count. Unique = covers >= 99% of rows."""
    if nd is None:
        return np.nan
    nd = float(nd)
    if nd < 0:
        return float(nd <= -0.99)
    return float(bool(rows) and nd >= 0.99 * rows)


def clustered_col(corr: Any) -> float:
    return np.nan if corr is None else float(abs(float(corr)) >= 0.9)


def test1e(scans: pd.DataFrame, db_stats: dict[str, Any]) -> None:
    print("\n1e  Which columns do the training index scans use? (index-scan filter examples)")
    print("    unique: distinct values cover >= 99% of the rows (primary keys and other unique columns);")
    print("    clustered: |correlation| >= 0.9 between column order and physical row order. Both make an index")
    print("    scan cheap even at higher selectivity; stats_found = share of examples whose column is in pg_stats.")
    d = scans[scans.filter_scan & (scans.kind == "index")]
    rows, allx = [], []
    for f, g in d.groupby("file", observed=True):
        sizes, cs = db_stats.get(f, ({}, {}))
        t, c = g.rel.astype(str).str.lower().values, g.col.astype(str).str.lower().values
        st = [cs.get(k, (None, None)) for k in zip(t, c)]
        x = pd.DataFrame({"col": [f"{a}.{b}" for a, b in zip(t, c)], "found": [s_[0] is not None for s_ in st],
                          "unique": [unique_col(s_[0], sizes.get(a)) for s_, a in zip(st, g.rel.astype(str))],
                          "clustered": [clustered_col(s_[1]) for s_ in st]})
        x["either"] = np.fmax(x.unique, x.clustered)
        allx.append(x)
        rows.append(summary_1e(f, x))
    if len(allx) > 1:
        rows.append(summary_1e("ALL FILES", pd.concat(allx)))
    if not rows:
        print("    no index-scan filter examples")
        return
    print(pd.DataFrame(rows).round(1).to_string(index=False))

    print("\n    Evaluation columns of the access-path task (Table 3), with their training index-scan examples:")
    ev = []
    for f, (sizes, cs) in db_stats.items():
        db = f.split(os.sep)[0]
        if db not in SCAN_COLUMNS:
            continue
        g = d[d.file.astype(str) == f]
        used = Counter(f"{a}.{b}" for a, b in zip(g.rel.astype(str).str.lower(), g.col.astype(str).str.lower()))
        for col in SCAN_COLUMNS[db]:
            t, c = col.lower().split(".")
            nd, corr = cs.get((t, c), (None, None))
            size = next((v for k, v in sizes.items() if k.lower() == t), None)
            ev.append({"file": f, "column": col, "n_distinct": nd, "unique": unique_col(nd, size),
                           "correlation": None if corr is None else round(float(corr), 3),
                           "clustered": clustered_col(corr), "train_index_examples": used.get(col.lower(), 0)})
    print(pd.DataFrame(ev).to_string(index=False) if ev else
          "    (none of the given training files is from IMDB, Baseball or TPC-H)")


def summary_1e(f: str, x: pd.DataFrame) -> dict[str, Any]:
    n = len(x)
    top = ", ".join(f"{k} ({v})" for k, v in Counter(x.col).most_common(3))
    return dict(file=f, n_index=n, n_columns=x.col.nunique(), **{"stats_found_%": 100 * x.found.mean(),
                "unique_%": 100 * x.unique.mean(), "clustered_%": 100 * x.clustered.mean(),
                "unique_or_clustered_%": 100 * x.either.mean()}, top_columns=top)


def test1d(queries: pd.DataFrame) -> None:
    print("\n1d  Timeouts: are timed-out training queries different? (operators from EXPLAIN without ANALYZE)")
    rows = []
    for f, g in list(queries.groupby("file", observed=True)) + ([("ALL FILES", queries)] if queries.file.nunique() > 1 else []):
        to = g[g.timeout]
        r = {"file": f, "queries": len(g), "timeouts": len(to), "timeouts_with_plan": int(to.has_verbose.sum())}
        g = g[g.has_verbose]
        for feat in ["has_nl", "has_index"]:
            x = g[feat].astype(bool)
            a, b = int((x & g.timeout).sum()), int((~x & g.timeout).sum())
            c, d_ = int((x & ~g.timeout).sum()), int((~x & ~g.timeout).sum())
            r[f"{feat}_timeout_%"] = 100 * a / (a + b) if a + b else np.nan
            r[f"{feat}_completed_%"] = 100 * c / (c + d_) if c + d_ else np.nan
            r[f"{feat}_fisher_p"] = fisher_exact([[a, b], [c, d_]])[1] if a + b and c + d_ else np.nan
        rows.append(r)
    df = pd.DataFrame(rows)
    print(df.round(4).to_string(index=False))
    if (df.timeouts == 0).all():
        print("    No timed-out queries are stored in these files, so 1d cannot be tested with them.")
    elif (df.timeouts_with_plan == 0).all():
        print("    Timed-out queries have no stored plan, so 1d cannot be tested with them.")


# ----------------------------------------------------------------------------- reference (retraining) data
def read_reference(index_path: str, seq_path: str, max_queries: int | None) -> pd.DataFrame:
    def collect(path: str, want: str) -> tuple[dict[str, dict[str, Any]], Counter]:
        it, dbstats = load_run(path, max_queries)
        sizes = table_sizes(dbstats)
        out, stats = {}, Counter()
        for q in it:
            stats["queries"] += 1
            key = normalize_sql(q.get("sql"))
            if q.get("timeout"):
                stats["timeout"] += 1
                out[key] = {"timeout": True, "sql": q.get("sql")}
                continue
            ap = first_executed_plan(q)
            recs = scan_records(ap, sizes) if ap else []
            if not recs:
                stats["no plan"] += 1
                continue
            r = recs[0]
            if r["kind"] != want:
                stats[f"hint not applied ({r['kind']} scan)"] += 1
                continue
            out[key] = dict(timeout=False, t=mean_exec_ms(q), sql=q.get("sql"), **r)
        return out, stats

    idx, si = collect(index_path, "index")
    seq, ss = collect(seq_path, "seq")
    print(f"  index-scan file: {dict(si)}")
    print(f"  seq-scan file:   {dict(ss)}")
    keys = set(idx) & set(seq)
    rows, dropped = [], Counter()
    for k in keys:
        a, b = idx[k], seq[k]
        if a["timeout"] or b["timeout"]:
            dropped["a plan timed out"] += 1
            continue
        if not (np.isfinite(a["t"]) and np.isfinite(b["t"]) and np.isfinite(a["s_cond"])):
            dropped["missing runtime or table size"] += 1
            continue
        rows.append({"rel": a["rel"], "col": a["col"], "s": a["s_cond"], "t_idx": a["t"], "t_seq": b["t"],
                         "idx_faster": a["t"] < b["t"], "sql": a["sql"]})
    print(f"  paired queries (same SQL in both files): {len(keys)}; usable: {len(rows)}; dropped: {dict(dropped)}")
    return pd.DataFrame(rows)


def crossovers(ref: pd.DataFrame, min_n: int) -> dict[tuple[Any, Any], dict[str, Any]]:
    """Crossover selectivity per (table, column), per table and pooled; the finest level with min_n queries."""
    out = {}
    tau, acc = threshold(ref.s.values, ref.idx_faster.values)
    out[(None, None)] = {"tau": tau, "acc": acc, "n": len(ref)}
    for t, g in ref.groupby("rel"):
        if len(g) >= min_n:
            tau, acc = threshold(g.s.values, g.idx_faster.values)
            out[(t, None)] = {"tau": tau, "acc": acc, "n": len(g)}
    for (t, c), g in ref.groupby(["rel", "col"]):
        if len(g) >= min_n:
            tau, acc = threshold(g.s.values, g.idx_faster.values)
            out[(t, c)] = {"tau": tau, "acc": acc, "n": len(g)}
    return out


def lookup(cx: dict[tuple[Any, Any], dict[str, Any]], rel: Any, col: Any) -> tuple[float, str | None]:
    for key, level in [((rel, col), "column"), ((rel, None), "table"), ((None, None), "pooled")]:
        if key in cx:
            return cx[key]["tau"], level
    return np.nan, None


def test1c(train_scans: pd.DataFrame, ref: pd.DataFrame, min_n: int) -> None:
    print("\n1c  Missing negative examples: index-scan examples in the region where an index scan is slower")
    cx = crossovers(ref, min_n)
    print(f"\n  Crossover selectivity s* (index faster iff s <= s*), from the retraining queries "
          f"(levels with at least {min_n} queries):")
    t = pd.DataFrame([{"table": k[0] or "(all tables)", "column": k[1] or "(all columns)", "n": v["n"],
                           "s_star": v["tau"], "rule_accuracy": v["acc"]} for k, v in cx.items()])
    print(t.sort_values(["table", "column"]).round(4).to_string(index=False))

    ref = ref.copy()
    ref[["tau", "level"]] = ref.apply(lambda r: pd.Series(lookup(cx, r.rel, r.col)), axis=1)
    ref["harmful"] = ref.s > ref.tau
    agree = (ref.harmful == ~ref.idx_faster).mean()
    print(f"\n  Check of the rule on the retraining queries (in-sample, so optimistic): 's > s*' agrees with "
          f"'index scan slower' for {100 * agree:.1f}% of the queries.")

    tables = sorted({k[0] for k in cx if k[0] is not None})
    tr = train_scans[train_scans.filter_scan & (train_scans.kind == "index") & train_scans.s_cond.notna()].copy()
    n_all = len(tr)
    tr = tr[tr.rel.isin(tables)]
    print(f"  Training index-scan filter examples: {n_all}; on tables covered by the retraining queries: {len(tr)}")
    if tr.empty:
        print("  No training index-scan examples on these tables: 1c cannot be computed.")
        return
    tr[["tau", "level"]] = tr.apply(lambda r: pd.Series(lookup(cx, r.rel, r.col)), axis=1)
    tr["harmful"] = tr.s_cond > tr.tau
    print("  crossover level used for the training examples:", dict(Counter(tr.level)))

    rows = []
    for name, d in [("all covered tables", tr), ("column-level crossover only", tr[tr.level == "column"])]:
        k, n = int(d.harmful.sum()), len(d)
        r_ = ref[ref.rel.isin(d.rel.unique())]
        kr, nr = int(r_.harmful.sum()), len(r_)
        lo, hi = (beta.ppf(0.025, k, n - k + 1) if k else 0.0), (beta.ppf(0.975, k + 1, n - k) if k < n else 1.0)
        rows.append({"scope": name, "train_n": n, "train_harmful": k, "train_share": k / n if n else np.nan,
                         "train_ci95": f"[{lo:.4f}, {hi:.4f}]", "train_upper95": cp_upper(k, n),
                         "ref_n": nr, "ref_harmful": kr, "ref_share": kr / nr if nr else np.nan,
                         "ref_share_true": (~r_.idx_faster).mean() if nr else np.nan,
                         "fisher_p_less": fisher_exact([[k, n - k], [kr, nr - kr]], alternative="less")[1]
                         if n and nr else np.nan})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    br = tr.assign(col=tr.col.astype(str), rel=tr.rel.astype(str)).groupby(["rel", "col"]).agg(
        n=("harmful", "size"), harmful=("harmful", "sum"), crossover_level=("level", "first"),
        s_star=("tau", "first"), median_s=("s_cond", "median")).reset_index()
    print("\n  Where the training index-scan examples are (by table and indexed column):")
    print(br.sort_values("n", ascending=False).round(4).to_string(index=False))
    print("  train_share: share of the training index-scan examples above s* (the 'negative examples');\n"
          "  train_upper95: one-sided 95% Clopper-Pearson upper bound on that share;\n"
          "  ref_share: same share in the retraining queries (index plans forced at all selectivities);\n"
          "  ref_share_true: share of retraining queries where the index scan was actually slower;\n"
          "  fisher_p_less: H0 the training share is not smaller than the retraining share.")


def leakage(ref: pd.DataFrame) -> None:
    print("\nLeakage check for Test 3a: retraining queries that filter on the IMDB evaluation columns of Fig. 14")
    ev = {tuple(c.split(".")) for c in SCAN_COLUMNS["imdb"]}
    hits = Counter()
    for sql in ref.sql:
        cols = {(t, c) for t, c in SQL_COL_RE.findall(sql or "")}
        for x in cols & ev:
            hits[".".join(x)] += 1
    tables = Counter(ref.rel)
    for c in SCAN_COLUMNS["imdb"]:
        t = c.split(".")[0]
        print(f"  {c:28s} queries filtering on it: {hits.get(c, 0):4d}   on its table: {tables.get(t, 0):4d}")
    print(f"  ({len(ref)} usable retraining queries in total)")


def run(a: argparse.Namespace) -> None:
    section("TEST 1: IS THE TRAINING DATA BIASED?")
    if ijson is None:
        print("note: ijson is not installed, so every file is loaded into memory at once (pip install ijson)")
    train_paths = (a.imdb_train_runs or []) + (a.train_runs or [])
    scans = queries = None
    if train_paths:
        print("\nReading training run files:")
        scans, queries, db_stats = read_training(train_paths, a.max_queries)
        if scans.empty:
            print("  no executed scan nodes found: check that these are run files with EXPLAIN ANALYZE output")
        else:
            test1a(scans)
            test1b(scans)
            test1e(scans, db_stats)
        test1d(queries)
    else:
        print("no training run files given: 1a, 1b, 1d skipped")

    if a.ref_index_runs and a.ref_seq_runs:
        print("\nReading the retraining (reference) run files:")
        ref = read_reference(a.ref_index_runs, a.ref_seq_runs, a.max_queries)
        if ref.empty:
            print("  no usable query pairs: 1c and the leakage check skipped")
            return
        leakage(ref)
        if a.imdb_train_runs and scans is not None and not scans.empty:
            imdb = {file_label(p) for p in a.imdb_train_runs}
            test1c(scans[scans.file.astype(str).isin(imdb)].copy(), ref, a.min_n)
        else:
            print("\n1c skipped: needs --imdb_train_runs (the retraining queries run on IMDB)")
    else:
        print("\n1c and the leakage check skipped: need --ref_index_runs and --ref_seq_runs")
