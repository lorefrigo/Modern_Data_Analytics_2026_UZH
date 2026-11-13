"""Test 1 of the training-data-bias claim in "How Good are Learned Cost Models, Really?"
(Heinrich et al., SIGMOD 2025; Section 5.5 and recommendation R3): is the training data itself biased?

Reads lcm-eval *run files*: a JSON with a 'query_list' (sql, verbose_plan, analyze_plans, timeout per query)
and 'database_stats', in raw or JSON EXPLAIN mode. parsed_plans files are not accepted.

  1a  composition: share of sequential, index and bitmap scans in the training plans (paper: about 90% / 10%)
  1b  where do index and sequential scans occur? selectivity distributions (two-sample KS test)
  1c  missing negative examples (the core test): among index-scan examples, the share in the region where an
      index scan is slower than a sequential scan, training data vs the authors' randomized retraining queries
      (Clopper-Pearson bound, one-sided Fisher's exact test). The region is estimated from the retraining
      queries, where both plans were executed for every query.
  1d  timeouts: do timed-out training queries more often contain nested-loop joins or index scans? (Fisher)
  1e  which columns do the training index scans use? unique and physically clustered columns (pg_stats),
      compared with the columns of the access-path evaluation (Table 3)
  plus a leakage check for Test 3a: do the retraining queries filter on the 6 IMDB evaluation columns?

Usage (from the MDA folder):
  python scripts/bias_data.py \\
      --imdb_train_runs lcm-eval/data/runs/raw/imdb/<training run file>.json \\
      --ref_index_runs  lcm-eval/data/runs/raw/imdb/<index_retraining run file>.json \\
      --ref_seq_runs    lcm-eval/data/runs/raw/imdb/<seq_retraining run file>.json \\
      [--train_runs <run files of other databases> ...] [--max_queries 2000] [--out results/bias_data.txt]

Install ijson (pip install ijson) to read large training files without loading them into memory.
"""
import argparse, datetime, json, os, platform, re, sys
from collections import Counter
import numpy as np, pandas as pd
import scipy
from scipy.stats import beta, fisher_exact, ks_2samp, norm

from lcm_recheck.config import SCAN_COLUMNS
from extract_runs import lines, exec_ms
from sigtests import section, Tee

try:
    import ijson
except ImportError:
    ijson = None

SCAN_KIND = {"Seq Scan": "seq", "Index Scan": "index", "Index Only Scan": "index", "Bitmap Heap Scan": "bitmap"}
NODE_RE = re.compile(r"^(?P<op>[A-Z][A-Za-z ]*?)(?: Backward)?(?: using (?P<idx>\S+))?"
                     r"(?: on (?P<rel>\S+)(?: (?P<alias>\S+))?)?\s+\(cost=")
ACT_RE = re.compile(r"\(actual time=[\d.]+\.\.[\d.]+ rows=(\d+) loops=(\d+)\)")
PARAM_RE = re.compile(r"(?:=|<|>)\s*\(?\s*(?:\$\d|\"?[A-Za-z_]\w*\"?\.\"?[A-Za-z_])")   # col = other.col / $1
COND_COL_RE = re.compile(r"\(*\s*(?:\"?\w+\"?\.)?\"?(\w+)\"?\)?(?:::[\w ]+?)?\s*(?:>=|<=|<>|!=|=|<|>|~~)")
SQL_COL_RE = re.compile(r"\"?(\w+)\"?\.\"?(\w+)\"?\s*(?:>=|<=|<>|!=|=|<|>|\bNOT LIKE\b|\bLIKE\b|\bIS\b|\bIN\b)", re.I)
HINT_RE = re.compile(r"/\*\+.*?\*/", re.S)


# ----------------------------------------------------------------------------- reading run files
def load_run(path, max_queries=None):
    """Returns (iterator over query entries, database_stats)."""
    with open(path, "rb") as f:
        head = f.read(4096).decode("utf-8", "ignore")
    if '"parsed_plans"' in head and '"query_list"' not in head:
        raise SystemExit(f"{path} is a parsed_plans file. Use the run file of the same workload from runs/raw "
                         f"(or runs/json) instead.")
    if ijson is not None:
        with open(path, "rb") as f:
            stats = next(ijson.items(f, "database_stats", use_float=True), None)

        def gen():
            with open(path, "rb") as f:
                for i, q in enumerate(ijson.items(f, "query_list.item", use_float=True)):
                    if max_queries and i >= max_queries:
                        break
                    yield q
        return gen(), stats
    run = json.load(open(path))
    if "query_list" not in run:
        raise SystemExit(f"{path} is not a run file (no 'query_list')")
    ql = run["query_list"][:max_queries] if max_queries else run["query_list"]
    return iter(ql), run.get("database_stats")


def table_sizes(stats):
    if not stats:
        return {}
    return {t["relname"]: float(t["reltuples"]) for t in stats.get("table_stats", []) if t.get("reltuples")}


def column_stats(stats):
    """{(table, column), lower case: (n_distinct, correlation)} from pg_stats."""
    if not stats:
        return {}
    return {(str(c["tablename"]).lower(), str(c["attname"]).lower()): (c.get("n_distinct"), c.get("correlation"))
            for c in stats.get("column_stats", []) if c.get("tablename") and c.get("attname")}


def clean_rel(rel):
    return rel.split(".")[-1].strip('"') if rel else rel


def plan_nodes(plan):
    """All operator nodes of one EXPLAIN output (raw text rows or JSON), with the fields used here."""
    if isinstance(plan, list) and plan and isinstance(plan[0], dict):
        plan = plan[0]
    if isinstance(plan, dict):
        out = []

        def walk(n):
            op, par = n.get("Node Type", ""), bool(n.get("Parallel Aware"))
            out.append(dict(op=op, kind=SCAN_KIND.get(op), parallel=par, rel=clean_rel(n.get("Relation Name")),
                            rows=n.get("Actual Rows"), loops=n.get("Actual Loops"),
                            removed=n.get("Rows Removed by Filter", 0) or 0,
                            cond=n.get("Index Cond") or n.get("Recheck Cond")))
            for c in n.get("Plans", []):
                walk(c)
        walk(plan.get("Plan", plan))
        return out
    out, cur = [], None
    for l in lines(plan):
        t = l.strip()
        if t.startswith("->"):
            t = t[2:].strip()
        if "(cost=" in t:
            m, act = NODE_RE.match(t), ACT_RE.search(t)
            op = m.group("op").strip() if m else ""
            par = op.startswith("Parallel ")
            op = op[len("Parallel "):] if par else op
            cur = dict(op=op, kind=SCAN_KIND.get(op), parallel=par, rel=clean_rel(m.group("rel") if m else None),
                       rows=int(act.group(1)) if act else None, loops=int(act.group(2)) if act else None,
                       removed=0, cond=None)
            out.append(cur)
        elif cur is not None:
            if t.startswith("Rows Removed by Filter:"):
                cur["removed"] = int(t.split(":")[1])
            elif t.startswith(("Index Cond:", "Recheck Cond:")):
                cur["cond"] = t.split(":", 1)[1].strip()
    return out


def scan_records(plan, sizes):
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
        recs.append(dict(kind=n["kind"], parallel=n["parallel"], rel=n["rel"], filter_scan=filt,
                         s_out=s_out, s_cond=s_cond if n["kind"] != "seq" else s_out,
                         col=m.group(1) if m else None))
    return recs


def first_executed_plan(q):
    aps = q.get("analyze_plans") or []
    return aps[0] if aps else None


def mean_exec_ms(q):
    t = [exec_ms(p) for p in (q.get("analyze_plans") or [])]
    t = [x for x in t if np.isfinite(x)]
    return float(np.mean(t)) if t else np.nan


def normalize_sql(sql):
    return re.sub(r"\s+", " ", HINT_RE.sub("", sql or "")).strip().rstrip(";").strip().lower()


# ----------------------------------------------------------------------------- statistics helpers
def wilson(k, n, conf=0.95):
    if n == 0:
        return np.nan, np.nan
    z, p = norm.ppf(1 - (1 - conf) / 2), k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def cp_upper(k, n, conf=0.95):
    """One-sided Clopper-Pearson upper bound: the largest rate still compatible with seeing only k of n."""
    return 1.0 if k >= n else float(beta.ppf(conf, k + 1, n - k))


def threshold(s, y):
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
def file_label(path):
    """'<database folder>/<file name>': the training files of all databases share the same file name."""
    return os.path.join(os.path.basename(os.path.dirname(os.path.abspath(path))), os.path.basename(path))


def read_training(paths, max_queries):
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
            queries.append(dict(file=name, q=i, timeout=to, has_verbose=ops is not None,
                                has_nl=("Nested Loop" in ops) if ops else None,
                                has_index=any(SCAN_KIND.get(o) == "index" for o in ops) if ops else None))
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


def compact(df):
    for c in ["file", "kind", "rel", "col"]:
        if c in df:
            df[c] = df[c].astype("category")
    for c in ["s_out", "s_cond"]:
        if c in df:
            df[c] = df[c].astype("float32")
    return df


def test1a(scans):
    print("\n1a  Composition of the scans in the training plans")
    print("    all = every executed scan node; filter = scans that filter a base table once (see 1b/1c).")
    rows = []
    for (f, sub), d in [((f, "all"), g) for f, g in scans.groupby("file", observed=True)] + \
                       [((f, "filter"), g[g.filter_scan]) for f, g in scans.groupby("file", observed=True)] + \
                       ([(("ALL FILES", "all"), scans), (("ALL FILES", "filter"), scans[scans.filter_scan])]
                        if scans.file.nunique() > 1 else []):
        n = len(d)
        r = dict(file=f, scans=sub, n=n)
        for k in ["seq", "index", "bitmap"]:
            c = int((d.kind == k).sum())
            lo, hi = wilson(c, n)
            r[f"{k}_%"] = 100 * c / n if n else np.nan
            r[f"{k}_ci95"] = f"[{100 * lo:.1f}, {100 * hi:.1f}]"
        rows.append(r)
    print(pd.DataFrame(rows).round(1).to_string(index=False))


def test1b(scans):
    print("\n1b  Selectivity (rows returned / table rows) of filter scans: index vs sequential")
    print("    H0: both come from the same distribution (two-sample Kolmogorov-Smirnov).")
    d = scans[scans.filter_scan & scans.s_out.notna()]
    rows = []
    for f, g in list(d.groupby("file", observed=True)) + ([("ALL FILES", d)] if d.file.nunique() > 1 else []):
        a, b = g.loc[g.kind == "index", "s_out"].values, g.loc[g.kind == "seq", "s_out"].values
        r = dict(file=f, n_index=len(a), n_seq=len(b))
        for name, v in [("index", a), ("seq", b)]:
            q = np.percentile(v, [10, 50, 90]) if len(v) else [np.nan] * 3
            r[f"{name}_p10/p50/p90"] = "/".join(f"{x:.4f}" for x in q)
        r["KS_D"], r["KS_p"] = ks_2samp(a, b) if len(a) and len(b) else (np.nan, np.nan)
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))


def unique_col(nd, rows):
    """n_distinct < 0 is minus the distinct fraction of rows; > 0 is a count. Unique = covers >= 99% of rows."""
    if nd is None:
        return np.nan
    nd = float(nd)
    if nd < 0:
        return float(nd <= -0.99)
    return float(bool(rows) and nd >= 0.99 * rows)


def clustered_col(corr):
    return np.nan if corr is None else float(abs(float(corr)) >= 0.9)


def test1e(scans, db_stats):
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
    for f in db_stats:
        db = f.split(os.sep)[0]
        if db not in SCAN_COLUMNS:
            continue
        sizes, cs = db_stats[f]
        g = d[d.file.astype(str) == f]
        used = Counter(f"{a}.{b}" for a, b in zip(g.rel.astype(str).str.lower(), g.col.astype(str).str.lower()))
        for col in SCAN_COLUMNS[db]:
            t, c = col.lower().split(".")
            nd, corr = cs.get((t, c), (None, None))
            size = next((v for k, v in sizes.items() if k.lower() == t), None)
            ev.append(dict(file=f, column=col, n_distinct=nd, unique=unique_col(nd, size),
                           correlation=None if corr is None else round(float(corr), 3),
                           clustered=clustered_col(corr), train_index_examples=used.get(col.lower(), 0)))
    print(pd.DataFrame(ev).to_string(index=False) if ev else
          "    (none of the given training files is from IMDB, Baseball or TPC-H)")


def summary_1e(f, x):
    n = len(x)
    top = ", ".join(f"{k} ({v})" for k, v in Counter(x.col).most_common(3))
    return dict(file=f, n_index=n, n_columns=x.col.nunique(), **{"stats_found_%": 100 * x.found.mean(),
                "unique_%": 100 * x.unique.mean(), "clustered_%": 100 * x.clustered.mean(),
                "unique_or_clustered_%": 100 * x.either.mean()}, top_columns=top)


def test1d(queries):
    print("\n1d  Timeouts: are timed-out training queries different? (operators from EXPLAIN without ANALYZE)")
    rows = []
    for f, g in list(queries.groupby("file", observed=True)) + ([("ALL FILES", queries)] if queries.file.nunique() > 1 else []):
        to = g[g.timeout]
        r = dict(file=f, queries=len(g), timeouts=len(to), timeouts_with_plan=int(to.has_verbose.sum()))
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
def read_reference(index_path, seq_path, max_queries):
    def collect(path, want):
        it, dbstats = load_run(path, max_queries)
        sizes = table_sizes(dbstats)
        out, stats = {}, Counter()
        for q in it:
            stats["queries"] += 1
            key = normalize_sql(q.get("sql"))
            if q.get("timeout"):
                stats["timeout"] += 1
                out[key] = dict(timeout=True, sql=q.get("sql"))
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
        rows.append(dict(rel=a["rel"], col=a["col"], s=a["s_cond"], t_idx=a["t"], t_seq=b["t"],
                         idx_faster=a["t"] < b["t"], sql=a["sql"]))
    print(f"  paired queries (same SQL in both files): {len(keys)}; usable: {len(rows)}; dropped: {dict(dropped)}")
    return pd.DataFrame(rows)


def crossovers(ref, min_n):
    """Crossover selectivity per (table, column), per table and pooled; the finest level with min_n queries."""
    out = {}
    tau, acc = threshold(ref.s.values, ref.idx_faster.values)
    out[(None, None)] = dict(tau=tau, acc=acc, n=len(ref))
    for t, g in ref.groupby("rel"):
        if len(g) >= min_n:
            tau, acc = threshold(g.s.values, g.idx_faster.values)
            out[(t, None)] = dict(tau=tau, acc=acc, n=len(g))
    for (t, c), g in ref.groupby(["rel", "col"]):
        if len(g) >= min_n:
            tau, acc = threshold(g.s.values, g.idx_faster.values)
            out[(t, c)] = dict(tau=tau, acc=acc, n=len(g))
    return out


def lookup(cx, rel, col):
    for key, level in [((rel, col), "column"), ((rel, None), "table"), ((None, None), "pooled")]:
        if key in cx:
            return cx[key]["tau"], level
    return np.nan, None


def test1c(train_scans, ref, min_n):
    print("\n1c  Missing negative examples: index-scan examples in the region where an index scan is slower")
    cx = crossovers(ref, min_n)
    print(f"\n  Crossover selectivity s* (index faster iff s <= s*), from the retraining queries "
          f"(levels with at least {min_n} queries):")
    t = pd.DataFrame([dict(table=k[0] or "(all tables)", column=k[1] or "(all columns)", n=v["n"],
                           s_star=v["tau"], rule_accuracy=v["acc"]) for k, v in cx.items()])
    print(t.sort_values(["table", "column"]).round(4).to_string(index=False))

    ref = ref.copy()
    ref[["tau", "level"]] = ref.apply(lambda r: pd.Series(lookup(cx, r.rel, r.col)), axis=1)
    ref["harmful"] = ref.s > ref.tau
    agree = (ref.harmful == ~ref.idx_faster).mean()
    print(f"\n  Check of the rule on the retraining queries (in-sample, so optimistic): 's > s*' agrees with "
          f"'index scan slower' for {100 * agree:.1f}% of the queries.")

    tables = sorted(set(k[0] for k in cx if k[0] is not None))
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
        rows.append(dict(scope=name, train_n=n, train_harmful=k, train_share=k / n if n else np.nan,
                         train_ci95=f"[{lo:.4f}, {hi:.4f}]", train_upper95=cp_upper(k, n),
                         ref_n=nr, ref_harmful=kr, ref_share=kr / nr if nr else np.nan,
                         ref_share_true=(~r_.idx_faster).mean() if nr else np.nan,
                         fisher_p_less=fisher_exact([[k, n - k], [kr, nr - kr]], alternative="less")[1]
                         if n and nr else np.nan))
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


def leakage(ref):
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


# ----------------------------------------------------------------------------- main
def main(a):
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--imdb_train_runs", nargs="+", help="training run files on IMDB (used for 1a-1d)")
    ap.add_argument("--train_runs", nargs="+", help="training run files of other databases (1a, 1b, 1d only)")
    ap.add_argument("--ref_index_runs", help="retraining run file with the forced index scans")
    ap.add_argument("--ref_seq_runs", help="retraining run file with the forced sequential scans")
    ap.add_argument("--min_n", type=int, default=20, help="queries needed for a table/column-level crossover")
    ap.add_argument("--max_queries", type=int, help="read only the first N queries of each file (quick check)")
    ap.add_argument("--out", help="also write the output to this file (overwritten)")
    a = ap.parse_args()
    out_file = None
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        out_file = open(a.out, "w", encoding="utf-8")
        sys.stdout = Tee(sys.__stdout__, out_file)
        print(f"# {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"# command: python {' '.join(sys.argv)}")
        print(f"# python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}, "
              f"scipy {scipy.__version__}, ijson {'yes' if ijson else 'no'}\n")
    try:
        main(a)
    finally:
        if out_file:
            sys.stdout = sys.__stdout__
            out_file.close()
            print(f"Output written to {a.out}")