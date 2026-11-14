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
import argparse

from lcm_recheck.report import tee_output
from lcm_recheck.runs import ijson
from lcm_recheck.tasks.bias_data import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imdb_train_runs", nargs="+", help="training run files on IMDB (used for 1a-1d)")
    ap.add_argument("--train_runs", nargs="+", help="training run files of other databases (1a, 1b, 1d only)")
    ap.add_argument("--ref_index_runs", help="retraining run file with the forced index scans")
    ap.add_argument("--ref_seq_runs", help="retraining run file with the forced sequential scans")
    ap.add_argument("--min_n", type=int, default=20, help="queries needed for a table/column-level crossover")
    ap.add_argument("--max_queries", type=int, help="read only the first N queries of each file (quick check)")
    ap.add_argument("--out", help="also write the output to this file (overwritten)")
    a = ap.parse_args()
    with tee_output(a.out, header_extra=f", ijson {'yes' if ijson else 'no'}", closing=f"Output written to {a.out}"):
        run(a)


if __name__ == "__main__":
    main()
