"""Tests 2 and 3a of the training-data-bias claim in "How Good are Learned Cost Models, Really?"
(Heinrich et al., SIGMOD 2025; Sections 5.5, 5.7 and recommendation R3).

Uses only the authors' published predictions for the access-path workload (scan_costs_percentiles),
the same files sigtests.py reads. No model is trained or run.

Test 2 (do the models show the bias?), 21 columns of Table 3, all LCMs, PG10 and PG16 for reference
  2a  selectivity sensitivity: within-column slope of "picks the index scan" on selectivity,
      compared with the slope of the optimal choice (cluster bootstrap over columns)
  2b  over-preference: does the model pick index scans more often than the optimum does?
      (exact McNemar test of model choice vs optimal choice, plus a cluster bootstrap check)
  2c  on the points where a sequential scan is faster: does the LCM wrongly pick the index scan
      more often than PG10? (exact McNemar test, plus a cluster bootstrap check)
Test 3a (does fine-tuning on diversified data help?), the 6 IMDB columns of Fig. 14, 5 fine-tuned LCMs
  per point: wrong->right vs right->wrong after fine-tuning (exact McNemar), on all points and on the
  points where a sequential scan is faster; per column and per seed breakdowns.

A "point" is one (column, selectivity) pair: query k of a column filters "col >= the (10k)-th percentile",
so it selects about s = 1 - k/10 of the table. Predictions are averaged over the seeds first, as the
authors do; the true runtimes are Flat Vector's labels for every model (as in sigtests.py).

Usage (from the MDA folder):
  python scripts/bias_models.py --eval_root lcm-eval/data/evaluation --out results/bias_models.txt
      [--retrain_root lcm-eval/data/retraining_evaluation] [--tests 2 3a]
"""
import argparse
import os

from lcm_recheck.config import BASE_SEED, B, default_retrain_root
from lcm_recheck.report import tee_output
from lcm_recheck.tasks.bias_models import test2, test3a


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--retrain_root", help="fine-tuned predictions (default: retraining_evaluation next to eval_root)")
    ap.add_argument("--tests", nargs="+", choices=["2", "3a"], default=["2", "3a"])
    ap.add_argument("--out", help="also write the output to this file (overwritten)")
    a = ap.parse_args()
    a.eval_root = os.path.expanduser(a.eval_root)
    a.retrain_root = os.path.expanduser(a.retrain_root or default_retrain_root(a.eval_root))
    with tee_output(a.out, header_extra=f"; bootstrap seed {BASE_SEED}, {B} resamples",
                    closing=f"Output written to {a.out}"):
        if "2" in a.tests:
            test2(a)
        if "3a" in a.tests:
            test3a(a)


if __name__ == "__main__":
    main()

