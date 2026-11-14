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
import argparse
import os

from lcm_recheck.config import BASE_SEED, default_retrain_root
from lcm_recheck.report import tee_output
from lcm_recheck.tasks.access import task_access
from lcm_recheck.tasks.common import summary
from lcm_recheck.tasks.join import task_join
from lcm_recheck.tasks.physical import task_physical
from lcm_recheck.tasks.versions import versions


def paper(a):
    tasks = {"join": task_join, "access": task_access, "physical": task_physical}
    for t in a.tasks:
        tasks[t](a)
    summary()
    print("\nVerdicts: runtime and balanced accuracy use the bootstrap interval at level 1 - 0.05/k "
          "(k = comparisons in the family; none below 10 units); pick rate uses the Holm-adjusted exact McNemar test.")


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
