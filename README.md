# Learned cost models, re-checked

A re-analysis of Heinrich et al., [*How Good are Learned Cost Models, Really? Insights from Query
Optimization Tasks*](https://doi.org/10.1145/3725309) (SIGMOD 2025), built on the authors' published code
and data. Seminar project for *Modern Data Analytics*, University of Zurich, autumn 2026.

The repository contains:
- significance tests of the paper's main claim on the authors' predictions;
- E1, PostgreSQL 18's cost model on the paper's join-order benchmark;
- E2, a design for executing the plans on two PostgreSQL versions (not run);
- tests of the training-data bias the paper uses to explain its access-path results.

## Key findings

- **The main claim holds outside one setting.** The claim is that no learned cost model (LCM) significantly
  beats PostgreSQL. 9 of 77 LCM-vs-PostgreSQL 10 comparisons show an LCM significantly better after
  family-wise correction, all in join-operator selection (Task 3): 8 with additional indexes (Section 6.6)
  and 1 pick rate (DACE on IMDB, Fig. 11). None is in join ordering or access-path selection.
- **The pinned join-order plans contain a hint bug.** The authors' pinned candidate-plan SQL for JOB-light
  writes `NestedLoop(`, but pg_hint_plan only knows `NestLoop(`. pg_hint_plan reports a syntax error and
  ignores every hint after it, including `Leading(...)`. This affects 771 of 7,863 SQL lines, in 20 of the
  58 queries.
- **With the hints fixed, PostgreSQL 18 behaves like the authors' PostgreSQL 16.** No hint errors remain.
  - PG18 picks a plan with the same PG10 runtime as the authors' Sc. PG16 in all 58 queries. That totals
    562.9 s, against 517.9 s for Sc. PG10 and 445.7 s for the fastest candidates. PG10's choices remain
    significantly better.
  - On the unfixed SQL, PG18's choices total 630.2 s.
- **The access-path training data is biased, and so are the models.**
  - In the IMDB training data, index scans are 12.5% of the filter scans. Their median selectivity is 0.047,
    against 0.356 for sequential scans. All 2,811 index-scan examples use unique or clustered columns, and
    none uses the six IMDB columns evaluated in Table 3.
  - On the 142 of 230 test points where a sequential scan is faster, PG10 never picks the index scan. The
    seven LCMs pick it on 10.6% (DACE) to 83.8% (Zero-Shot) of those points. For all seven the difference
    from PG10 is significant (exact McNemar test, Holm-corrected). For six of them it also holds when the
    bootstrap resamples whole columns.

The full outputs are in `results/sigtests_paper.txt`, `results/bias_data_imdb.txt` and `results/bias_models.txt`.

## Repository layout

| Path | Contents |
|---|---|
| `scripts/` | Command-line scripts (`uv run python scripts/<name>.py --help`) |
| `src/lcm_recheck/` | The package: configuration, run-file parsing, statistics, the tests in `tasks/` |
| `results/` | Committed outputs: test results, extracted runtimes and costs, generated prediction files |
| `tests/` | Unit tests on small fixtures (no real data, no database needed) |
| `tools/` | Maintainer tools for regression checks |

## Quick start (no database)

You need [uv](https://docs.astral.sh/uv/) and git.

```bash
git clone https://github.com/lorefrigo/Modern_Data_Analytics_2026_UZH.git MDA && cd MDA
uv sync
git clone https://github.com/DataManagementLab/lcm-eval.git lcm-eval
git -C lcm-eval checkout e024859de0d0b10f686e7da69759656d2a743057
uv run python scripts/osf_fetch.py --match evaluation.zip   # the authors' predictions, 40.2 MB, into lcm-eval/data/
uv run python scripts/check_downloads.py                     # which prediction files are present
```

Then rerun the significance tests (about 1 minute) and bias Tests 2 and 3a (a few seconds). Compare the
results with the committed files; only the header lines starting with `#` should differ:

```bash
uv run python scripts/sigtests.py paper --eval_root lcm-eval/data/evaluation --out results/check/sigtests_paper.txt
uv run python scripts/bias_models.py --eval_root lcm-eval/data/evaluation --out results/check/bias_models.txt
diff <(grep -v '^#' results/sigtests_paper.txt) <(grep -v '^#' results/check/sigtests_paper.txt)
diff <(grep -v '^#' results/bias_models.txt) <(grep -v '^#' results/check/bias_models.txt)
```

`osf_fetch.py` shows the size and asks before downloading. The fine-tuned predictions of Fig. 14 (bias
Test 3a) are not in `evaluation.zip`. Without them, both scripts skip that part, as the committed outputs do.

## Full E1 reproduction (PostgreSQL 18)

Requirements:
- PostgreSQL 18 (we used Postgres.app on macOS), with pg_hint_plan 1.8.0 in `shared_preload_libraries`, on
  port 5418.
- Planner settings `effective_cache_size = 10GB`, `geqo = off`, `enable_memoize = off` and `jit = off`;
  other planner settings at their defaults. `load_imdb.py --check_only` checks all of them.
- From OSF: `runs.zip` (PG10 runtimes, 1.4 GB), `workloads.zip` (pinned SQL, 0.3 GB) and `datasets.zip`
  (IMDB, 2.7 GB).

```bash
uv run python scripts/osf_fetch.py --any runs.zip --any workloads.zip --any datasets.zip
uv run python scripts/load_imdb.py --port 5418                  # create and load the database imdb
uv run python scripts/load_imdb.py --port 5418 --check_only     # row counts, planner settings, hints
Q58=($(uv run python -c "from lcm_recheck.config import JOB_LIGHT_IDS; print(*JOB_LIGHT_IDS)"))
uv run python scripts/fix_hints.py                              # NestedLoop( -> NestLoop( in a copy of the pinned SQL
uv run python scripts/run_plans.py --explain_only --port 5418 --sql_dir results/sql_pinned_fixed --out_dir results/runs_pg18_explain_fixed --queries "${Q58[@]}"
uv run python scripts/extract_runs.py --runs results/runs_pg18_explain_fixed --version pg18_fixed --out results/check/pg18_explain_fixed.csv
uv run python scripts/make_predcsv.py --label postgres_18 --pg10 results/pg10_paper.csv --costs results/check/pg18_explain_fixed.csv --eval_root lcm-eval/data/evaluation --out_root results/check/predictions
uv run python scripts/diagnose_hints.py --pg18 results/check/pg18_explain_fixed.csv --port 5418 --eval_root lcm-eval/data/evaluation --extra_root results/check/predictions --out results/check/pg18_hint_status_fixed.csv
uv run python scripts/sigtests.py paper --eval_root lcm-eval/data/evaluation --extra_root results/check/predictions --tasks join --out results/check/sigtests_e1.txt
```

`run_plans.py --explain_only` only plans the queries and executes none. Each step after loading took seconds
on a laptop; `sigtests.py --tasks join` took about 15 s.

Each output can be compared with its committed counterpart, for example
`cmp results/check/pg18_explain_fixed.csv results/pg18_explain_fixed.csv` and
`diff -r results/check/predictions/postgres_18 results/predictions/postgres_18`.

For the unfixed SQL (`postgres_18_asis`), extracting the PG10 runtimes yourself, and the E2 design, see
`AGENTS.md`.

## Bias Test 1: the training data

`bias_data.py` reads the authors' raw training run files from `runs.zip`. These are
`runs/raw/<database>/workload_100k_s1_c8220.json` for the 20 training databases, plus the IMDB index-scan
and sequential-scan retraining runs.

```bash
R=lcm-eval/data/runs/raw
uv run python scripts/bias_data.py --imdb_train_runs $R/imdb/workload_100k_s1_c8220.json --ref_index_runs $R/imdb/index_retraining/index_retraining.json --ref_seq_runs $R/imdb/seq_retraining/seq_retraining.json --out results/check/bias_data_imdb.txt
uv run python scripts/bias_data.py --train_runs $R/{accidents,airline,baseball,basketball,carcinogenesis,consumer,credit,employee,fhnk,financial,geneea,genome,hepatitis,movielens,seznam,ssb,tournament,tpc_h_pk,walmart}/workload_100k_s1_c8220.json --out results/check/bias_data_19dbs.txt
```

The IMDB run takes a few seconds and the 19 other databases under a minute. Compare the results with
`results/bias_data_imdb.txt` and `results/bias_data_19dbs.txt` as above.

## For contributors

```bash
uv run pytest           # unit tests; the one database test skips itself without a server on port 5418
uv run ruff check .
```

`AGENTS.md` describes the code, the statistical protocol and the conventions in detail.
`tools/verify_golden.sh` is a maintainer tool: it compares every output with local reference files in
`results/golden/`, which are not in the repository.

## Data and license

The authors' code ([lcm-eval](https://github.com/DataManagementLab/lcm-eval), Apache License 2.0) and their
data ([OSF project rb5tn](https://osf.io/rb5tn/)) are not redistributed here; the commands above download
them. These committed files are derived from the authors' data:
- `results/pg10_paper.csv`, `results/pg16_paper.csv`: extracted from the authors' PG10 and PG16 run files,
  including their SQL text and runtimes.
- `results/pg18_explain.csv`, `results/pg18_explain_fixed.csv`, `results/pg18_hint_status*.csv`: our PG18
  costs and hint diagnosis for the authors' pinned SQL. The explain files include that SQL.
- `results/predictions/postgres_18*/`: our PG18 costs, with PG10 runtimes from the authors' runs as labels.
- `results/sigtests_paper.txt`, `results/bias_*.txt`: test results computed from the authors' predictions
  and run files.
- `results/osf_listing.tsv`: the file names and sizes of the OSF project.

This repository's own code is licensed under the Apache License 2.0 (see `LICENSE`).

## Citation and acknowledgments

The paper under study:

```bibtex
@article{10.1145/3725309,
  author = {Heinrich, Roman and Luthra, Manisha and Wehrstein, Johannes and Kornmayer, Harald and Binnig, Carsten},
  title = {How Good are Learned Cost Models, Really? Insights from Query Optimization Tasks},
  year = {2025},
  issue_date = {June 2025},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  volume = {3},
  number = {3},
  url = {https://doi.org/10.1145/3725309},
  doi = {10.1145/3725309},
  journal = {Proc. ACM Manag. Data},
  month = jun,
  articleno = {172},
  numpages = {27},
}
```

We thank the authors for publishing their code, predictions and run files; this re-analysis is possible
only because of them. Thanks also to the maintainers of [pg_hint_plan](https://github.com/ossc-db/pg_hint_plan).
