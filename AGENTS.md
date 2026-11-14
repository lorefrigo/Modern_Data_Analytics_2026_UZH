# AGENTS.md

Instructions for coding agents (Claude Code reads this through `CLAUDE.md`). Read this file completely
before changing anything.

## What this repository is

Seminar project (Modern Data Analytics, UZH, autumn 2026) on Heinrich et al., *How Good are Learned Cost
Models, Really?* (SIGMOD 2025). It contains four pieces of work, all on top of the authors' code and data:

1. **Re-analysis.** Significance tests of the paper's main claim ("in none of the tasks LCMs can
   significantly beat traditional approaches") on the authors' published predictions, for all three tasks:
   join ordering (JOB-light), access paths, join operators.
2. **E1.** PostgreSQL 18's cost model scored on the authors' PG10 runtimes, on the 58 JOB-light queries
   with the PG10-pinned candidate plans (7,863 SQL lines). Finding: in the pinned SQL, pg_hint_plan does
   not know the keyword `NestedLoop` (the correct one is `NestLoop`). It reports a syntax error and
   ignores every hint after it, including `Leading(...)`, on 771 lines in 20 queries. The fix is a plain
   text replacement, applied to a copy of the SQL.
3. **E2 (designed, not run).** Execute the candidate plans on PG10 and PG18 on the same machine and compare
   each version's plan selection against its own runtimes (`sigtests.py versions`).
4. **Training-data bias** (Section 5.5, Section 5.7, recommendation R3), access-path task only:
   - **Test 1** (`bias_data.py`): is the training data itself biased? Reads the authors' raw training run
     files. 1a scan composition, 1b selectivity of index vs sequential scans (KS), 1c missing negative
     examples (Clopper-Pearson, Fisher) against the index/seq retraining queries, 1d timeouts (Fisher),
     1e unique and clustered index columns. Also a leakage check for Test 3a. No random numbers.
   - **Test 2** (`bias_models.py`): do the models show the bias? 21 columns of Table 3, all LCMs, PG10 and
     PG16. 2a selectivity slope, 2b over-preference for index scans, 2c false index picks vs PG10. Exact
     McNemar tests, Holm across LCMs, and a cluster bootstrap over **columns**.
   - **Test 3a** (`bias_models.py`): does fine-tuning on diversified data fix the choices? 6 IMDB columns
     of Fig. 14, 5 fine-tuned LCMs, exact McNemar before vs after.

   ```bash
   uv run python scripts/bias_data.py --imdb_train_runs lcm-eval/data/runs/raw/imdb/workload_100k_s1_c8220.json \
       --ref_index_runs lcm-eval/data/runs/raw/imdb/index_retraining/index_retraining.json \
       --ref_seq_runs lcm-eval/data/runs/raw/imdb/seq_retraining/seq_retraining.json --out results/bias_data_imdb.txt
   # the 19 other training databases (1a, 1b, 1d, 1e only)
   uv run python scripts/bias_data.py --train_runs lcm-eval/data/runs/raw/{accidents,airline,baseball,basketball,carcinogenesis,consumer,credit,employee,fhnk,financial,geneea,genome,hepatitis,movielens,seznam,ssb,tournament,tpc_h_pk,walmart}/workload_100k_s1_c8220.json \
       --out results/bias_data_19dbs.txt
   uv run python scripts/bias_models.py --eval_root lcm-eval/data/evaluation --out results/bias_models.txt
   ```

## External dependencies (not in this repository)

- **Authors' code:** https://github.com/DataManagementLab/lcm-eval, cloned into `lcm-eval/` at commit
  `e024859de0d0b10f686e7da69759656d2a743057`. Gitignored. **Read-only:** never edit tracked files there.
- **Authors' data:** OSF project `rb5tn`, downloaded selectively with `scripts/osf_fetch.py` into
  `lcm-eval/data/` (the layout the authors' code expects).
- **PostgreSQL 18** (Postgres.app on macOS, server `PG18-E1`, port 5418) with pg_hint_plan 1.8.0 loaded via
  `shared_preload_libraries`. Cost-relevant settings: `effective_cache_size = 10GB`, `geqo = off`,
  `enable_memoize = off`, `jit = off`. Database `imdb`, loaded by `scripts/load_imdb.py`.
  Binaries: `$PG18 = /Applications/Postgres.app/Contents/Versions/18/bin`.
- **Python** managed by uv. Always run code with `uv run`. Add dependencies with `uv add`. Keep the
  dependency set small: never import the authors' training stack (torch, DGL).

## Current layout (before the refactoring)

All commands run from the repository root.

| File | Purpose |
|---|---|
| `src/lcm_recheck/config.py` | Single source of constants: paths, PostgreSQL defaults, model lists, the 58 JOB-light ids, workload lists, `B`, seeds, folder names, `PRED_ROOT` (`results/predictions`), `default_retrain_root()`, `model_root()` (a model's predictions come from `extra_root` if it has a folder there, else from `eval_root`), `shadowed_models()` |
| `src/lcm_recheck/db.py` | Shared connection arguments (`add_db_args()`: `--host --port --user --password --db`) and `connect()` |
| `src/lcm_recheck/hints.py` | `leading()`: content of the `Leading(...)` hint, whitespace-normalised |
| `src/lcm_recheck/report.py` | `section()`, `Tee` and `tee_output()`: the `--out` file with its header (date, command, versions + a script-specific suffix) and the script's closing line |
| `src/lcm_recheck/runs.py` | Run-file parsing: `load_run()` (ijson streaming or `json.load`; errors on files without `query_list` and on parsed_plans files), `json_root()` (the three JSON plan shapes `[{"Plan": …}]`, `{"Plan": …}`, bare node), `lines()`, `exec_ms()`, `top_cost()`, `plan_shape()`, `plan_nodes()` |
| `src/lcm_recheck/stats.py` | Statistics helpers, no module-level generator: `family_rng()`/`resample_idx()` (one generator and one resample-index matrix per test family), `holm()`, `boot()`/`boot_fast()` (apply the index matrix they are given), `ci()`, `boot_p()`, `ci_p()`, `wilcoxon_p()`, `mcnemar()`, `wilson()`, `cp_upper()` |
| `src/lcm_recheck/tasks/common.py` | Shared by the `sigtests.py` tasks: the `SUMMARY` list (`record()`, `summary()`), `compare()`/`compare_pick()` (bootstrap with the matrix of the `family` they are given), `finish_family()`, `show()`, `load_workload()`, `with_common_labels()` |
| `src/lcm_recheck/tasks/join.py`, `access.py`, `physical.py`, `versions.py` | `sigtests.py paper` Tasks 1, 2, 3 and `sigtests.py versions` |
| `src/lcm_recheck/tasks/bias_data.py` | Test 1 of the bias tests (run-file records, crossover rule, 1a-1e, leakage check) |
| `src/lcm_recheck/tasks/bias_models.py` | Tests 2 and 3a of the bias tests; Test 2 uses the one index matrix of family `bias_models/test2` |
| `tests/test_config.py` | Consistency checks of `config.py` (no data, no database); pins the full list of `FAMILY_IDS` |
| `tests/test_db.py` | `db.py` argument defaults and `connect()`; one `db`-marked test against port 5418 |
| `tests/test_hints.py`, `tests/test_report.py`, `tests/test_runs.py`, `tests/test_stats.py` | Unit tests of the package helpers (no data, no database); `test_stats.py` also pins the family seeding (`[0, crc32(family_id)]`) and how `boot()`/`boot_fast()` apply the index matrix |
| `tests/test_tasks.py` | Wiring of `tasks/`: no module-level generator anywhere; which family ids `compare()`, `compare_pick()`, `physical_family()`, `versions()` and `bias_models.test2()` use; members of a family share one matrix; no dependence on call order; `SUMMARY` order |
| `tests/test_scripts.py` | Every script only defines things on import (nothing runs or prints) and has a `main()` |
| `tools/verify_golden.sh` | Runs every regression check of "Verification" against `results/golden/`; one line per check |
| `tools/dump_defaults.py` | Prints the argparse defaults of a script without running it (used by `verify_golden.sh`) |
| `tools/check_json_vs_raw.py` | Real-data check of `runs.py`: the authors' PG10 JOB-light runs in JSON mode (`runs/json`, read from `lcm-eval/data/runs.zip` in memory) vs raw mode; `top_cost` and the `plan_shape` grouping must agree (used by `verify_golden.sh`) |
| `scripts/osf_fetch.py` | List or selectively download OSF files (shows sizes first) |
| `scripts/check_downloads.py` | Report which prediction files are present; constants from `config.py` |
| `scripts/load_imdb.py` | Create and load `imdb`; check row counts, planner settings and that hints work |
| `scripts/run_plans.py` | EXPLAIN (and optionally EXPLAIN ANALYZE) candidate plans; writes the authors' run-file format; resumable |
| `scripts/extract_runs.py` | Run files to one CSV row per candidate plan (cost, runtime, plan-shape hash); parsing in `runs.py` |
| `scripts/make_predcsv.py` | PG10 runtimes + one version's explain-only costs to prediction CSVs in `<out_root>/<label>/imdb/join_order_full/` (`--out_root` default `results/predictions`; refuses to replace an existing label folder without `--overwrite`) |
| `scripts/diagnose_hints.py` | Re-plan with pg_hint_plan debug output; find hint syntax errors and unused hints; Sc. PG18 from `--extra_root` (default `results/predictions`) |
| `scripts/check_pg16_ties.py` | Test whether the authors' Sc. PG16 shows the same broken-hint ties; generated models from `--extra_root` (default `results/predictions`) |
| `scripts/check_pg16_labels.py` | Which runtimes and candidate set the authors used for Sc. PG10 / Sc. PG16 (four hypotheses); prints only |
| `scripts/check_pinned.py` | Whether Sc. PG16 was built from the pinned SQL (maps pinned lines to PG10 plans via `Leading(...)`); prints only |
| `scripts/inspect_e1_inputs.py` | E1 input check: which SQL each authors' run used, and whether plan positions line up across folders; no arguments, prints only |
| `scripts/sigtests.py` | All significance tests; subcommands `paper` and `versions` (CLI; the tests are in `tasks/`). `paper --extra_root` (default `results/predictions`): Task 1 reads a model from there if it has a folder there (a copy in `--eval_root` is then ignored, with a note on stderr) |
| `scripts/bias_data.py` | Training-data bias, Test 1 (and the Test 3a leakage check), from raw training run files; streams them via `runs.load_run` (ijson) (CLI; the tests are in `tasks/bias_data.py`) |
| `scripts/bias_models.py` | Training-data bias, Tests 2 and 3a, from the access-path predictions (CLI; the tests are in `tasks/bias_models.py`) |
| `results/pg10_paper.csv` | `extract_runs.py` output for the authors' PG10 runs (E1 labels) |
| `results/pg16_paper.csv` | `extract_runs.py` output for the authors' PG16 runs (`--version pg16_paper`); input to `check_pg16_labels.py` |
| `results/pg18_explain.csv`, `results/pg18_explain_fixed.csv` | PG18 costs on the original and the fixed pinned SQL |
| `results/pg18_hint_status.csv`, `results/pg18_hint_status_fixed.csv` | `diagnose_hints.py` output for both variants |
| `results/predictions/postgres_18/`, `results/predictions/postgres_18_asis/` | `make_predcsv.py` output (Sc. PG18 on the fixed and on the original pinned SQL), same layout as an eval root; committed |
| `results/sigtests_paper.txt` | `sigtests.py paper` output, all tasks |
| `results/bias_data_imdb.txt` | `bias_data.py` on IMDB with the retraining queries (Tests 1a-1e, leakage check) |
| `results/bias_data_19dbs.txt` | `bias_data.py` on the 19 other training databases (1a, 1b, 1d, 1e) |
| `results/bias_models.txt` | `bias_models.py` output (Tests 2 and 3a) |
| `results/osf_listing.tsv` | OSF archive sizes from `osf_fetch.py` |
| `results/lcm-eval-commit.txt` | The pinned `lcm-eval` commit hash |
| `results/runs_*/` | Raw run files (gitignored) |
| `results/sql_pinned_fixed/` | Pinned SQL with `NestedLoop(` replaced by `NestLoop(` (gitignored, regenerable) |
| `results/golden/` | Local regression reference, see "Verification" (gitignored): copies of the committed CSVs and `sigtests_paper.txt`, plus `sigtests_paper_all.txt`, `bias_models_rerun.txt`, `rerun_pg18_explain_fixed.csv`, `check_json_vs_raw.stdout` and `predictions/postgres_18{,_asis}/` |

Known problems to fix:

- `check_pg16_labels.py`, `check_pinned.py`, `check_pg16_ties.py`, `inspect_e1_inputs.py` and
  `check_downloads.py` have no `--out`.
- The SQL fix exists only as a shell loop.
- `extract_runs.py` averages a plan's repetitions with `sum/len`, so one repetition without an `Execution Time`
  (NaN from `exec_ms()`) makes `runtime_s` NaN; `bias_data.py`'s `mean_exec_ms()` drops such repetitions.
- `plan_nodes()` names aggregate nodes differently per format: `Finalize Aggregate`/`Partial Aggregate` in text
  mode, `Aggregate` in JSON (9,903 plan pairs each in `check_json_vs_raw.py`); no current test depends on them.
- `plan_shape()` masks every number in JSON mode too (all 10,351 of the authors' JSON hashes are unchanged when every
  numeric field is perturbed). But JSON plans from `EXPLAIN ANALYZE` carry run-dependent structure (the `Workers`
  list, one entry per launched worker; `Sort Method`; `Sort Space Type`), and floats of 1e16 or more are dumped as
  `#e+#`, so identical plans can get different JSON hashes. `extract_runs.py`'s duplicate-plan check can then miss
  duplicates on JSON runs. The grouping check of `check_json_vs_raw.py` is unaffected: these fields only split
  plans, never merge them.

## E1 pipeline (current commands)

The 58 query ids are defined only in `src/lcm_recheck/config.py` (`JOB_LIGHT_IDS`). Below, the array `Q58` holds
that list (`"${Q58[@]}"` works in bash and zsh).

```bash
Q58=($(uv run python -c "from lcm_recheck.config import JOB_LIGHT_IDS; print(*JOB_LIGHT_IDS)"))
# PG10 runtimes from the authors' raw runs (check_downloads.py prints the exact --runs path)
uv run python scripts/extract_runs.py --runs <authors' PG10 run folder> --version pg10_paper --out results/pg10_paper.csv
# fixed copy of the pinned SQL
mkdir -p results/sql_pinned_fixed
for f in lcm-eval/data/workloads/evaluation/imdb/join_order_full_pg_10_hinted/*.sql; do
  sed 's/NestedLoop(/NestLoop(/g' "$f" > "results/sql_pinned_fixed/$(basename "$f")"; done
# PG18 costs (explain only, a few minutes)
uv run python scripts/run_plans.py --explain_only --port 5418 --sql_dir results/sql_pinned_fixed --out_dir results/runs_pg18_explain_fixed --queries "${Q58[@]}"
uv run python scripts/extract_runs.py --runs results/runs_pg18_explain_fixed --version pg18_fixed --out results/pg18_explain_fixed.csv
# prediction CSVs into results/predictions/postgres_18/ (eval_root is only read: Sc. PG10/PG16 for the printed comparison)
uv run python scripts/make_predcsv.py --label postgres_18 --pg10 results/pg10_paper.csv --costs results/pg18_explain_fixed.csv --eval_root lcm-eval/data/evaluation
uv run python scripts/diagnose_hints.py --pg18 results/pg18_explain_fixed.csv --port 5418 --eval_root lcm-eval/data/evaluation --out results/pg18_hint_status_fixed.csv
uv run python scripts/sigtests.py paper --eval_root lcm-eval/data/evaluation --tasks join --out results/sigtests_e1.txt
```

`diagnose_hints.py` and `sigtests.py paper` read `postgres_18` from `results/predictions/` (their `--extra_root`
default). The unfixed variant uses the original SQL folder, writes `runs_pg18_explain` and `pg18_explain.csv`, and its
prediction folder is `postgres_18_asis` (`make_predcsv.py --label postgres_18_asis --costs results/pg18_explain.csv ...`).

## Statistical protocol (never change silently)

- The unit of resampling is the unit within which observations are correlated: the **query** for Tasks 1
  and 3 (plans of the same query are never resampled individually), the **column** for Task 2 and for bias
  Tests 2 and 3a.
- Bootstrap intervals use `B = 10_000` resamples. Paired tests are Wilcoxon signed-rank tests on per-query
  differences (log ratios for runtimes). Multiple testing is handled per family as currently implemented:
  Holm for p-values, Bonferroni-adjusted bootstrap intervals.
- LCM results are averaged over seeds 0, 1, 2, as in the paper; per-seed results are also reported.
- Timed-out plans are dropped, as in the paper, unless `--timeout_penalty` is given.
- **Randomness (decided: option B).** No module-level random generator. One generator per test family,
  seeded from the base seed 0 and a stable family identifier:
  `np.random.default_rng([0, zlib.crc32(family_id.encode())])`. Never use Python's `hash()` for this,
  because it is randomized per process. Within a family all models use the same resample indices, because
  the comparisons with PG10 and with the optimum are paired. Results must then not depend on which tasks
  are run or in which order.
  - The family ids (`config.FAMILY_IDS`, pinned in `tests/test_config.py`) are part of this protocol:
    renaming one changes its bootstrap numbers and needs the same treatment as the option-B commit.
  - The option-B commit may change bootstrap intervals, bootstrap p-values and the verdicts that rest on
    them (runtime and balanced-accuracy verdicts in `sigtests.py`, Tests 2a and 2b in `bias_models.py`).
    Exact tests (Wilcoxon, McNemar), totals and point estimates must not change. The commit message lists
    every verdict that flips.
  - Do it in a dedicated commit that reports how far each interval moved.
  - After that commit, regenerate `results/golden/sigtests_*` and the committed `results/sigtests_*.txt`.
- Every file written with `--out` starts with a header: date, exact command, package versions.

Ask the human before changing anything in this section.

## Refactoring plan

Do one step per commit and verify each step (see "Verification") before the next one.

1. `src/lcm_recheck/config.py`: the single source for paths, default ports, the 58 JOB-light ids, the model
   lists, seeds, workload lists, the `scan_costs_percentiles` folder, the `retraining_evaluation` default
   and `B`. All scripts import from it, including `bias_*.py`, `check_pinned.py` and `inspect_e1_inputs.py`.
2. `src/lcm_recheck/db.py`: shared connection arguments (`--host --port --user --password --db`) and
   `connect()`.
3. Wrap all top-level code in `main()` functions (including `check_pg16_labels.py`, `check_pinned.py` and
   `inspect_e1_inputs.py`). `check_downloads.py`, `bias_data.py` and `bias_models.py` import from the
   package, not from `sigtests.py` or `extract_runs.py`. Move `leading()`, `Tee`, `section()` and the
   `--out` header into the package.
4. `src/lcm_recheck/runs.py`: run-file parsing, moved out of `extract_runs.py` (also `lines`/`exec_ms`, which
   `bias_data.py` uses, and `bias_data.py`'s `load_run`/`plan_nodes`).
5. Split `sigtests.py` into `stats.py` (bootstrap, Wilcoxon, corrections) and `tasks/join.py`,
   `tasks/access.py`, `tasks/physical.py`, `tasks/versions.py`. Keep the `sigtests.py` CLI and its output
   format.
   - Same step for the bias tests: their statistics helpers (`mcnemar`, `boot_p`, `ci`, `wilson`,
     `cp_upper`) go to `stats.py`, the tests to `tasks/bias_data.py` and `tasks/bias_models.py`. Keep both
     CLIs and their output format.
6. Option B randomness (see above), in its own commit, for `sigtests.py` and `bias_models.py`.
   `bias_models.py` Test 2 must keep one shared index matrix across models (the comparisons with the
   optimum and with PG10 are paired).
   - In the same commit, the Task 1 H2 (Q-error) bootstrap in `sigtests.py` moves from `b=2000` to `B`.
     If the full `sigtests.py paper` run then takes much longer, report the runtime and ask the human;
     keep 2,000 only if they agree, and then document it as an exception in "Statistical protocol".
7. Generalise `make_pg18_predcsv.py` into `make_predcsv.py --label postgres_18 ...`, writing to
   `results/predictions/<label>/imdb/join_order_full/`.
   - `sigtests.py paper` gets `--extra_root results/predictions`, so that no generated file lives inside
     `lcm-eval/` any more.
   - Commit the generated prediction CSVs, which are small.
8. `scripts/fix_hints.py`: replaces the shell loop that writes `results/sql_pinned_fixed/`. It also reports
   how many lines it changed (771 expected).
9. Console entry points in `pyproject.toml` (e.g. `uv run lcm-sigtests paper ...`), plus `pytest` and
   `ruff` as dev dependencies.
10. Update the README's command section from the final code. Before the README is written, in this order:
    - One commit across all scripts: give every command-line option a `--help` text. No default changes;
      `tools/verify_golden.sh` then reports DIFFERENT only for the `--help` checks of the scripts touched.
    - `tee_output()` (`src/lcm_recheck/report.py`) prints the script's closing line ("Output written to
      ...") only when the run finished without an exception. Today it is also printed on errors, as the
      scripts did before step 3. Stdout is still restored and the file still closed in `finally`. Update
      the test in `tests/test_report.py` that pins today's behaviour.

Scripts stay thin CLIs in `scripts/`; logic lives in `src/lcm_recheck/`.

## Verification (after every change)

1. Until step 9 adds `ruff` and `pytest` as dev dependencies, use exactly these commands:
   ```bash
   uvx ruff check src tests tools/check_json_vs_raw.py   # must be clean
   uvx ruff check scripts                                # at most 30 findings (the count before step 7)
   uv run --with pytest pytest
   ```
   - Unit tests use tiny fixture run files in `tests/fixtures/`, never the real data.
   - Tests that need PostgreSQL are marked `@pytest.mark.db` and skip themselves when port 5418 is
     unreachable.
2. Regression against `results/golden/` (local, gitignored, created before the refactoring).
   `tools/verify_golden.sh` runs every check below and prints one line per check (IDENTICAL, DIFFERENT,
   ERROR or SKIPPED); exit status 1 if any check is DIFFERENT or ERROR. It writes only to a new temp folder,
   kept (path printed) if a check fails, and skips the golden checks if `results/golden/` is missing.
   ```bash
   tools/verify_golden.sh                 # all checks, about 3 min; database checks SKIPPED if no server
   tools/verify_golden.sh --no-db         # without the database
   tools/verify_golden.sh --db-only       # only the database checks
   tools/verify_golden.sh --quick         # only argparse defaults and --help, against --base (seconds)
   tools/verify_golden.sh --base <commit> # reference commit for defaults, --help and make_predcsv stdout
   ```
   - No step changes a command-line default unless it says so. `tools/dump_defaults.py` prints every argparse
     default of a script; the tool compares these defaults and `--help` with the scripts and `src/` at
     `--base` (default `HEAD`, so run it before committing).
   - When a step renames a script or changes a golden command (e.g. steps 7 and 8), update the tool in the
     same commit.
   - CSV outputs (`extract_runs`, prediction CSVs) must be identical. Use `cmp`, or pandas with exact
     comparison for floats that are only re-serialised.
   - `make_predcsv.py` (both labels, into a temp `--out_root`) and the committed `results/predictions/<label>/`
     must be byte-identical to `results/golden/predictions/<label>/` (`diff -r`).
   - Checks that read generated predictions (`sigtests.py paper`, `check_pg16_ties.py`, `diagnose_hints.py`) get
     a temp eval root that links every folder of `lcm-eval/data/evaluation/` except those also in
     `results/predictions/`, plus `--extra_root results/predictions` (and, for `sigtests.py`, `--retrain_root
     lcm-eval/data/retraining_evaluation`, so that the printed path is unchanged). This proves they read the
     generated predictions from `results/predictions/`.
   - `extract_runs.py` on the full run folders must reproduce the committed CSVs (`cmp`):
     `results/runs_pg18_explain_fixed` (`--version pg18_fixed`) → `results/pg18_explain_fixed.csv`,
     `results/runs_pg18_explain` (`pg18_explain`) → `results/pg18_explain.csv`, and, if downloaded, the
     authors' runs `RUNS_PG10_DIR` (`pg10_paper`) → `results/pg10_paper.csv` and `RUNS_PG16_DIR`
     (`pg16_paper`) → `results/pg16_paper.csv`.
   - `sigtests` outputs must be identical apart from the header lines; bootstrap intervals, bootstrap
     p-values and the verdicts resting on them may differ only in the option-B commit (see "Statistical
     protocol"). This includes `sigtests.py versions --a results/pg10_paper.csv --b results/pg16_paper.csv`,
     without and with `--timeout_penalty 30`, against `results/golden/sigtests_versions.txt` and
     `results/golden/sigtests_versions_penalty.txt`.
   - `bias_models.py --eval_root lcm-eval/data/evaluation` must match `results/golden/bias_models_rerun.txt`
     apart from the header lines; bootstrap intervals and bootstrap p-values may differ only in the
     option-B commit. After that commit, regenerate the golden file and the committed
     `results/bias_models.txt`.
   - `bias_data.py` outputs (the two commands in "What this repository is") must be identical to
     `results/golden/bias_data_imdb_rerun.txt` and `results/golden/bias_data_19dbs_rerun.txt` (which match
     the committed `results/bias_data_*.txt`) apart from the header lines (no randomness). Compare with
     `diff <(grep -v -e '^#' -e '^Output written' A) <(grep -v -e '^#' -e '^Output written' B)`.
     Runtimes: IMDB about 5 s, 19 databases about 80 s.
   - `check_pg16_labels.py`, `check_pinned.py`, `check_pg16_ties.py`, `inspect_e1_inputs.py` and
     `check_downloads.py` only print: their stdout (arguments as in their docstrings) must be identical to
     `results/golden/<script>.stdout`, saved before the refactoring and confirmed deterministic.
   - `tools/check_json_vs_raw.py` (no arguments, about 12 s): stdout identical to
     `results/golden/check_json_vs_raw.stdout` (saved before `json_root()` existed; `top_cost` equal on 10,351 of
     10,351 plans, `plan_shape` grouping equal in 70 of 70 files).
   - DB steps:
     - Rerun `run_plans.py --explain_only` on queries `0 33 55` into `/tmp` and compare `est_cost`
       with `results/pg18_explain_fixed.csv` (0 differences expected).
     - `load_imdb.py --port 5418 --check_only`: stdout identical to `results/golden/load_imdb.stdout` (its
       `Server:` line holds the exact server version, so a Postgres.app update changes that line only).
     - `diagnose_hints.py --pg18 results/pg18_explain_fixed.csv --port 5418 --eval_root
       lcm-eval/data/evaluation --out /tmp/hint_status_check.csv`: `cmp` with
       `results/pg18_hint_status_fixed.csv` (identical), and its stdout identical to
       `results/golden/diagnose_hints.stdout` (deterministic; it prints the `--out` path, so use exactly
       this one).
3. The commit message states what changed and whether any number changed, and why.

## How to extend the code

- **Another PostgreSQL version (e.g. PG17 or PG19).**
  1. Set up a server on its own port (e.g. 5417) with the same settings and pg_hint_plan.
  2. Load the data with `load_imdb.py --port`.
  3. Run `run_plans.py --explain_only --port <port>` on `results/sql_pinned_fixed` (`--port` has no default).
  4. Run `extract_runs.py`, then `make_predcsv.py --label postgres_<v>`.
  5. `sigtests.py paper` picks the label up in the "PG version vs PG10" family.
  6. Add the port and label to `config.py`; nothing else should need to change.
- **E2 (executing plans).**
  1. Use `run_plans.py` without `--explain_only` (`--reps 3 --timeout 30`), one server running at a time,
     with a warm-up query first.
  2. Run `extract_runs.py` per version, then `sigtests.py versions --a ... --b ...`, once more with
     `--timeout_penalty 30` as a sensitivity check.
  3. Executing all 58 queries takes days. Start with the queries that have at most 4 tables, and never
     start an execution run without asking the human.
- **Another learned model or prediction folder.** Add it to the model lists in `config.py`;
  `check_downloads.py` and `sigtests.py` must pick it up from there.
- **Another statistical test or task.**
  1. Add a module in `src/lcm_recheck/tasks/` using the helpers in `stats.py` and its own named generator.
  2. Register it in the `sigtests.py` CLI.
  3. Add a fixture-based test.
  4. Document it in the README.

## Conventions

- Scripts use argparse with `--help` text. Defaults come from `config.py`. No absolute paths or usernames
  in code or docs.
- Outputs go to `results/`. Never overwrite an input. Long runs are resumable: skip existing outputs, as
  `run_plans.py` does.
- Every script starts with a docstring: what it does, then a usage example.
- New code uses type hints and passes `ruff`.
- Never commit `lcm-eval/`, downloaded data, raw run files or `results/golden/`.
- When a command changes, update `README.md` and this file in the same commit.

## Ask the human first before

- deleting or overwriting anything in `results/` or `lcm-eval/data/`;
- changing the statistical protocol;
- running anything that executes plans (`EXPLAIN ANALYZE`);
- changing PostgreSQL server settings, or starting or stopping servers;
- adding a dependency beyond the scientific Python stack, psycopg2 and osfclient.
