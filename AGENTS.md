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
| `scripts/osf_fetch.py` | List or selectively download OSF files (shows sizes first) |
| `scripts/check_downloads.py` | Report which prediction files are present; imports constants from `sigtests.py` |
| `scripts/load_imdb.py` | Create and load `imdb`; check row counts, planner settings and that hints work |
| `scripts/run_plans.py` | EXPLAIN (and optionally EXPLAIN ANALYZE) candidate plans; writes the authors' run-file format; resumable |
| `scripts/extract_runs.py` | Run files to one CSV row per candidate plan (cost, runtime, plan-shape hash) |
| `scripts/make_pg18_predcsv.py` | PG10 runtimes + PG18 costs to prediction CSVs in `<eval_root>/postgres_18/` |
| `scripts/diagnose_hints.py` | Re-plan with pg_hint_plan debug output; find hint syntax errors and unused hints |
| `scripts/check_pg16_ties.py` | Test whether the authors' Sc. PG16 shows the same broken-hint ties |
| `scripts/check_pg16_labels.py` | Which runtimes and candidate set the authors used for Sc. PG10 / Sc. PG16 (four hypotheses); prints only |
| `scripts/check_pinned.py` | Whether Sc. PG16 was built from the pinned SQL (maps pinned lines to PG10 plans via `Leading(...)`); prints only |
| `scripts/inspect_e1_inputs.py` | E1 input check: which SQL each authors' run used, and whether plan positions line up across folders; no arguments, prints only |
| `scripts/sigtests.py` | All significance tests; subcommands `paper` and `versions` |
| `scripts/bias_data.py` | Training-data bias, Test 1 (and the Test 3a leakage check), from raw training run files; imports from `extract_runs.py` and `sigtests.py`; uses `ijson` |
| `scripts/bias_models.py` | Training-data bias, Tests 2 and 3a, from the access-path predictions; imports from `sigtests.py` |
| `results/pg10_paper.csv` | `extract_runs.py` output for the authors' PG10 runs (E1 labels) |
| `results/pg16_paper.csv` | `extract_runs.py` output for the authors' PG16 runs (`--version pg16_paper`); input to `check_pg16_labels.py` |
| `results/pg18_explain.csv`, `results/pg18_explain_fixed.csv` | PG18 costs on the original and the fixed pinned SQL |
| `results/pg18_hint_status.csv`, `results/pg18_hint_status_fixed.csv` | `diagnose_hints.py` output for both variants |
| `results/sigtests_paper.txt` | `sigtests.py paper` output, all tasks |
| `results/bias_data_imdb.txt` | `bias_data.py` on IMDB with the retraining queries (Tests 1a-1e, leakage check) |
| `results/bias_data_19dbs.txt` | `bias_data.py` on the 19 other training databases (1a, 1b, 1d, 1e) |
| `results/bias_models.txt` | `bias_models.py` output (Tests 2 and 3a) |
| `results/osf_listing.tsv` | OSF archive sizes from `osf_fetch.py` |
| `results/lcm-eval-commit.txt` | The pinned `lcm-eval` commit hash |
| `results/runs_*/` | Raw run files (gitignored) |
| `results/sql_pinned_fixed/` | Pinned SQL with `NestedLoop(` replaced by `NestLoop(` (gitignored, regenerable) |
| `results/golden/` | Local regression reference, see "Verification" (gitignored): copies of the committed CSVs and `sigtests_paper.txt`, plus `sigtests_paper_all.txt`, `bias_models_rerun.txt`, `rerun_pg18_explain_fixed.csv` and `predictions/postgres_18{,_asis}/` |

Known problems to fix:

- Constants are duplicated: the 58 JOB-light ids, the model lists and the workload lists.
  - The 58 ids are also hard-coded (`IDS`) in `check_pinned.py`, `inspect_e1_inputs.py` and
    `make_pg18_predcsv.py`.
  - `bias_models.py` redefines `B = 10_000` and the folder name `scan_costs_percentiles` (also literal in
    `sigtests.py` and `check_downloads.py`).
  - The default `retraining_evaluation` path next to `eval_root` is derived separately in `sigtests.py`,
    `bias_models.py` and `check_downloads.py`.
  - `inspect_e1_inputs.py` hard-codes the workload and run folders under `lcm-eval/data/`.
- Helper code is duplicated: `leading()` (parses the `Leading(...)` hint) in `check_pinned.py`,
  `inspect_e1_inputs.py`, `check_pg16_ties.py` and `make_pg18_predcsv.py`; the `--out` header and `Tee`
  set-up in `sigtests.py`, `bias_data.py` and `bias_models.py`.
- `check_downloads.py` depends on `sigtests.py` being importable from the same folder; so do
  `bias_data.py` (also on `extract_runs.py`) and `bias_models.py`.
- `check_pg16_labels.py`, `check_pinned.py` and `inspect_e1_inputs.py` run at import time and have no
  `--out`.
- Connection arguments are repeated in `run_plans.py`, `load_imdb.py` and `diagnose_hints.py`.
- `make_pg18_predcsv.py` runs at import time and is hard-wired to PG18.
- The SQL fix exists only as a shell loop.
- Generated prediction folders (`postgres_18`, `postgres_18_asis`) are written **inside**
  `lcm-eval/data/evaluation/`, which is gitignored and belongs to the authors.
- `sigtests.py` uses one module-level random generator, so bootstrap results depend on the order of calls.
  `bias_models.py` has its own module-level `RNG = np.random.default_rng(0)`; Test 2 draws one index matrix
  from it and shares it across 2a, 2b, 2c and all models.

## E1 pipeline (current commands)

The 58 query ids are listed in `scripts/sigtests.py` and `scripts/check_downloads.py`; after the
refactoring they must come from `config.py` only. Below, `$Q58` stands for that list.

```bash
# PG10 runtimes from the authors' raw runs (check_downloads.py prints the exact --runs path)
uv run python scripts/extract_runs.py --runs <authors' PG10 run folder> --version pg10_paper --out results/pg10_paper.csv
# fixed copy of the pinned SQL
mkdir -p results/sql_pinned_fixed
for f in lcm-eval/data/workloads/evaluation/imdb/join_order_full_pg_10_hinted/*.sql; do
  sed 's/NestedLoop(/NestLoop(/g' "$f" > "results/sql_pinned_fixed/$(basename "$f")"; done
# PG18 costs (explain only, a few minutes)
uv run python scripts/run_plans.py --explain_only --port 5418 --sql_dir results/sql_pinned_fixed --out_dir results/runs_pg18_explain_fixed --queries $Q58
uv run python scripts/extract_runs.py --runs results/runs_pg18_explain_fixed --version pg18_fixed --out results/pg18_explain_fixed.csv
uv run python scripts/make_pg18_predcsv.py --pg10 results/pg10_paper.csv --pg18 results/pg18_explain_fixed.csv --eval_root lcm-eval/data/evaluation
uv run python scripts/diagnose_hints.py --pg18 results/pg18_explain_fixed.csv --port 5418 --eval_root lcm-eval/data/evaluation --out results/pg18_hint_status_fixed.csv
uv run python scripts/sigtests.py paper --eval_root lcm-eval/data/evaluation --tasks join --out results/sigtests_e1.txt
```

The unfixed variant uses the original SQL folder, writes `runs_pg18_explain` and `pg18_explain.csv`, and its
prediction folder is kept as `postgres_18_asis`.

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
10. Update the README's command section from the final code.

Scripts stay thin CLIs in `scripts/`; logic lives in `src/lcm_recheck/`.

## Verification (after every change)

1. `uv run ruff check .` and `uv run pytest`.
   - Unit tests use tiny fixture run files in `tests/fixtures/`, never the real data.
   - Tests that need PostgreSQL are marked `@pytest.mark.db` and skip themselves when port 5418 is
     unreachable.
2. Regression against `results/golden/` (local, gitignored, created before the refactoring):
   - CSV outputs (`extract_runs`, prediction CSVs) must be identical. Use `cmp`, or pandas with exact
     comparison for floats that are only re-serialised.
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
  3. Run `run_plans.py --explain_only` on `results/sql_pinned_fixed`.
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
