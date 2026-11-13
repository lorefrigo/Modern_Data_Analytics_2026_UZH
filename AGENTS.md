# AGENTS.md

Instructions for coding agents (Claude Code reads this through `CLAUDE.md`). Read this file completely
before changing anything.

## What this repository is

Seminar project (Modern Data Analytics, UZH, autumn 2026) on Heinrich et al., *How Good are Learned Cost
Models, Really?* (SIGMOD 2025). It contains three pieces of work, all on top of the authors' code and data:

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
| `scripts/sigtests.py` | All significance tests; subcommands `paper` and `versions` |
| `results/` | Derived outputs: `pg10_paper.csv`, `pg18_explain*.csv`, `pg18_hint_status*.csv`, `sigtests_*.txt`, `osf_listing.tsv` |
| `results/runs_*/` | Raw run files (gitignored) |
| `results/sql_pinned_fixed/` | Pinned SQL with `NestedLoop(` replaced by `NestLoop(` (gitignored, regenerable) |
| `results/golden/` | Local regression reference, see "Verification" (gitignored) |

Known problems to fix:

- Constants are duplicated: the 58 JOB-light ids, the model lists and the workload lists.
- `check_downloads.py` depends on `sigtests.py` being importable from the same folder.
- Connection arguments are repeated in `run_plans.py`, `load_imdb.py` and `diagnose_hints.py`.
- `make_pg18_predcsv.py` runs at import time and is hard-wired to PG18.
- The SQL fix exists only as a shell loop.
- Generated prediction folders (`postgres_18`, `postgres_18_asis`) are written **inside**
  `lcm-eval/data/evaluation/`, which is gitignored and belongs to the authors.
- `sigtests.py` uses one module-level random generator, so bootstrap results depend on the order of calls.

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

- The unit of resampling is the **query**: plans of the same query are correlated, so they are never
  resampled individually.
- Bootstrap intervals use `B = 10_000` resamples. Paired tests are Wilcoxon signed-rank tests on per-query
  differences (log ratios for runtimes). Multiple testing is handled per family as currently implemented:
  Holm for p-values, Bonferroni-adjusted bootstrap intervals.
- LCM results are averaged over seeds 0, 1, 2, as in the paper; per-seed results are also reported.
- Timed-out plans are dropped, as in the paper, unless `--timeout_penalty` is given.
- **Randomness (decided: option B).** No module-level or shared random generator. Every bootstrap gets
  its own generator, seeded from the base seed 0 and a stable identifier of the test:
  `np.random.default_rng([0, zlib.crc32(test_id.encode())])`. Never use Python's `hash()` for this,
  because it is randomized per process. Results must then not depend on which tasks are run or in which
  order.
  - Introducing this changes the bootstrap intervals once, but not the p-values, totals or verdicts.
  - Do it in a dedicated commit that reports how far each interval moved.
  - After that commit, regenerate `results/golden/sigtests_*` and the committed `results/sigtests_*.txt`.
- Every file written with `--out` starts with a header: date, exact command, package versions.

Ask the human before changing anything in this section.

## Refactoring plan

Do one step per commit and verify each step (see "Verification") before the next one.

1. `src/lcm_recheck/config.py`: the single source for paths, default ports, the 58 JOB-light ids, the model
   lists, seeds, workload lists and `B`. All scripts import from it.
2. `src/lcm_recheck/db.py`: shared connection arguments (`--host --port --user --password --db`) and
   `connect()`.
3. Wrap all top-level code in `main()` functions. `check_downloads.py` imports from the package, not from
   `sigtests.py`.
4. `src/lcm_recheck/runs.py`: run-file parsing, moved out of `extract_runs.py`.
5. Split `sigtests.py` into `stats.py` (bootstrap, Wilcoxon, corrections) and `tasks/join.py`,
   `tasks/access.py`, `tasks/physical.py`, `tasks/versions.py`. Keep the `sigtests.py` CLI and its output
   format.
6. Option B randomness (see above), in its own commit.
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
   - `sigtests` outputs must be identical apart from the header lines; bootstrap intervals may differ
     only in the option-B commit.
   - DB steps: rerun `run_plans.py --explain_only` on queries `0 33 55` into `/tmp` and compare `est_cost`
     with `results/pg18_explain_fixed.csv` (0 differences expected).
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
