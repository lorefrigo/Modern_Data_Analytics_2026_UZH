#!/usr/bin/env bash
# Regression check of the refactoring against results/golden/ (see "Verification" in AGENTS.md).
#
# Prints one line per check: IDENTICAL, DIFFERENT, ERROR (a command failed) or SKIPPED (input missing).
#   1. command line: argparse defaults (tools/dump_defaults.py) and --help of every script, against the
#      scripts and src/ at --base (default HEAD);
#   2. golden outputs without the database: sigtests paper/versions, bias_models, bias_data, extract_runs on the
#      full run folders (PG18 with --no_sql: cmp with the committed CSVs; the authors' PG10/PG16 runs: cmp with
#      results/golden/), the committed PG18 files against the golden ones without the authors' SQL and hint text,
#      the stdout of the check_* scripts and of
#      tools/check_json_vs_raw.py, make_predcsv CSVs and the committed results/predictions/ against
#      results/golden/predictions/ (and its stdout against the --base code), fix_hints into a temp folder against
#      results/sql_pinned_fixed/ (diff -r) and its totals (771 lines in 20 of 58 files);
#   3. database checks (read-only, no EXPLAIN ANALYZE): run_plans --explain_only est_cost, load_imdb
#      --check_only, diagnose_hints. Skipped with a note if the server is not reachable (never started here).
# Checks that read generated predictions (sigtests paper, check_pg16_ties, diagnose_hints) get an eval root without
# the generated folders (lcm-eval/ as after the cleanup), so they must read them from results/predictions/.
# Writes only to a new temp folder, deleted at the end unless a check fails or --keep is given.
# Skips the golden checks cleanly if results/golden/ is missing. Exit status 1 if any check is DIFFERENT or ERROR.
#
# Usage (from anywhere inside the repository):
#   tools/verify_golden.sh                    # all checks (about 3 min)
#   tools/verify_golden.sh --no-db            # without the database
#   tools/verify_golden.sh --db-only          # only the database checks
#   tools/verify_golden.sh --quick            # only argparse defaults and --help (seconds)
#   tools/verify_golden.sh --base HEAD~1      # command line compared with another commit
set -u -o pipefail
export PYTHONDONTWRITEBYTECODE=1

usage() { sed -n '2,/^set /p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//'; exit "${1:-0}"; }
BASE=HEAD; RUN_NODB=1; RUN_DB=1; QUICK=0; KEEP=0; PORT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --base) BASE=$2; shift ;;
    --no-db) RUN_DB=0 ;;
    --db-only) RUN_NODB=0 ;;
    --quick) QUICK=1 ;;
    --port) PORT=$2; shift ;;
    --keep) KEEP=1 ;;
    -h|--help) usage 0 ;;
    *) echo "unknown option: $1" >&2; usage 2 ;;
  esac
  shift
done

cd "$(git rev-parse --show-toplevel)" || exit 2
git rev-parse --verify --quiet "$BASE^{commit}" > /dev/null || { echo "not a commit: $BASE" >&2; exit 2; }
T=$(mktemp -d "${TMPDIR:-/tmp}/verify_golden.XXXXXX") || exit 2
trap 'rm -rf "$T"; exit 130' INT TERM
G=${GOLDEN_DIR:-results/golden}      # GOLDEN_DIR is only for testing the skip path
EV=lcm-eval/data/evaluation
PY="uv run --frozen python"
N_OK=0; N_BAD=0; N_SKIP=0

result() {   # status name [note]
  printf '%-10s %s%s\n' "$1" "$2" "${3:+   ($3)}"
  case "$1" in IDENTICAL) N_OK=$((N_OK + 1)) ;; SKIPPED) N_SKIP=$((N_SKIP + 1)) ;; *) N_BAD=$((N_BAD + 1)) ;; esac
}
strip() { grep -v -e '^#' -e '^Output written' "$1"; }
compare() {  # name id file_a file_b: exact comparison, diff saved as $T/<id>.diff
  if diff "$3" "$4" > "$T/$2.diff"; then result IDENTICAL "$1"; else result DIFFERENT "$1" "diff: $T/$2.diff"; fi
}
need() {     # name file...: SKIPPED unless every file or folder exists
  local name=$1; shift
  for f in "$@"; do [ -e "$f" ] || { result SKIPPED "$name" "missing $f"; return 1; }; done
}
run() {      # name id command...: stdout to $T/<id>.out, stderr to $T/<id>.log; ERROR if it fails
  local name=$1 id=$2; shift 2
  "$@" > "$T/$id.out" 2> "$T/$id.log" || { result ERROR "$name" "exit $?, log: $T/$id.log"; return 1; }
}
check_out() {     # name golden id command... (writes $T/<id>.txt via --out): compare without header lines
  local name=$1 golden=$2 id=$3; shift 3
  need "$name" "$golden" || return
  run "$name" "$id" "$@" || return
  strip "$T/$id.txt" > "$T/$id.new"; strip "$golden" > "$T/$id.ref"
  compare "$name" "$id" "$T/$id.new" "$T/$id.ref"
}
check_stdout() {  # name golden id command...: stdout compared exactly
  local name=$1 golden=$2 id=$3; shift 3
  need "$name" "$golden" || return
  run "$name" "$id" "$@" || return
  compare "$name" "$id" "$T/$id.out" "$golden"
}

# ----------------------------------------------------------------------------- 1. command line vs --base
echo "== command line: working tree vs $BASE ($(git rev-parse --short "$BASE"))"
mkdir -p "$T/base"
git archive "$BASE" scripts | tar -x -C "$T/base"
if git cat-file -e "$BASE:src" 2> /dev/null; then git archive "$BASE" src | tar -x -C "$T/base"; fi
BASE_PY="env PYTHONPATH=$T/base/src $PY"       # the --base scripts import the --base lcm_recheck
mkdir -p "$T/cli"
scripts=$( (cd scripts && grep -l argparse ./*.py; cd "$T/base/scripts" && grep -l argparse ./*.py) 2> /dev/null \
           | sed 's#^\./##' | sort -u)
for s in $scripts; do
  if [ ! -f "scripts/$s" ] || [ ! -f "$T/base/scripts/$s" ]; then
    result DIFFERENT "defaults + --help scripts/$s" "only in $([ -f "scripts/$s" ] && echo 'working tree' || echo "$BASE")"
    continue
  fi
  id=cli/${s%.py}
  if $BASE_PY tools/dump_defaults.py "$T/base/scripts/$s" > "$T/$id.defaults_base" 2> "$T/$id.log" &&
     $PY tools/dump_defaults.py "scripts/$s" > "$T/$id.defaults_new" 2>> "$T/$id.log"; then
    compare "defaults scripts/$s ($(wc -l < "$T/$id.defaults_new" | tr -d ' '))" "$id.defaults" \
      "$T/$id.defaults_base" "$T/$id.defaults_new"
  else
    result ERROR "defaults scripts/$s" "log: $T/$id.log"
  fi
  $BASE_PY "$T/base/scripts/$s" --help > "$T/$id.help_base" 2>&1
  $PY "scripts/$s" --help > "$T/$id.help_new" 2>&1      # argparse prints only the basename, so no path to strip
  compare "--help scripts/$s" "$id.help" "$T/$id.help_base" "$T/$id.help_new"
done

GOLDEN=1
if [ "$QUICK" = 1 ]; then
  RUN_NODB=0; RUN_DB=0
elif [ ! -d "$G" ]; then
  echo; result SKIPPED "all golden comparisons" "$G/ not found"; RUN_NODB=0; RUN_DB=0; GOLDEN=0
fi

# $EV without the folders that also exist in $PRED (generated predictions): what lcm-eval/ looks like after the
# cleanup. Checks reading generated predictions use it as --eval_root, so they can only find them in $PRED.
if [ "$RUN_NODB" = 1 ] || [ "$RUN_DB" = 1 ]; then
  PRED=$($PY -c "from lcm_recheck.config import PRED_ROOT; print(PRED_ROOT)")
  RETRAIN=$($PY -c "from lcm_recheck.config import default_retrain_root; print(default_retrain_root('$EV'))")
  mkdir -p "$T/eval"
  for d in "$EV"/*/; do
    d=${d%/}; m=$(basename "$d")
    [ -d "$d" ] && [ ! -e "$PRED/$m" ] && ln -s "$PWD/$d" "$T/eval/$m"
  done
fi

# ----------------------------------------------------------------------------- 2. golden outputs, no database
if [ "$RUN_NODB" = 1 ]; then
  echo; echo "== golden outputs without the database"
  check_out "sigtests paper (all tasks)" $G/sigtests_paper_all.txt sigtests_paper \
    $PY scripts/sigtests.py paper --eval_root "$T/eval" --retrain_root "$RETRAIN" --extra_root "$PRED" \
    --out "$T/sigtests_paper.txt"
  check_out "sigtests versions" $G/sigtests_versions.txt sigtests_versions \
    $PY scripts/sigtests.py versions --a results/pg10_paper.csv --b results/pg16_paper.csv --out "$T/sigtests_versions.txt"
  check_out "sigtests versions --timeout_penalty 30" $G/sigtests_versions_penalty.txt sigtests_versions_penalty \
    $PY scripts/sigtests.py versions --a results/pg10_paper.csv --b results/pg16_paper.csv --timeout_penalty 30 \
    --out "$T/sigtests_versions_penalty.txt"
  check_out "bias_models" $G/bias_models_rerun.txt bias_models \
    $PY scripts/bias_models.py --eval_root $EV --out "$T/bias_models.txt"
  R=lcm-eval/data/runs/raw
  check_out "bias_data imdb" $G/bias_data_imdb_rerun.txt bias_data_imdb \
    $PY scripts/bias_data.py --imdb_train_runs $R/imdb/workload_100k_s1_c8220.json \
    --ref_index_runs $R/imdb/index_retraining/index_retraining.json \
    --ref_seq_runs $R/imdb/seq_retraining/seq_retraining.json --out "$T/bias_data_imdb.txt"
  dbs19=""
  for db in accidents airline baseball basketball carcinogenesis consumer credit employee fhnk financial geneea \
            genome hepatitis movielens seznam ssb tournament tpc_h_pk walmart; do
    dbs19="$dbs19 $R/$db/workload_100k_s1_c8220.json"
  done
  # shellcheck disable=SC2086
  check_out "bias_data 19 databases" $G/bias_data_19dbs_rerun.txt bias_data_19dbs \
    $PY scripts/bias_data.py --train_runs $dbs19 --out "$T/bias_data_19dbs.txt"

  # extract_runs on the full run folders (--version labels as originally used): our PG18 runs with --no_sql against
  # the committed CSVs, the authors' PG10/PG16 runs against the golden copies (those CSVs are not committed)
  RUNS_PG10=$($PY -c "from lcm_recheck.config import RUNS_PG10_DIR; print(RUNS_PG10_DIR)")
  RUNS_PG16=$($PY -c "from lcm_recheck.config import RUNS_PG16_DIR; print(RUNS_PG16_DIR)")
  for v in "results/runs_pg18_explain_fixed pg18_fixed results/pg18_explain_fixed.csv --no_sql" \
           "results/runs_pg18_explain pg18_explain results/pg18_explain.csv --no_sql" \
           "$RUNS_PG10 pg10_paper $G/pg10_paper.csv" \
           "$RUNS_PG16 pg16_paper $G/pg16_paper.csv"; do
    set -- $v
    name="extract_runs --version $2${4:+ $4} vs $3"
    need "$name" "$1" "$3" || continue
    run "$name" "er_$2" $PY scripts/extract_runs.py --runs "$1" --version "$2" ${4:+"$4"} --out "$T/er_$2.csv" \
      || continue
    if cmp -s "$T/er_$2.csv" "$3"; then
      result IDENTICAL "$name ($(ls "$1"/job_light_*.json | wc -l | tr -d ' ') run files)"
    else
      result DIFFERENT "$name" "file: $T/er_$2.csv"
    fi
  done

  # the committed PG18 files are the golden ones without the authors' SQL (replaced by its hash) and hint text
  if need "committed PG18 files vs golden" $G/pg18_explain.csv $G/pg18_explain_fixed.csv $G/pg18_hint_status.csv \
       $G/pg18_hint_status_fixed.csv; then
    if $PY - "$G" > "$T/reduced.txt" 2>&1 <<'PYEOF'
import io
import sys

import pandas as pd

from lcm_recheck.hints import sql_hash

G, bad = sys.argv[1], 0
for name in ["pg18_explain", "pg18_explain_fixed", "pg18_hint_status", "pg18_hint_status_fixed"]:
    g = pd.read_csv(f"{G}/{name}.csv", dtype=str, keep_default_na=False)
    with open(f"results/{name}.csv") as f:
        text = f.read()
    if "sql" in g:
        exp = g.assign(sql=g.sql.map(sql_hash)).rename(columns={"sql": "sql_sha1"})
    else:
        exp = g[["query", "plan_idx", "dup_plan", "syntax_error", "leading_used", "dump_found"]]
    b = io.StringIO()
    exp.to_csv(b, index=False)
    ok = b.getvalue() == text
    bad += not ok
    print(f"{name}: {'identical' if ok else 'DIFFERENT'}")
sys.exit(bad > 0)
PYEOF
    then result IDENTICAL "committed PG18 files vs golden without SQL and hint text (4 files)"
    else result DIFFERENT "committed PG18 files vs golden without SQL and hint text" \
      "$(grep -v identical "$T/reduced.txt" | head -1)"
    fi
  fi

  check_stdout "check_downloads stdout" $G/check_downloads.stdout check_downloads $PY scripts/check_downloads.py
  check_stdout "inspect_e1_inputs stdout" $G/inspect_e1_inputs.stdout inspect_e1_inputs $PY scripts/inspect_e1_inputs.py
  check_stdout "check_pinned stdout" $G/check_pinned.stdout check_pinned $PY scripts/check_pinned.py \
    --pg10 results/pg10_paper.csv --pinned_dir lcm-eval/data/workloads/evaluation/imdb/join_order_full_pg_10_hinted \
    --eval_root $EV
  check_stdout "check_pg16_labels stdout" $G/check_pg16_labels.stdout check_pg16_labels $PY scripts/check_pg16_labels.py \
    --pg10 results/pg10_paper.csv --pg16 results/pg16_paper.csv --eval_root $EV
  check_stdout "check_pg16_ties stdout" $G/check_pg16_ties.stdout check_pg16_ties $PY scripts/check_pg16_ties.py \
    --pg10 results/pg10_paper.csv --pg18 results/pg18_explain.csv --status results/pg18_hint_status.csv \
    --eval_root "$T/eval" --extra_root "$PRED"
  # JSON-mode vs raw-mode parsing in lcm_recheck.runs on the authors' PG10 runs (runs/json read from runs.zip in memory)
  if need "check_json_vs_raw stdout" lcm-eval/data/runs/raw/imdb/join_order_full lcm-eval/data/runs.zip; then
    check_stdout "check_json_vs_raw stdout" $G/check_json_vs_raw.stdout check_json_vs_raw $PY tools/check_json_vs_raw.py
  fi

  # fix_hints into a scratch folder: byte-identical to results/sql_pinned_fixed/ (made by the former shell loop),
  # and the reported totals must be the known ones
  SQL_PINNED=$($PY -c "from lcm_recheck.config import SQL_PINNED_DIR; print(SQL_PINNED_DIR)")
  if need "fix_hints" "$SQL_PINNED" results/sql_pinned_fixed &&
     run "fix_hints" fix_hints $PY scripts/fix_hints.py --sql_dir "$SQL_PINNED" --out_dir "$T/sql_fixed"; then
    if diff -r "$T/sql_fixed" results/sql_pinned_fixed > "$T/fix_hints.diff"; then
      result IDENTICAL "fix_hints vs results/sql_pinned_fixed ($(ls "$T/sql_fixed" | wc -l | tr -d ' ') files)"
    else
      result DIFFERENT "fix_hints vs results/sql_pinned_fixed" "diff: $T/fix_hints.diff"
    fi
    expected="771 lines changed in 20 of 58 files (819 replacements of NestedLoop( by NestLoop(); written to $T/sql_fixed"
    if [ "$(tail -1 "$T/fix_hints.out")" = "$expected" ]; then
      result IDENTICAL "fix_hints totals (771 lines in 20 of 58 files)"
    else
      result DIFFERENT "fix_hints totals" "got: $(tail -1 "$T/fix_hints.out")"
    fi
  fi

  # make_predcsv into a scratch out_root, and the committed $PRED: both byte-identical to $G/predictions/
  # ($EV is only read, for the Sc. PG10 / Sc. PG16 comparison in stdout)
  if need "make_predcsv" $EV/postgres $EV/postgres_16 $G/predictions/postgres_18 $G/predictions/postgres_18_asis \
       results/pg10_paper.csv results/sql_pinned_fixed "$SQL_PINNED"; then
    ok=1
    for v in "postgres_18 results/pg18_explain_fixed.csv results/sql_pinned_fixed" \
             "postgres_18_asis results/pg18_explain.csv $SQL_PINNED"; do
      set -- $v
      if run "make_predcsv --label $1" "mk_new_$1" $PY scripts/make_predcsv.py --label "$1" \
           --pg10 results/pg10_paper.csv --costs "$2" --sql_dir "$3" --eval_root $EV --out_root "$T/pred_new"; then
        if diff -r "$T/pred_new/$1" "$G/predictions/$1" > "$T/mk_$1.diff"; then
          result IDENTICAL "make_predcsv CSVs $1 ($(ls "$T/pred_new/$1/imdb/join_order_full" | wc -l | tr -d ' ') files)"
        else
          result DIFFERENT "make_predcsv CSVs $1" "diff: $T/mk_$1.diff"
        fi
      else
        ok=0
      fi
      if need "committed $PRED/$1 vs golden" "$PRED/$1"; then
        if diff -r "$PRED/$1" "$G/predictions/$1" > "$T/committed_$1.diff"; then
          result IDENTICAL "committed $PRED/$1 vs golden ($(ls "$PRED/$1/imdb/join_order_full" | wc -l | tr -d ' ') files)"
        else
          result DIFFERENT "committed $PRED/$1 vs golden" "diff: $T/committed_$1.diff"
        fi
      fi
    done
    if [ "$ok" = 1 ]; then
      if [ ! -f "$T/base/scripts/make_predcsv.py" ]; then
        result SKIPPED "make_predcsv stdout vs $BASE" "no scripts/make_predcsv.py in $BASE"
      else
        # the --base code gets the golden CSVs, which still hold the SQL (code before --sql_dir needs it)
        for v in "postgres_18 $G/pg18_explain_fixed.csv" "postgres_18_asis $G/pg18_explain.csv"; do
          set -- $v
          $BASE_PY "$T/base/scripts/make_predcsv.py" --label "$1" --pg10 results/pg10_paper.csv --costs "$2" \
            --eval_root $EV --out_root "$T/pred_base" > "$T/mk_base_$1.out" 2> "$T/mk_base_$1.log" || ok=0
        done
        if [ "$ok" = 1 ]; then
          cat "$T"/mk_base_postgres_18.out "$T"/mk_base_postgres_18_asis.out | sed "s#$T/pred_base#ROOT#g" > "$T/mk.ref"
          cat "$T"/mk_new_postgres_18.out "$T"/mk_new_postgres_18_asis.out | sed "s#$T/pred_new#ROOT#g" > "$T/mk.new"
          compare "make_predcsv stdout vs $BASE" mk_stdout "$T/mk.new" "$T/mk.ref"
        else
          result ERROR "make_predcsv stdout vs $BASE" "log: $T/mk_base_*.log"
        fi
      fi
    fi
  fi
fi

# ----------------------------------------------------------------------------- 3. database checks
if [ "$RUN_DB" = 1 ]; then
  echo; echo "== database checks (read-only)"
  [ -n "$PORT" ] || PORT=$($PY -c "from lcm_recheck.config import PORTS, PG18; print(PORTS[PG18])")
  PG_ISREADY=/Applications/Postgres.app/Contents/Versions/18/bin/pg_isready
  command -v pg_isready > /dev/null && [ ! -x "$PG_ISREADY" ] && PG_ISREADY=pg_isready
  if ! "$PG_ISREADY" -q -p "$PORT"; then
    result SKIPPED "database checks" "no server on port $PORT; start it in Postgres.app"
  else
    if need "run_plans est_cost (queries 0 33 55)" results/sql_pinned_fixed results/pg18_explain_fixed.csv &&
       run "run_plans --explain_only" run_plans $PY scripts/run_plans.py --explain_only --port "$PORT" \
         --sql_dir results/sql_pinned_fixed --out_dir "$T/runs" --queries 0 33 55 &&
       run "extract_runs on run_plans output" extract_runs $PY scripts/extract_runs.py --runs "$T/runs" \
         --version check --out "$T/runs.csv"; then
      $PY - "$T/runs.csv" results/pg18_explain_fixed.csv > "$T/est_cost.txt" 2>&1 <<'EOF'
import sys
import pandas as pd
new, ref = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
m = new.merge(ref, on=["query", "plan_idx"], suffixes=("_new", "_ref"))
diff = int((m.est_cost_new != m.est_cost_ref).sum()) + len(new) - len(m)
print(f"{len(new)} plans of {new['query'].nunique()} queries, {diff} differences")
sys.exit(diff > 0)
EOF
      st=$?
      [ $st = 0 ] && result IDENTICAL "run_plans est_cost vs pg18_explain_fixed.csv" "$(cat "$T/est_cost.txt")" \
                  || result DIFFERENT "run_plans est_cost vs pg18_explain_fixed.csv" "$(tail -1 "$T/est_cost.txt")"
    fi
    if [ "$GOLDEN" = 1 ]; then
      check_stdout "load_imdb --port $PORT --check_only" $G/load_imdb.stdout load_imdb \
        $PY scripts/load_imdb.py --port "$PORT" --check_only
      check_stdout "load_imdb --check_only (default port)" $G/load_imdb.stdout load_imdb_default \
        $PY scripts/load_imdb.py --check_only
      # the golden stdout prints the --out path used when it was made: /tmp/hint_status_check.csv
      if need "diagnose_hints" $G/diagnose_hints.stdout results/pg18_hint_status_fixed.csv results/sql_pinned_fixed &&
         run "diagnose_hints" diagnose_hints $PY scripts/diagnose_hints.py --pg18 results/pg18_explain_fixed.csv \
           --sql_dir results/sql_pinned_fixed --port "$PORT" --eval_root "$T/eval" --extra_root "$PRED" \
           --out "$T/hint_status_check.csv"; then
        sed "s#$T/hint_status_check.csv#/tmp/hint_status_check.csv#g" "$T/diagnose_hints.out" > "$T/diagnose_hints.new"
        compare "diagnose_hints stdout" diagnose_hints_stdout "$T/diagnose_hints.new" $G/diagnose_hints.stdout
        if cmp -s "$T/hint_status_check.csv" results/pg18_hint_status_fixed.csv; then
          result IDENTICAL "diagnose_hints CSV vs pg18_hint_status_fixed.csv"
        else
          result DIFFERENT "diagnose_hints CSV vs pg18_hint_status_fixed.csv" "file: $T/hint_status_check.csv"
        fi
      fi
    fi
  fi
fi

# ----------------------------------------------------------------------------- summary
echo; echo "$N_OK identical, $N_BAD different or failed, $N_SKIP skipped"
if [ "$N_BAD" = 0 ] && [ "$KEEP" = 0 ]; then rm -rf "$T"; else echo "outputs kept in $T"; fi
[ "$N_BAD" = 0 ]
