"""Run the authors' JOB-light candidate plans against PostgreSQL, without their training stack.

Writes the same run-file format as the authors' run_benchmark.py (a JSON with a 'query_list'
holding sql, verbose_plan, analyze_plans and timeout per plan), so extract_runs.py reads both.
It follows their protocol: EXPLAIN VERBOSE first, then EXPLAIN ANALYZE `--reps` times;
a statement timeout marks the plan as timed out. Needs only psycopg2.

E1 (costs only):     python scripts/run_plans.py --explain_only --port 5418 --sql_dir <WL> --out_dir runs_pg18_explain --queries 0 1 2
E2 (execute plans):  python scripts/run_plans.py --reps 3 --timeout 30 --port 5418 --sql_dir <WL> --out_dir runs_pg18 --queries 0 1 2
"""
import argparse, json, os, time
import psycopg2
from psycopg2 import errors
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


def rows(cur, sql):
    cur.execute(sql)
    return [list(r) for r in cur.fetchall()]          # raw mode: one [line] per row, like the authors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sql_dir", required=True, help="folder with job_light_<k>.sql (one SQL per line)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--queries", nargs="+", required=True, help="query ids, e.g. 0 1 2")
    ap.add_argument("--explain_only", action="store_true")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=30, help="seconds per statement")
    ap.add_argument("--host", default="localhost"); ap.add_argument("--port", type=int, default=5432)
    ap.add_argument("--user", default="postgres"); ap.add_argument("--password", default="")
    ap.add_argument("--db", default="imdb")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    conn = psycopg2.connect(host=a.host, port=a.port, user=a.user, password=a.password, dbname=a.db)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute("SELECT version()"); server_version = cur.fetchone()[0]
    cur.execute(f"SET statement_timeout = {a.timeout * 1000}")
    print(server_version)

    for k in a.queries:
        target = os.path.join(a.out_dir, f"job_light_{k}.json")
        if os.path.exists(target):
            print(f"job_light_{k}: exists, skipping (delete the file to re-run)")
            continue
        with open(os.path.join(a.sql_dir, f"job_light_{k}.sql")) as f:
            sqls = [l.strip() for l in f if l.strip()]
        start, query_list = time.time(), []
        for i, sql in enumerate(sqls):
            entry = dict(sql=sql, hint="", verbose_plan=None, analyze_plans=[], timeout=False)
            try:
                # same statement form as the authors' runner: the /*+ ... */ hint stays inside the SQL,
                # after EXPLAIN (pg_hint_plan honours it there; check the duplicate-plan count anyway)
                entry["verbose_plan"] = rows(cur, f"EXPLAIN (VERBOSE TRUE, ANALYZE FALSE) {sql}")
                if not a.explain_only:
                    for _ in range(a.reps):
                        entry["analyze_plans"].append(rows(cur, f"EXPLAIN (ANALYZE TRUE) {sql}"))
            except errors.QueryCanceled:
                entry["timeout"] = True
            query_list.append(entry)
            print(f"\rjob_light_{k}: {i + 1}/{len(sqls)} plans", end="", flush=True)
        run = dict(query_list=query_list, server_version=server_version,
                   run_kwargs=dict(explain_only=a.explain_only, reps=a.reps, timeout=a.timeout),
                   total_time_secs=time.time() - start)
        with open(target + ".tmp", "w") as out:
            json.dump(run, out)
        os.replace(target + ".tmp", target)           # only complete files count as done
        print(f"  ({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
