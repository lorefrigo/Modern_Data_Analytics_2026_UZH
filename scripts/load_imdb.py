"""Load the IMDB dataset into PostgreSQL for E1/E2, without the authors' torch/DGL stack. Needs only psycopg2.

Mirrors load_database() in lcm-eval/src/cross_db_benchmark/benchmark_tools/postgres/database_connection.py:
  1. create the database,
  2. run schema_sql/postgres.sql as is (it creates 21 tables with primary keys only; the 6 tables that are
     not in schema.json stay empty, exactly as in the authors' setup),
  3. COPY the 15 tables listed in schema.json, in that order, with its db_load_kwargs
     (DELIMITER ',' QUOTE '"' ESCAPE '\\' NULL '' CSV HEADER), primary keys in place during the load,
  4. VACUUM ANALYZE.
Then it checks row counts against table_lengths.json, prints the planner settings that matter for E1,
and tests that pg_hint_plan hints are honoured.

Differences from the authors that cannot change statistics or cost estimates:
  - COPY ... FROM STDIN (client side) instead of COPY ... FROM '<file>', so the server needs no read
    access to your project folder (macOS privacy rules often block that); the parsing options are the same.
  - synchronous_commit = off for this session only (affects durability during the load, nothing else).

Run from the MDA folder:
  uv run python scripts/load_imdb.py --port 5418                     # load (refuses to touch an existing DB)
  uv run python scripts/load_imdb.py --port 5418 --force             # drop and reload
  uv run python scripts/load_imdb.py --port 5418 --check_only        # only the checks
"""
import argparse, json, os, sys, time
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from lcm_recheck.config import IMDB_CSV_DIR, IMDB_DATASET_DIR, PG_HOST, PG_USER, PG_PASSWORD, PG_DB, PORTS, PG18

AUTHORS_COPY_OPTS = "DELIMITER ',' QUOTE '\"' ESCAPE '\\' NULL '' CSV HEADER"

# Planner-relevant settings: the authors' PG16 config (src/conf/postgres/modified-postgresql16.conf)
# plus defaults. jit is not in their config; it is decided after planning and never changes a cost.
EXPECTED = {
    "effective_cache_size": "10GB", "geqo": "off", "enable_memoize": "off", "jit": "off",
    "work_mem": "4MB", "seq_page_cost": "1", "random_page_cost": "4", "cpu_tuple_cost": "0.01",
    "cpu_index_tuple_cost": "0.005", "cpu_operator_cost": "0.0025", "default_statistics_target": "100",
    "max_parallel_workers_per_gather": "2", "join_collapse_limit": "8", "from_collapse_limit": "8",
}


class Progress:
    """File wrapper that prints how much of the CSV has been sent."""
    def __init__(self, f, total, label):
        self.f, self.total, self.label, self.done, self.next_pct = f, max(total, 1), label, 0, 0

    def read(self, n=-1):
        chunk = self.f.read(n)
        self.done += len(chunk)
        pct = int(100 * self.done / self.total)
        if pct >= self.next_pct:
            print(f"\r  {self.label:<16} {pct:3d}%", end="", flush=True)
            self.next_pct = pct + 2
        return chunk

    def readline(self, n=-1):
        return self.f.readline(n)


def connect(a, db):
    conn = psycopg2.connect(host=a.host, port=a.port, user=a.user, password=a.password, dbname=db)
    conn.set_client_encoding("UTF8")
    return conn


def load(a, tables, copy_opts, schema_sql):
    admin = connect(a, "postgres")
    admin.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = admin.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (a.db,))
    if cur.fetchone():
        if not a.force:
            sys.exit(f"Database '{a.db}' already exists. Use --check_only to check it or --force to reload it.")
        print(f"Dropping database {a.db}")
        cur.execute(f'DROP DATABASE "{a.db}" WITH (FORCE)')
    cur.execute(f"CREATE DATABASE \"{a.db}\" TEMPLATE template0 ENCODING 'UTF8'")
    admin.close()

    conn = connect(a, a.db)
    cur = conn.cursor()
    cur.execute(schema_sql)
    conn.commit()
    cur.execute("SET synchronous_commit = off")

    t_all = time.time()
    for t in tables:
        path = os.path.join(a.csv_dir, f"{t}.csv")
        t0 = time.time()
        with open(path, "rb") as f:
            cur.copy_expert(f'COPY "{t}" FROM STDIN {copy_opts}', Progress(f, os.path.getsize(path), t),
                            size=1 << 20)
        conn.commit()
        print(f"\r  {t:<16} done in {time.time() - t0:6.1f} s")
    print(f"COPY finished in {(time.time() - t_all) / 60:.1f} min")

    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    t0 = time.time()
    print("VACUUM ANALYZE ...", end="", flush=True)
    cur.execute("VACUUM ANALYZE")
    print(f" done in {time.time() - t0:.0f} s")
    conn.close()


def check(a, lengths):
    conn = connect(a, a.db)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    ok = True

    cur.execute("SELECT version()")
    print(f"\nServer: {cur.fetchone()[0]}")

    print("\nRow counts vs table_lengths.json")
    for t, expected in lengths.items():
        cur.execute(f'SELECT count(*) FROM "{t}"')
        n = cur.fetchone()[0]
        flag = "ok" if n == expected else f"MISMATCH ({n - expected:+d})"
        ok &= n == expected
        print(f"  {t:<16} {n:>11,d}  expected {expected:>11,d}  {flag}")
    cur.execute("SELECT relname FROM pg_stat_user_tables WHERE n_live_tup = 0 ORDER BY relname")
    print(f"  empty tables (expected, not in schema.json): {', '.join(r[0] for r in cur.fetchall())}")
    cur.execute("SELECT count(*) FROM pg_stat_user_tables WHERE last_analyze IS NULL AND last_autoanalyze IS NULL "
                "AND n_live_tup > 0")
    never = cur.fetchone()[0]
    ok &= never == 0
    print(f"  loaded tables never analyzed: {never}")

    print("\nPlanner settings (expected = authors' config + defaults)")
    for name, want in EXPECTED.items():
        cur.execute(f"SHOW {name}")
        got = cur.fetchone()[0]
        flag = "ok" if got == want else "DIFFERENT"
        ok &= got == want
        print(f"  {name:<32} {got:<8} expected {want:<8} {flag}")
    cur.execute("SHOW shared_preload_libraries")
    spl = cur.fetchone()[0]
    print(f"  {'shared_preload_libraries':<32} {spl}")

    # pg_hint_plan works through shared_preload_libraries alone (CREATE EXTENSION is only for the hint table).
    q = "SELECT * FROM title t WHERE t.id = 1"
    cur.execute(f"EXPLAIN (COSTS OFF) {q}")
    plain = " ".join(r[0] for r in cur.fetchall())
    cur.execute(f"EXPLAIN (COSTS OFF) /*+ SeqScan(t) */ {q}")
    hinted = " ".join(r[0] for r in cur.fetchall())
    honoured = "Index" in plain and "Seq Scan" in hinted
    ok &= honoured
    print(f"\npg_hint_plan test: without hint -> {plain.split('  ')[0].strip()!r}, "
          f"with SeqScan(t) -> {hinted.split('  ')[0].strip()!r}: {'hints honoured' if honoured else 'HINT IGNORED'}")
    conn.close()
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED (see above)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default=IMDB_CSV_DIR, help="folder with the 15 <table>.csv files")
    ap.add_argument("--dataset_dir", default=IMDB_DATASET_DIR,
                    help="repo folder with schema.json, table_lengths.json, schema_sql/postgres.sql")
    ap.add_argument("--host", default=PG_HOST); ap.add_argument("--port", type=int, default=PORTS[PG18])
    ap.add_argument("--user", default=PG_USER); ap.add_argument("--password", default=PG_PASSWORD)
    ap.add_argument("--db", default=PG_DB)
    ap.add_argument("--force", action="store_true", help="drop and reload an existing database")
    ap.add_argument("--check_only", action="store_true")
    a = ap.parse_args()

    with open(os.path.join(a.dataset_dir, "schema.json")) as f:
        schema = json.load(f)
    with open(os.path.join(a.dataset_dir, "table_lengths.json")) as f:
        lengths = json.load(f)
    with open(os.path.join(a.dataset_dir, "schema_sql", "postgres.sql")) as f:
        schema_sql = f.read()
    tables = schema["tables"]
    copy_opts = schema["db_load_kwargs"]["postgres"].strip().rstrip(";")
    assert copy_opts == AUTHORS_COPY_OPTS, f"unexpected COPY options in schema.json: {copy_opts}"
    assert set(tables) == set(lengths), "schema.json and table_lengths.json list different tables"

    if not a.check_only:
        missing = [t for t in tables if not os.path.isfile(os.path.join(a.csv_dir, f"{t}.csv"))]
        if missing:
            sys.exit(f"Missing CSVs in {a.csv_dir}: {', '.join(missing)}")
        gb = sum(os.path.getsize(os.path.join(a.csv_dir, f"{t}.csv")) for t in tables) / 1e9
        print(f"{len(tables)} CSVs, {gb:.1f} GB, from {a.csv_dir}")
        try:
            load(a, tables, copy_opts, schema_sql)
        except psycopg2.Error as e:
            sys.exit(f"\nLoad failed: {e}\nThe database '{a.db}' is incomplete; fix the cause and rerun with --force.")

    sys.exit(0 if check(a, lengths) else 1)


if __name__ == "__main__":
    main()