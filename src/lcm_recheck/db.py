"""Shared PostgreSQL connection arguments (--host --port --user --password --db) and connect().

Defaults come from lcm_recheck.config. Each script states its own --port default; port=None makes --port
required, so the server (and with it the PostgreSQL version) is always named on the command line.

Usage:
  from lcm_recheck.config import PG18, PORTS
  from lcm_recheck.db import add_db_args, connect
  ap = argparse.ArgumentParser()
  add_db_args(ap, port=PORTS[PG18])
  a = ap.parse_args()
  conn = connect(a)                       # database from --db
  admin = connect(a, dbname="postgres")   # another database on the same server
"""
import argparse

import psycopg2
from psycopg2.extensions import connection

from lcm_recheck.config import PG_DB, PG_HOST, PG_PASSWORD, PG_USER


def add_db_args(ap: argparse.ArgumentParser, *, port: int | None) -> None:
    """Add --host --port --user --password --db to `ap`; --port is required if `port` is None."""
    ap.add_argument("--host", default=PG_HOST)
    if port is None:
        ap.add_argument("--port", type=int, required=True)
    else:
        ap.add_argument("--port", type=int, default=port)
    ap.add_argument("--user", default=PG_USER)
    ap.add_argument("--password", default=PG_PASSWORD)
    ap.add_argument("--db", default=PG_DB)


def connect(a: argparse.Namespace, dbname: str | None = None) -> connection:
    """Open a connection with the arguments added by add_db_args(); `dbname` overrides --db."""
    return psycopg2.connect(host=a.host, port=a.port, user=a.user, password=a.password,
                            dbname=a.db if dbname is None else dbname)
