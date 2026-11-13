"""lcm_recheck.db: shared connection arguments and connect(). Only test_live_server needs PostgreSQL."""
import argparse
import socket

import pytest

from lcm_recheck import config as c
from lcm_recheck import db


def parser(port: int | None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    db.add_db_args(ap, port=port)
    return ap


def test_defaults() -> None:
    a = parser(5418).parse_args([])
    assert vars(a) == {"host": c.PG_HOST, "port": 5418, "user": c.PG_USER, "password": c.PG_PASSWORD,
                       "db": c.PG_DB}


def test_port_parsed_as_int() -> None:
    assert parser(5418).parse_args(["--port", "5417"]).port == 5417


def test_port_required_without_default() -> None:
    ap = parser(None)
    with pytest.raises(SystemExit):
        ap.parse_args([])
    assert ap.parse_args(["--port", "5418"]).port == 5418


def test_connect_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(db.psycopg2, "connect", lambda **kw: calls.append(kw))
    a = parser(5418).parse_args(["--password", "pw"])
    db.connect(a)
    db.connect(a, dbname="postgres")
    base = {"host": c.PG_HOST, "port": 5418, "user": c.PG_USER, "password": "pw"}
    assert calls == [{**base, "dbname": c.PG_DB}, {**base, "dbname": "postgres"}]


def reachable(port: int) -> bool:
    try:
        socket.create_connection((c.PG_HOST, port), timeout=1).close()
    except OSError:
        return False
    return True


@pytest.mark.db
def test_live_server() -> None:
    port = c.PORTS[c.PG18]
    if not reachable(port):
        pytest.skip(f"no PostgreSQL server on port {port}")
    conn = db.connect(parser(port).parse_args([]))
    try:
        cur = conn.cursor()
        cur.execute("SELECT version()")
        assert cur.fetchone()[0].startswith("PostgreSQL 18")
    finally:
        conn.close()
