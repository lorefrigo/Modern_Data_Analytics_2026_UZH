"""lcm_recheck.runs: lines() and exec_ms() on small raw-mode and JSON-mode plans."""
import math

from lcm_recheck.runs import exec_ms, lines

RAW_ANALYZE = [["Seq Scan on title t  (cost=0.00..10.00 rows=1 width=4) (actual time=0.01..0.02 rows=1 loops=1)"],
               ["Planning Time: 0.100 ms"],
               ["Execution Time: 12.345 ms"]]


def test_lines_raw_rows_and_strings() -> None:
    assert lines([["a"], ("b",), "c", 5]) == ["a", "b", "c", "5"]


def test_exec_ms_raw() -> None:
    assert exec_ms(RAW_ANALYZE) == 12.345


def test_exec_ms_case_insensitive() -> None:
    assert exec_ms([["execution time: 7 ms"]]) == 7.0


def test_exec_ms_json() -> None:
    assert exec_ms({"Plan": {}, "Execution Time": 3.5}) == 3.5


def test_exec_ms_missing_is_nan() -> None:
    assert math.isnan(exec_ms([["Planning Time: 0.1 ms"]]))
