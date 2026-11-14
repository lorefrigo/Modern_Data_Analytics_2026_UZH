"""Parsing of lcm-eval run files (raw or JSON EXPLAIN mode).

Usage:
  from lcm_recheck.runs import exec_ms, lines
  ms = exec_ms(entry["analyze_plans"][0])
"""
import re
from typing import Any

EXEC_RE = re.compile(r"execution time:\s*([\d.]+)\s*ms", re.IGNORECASE)


def lines(plan: list[Any]) -> list[str]:
    """Raw-mode plans are lists of 1-element rows; return them as a list of strings."""
    return [r[0] if isinstance(r, (list, tuple)) else str(r) for r in plan]


def exec_ms(analyze_plan: dict[str, Any] | list[Any]) -> float:
    if isinstance(analyze_plan, dict):
        return float(analyze_plan["Execution Time"])
    for line in lines(analyze_plan):
        m = EXEC_RE.search(line)
        if m:
            return float(m.group(1))
    return float("nan")
