"""pg_hint_plan hints in the candidate-plan SQL.

Usage:
  from lcm_recheck.hints import leading
  leading("/*+ HashJoin(t mi) Leading((t mi)) */ SELECT ...")   # -> "(t mi)"
"""


def leading(s: str) -> str | None:
    """Content of the Leading(...) hint, with nested parentheses; None if absent."""
    i = s.find("Leading(")
    if i < 0:
        return None
    depth, j = 0, i + len("Leading")
    for k in range(j, len(s)):
        depth += {"(": 1, ")": -1}.get(s[k], 0)
        if depth == 0:
            return " ".join(s[j + 1:k].split())
    return None
