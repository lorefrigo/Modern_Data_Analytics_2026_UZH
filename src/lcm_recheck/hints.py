"""pg_hint_plan hints in the candidate-plan SQL.

Usage:
  from lcm_recheck.hints import leading, fix_nestloop
  leading("/*+ HashJoin(t mi) Leading((t mi)) */ SELECT ...")   # -> "(t mi)"
  fixed, n_lines, n_repl = fix_nestloop(open(path, "rb").read())   # NestedLoop( -> NestLoop(
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


# The pinned SQL uses "NestedLoop(", which pg_hint_plan does not know: it reports a syntax error and ignores every
# hint after it, including Leading(...). The correct keyword is "NestLoop(".
BROKEN_HINT = b"NestedLoop("
FIXED_HINT = b"NestLoop("


def fix_nestloop(data: bytes) -> tuple[bytes, int, int]:
    """Replace every NestedLoop( by NestLoop( in the bytes of a SQL file.

    Returns (fixed bytes, number of changed lines, number of replacements). Works on bytes, so line endings
    and everything else stay exactly as they were (same output as sed 's/NestedLoop(/NestLoop(/g').
    """
    n_lines = sum(BROKEN_HINT in line for line in data.splitlines())
    return data.replace(BROKEN_HINT, FIXED_HINT), n_lines, data.count(BROKEN_HINT)
