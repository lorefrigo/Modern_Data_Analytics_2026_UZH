"""lcm_recheck.hints: leading() (content of the Leading(...) hint) and fix_nestloop()."""
from lcm_recheck.hints import fix_nestloop, leading


def test_nested_parentheses() -> None:
    s = "/*+ NestLoop(t mi) HashJoin(t mi mc) Leading(((t mi) mc)) */ SELECT 1;"
    assert leading(s) == "((t mi) mc)"


def test_whitespace_normalised() -> None:
    assert leading("/*+ Leading(( t\n\t mi )  mc) */") == "( t mi ) mc"


def test_absent() -> None:
    assert leading("/*+ HashJoin(t mi) */ SELECT 1;") is None


def test_unbalanced() -> None:
    assert leading("/*+ Leading((t mi) */") is None


def test_fix_nestloop_counts_lines_and_replacements() -> None:
    data = b"/*+ NestedLoop(t mi) NestedLoop(t mi mc) */ SELECT 1;\n/*+ HashJoin(t mi) */ SELECT 2;\n" \
           b"/*+ NestedLoop(mc t) */ SELECT 3;\n"
    fixed, n_lines, n_repl = fix_nestloop(data)
    assert fixed == data.replace(b"NestedLoop(", b"NestLoop(")
    assert (n_lines, n_repl) == (2, 3)


def test_fix_nestloop_leaves_other_text_unchanged() -> None:
    data = b"/*+ NestLoop(t mi) */ -- NestedLoop without parenthesis\r\nSELECT 1;"   # CRLF, no final newline
    assert fix_nestloop(data) == (data, 0, 0)
    fixed, n_lines, _ = fix_nestloop(b"/*+ NestedLoop(t mi) */\r\nSELECT 1;")
    assert fixed == b"/*+ NestLoop(t mi) */\r\nSELECT 1;" and n_lines == 1
