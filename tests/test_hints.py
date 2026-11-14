"""lcm_recheck.hints.leading(): content of the Leading(...) hint."""
from lcm_recheck.hints import leading


def test_nested_parentheses() -> None:
    s = "/*+ NestLoop(t mi) HashJoin(t mi mc) Leading(((t mi) mc)) */ SELECT 1;"
    assert leading(s) == "((t mi) mc)"


def test_whitespace_normalised() -> None:
    assert leading("/*+ Leading(( t\n\t mi )  mc) */") == "( t mi ) mc"


def test_absent() -> None:
    assert leading("/*+ HashJoin(t mi) */ SELECT 1;") is None


def test_unbalanced() -> None:
    assert leading("/*+ Leading((t mi) */") is None
