"""lcm_recheck.report: section(), Tee and tee_output() (the --out header of sigtests/bias_* scripts)."""
import io
import sys
from pathlib import Path

import pytest

from lcm_recheck.report import Tee, section, tee_output


def test_section(capsys: pytest.CaptureFixture[str]) -> None:
    section("TASK 1")
    assert capsys.readouterr().out == "\n" + "=" * 100 + "\nTASK 1\n" + "=" * 100 + "\n"


def test_tee_writes_to_all_streams() -> None:
    a, b = io.StringIO(), io.StringIO()
    t = Tee(a, b)
    t.write("x\n")
    t.flush()
    assert a.getvalue() == b.getvalue() == "x\n"


def test_no_path_changes_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    with tee_output(None, header_extra="; extra", closing="done"):
        print("body")
    assert capsys.readouterr().out == "body\n"


@pytest.fixture
def real_stdout(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    """tee_output tees to sys.__stdout__ and restores it, so replace both with one buffer."""
    buf = io.StringIO()
    monkeypatch.setattr(sys, "__stdout__", buf)
    monkeypatch.setattr(sys, "stdout", buf)
    monkeypatch.setattr(sys, "argv", ["sigtests.py", "paper", "--out", "x.txt"])
    return buf


def test_file_header_body_closing(tmp_path: Path, real_stdout: io.StringIO) -> None:
    out = tmp_path / "sub" / "out.txt"                      # the folder is created
    with tee_output(str(out), header_extra="; bootstrap seed 0", closing=f"Output written to {out}"):
        print("body")
    text = out.read_text(encoding="utf-8").split("\n")
    assert text[0].startswith("# 20")
    assert text[1] == "# command: python sigtests.py paper --out x.txt"
    assert text[2].startswith("# python ") and text[2].endswith("; bootstrap seed 0")
    assert text[3:] == ["", "body", ""]
    assert sys.stdout is sys.__stdout__
    term = real_stdout.getvalue()
    assert term.endswith(f"\nbody\nOutput written to {out}\n")      # closing line on the terminal only
    assert "Output written" not in out.read_text(encoding="utf-8")


def test_overwrites(tmp_path: Path, real_stdout: io.StringIO) -> None:
    out = tmp_path / "out.txt"
    out.write_text("old content\n", encoding="utf-8")
    with tee_output(str(out), header_extra="", closing="done"):
        print("new")
    assert "old content" not in out.read_text(encoding="utf-8")


def test_exception_restores_and_closes_without_closing_line(tmp_path: Path, real_stdout: io.StringIO) -> None:
    out = tmp_path / "out.txt"
    with pytest.raises(RuntimeError, match="boom"), tee_output(str(out), header_extra="", closing="CLOSING"):
        print("partial")
        raise RuntimeError("boom")
    assert sys.stdout is sys.__stdout__
    assert out.read_text(encoding="utf-8").endswith("\npartial\n")    # flushed and closed
    assert real_stdout.getvalue().endswith("partial\n")               # no closing line after an error
    assert "CLOSING" not in real_stdout.getvalue()
