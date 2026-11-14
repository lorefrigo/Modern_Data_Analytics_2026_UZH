"""Every script only defines things on import: nothing runs and nothing is printed unless run as __main__."""
import runpy
import sys
from pathlib import Path

import pytest

SCRIPTS = sorted((Path(__file__).resolve().parents[1] / "scripts").glob("*.py"))


def test_scripts_found() -> None:
    assert len(SCRIPTS) >= 14


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_import_runs_nothing(path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", [str(path)])
    ns = runpy.run_path(str(path), run_name="not_main")
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""
    assert callable(ns.get("main")), f"{path.name} has no main()"
