"""scripts/fix_hints.py on a tiny SQL folder: byte-exact copies, counts, and no overwrite without --overwrite."""
import runpy
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fix_hints.py"
FILES = {
    "job_light_10.sql": b"/*+ NestedLoop(t mi) NestedLoop(t mi mc) Leading(((t mi) mc)) */ SELECT 1;\n"
                        b"/*+ HashJoin(t mi) Leading((t mi)) */ SELECT 2;\n"
                        b"/*+ NestedLoop(mc t) Leading((mc t)) */ SELECT 3;\n",
    "job_light_2.sql": b"/*+ NestLoop(t mi) Leading((t mi)) */ SELECT 1;\n",
}


def run(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), *args])
    runpy.run_path(str(SCRIPT), run_name="not_main")["main"]()


@pytest.fixture
def sql_dir(tmp_path: Path) -> Path:
    d = tmp_path / "sql"
    d.mkdir()
    for name, data in FILES.items():
        (d / name).write_bytes(data)
    return d


def test_fix_hints(sql_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                   capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out"
    run(monkeypatch, "--sql_dir", str(sql_dir), "--out_dir", str(out))
    assert sorted(p.name for p in out.iterdir()) == sorted(FILES)
    for name, data in FILES.items():
        assert (out / name).read_bytes() == data.replace(b"NestedLoop(", b"NestLoop(")
    assert capsys.readouterr().out.splitlines() == [
        "job_light_10.sql: 2 lines",
        f"2 lines changed in 1 of 2 files (3 replacements of NestedLoop( by NestLoop(); written to {out}",
    ]


def test_no_overwrite_without_flag(sql_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                   capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "job_light_10.sql").write_bytes(b"keep")
    with pytest.raises(SystemExit, match="--overwrite"):
        run(monkeypatch, "--sql_dir", str(sql_dir), "--out_dir", str(out))
    assert [p.name for p in out.iterdir()] == ["job_light_10.sql"] and (out / "job_light_10.sql").read_bytes() == b"keep"
    run(monkeypatch, "--sql_dir", str(sql_dir), "--out_dir", str(out), "--overwrite")
    assert (out / "job_light_10.sql").read_bytes() == FILES["job_light_10.sql"].replace(b"NestedLoop(", b"NestLoop(")


def test_out_dir_must_differ(sql_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit, match="must differ"):
        run(monkeypatch, "--sql_dir", str(sql_dir), "--out_dir", str(sql_dir), "--overwrite")
    assert (sql_dir / "job_light_10.sql").read_bytes() == FILES["job_light_10.sql"]
