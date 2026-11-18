"""scripts/osf_fetch.py --extract on invented zips: only matching members, no __MACOSX, zip kept, no re-download,
nested zips unpacked."""
import io
import runpy
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "osf_fetch.py"
MEMBERS = {"runs/raw/a/x.json": b'{"query_list": []}', "runs/raw/a/y.json": b"{}",
           "runs/raw/b/z.json": b"{}", "runs/json/a/x.json": b"{}", "__MACOSX/runs/raw/a/._x.json": b"mac"}


def zip_bytes() -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("runs/raw/a/", b"")                     # a directory entry
        for name, data in MEMBERS.items():
            z.writestr(name, data)
    return b.getvalue()


class FakeFile:
    def __init__(self, allow_download: bool) -> None:
        self.allow_download, self.downloads = allow_download, 0

    def write_to(self, fp) -> None:
        assert self.allow_download, "downloaded although the zip exists"
        self.downloads += 1
        fp.write(zip_bytes())


def run(monkeypatch: pytest.MonkeyPatch, fake: FakeFile, *args: str) -> None:
    main = runpy.run_path(str(SCRIPT), run_name="not_main")["main"]
    main.__globals__["walk"] = lambda project: iter([("runs.zip", 100, fake), ("other.zip", 100, fake)])
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--match", "runs.zip", "--yes", *args])
    main()


def files(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def test_extract_only_matching_members(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                       capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeFile(allow_download=True)
    run(monkeypatch, fake, "--dest", str(tmp_path), "--extract", "runs/raw/a/")
    assert fake.downloads == 1
    assert files(tmp_path) == ["runs.zip", "runs/raw/a/x.json", "runs/raw/a/y.json"]
    assert (tmp_path / "runs/raw/a/x.json").read_bytes() == MEMBERS["runs/raw/a/x.json"]
    assert "extracted 2 files (0.0 MB) under runs/raw/a/" in capsys.readouterr().out


def test_existing_zip_is_not_downloaded_again(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                              capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "runs.zip").write_bytes(zip_bytes())
    run(monkeypatch, FakeFile(allow_download=False), "--dest", str(tmp_path),
        "--extract", "runs/raw/b/", "runs/json/")       # several prefixes after one --extract
    assert files(tmp_path) == ["runs.zip", "runs/json/a/x.json", "runs/raw/b/z.json"]
    out = capsys.readouterr().out
    assert "exists, not downloaded again: runs.zip" in out and "extracted 2 files" in out


def test_no_matching_member(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                            capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "runs.zip").write_bytes(zip_bytes())
    run(monkeypatch, FakeFile(allow_download=False), "--dest", str(tmp_path), "--extract", "workloads/")
    assert files(tmp_path) == ["runs.zip"]
    assert "no member of runs.zip starts with workloads/" in capsys.readouterr().out


def test_nested_zip_is_unpacked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                capsys: pytest.CaptureFixture[str]) -> None:
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("db1/", b"")
        z.writestr("db1/t.csv", b"1,2\n")
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as z:
        z.writestr("datasets/db1.zip", inner.getvalue())
        z.writestr("datasets/db2.zip", inner.getvalue())
    (tmp_path / "runs.zip").write_bytes(outer.getvalue())
    run(monkeypatch, FakeFile(allow_download=False), "--dest", str(tmp_path), "--extract", "datasets/db1.zip")
    assert files(tmp_path) == ["datasets/db1/t.csv", "runs.zip"]
    assert "unpacked db1.zip (2 entries)" in capsys.readouterr().out
