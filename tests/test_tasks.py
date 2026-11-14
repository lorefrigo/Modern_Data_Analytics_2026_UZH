"""Wiring of lcm_recheck.tasks: option B resampling (one index matrix per test family, no module-level
generator) and the SUMMARY collection. No data, no database (tiny files in tmp_path only).
"""
import argparse
import importlib
import types
from typing import Any

import numpy as np
import pandas as pd
import pytest

from lcm_recheck import stats
from lcm_recheck.config import FAMILY_IDS, LABEL_SOURCE, REF, SCAN_FOLDER, B
from lcm_recheck.stats import boot_fast, resample_idx
from lcm_recheck.tasks import (
    access,
    bias_data,
    bias_models,
    common,
    join,
    physical,
    versions,
)

TASK_MODULES = ["common", "join", "access", "physical", "versions", "bias_data", "bias_models"]


@pytest.mark.parametrize("name", TASK_MODULES)
def test_import_prints_nothing(name: str, capsys: pytest.CaptureFixture[str]) -> None:
    importlib.import_module(f"lcm_recheck.tasks.{name}")
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""


def generators(mod: types.ModuleType) -> list[np.random.Generator]:
    return [v for v in vars(mod).values() if isinstance(v, np.random.Generator)]


@pytest.mark.parametrize("mod", [stats, common, join, access, physical, versions, bias_data, bias_models])
def test_no_module_level_generator(mod: types.ModuleType) -> None:
    assert generators(mod) == []
    assert not hasattr(mod, "RNG")


def spy(monkeypatch: pytest.MonkeyPatch, mod: types.ModuleType) -> list[tuple[str, int, np.ndarray]]:
    """Record every (family_id, n, matrix) that `mod` asks resample_idx for."""
    calls: list[tuple[str, int, np.ndarray]] = []

    def wrapped(family_id: str, n: int, b: int = B) -> np.ndarray:
        idx = resample_idx(family_id, n, b)
        calls.append((family_id, n, idx))
        return idx
    monkeypatch.setattr(mod, "resample_idx", wrapped)
    return calls


X, REF_X = np.arange(1.0, 13.0), np.arange(1.0, 13.0)[::-1] + 0.5
FAM_A, FAM_B = "sigtests/join/H1/LCM vs PG10", "sigtests/access/T2a/LCM vs PG10"


@pytest.mark.parametrize("kind", ["score", "runtime"])
def test_compare_bootstraps_with_the_family_matrix(kind: str) -> None:
    got = common.compare(X, REF_X, 2, kind, family=FAM_A)
    stat = np.sum if kind == "runtime" else np.mean
    samples = boot_fast(X - REF_X, stat, idx=resample_idx(FAM_A, len(X)))
    assert (got["ci95_lo"], got["ci95_hi"]) == tuple(np.percentile(samples, [2.5, 97.5]))
    assert (got["fwer_ci_lo"], got["fwer_ci_hi"]) == tuple(np.percentile(samples, [100 * 0.0125, 100 * 0.9875]))


def test_compare_needs_a_family() -> None:
    with pytest.raises(TypeError):
        common.compare(X, REF_X, 2, "score")  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="unknown test family"):
        common.compare(X, REF_X, 2, "score", family="not registered")


def test_members_of_a_family_share_one_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = spy(monkeypatch, common)
    common.compare(X, REF_X, 2, "runtime", family=FAM_A)
    common.compare(X * 1.1, REF_X, 2, "runtime", family=FAM_A)
    (f1, n1, i1), (f2, n2, i2) = calls
    assert (f1, n1) == (f2, n2) == (FAM_A, 12)
    np.testing.assert_array_equal(i1, i2)


def test_results_do_not_depend_on_call_order() -> None:
    first = common.compare(X, REF_X, 3, "runtime", family=FAM_A)
    common.compare(X, REF_X, 3, "score", family=FAM_B)
    common.compare_pick(X > 6, REF_X > 6, family="sigtests/physical/Fig. 11: LCM vs PG10/pick rate/IMDB")
    assert common.compare(X, REF_X, 3, "runtime", family=FAM_A) == first


def test_compare_pick_bootstraps_with_the_family_matrix() -> None:
    x = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 1], bool)
    ref = np.array([0, 1, 0, 0, 0, 1, 0, 0, 1, 1], bool)
    fam = "sigtests/physical/Fig. 11: LCM vs PG10/pick rate/IMDB"
    got = common.compare_pick(x, ref, family=fam)
    samples = boot_fast(x.astype(float) - ref.astype(float), np.mean, idx=resample_idx(fam, 10))
    assert (got["ci95_lo"], got["ci95_hi"]) == tuple(np.percentile(samples, [2.5, 97.5]))
    assert (got["only_lcm_optimal"], got["only_pg10_optimal"]) == (3, 0)
    assert got["mcnemar_p"] == pytest.approx(2 * 0.5 ** 3)


def test_physical_family_ids(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    calls = spy(monkeypatch, common)
    rng = np.random.default_rng(1)

    def scores(qs: list[str]) -> dict[str, dict[str, Any]]:
        return {m: {q: {"runtime": float(rng.uniform(1, 2)), "pick": bool(rng.integers(0, 2))} for q in qs}
                for m in (REF, "flat", "dace")}
    by_ds = {"IMDB": (scores([f"q{i}" for i in range(12)]), [f"q{i}" for i in range(12)]),
             "TPC-H": (scores([f"t{i}" for i in range(11)]), [f"t{i}" for i in range(11)])}
    physical.physical_family(by_ds, ["flat", "dace"], 4, "Fig. 11: LCM vs PG10", "3 join operator")
    capsys.readouterr()
    fam = "sigtests/physical/Fig. 11: LCM vs PG10"
    assert [(f, n) for f, n, _ in calls] == [
        (f"{fam}/selected runtime/IMDB", 12), (f"{fam}/pick rate/IMDB", 12),
        (f"{fam}/selected runtime/IMDB", 12), (f"{fam}/pick rate/IMDB", 12),
        (f"{fam}/selected runtime/TPC-H", 11), (f"{fam}/pick rate/TPC-H", 11),
        (f"{fam}/selected runtime/TPC-H", 11), (f"{fam}/pick rate/TPC-H", 11)]


def test_versions_family_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Any,
                            capsys: pytest.CaptureFixture[str]) -> None:
    calls = spy(monkeypatch, versions)
    rows = [{"query": f"q{q}", "est_cost": c, "runtime_s": 1.0 + q + 0.5 * c, "timeout": False}
            for q in range(8) for c in range(3)]
    pd.DataFrame(rows).assign(version="a").to_csv(tmp_path / "a.csv", index=False)
    pd.DataFrame(rows).assign(version="b", est_cost=lambda d: -d.est_cost).to_csv(tmp_path / "b.csv", index=False)
    versions.versions(argparse.Namespace(a=tmp_path / "a.csv", b=tmp_path / "b.csv", timeout_penalty=None))
    capsys.readouterr()
    assert [(f, n) for f, n, _ in calls] == [("sigtests/versions", 8)]


def test_bias_models_test2_draws_one_matrix(monkeypatch: pytest.MonkeyPatch, tmp_path: Any,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    calls = spy(monkeypatch, bias_models)
    rng = np.random.default_rng(2)
    for col in ("aka_name.person_id", "cast_info.nr_order", "title.episode_nr"):
        ls, li = rng.uniform(1, 2, 11), rng.uniform(0.5, 2.5, 11)
        for m in (REF, LABEL_SOURCE):
            d = tmp_path / m / "imdb" / SCAN_FOLDER
            d.mkdir(parents=True, exist_ok=True)
            for side, lab in (("seq", ls), ("index", li)):
                pd.DataFrame({"query_index": range(11), "label": lab, "prediction": lab * rng.uniform(0.5, 2, 11)}
                             ).to_csv(d / f"{side}.{col}_0_test_pred.csv", index=False)
    bias_models.test2(argparse.Namespace(eval_root=str(tmp_path)))
    assert "2c" in capsys.readouterr().out
    assert [(f, n) for f, n, _ in calls] == [("bias_models/test2", 3)]


def test_every_used_family_is_registered() -> None:
    assert {"sigtests/versions", "bias_models/test2", FAM_A, FAM_B} <= set(FAMILY_IDS)


def test_record_appends_in_call_order(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    collected: list[dict] = []
    monkeypatch.setattr(common, "SUMMARY", collected)
    common.record("1 join order", "LCM vs PG10", "selected runtime",
                  pd.DataFrame({"model": ["a", "b"], "verdict": ["PG10 better", "LCM better"]}))
    common.record("3 join operator", "LCM vs PG10", "pick rate",
                  pd.DataFrame({"dataset": ["IMDB"], "model": ["c"], "verdict": ["no clear difference"]}))
    assert [(r["task"], r["dataset"], r["model"], r["verdict"]) for r in collected] == [
        ("1 join order", "", "a", "PG10 better"),
        ("1 join order", "", "b", "LCM better"),
        ("3 join operator", "IMDB", "c", "no clear difference")]
    common.summary()
    out = capsys.readouterr().out
    assert "significantly better than PG10 (after family-wise correction): 1" in out


def test_summary_without_comparisons(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(common, "SUMMARY", [])
    common.summary()
    assert "no comparisons were run" in capsys.readouterr().out
