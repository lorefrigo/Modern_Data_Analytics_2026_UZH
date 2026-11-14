"""Wiring of lcm_recheck.tasks: which generator each task draws from, and the SUMMARY collection.

Until option B (refactoring step 6) sigtests has exactly one generator, tasks.common.RNG, and bias_models
has its own, tasks.bias_models.RNG. No data, no database.
"""
import importlib
import types

import numpy as np
import pandas as pd
import pytest

from lcm_recheck import stats
from lcm_recheck.config import BASE_SEED
from lcm_recheck.stats import boot, boot_fast
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


def test_sigtests_tasks_share_one_generator() -> None:
    assert join.RNG is common.RNG
    assert versions.RNG is common.RNG
    for mod in (common, join, access, physical, versions):
        assert all(g is common.RNG for g in generators(mod)), mod.__name__


def test_bias_models_has_its_own_generator() -> None:
    assert generators(bias_models) == [bias_models.RNG]
    assert bias_models.RNG is not common.RNG
    assert generators(bias_data) == []
    assert generators(stats) == []


def test_generators_are_seeded_with_base_seed() -> None:
    # the seed sequence records the seed and does not change with draws
    assert common.RNG.bit_generator.seed_seq.entropy == BASE_SEED
    assert bias_models.RNG.bit_generator.seed_seq.entropy == BASE_SEED


def test_compare_draws_from_common_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    x, ref = np.arange(1.0, 13.0), np.arange(1.0, 13.0)[::-1] + 0.5
    d = x - ref
    for legacy, kind in [(False, "score"), (False, "runtime"), (True, "runtime")]:
        monkeypatch.setattr(common, "RNG", np.random.default_rng(123))
        got = common.compare(x, ref, 2, kind, legacy=legacy)
        stat = np.sum if kind == "runtime" else np.mean
        r = np.random.default_rng(123)
        samples = boot(list(d), stat, rng=r) if legacy else boot_fast(d, stat, rng=r)
        assert got["ci95_lo"] == np.percentile(samples, 2.5)
        assert got["ci95_hi"] == np.percentile(samples, 97.5)


def test_compare_pick_draws_from_common_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    x = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 1], bool)
    ref = np.array([0, 1, 0, 0, 0, 1, 0, 0, 1, 1], bool)
    monkeypatch.setattr(common, "RNG", np.random.default_rng(7))
    got = common.compare_pick(x, ref)
    samples = boot_fast(x.astype(float) - ref.astype(float), np.mean, rng=np.random.default_rng(7))
    assert (got["ci95_lo"], got["ci95_hi"]) == tuple(np.percentile(samples, [2.5, 97.5]))
    assert (got["only_lcm_optimal"], got["only_pg10_optimal"]) == (3, 0)
    assert got["mcnemar_p"] == pytest.approx(2 * 0.5 ** 3)


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
