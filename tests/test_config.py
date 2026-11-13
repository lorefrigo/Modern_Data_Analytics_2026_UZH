"""Consistency of the constants in lcm_recheck.config (no data, no database)."""
from lcm_recheck import config as c


def test_job_light_ids() -> None:
    assert len(c.JOB_LIGHT_IDS) == 58
    assert list(c.JOB_LIGHT_IDS) == sorted(set(c.JOB_LIGHT_IDS))
    assert c.JOB_LIGHT_QUERIES == tuple(f"job_light_{i}" for i in c.JOB_LIGHT_IDS)


def test_scan_columns() -> None:
    assert {db: len(cols) for db, cols in c.SCAN_COLUMNS.items()} == {"imdb": 6, "baseball": 7, "tpc_h_pk": 8}
    assert sum(len(set(cols)) for cols in c.SCAN_COLUMNS.values()) == 21


def test_physical_workloads() -> None:
    for db, folder, names in c.PHYSICAL.values():
        assert folder == "physical_plan"
        assert len(names) == len(set(names)) == 100
    db, folder, names = c.PHYSICAL_INDEX
    assert (db, folder) == ("imdb", "physical_index_plan")
    assert names == c.PHYSICAL["IMDB"][2]


def test_models() -> None:
    assert c.REF not in c.LCMS
    assert set(c.FINE_TUNED_LCMS) <= set(c.LCMS) and set(c.INDEX_LCMS) <= set(c.LCMS)
    assert c.LABEL_SOURCE in c.LCMS
    assert c.PG_VERSIONS == [c.PG16, c.PG18]


def test_protocol_constants() -> None:
    assert c.B == 10_000
    assert c.BASE_SEED == 0
    assert c.SEEDS == (0, 1, 2)


def test_default_retrain_root() -> None:
    expected = "lcm-eval/data/retraining_evaluation"
    assert c.default_retrain_root("lcm-eval/data/evaluation") == expected
    assert c.default_retrain_root("lcm-eval/data/evaluation/") == expected
