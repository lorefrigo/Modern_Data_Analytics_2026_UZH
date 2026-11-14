"""Consistency of the constants in lcm_recheck.config (no data, no database)."""
from pathlib import Path

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


def test_pred_root() -> None:
    assert c.PRED_ROOT == "results/predictions"
    assert not c.PRED_ROOT.startswith("lcm-eval")


def test_model_root_prefers_extra_root(tmp_path: Path) -> None:
    ev, ex = tmp_path / "eval", tmp_path / "extra"
    for d in (ev / "postgres", ev / "postgres_18", ex / "postgres_18", ex / "postgres_19"):
        d.mkdir(parents=True)
    (ex / "not_a_model.txt").write_text("")
    assert c.model_root(str(ev), str(ex), "postgres_18") == str(ex)       # in both: extra_root wins
    assert c.model_root(str(ev), str(ex), "postgres_19") == str(ex)       # only in extra_root
    assert c.model_root(str(ev), str(ex), "postgres") == str(ev)          # only in eval_root
    assert c.model_root(str(ev), str(ex), "missing") == str(ev)
    assert c.model_root(str(ev), None, "postgres_18") == str(ev)
    assert c.model_root(str(ev), str(tmp_path / "nowhere"), "postgres_18") == str(ev)
    assert c.shadowed_models(str(ev), str(ex)) == ["postgres_18"]
    assert c.shadowed_models(str(ev), None) == []
    assert c.shadowed_models(str(ev), str(tmp_path / "nowhere")) == []


def test_family_ids_pinned() -> None:
    """The family ids seed the bootstrap generators (statistical protocol): renaming, adding or removing one
    changes bootstrap numbers and needs the same treatment as refactoring step 6. Update this list only then."""
    fig11 = [f"sigtests/physical/Fig. 11: {cmp} vs PG10/{metric}/{ds}"
             for cmp in ("LCM", "PG16") for metric in ("selected runtime", "pick rate")
             for ds in ("IMDB", "TPC-H", "Baseball")]
    assert fig11[:3] == ["sigtests/physical/Fig. 11: LCM vs PG10/selected runtime/IMDB",
                         "sigtests/physical/Fig. 11: LCM vs PG10/selected runtime/TPC-H",
                         "sigtests/physical/Fig. 11: LCM vs PG10/selected runtime/Baseball"]
    assert c.FAMILY_IDS == (
        "sigtests/join/H1/LCM vs PG10",
        "sigtests/join/H1/PG version vs PG10",
        "sigtests/join/H2/LCM vs PG10",
        "sigtests/join/H1b/LCM (act. card.) vs PG10",
        "sigtests/access/T2a/LCM vs PG10",
        "sigtests/access/T2a/PG16 vs PG10",
        "sigtests/access/T2b/LCM vs PG10",
        "sigtests/access/T2b/PG16 vs PG10",
        "sigtests/access/T2c/balanced accuracy",
        "sigtests/access/T2c/selected runtime",
        *fig11,
        "sigtests/physical/Section 6.6: LCM vs PG10/selected runtime/IMDB+indexes",
        "sigtests/physical/Section 6.6: LCM vs PG10/pick rate/IMDB+indexes",
        "sigtests/physical/Section 6.6: PG16 vs PG10/selected runtime/IMDB+indexes",
        "sigtests/physical/Section 6.6: PG16 vs PG10/pick rate/IMDB+indexes",
        "sigtests/versions",
        "bias_models/test2",
    )
    assert len(set(c.FAMILY_IDS)) == len(c.FAMILY_IDS) == 28
