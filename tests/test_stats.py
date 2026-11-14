"""lcm_recheck.stats against hand-computed values, and the per-family resampling of option B."""
import zlib

import numpy as np
import pytest

from lcm_recheck.config import FAMILY_IDS, B
from lcm_recheck.stats import (
    boot,
    boot_fast,
    boot_p,
    ci,
    ci_p,
    cp_upper,
    family_rng,
    holm,
    mcnemar,
    resample_idx,
    wilcoxon_p,
    wilson,
)


def test_holm_hand_computed() -> None:
    # sorted: 0.01*3 = 0.03, 0.03*2 = 0.06, 0.04*1 = 0.04 -> monotone 0.06; returned in input order
    np.testing.assert_allclose(holm([0.04, 0.01, 0.03]), [0.06, 0.03, 0.06])


def test_holm_capped_at_one() -> None:
    np.testing.assert_allclose(holm([0.5, 0.6]), [1.0, 1.0])


def test_holm_single() -> None:
    np.testing.assert_allclose(holm([0.2]), [0.2])


# ----------------------------------------------------------------------------- bootstrap p and intervals
def test_boot_p_hand_computed() -> None:
    assert boot_p(np.array([-1.0, 1.0, 2.0, 3.0])) == 0.5      # 2 * min(1/4, 3/4)
    assert boot_p(np.array([1.0, 2.0])) == 0.0
    assert boot_p(np.array([-1.0, 0.0, 1.0])) == 1.0           # 2 * 2/3, capped


def test_ci_default_is_exactly_2_5_and_97_5() -> None:
    s = np.random.default_rng(5).standard_normal(1001)
    np.testing.assert_array_equal(ci(s), np.percentile(s, [2.5, 97.5]))   # bias_models' former ci()
    np.testing.assert_array_equal(ci(np.arange(101.0)), [2.5, 97.5])
    np.testing.assert_array_equal(ci(np.arange(101.0), alpha=0.1), [5.0, 95.0])


def test_ci_p_combines_ci_and_boot_p() -> None:
    s = np.arange(101.0) - 50
    obs, lo, hi, p = ci_p(s, 7.0)
    assert (obs, lo, hi, p) == (7.0, -47.5, 47.5, 1.0)
    _, lo, hi, _ = ci_p(s, 7.0, alpha=0.1)
    assert (lo, hi) == (-45.0, 45.0)


# ----------------------------------------------------------------------------- exact tests
def test_mcnemar() -> None:
    assert mcnemar(0, 0) == 1.0
    assert mcnemar(8, 0) == pytest.approx(2 * 0.5 ** 8)
    assert mcnemar(3, 7) == pytest.approx(mcnemar(7, 3))


def test_wilcoxon_p() -> None:
    p, nz = wilcoxon_p([1.0, 2.0, 3.0, 0.0, 0.0])
    assert np.isnan(p) and nz == 3                              # fewer than 6 non-zero differences
    p, nz = wilcoxon_p([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert nz == 6 and p == pytest.approx(2 / 64)              # exact: all 6 signs positive


# ----------------------------------------------------------------------------- proportions
def test_wilson() -> None:
    assert all(np.isnan(wilson(0, 0)))
    lo, hi = wilson(5, 10)
    assert lo == pytest.approx(0.236593, abs=1e-6) and hi == pytest.approx(0.763407, abs=1e-6)


def test_cp_upper() -> None:
    assert cp_upper(0, 10) == pytest.approx(1 - 0.05 ** (1 / 10))
    assert cp_upper(10, 10) == 1.0


# ----------------------------------------------------------------------------- option B resampling
VALUES = np.array([10.0, 20.0, 30.0, 40.0, 55.0])
FAM = "sigtests/versions"


def test_family_rng_seed_is_base_seed_and_crc32() -> None:
    assert zlib.crc32(FAM.encode()) == 37512926          # pinned: crc32 is stable across processes, hash() is not
    assert family_rng(FAM).bit_generator.seed_seq.entropy == [0, 37512926]
    for fid in FAMILY_IDS:
        assert family_rng(fid).bit_generator.seed_seq.entropy == [0, zlib.crc32(fid.encode())]


def test_family_rng_refuses_unregistered_ids() -> None:
    with pytest.raises(ValueError, match="unknown test family"):
        family_rng("sigtests/versions ")


def test_family_rng_is_fresh_each_call() -> None:
    a, b = family_rng(FAM), family_rng(FAM)
    assert a is not b
    np.testing.assert_array_equal(a.integers(0, 1 << 30, 5), b.integers(0, 1 << 30, 5))


def test_resample_idx_is_one_draw_from_the_family_generator() -> None:
    idx = resample_idx(FAM, 5, b=7)
    assert idx.shape == (7, 5)
    np.testing.assert_array_equal(idx, family_rng(FAM).integers(0, 5, (7, 5)))
    np.testing.assert_array_equal(idx, resample_idx(FAM, 5, b=7))   # same family and n: identical matrix
    assert resample_idx(FAM, 58).shape == (B, 58)                       # default b = B


def test_resample_idx_differs_between_families() -> None:
    a = resample_idx("sigtests/join/H1/LCM vs PG10", 58)
    b = resample_idx("sigtests/join/H1/PG version vs PG10", 58)
    assert not np.array_equal(a, b)


def test_boot_applies_idx_row_by_row() -> None:
    idx = np.array([[0, 0, 0, 0, 0], [4, 3, 2, 1, 0], [1, 1, 4, 4, 2]])
    expected = [10.0, VALUES.mean(), np.mean([20, 20, 55, 55, 30])]
    np.testing.assert_array_equal(boot(list(VALUES), np.mean, idx=idx), expected)
    np.testing.assert_array_equal(boot_fast(VALUES, np.mean, idx=idx), expected)


def test_boot_and_boot_fast_agree_on_family_indices() -> None:
    idx = resample_idx(FAM, 5, b=50)
    np.testing.assert_allclose(boot(list(VALUES), np.sum, idx=idx), boot_fast(VALUES, np.sum, idx=idx))


def test_boot_rejects_idx_of_wrong_width() -> None:
    idx = resample_idx(FAM, 4, b=3)
    with pytest.raises(ValueError, match="does not fit 5 units"):
        boot_fast(VALUES, np.sum, idx=idx)
    with pytest.raises(ValueError, match="does not fit 5 units"):
        boot(list(VALUES), np.sum, idx=idx)


def test_boot_needs_idx() -> None:
    with pytest.raises(TypeError):
        boot_fast(VALUES, np.sum)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        boot_fast(VALUES, np.sum, rng=np.random.default_rng(0))  # type: ignore[call-arg]
