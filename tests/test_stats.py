"""lcm_recheck.stats against hand-computed values, and the exact draw pattern of the bootstrap helpers."""
import numpy as np
import pytest

from lcm_recheck.stats import (
    boot,
    boot_fast,
    boot_p,
    ci,
    ci_p,
    cp_upper,
    holm,
    mcnemar,
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


# ----------------------------------------------------------------------------- draw patterns (pinned)
VALUES = np.array([10.0, 20.0, 30.0, 40.0, 55.0])


def test_boot_draw_pattern() -> None:
    """boot draws b times rng.integers(0, n, n), in order, from the generator it is given."""
    ref = np.random.default_rng(0)
    expected = [np.mean([VALUES[j] for j in ref.integers(0, 5, 5)]) for _ in range(7)]
    rng = np.random.default_rng(0)
    np.testing.assert_array_equal(boot(list(VALUES), np.mean, rng=rng, b=7), expected)
    # the caller's generator has advanced exactly as far as the reference one
    assert rng.integers(0, 1 << 30) == ref.integers(0, 1 << 30)


def test_boot_fast_draw_pattern() -> None:
    """boot_fast draws once rng.integers(0, n, (b, n)) from the generator it is given."""
    ref = np.random.default_rng(0)
    expected = VALUES[ref.integers(0, 5, (7, 5))].sum(axis=1)
    rng = np.random.default_rng(0)
    np.testing.assert_array_equal(boot_fast(VALUES, np.sum, rng=rng, b=7), expected)
    assert rng.integers(0, 1 << 30) == ref.integers(0, 1 << 30)


def test_boot_needs_a_generator() -> None:
    with pytest.raises(TypeError):
        boot_fast(VALUES, np.sum)  # type: ignore[call-arg]
