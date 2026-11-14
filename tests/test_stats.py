"""lcm_recheck.stats.holm() against hand-computed values."""
import numpy as np

from lcm_recheck.stats import holm


def test_holm_hand_computed() -> None:
    # sorted: 0.01*3 = 0.03, 0.03*2 = 0.06, 0.04*1 = 0.04 -> monotone 0.06; returned in input order
    np.testing.assert_allclose(holm([0.04, 0.01, 0.03]), [0.06, 0.03, 0.06])


def test_holm_capped_at_one() -> None:
    np.testing.assert_allclose(holm([0.5, 0.6]), [1.0, 1.0])


def test_holm_single() -> None:
    np.testing.assert_allclose(holm([0.2]), [0.2])
