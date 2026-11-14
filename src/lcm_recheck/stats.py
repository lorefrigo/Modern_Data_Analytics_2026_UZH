"""Statistics helpers shared by the significance tests (sigtests.py) and the bias tests (bias_*.py).

No module-level random generator (option B of the statistical protocol). Each test family has its own
generator, family_rng(family_id), seeded with [BASE_SEED, crc32(family_id)]; resample_idx() draws the family's
one (B, n) resample-index matrix from it. The matrix depends only on (family_id, n), so every model of a family
resamples the same units, and no result depends on which tests ran before.

Usage:
  from lcm_recheck.stats import holm, boot_fast, ci_p, resample_idx
  holm([0.01, 0.04, 0.03])                                       # Holm-adjusted p-values, in input order
  samples = boot_fast(d, np.mean, idx=resample_idx("sigtests/access/T2a/LCM vs PG10", len(d)))
  observed, lo, hi, p = ci_p(samples, d.mean())
"""
import zlib
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.stats import beta, binomtest, norm, wilcoxon

from lcm_recheck.config import BASE_SEED, FAMILY_IDS, B


def holm(p: Sequence[float] | np.ndarray) -> np.ndarray:
    """Holm-adjusted p-values (step-down, monotone, capped at 1), in input order."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = run
    return adj


# ----------------------------------------------------------------------------- bootstrap
def family_rng(family_id: str) -> np.random.Generator:
    """A fresh generator for one test family, seeded with [BASE_SEED, crc32(family_id)].

    crc32, not hash(): hash() of a str is randomised per process. Only ids in config.FAMILY_IDS are accepted."""
    if family_id not in FAMILY_IDS:
        raise ValueError(f"unknown test family {family_id!r}; register it in config.FAMILY_IDS")
    return np.random.default_rng([BASE_SEED, zlib.crc32(family_id.encode())])


def resample_idx(family_id: str, n: int, b: int = B) -> np.ndarray:
    """The family's (b, n) resample-index matrix: one draw family_rng(family_id).integers(0, n, (b, n)).

    Row r is resample r (n units drawn with replacement). Same (family_id, n, b) -> identical matrix."""
    return family_rng(family_id).integers(0, n, (b, n))


def _check_idx(idx: np.ndarray, n: int) -> None:
    if idx.ndim != 2 or idx.shape[1] != n:
        raise ValueError(f"resample-index matrix of shape {idx.shape} does not fit {n} units")


def boot(per_unit: Sequence[Any], stat: Callable[[list[Any]], float], *, idx: np.ndarray) -> np.ndarray:
    """Bootstrap of per-unit objects (lists of arrays, tuples): stat of each resample, one per row of idx."""
    _check_idx(idx, len(per_unit))
    return np.array([stat([per_unit[j] for j in row]) for row in idx])


def boot_fast(values: Sequence[float] | np.ndarray, stat: Callable[..., np.ndarray], *,
              idx: np.ndarray) -> np.ndarray:
    """Vectorised bootstrap of numeric per-unit values: stat(values[idx], axis=1)."""
    values = np.asarray(values, float)
    _check_idx(idx, len(values))
    return stat(values[idx], axis=1)


def boot_p(samples: np.ndarray) -> float:
    """Two-sided bootstrap p-value for H0: statistic = 0."""
    return min(1.0, 2 * min((samples <= 0).mean(), (samples >= 0).mean()))


def ci(samples: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Percentile interval at level 1 - alpha: [lo, hi] (alpha = 0.05 gives exactly [2.5, 97.5])."""
    return np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def ci_p(samples: np.ndarray, observed: float, alpha: float = 0.05) -> tuple[float, float, float, float]:
    """(observed, lo, hi, p): percentile interval at level 1 - alpha and the two-sided bootstrap p-value."""
    lo, hi = ci(samples, alpha)
    p = boot_p(samples)
    return observed, lo, hi, p


# ----------------------------------------------------------------------------- exact tests
def wilcoxon_p(w: Sequence[float] | np.ndarray) -> tuple[float, int]:
    """Wilcoxon signed-rank p-value (zero_method="wilcox") and the number of non-zero differences;
    NaN if fewer than 6 differences are non-zero."""
    nz = int((np.abs(w) > 1e-12).sum())
    p = wilcoxon(w, zero_method="wilcox").pvalue if nz >= 6 else np.nan
    return p, nz


def mcnemar(b: int, c: int) -> float:
    """Exact McNemar test: under H0 each of the b + c discordant points is a fair coin flip."""
    return binomtest(int(b), int(b + c), 0.5).pvalue if b + c else 1.0


# ----------------------------------------------------------------------------- proportions
def wilson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for k successes out of n; (nan, nan) for n = 0."""
    if n == 0:
        return np.nan, np.nan
    z, p = norm.ppf(1 - (1 - conf) / 2), k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def cp_upper(k: int, n: int, conf: float = 0.95) -> float:
    """One-sided Clopper-Pearson upper bound: the largest rate still compatible with seeing only k of n."""
    return 1.0 if k >= n else float(beta.ppf(conf, k + 1, n - k))
