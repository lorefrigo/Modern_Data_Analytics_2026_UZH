"""Statistics helpers shared by the significance tests (sigtests.py) and the bias tests (bias_*.py).

No random generator lives here: the bootstrap helpers draw from the generator they are given.

Usage:
  from lcm_recheck.stats import holm, boot_fast, ci_p
  holm([0.01, 0.04, 0.03])                     # Holm-adjusted p-values, in input order
  samples = boot_fast(d, np.mean, rng=rng)     # rng: a numpy Generator owned by the caller
  observed, lo, hi, p = ci_p(samples, d.mean())
"""
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.stats import beta, binomtest, norm, wilcoxon

from lcm_recheck.config import B


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
def boot(per_query: Sequence[Any], stat: Callable[[list[Any]], float], *, rng: np.random.Generator,
         b: int = B) -> np.ndarray:
    """Resample units with replacement; per_query is a list of per-unit objects (used by Task 1).

    Draws b times rng.integers(0, n, n), one resample after the other."""
    n = len(per_query)
    return np.array([stat([per_query[j] for j in rng.integers(0, n, n)]) for _ in range(b)])


def boot_fast(values: Sequence[float] | np.ndarray, stat: Callable[..., np.ndarray], *,
              rng: np.random.Generator, b: int = B) -> np.ndarray:
    """Vectorised version for numeric arrays (used by Tasks 2 and 3); one draw rng.integers(0, n, (b, n))."""
    values = np.asarray(values, float)
    idx = rng.integers(0, len(values), (b, len(values)))
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
