"""Statistics helpers shared by the significance tests.

Usage:
  from lcm_recheck.stats import holm
  holm([0.01, 0.04, 0.03])   # Holm-adjusted p-values, in input order
"""
from collections.abc import Sequence

import numpy as np


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
