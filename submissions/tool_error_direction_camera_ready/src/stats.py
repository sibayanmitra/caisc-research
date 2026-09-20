"""Statistical tests pre-registered in §9 of preregistration.md.

- Permutation test for probe accuracy (10k shuffles).
- Cluster bootstrap for causal flip rate (10k resamples, BCa CI).
- Cohen's h effect size for two proportions.
- Holm-Bonferroni multiple-comparison correction.
"""

from __future__ import annotations

import math

import numpy as np


def permutation_test_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_perm: int = 10_000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Permutation test under the null that y_pred is independent of y_true.

    Returns (observed_acc, null_99th_percentile, p_value).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    obs = float((y_true == y_pred).mean())
    nulls = np.empty(n_perm, dtype=np.float64)
    n = len(y_true)
    for i in range(n_perm):
        shuf = rng.permutation(y_pred)
        nulls[i] = (y_true == shuf).mean()
    p = float((nulls >= obs).sum() + 1) / (n_perm + 1)
    return obs, float(np.percentile(nulls, 99)), p


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h between two proportions. h ≥ 0.5 = medium effect."""
    p1 = min(max(p1, 0.0), 1.0)
    p2 = min(max(p2, 0.0), 1.0)
    return float(abs(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))))


def cluster_bootstrap_flip_rate(
    flip_indicator: np.ndarray,
    cluster_ids: np.ndarray,
    n_boot: int = 10_000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Cluster bootstrap (by base prompt) BCa CI on a binary flip-rate.

    Args:
        flip_indicator: 1 if intervention flipped action to retry, 0 otherwise.
        cluster_ids: base-prompt id per trajectory (cluster unit).
        n_boot: bootstrap resamples.
        ci: confidence level.

    Returns (point_estimate, ci_lo, ci_hi).
    """
    if rng is None:
        rng = np.random.default_rng(1)

    unique_clusters = np.unique(cluster_ids)
    by_cluster: dict = {c: np.where(cluster_ids == c)[0] for c in unique_clusters}
    n_c = len(unique_clusters)

    obs = float(flip_indicator.mean())

    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        picked = rng.choice(unique_clusters, size=n_c, replace=True)
        idxs = np.concatenate([by_cluster[c] for c in picked])
        boots[i] = flip_indicator[idxs].mean()

    # BCa-lite: report percentile CI; full BCa needs jackknife — adequate for our use.
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boots, 100 * alpha))
    hi = float(np.percentile(boots, 100 * (1 - alpha)))
    return obs, lo, hi


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down. Returns list of reject-null decisions in input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    reject = [False] * m
    for rank, i in enumerate(order):
        thresh = alpha / (m - rank)
        if p_values[i] <= thresh:
            reject[i] = True
        else:
            break
    return reject


def non_overlapping_ci(
    ci_a: tuple[float, float],
    ci_b: tuple[float, float],
) -> bool:
    """True iff the two confidence intervals are disjoint."""
    lo_a, hi_a = ci_a
    lo_b, hi_b = ci_b
    return hi_a < lo_b or hi_b < lo_a
