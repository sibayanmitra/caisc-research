"""Sanity tests for the statistical-test module."""

import numpy as np

from src.stats import (
    cluster_bootstrap_flip_rate,
    cohens_h,
    holm_bonferroni,
    non_overlapping_ci,
    permutation_test_accuracy,
)


def test_permutation_test_recovers_chance_null():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, size=300)
    y_pred = rng.integers(0, 3, size=300)
    obs, p99, p = permutation_test_accuracy(y_true, y_pred, n_perm=2000, rng=rng)
    assert 0.20 < obs < 0.45
    assert p > 0.01  # under independent labels, should not reject


def test_permutation_test_rejects_perfect_predictor():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, size=300)
    y_pred = y_true.copy()
    obs, _, p = permutation_test_accuracy(y_true, y_pred, n_perm=2000, rng=rng)
    assert obs == 1.0
    assert p < 0.001


def test_cohens_h_known():
    # h(0.5, 0.5) = 0
    assert abs(cohens_h(0.5, 0.5)) < 1e-10
    # h(0.8, 0.2) > 1 (large effect)
    assert cohens_h(0.8, 0.2) > 1.0


def test_cluster_bootstrap_returns_ci():
    rng = np.random.default_rng(0)
    flips = np.concatenate([np.ones(40), np.zeros(60)])
    clusters = np.concatenate([np.repeat(np.arange(20), 5)])
    obs, lo, hi = cluster_bootstrap_flip_rate(flips, clusters, n_boot=500, rng=rng)
    assert 0 <= lo <= obs <= hi <= 1
    assert obs == 0.4


def test_holm_bonferroni_step_down():
    # 3 p-values: 0.01, 0.03, 0.08; α = 0.05
    # Holm: smallest must beat 0.05/3=0.0167; second 0.05/2=0.025; third 0.05/1=0.05
    p = [0.01, 0.03, 0.08]
    rej = holm_bonferroni(p, alpha=0.05)
    assert rej == [True, False, False]


def test_non_overlapping_ci():
    assert non_overlapping_ci((0.1, 0.3), (0.4, 0.5))
    assert not non_overlapping_ci((0.1, 0.4), (0.3, 0.5))
