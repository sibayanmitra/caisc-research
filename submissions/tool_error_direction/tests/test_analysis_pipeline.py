"""Integration test: probe + direction extraction on synthetic activations.

Validates that the analysis path produces sensible results when the signal is real
(separable classes) and chance-level when the signal is random.
"""

import numpy as np

from src.probe import (
    CLASSES,
    encode_labels,
    fit_probes_all_layers,
    mean_difference_directions,
    subspace_pca_top1_variance,
)


def _synthetic_data(n_per_class: int = 100, n_layers: int = 4, d: int = 64, signal_layer: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = n_per_class * 3
    H = rng.standard_normal((n, n_layers, d)).astype(np.float32) * 0.5
    labels = []
    # Embed separable signal in `signal_layer` only
    means = rng.standard_normal((3, d)).astype(np.float32) * 2.0
    for ci, c in enumerate(CLASSES):
        for _ in range(n_per_class):
            labels.append(c)
    labels_arr = np.array(labels)
    rng.shuffle(np.arange(n))  # not used directly; CLASSES order kept
    perm = rng.permutation(n)
    labels_arr = labels_arr[perm]
    H = H[perm]
    # add signal to signal_layer for each class
    for i, c in enumerate(labels_arr):
        ci = CLASSES.index(c)
        H[i, signal_layer] += means[ci]
    y = encode_labels(labels_arr.tolist())
    return H, y


def test_probe_recovers_signal_layer():
    """A probe trained at the signal layer should be substantially above chance; other layers near chance."""
    H, y = _synthetic_data(n_per_class=120, n_layers=4, d=64, signal_layer=2, seed=42)
    # 70/30 train/test split
    n = H.shape[0]
    rng = np.random.default_rng(7)
    perm = rng.permutation(n)
    cut = int(n * 0.7)
    tr, te = perm[:cut], perm[cut:]
    results = fit_probes_all_layers(H[tr], y[tr], H[te], y[te], seed=0)
    accs = [r.test_acc for r in results]
    # signal layer >> chance; non-signal layers near chance (0.33)
    assert accs[2] > 0.7, f"signal layer too weak: {accs}"
    assert all(a < 0.5 for i, a in enumerate(accs) if i != 2), f"non-signal layers leaked: {accs}"


def test_probe_chance_on_random_labels():
    H, y = _synthetic_data(n_per_class=120, n_layers=2, d=32, signal_layer=0, seed=1)
    rng = np.random.default_rng(2)
    y_shuf = rng.permutation(y)
    n = H.shape[0]
    perm = rng.permutation(n)
    cut = int(n * 0.7)
    tr, te = perm[:cut], perm[cut:]
    results = fit_probes_all_layers(H[tr], y_shuf[tr], H[te], y_shuf[te], seed=0)
    accs = [r.test_acc for r in results]
    assert all(a < 0.5 for a in accs), f"shuffled probe should be near chance: {accs}"


def test_mean_diff_directions_separable():
    H, y = _synthetic_data(n_per_class=120, n_layers=4, d=64, signal_layer=2, seed=3)
    dirs = mean_difference_directions(H, y, layer=2)
    # All 3 directions should have substantial norm
    for c in CLASSES:
        assert np.linalg.norm(dirs[c]) > 1.0, f"direction {c} too small"


def test_subspace_pca_top1_high_for_separable():
    H, y = _synthetic_data(n_per_class=120, n_layers=4, d=64, signal_layer=2, seed=4)
    var = subspace_pca_top1_variance(H, y, layer=2)
    # 3 separable classes — top-1 should explain a significant share
    assert var > 0.30, f"PCA top1 too low: {var}"
