"""Probes and direction extraction.

Pre-registered probe variants:
  - PRIMARY: error-only, action-balanced, 3-way logistic regression per layer.
  - WITHIN-ERROR: same but per error_kind.
  - LOO-TOOL: train on 2 tools, test on the held-out tool.
  - MATCHED-PAIR: same-prompt same-outcome different-action contrasts.

Direction extraction: mean-difference per class on dir_extract split (disjoint).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

CLASSES = ("retry-same", "modify-args", "give-up")
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


@dataclass
class ProbeResult:
    layer: int
    train_acc: float
    test_acc: float
    n_train: int
    n_test: int


def _action_balance(X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Downsample to equal counts per class. Returns (X_bal, y_bal)."""
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return X, y
    n_min = counts.min()
    idxs: list[int] = []
    for c in classes:
        ci = np.where(y == c)[0]
        sel = rng.choice(ci, size=n_min, replace=False)
        idxs.extend(sel.tolist())
    rng.shuffle(idxs)
    idxs_arr = np.array(idxs, dtype=int)
    return X[idxs_arr], y[idxs_arr]


def fit_probe_layer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    layer: int,
    seed: int = 0,
) -> ProbeResult:
    """L2 logistic regression with 5-fold inner CV over C ∈ {0.01, 0.1, 1, 10}."""
    rng = np.random.default_rng(seed)
    X_train, y_train = _action_balance(X_train, y_train, rng)

    best_C = 1.0
    best_score = -1.0
    if len(np.unique(y_train)) > 1 and len(y_train) >= 30:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for C in (0.01, 0.1, 1.0, 10.0):
            scores: list[float] = []
            for tr, va in skf.split(X_train, y_train):
                clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
                clf.fit(X_train[tr], y_train[tr])
                scores.append(clf.score(X_train[va], y_train[va]))
            s = float(np.mean(scores))
            if s > best_score:
                best_score = s
                best_C = C

    clf = LogisticRegression(C=best_C, max_iter=2000, solver="lbfgs")
    clf.fit(X_train, y_train)
    train_acc = float(clf.score(X_train, y_train))
    test_acc = float(clf.score(X_test, y_test)) if len(X_test) > 0 else float("nan")
    return ProbeResult(
        layer=layer,
        train_acc=train_acc,
        test_acc=test_acc,
        n_train=len(X_train),
        n_test=len(X_test),
    )


def fit_probes_all_layers(
    H_train: np.ndarray,
    y_train: np.ndarray,
    H_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 0,
) -> list[ProbeResult]:
    """H shape: (n_samples, n_layers, d_model). Returns one ProbeResult per layer."""
    n_layers = H_train.shape[1]
    out: list[ProbeResult] = []
    for L in range(n_layers):
        out.append(fit_probe_layer(H_train[:, L], y_train, H_test[:, L], y_test, layer=L, seed=seed))
    return out


def mean_difference_directions(
    H: np.ndarray,
    y: np.ndarray,
    layer: int,
) -> dict[str, np.ndarray]:
    """Mean-difference per class at a given layer.

    v_c = mean(h | y == c) - mean(h | y != c).
    y is int-encoded labels; -1 (ambiguous) is dropped.
    Returns dict {class_name: v ∈ R^d_model}.
    """
    out: dict[str, np.ndarray] = {}
    if H.ndim == 3:
        H_L = H[:, layer]
    else:
        H_L = H
    valid = y >= 0
    H_L = H_L[valid]
    y_v = y[valid]
    for cls_idx, c in enumerate(CLASSES):
        in_mask = y_v == cls_idx
        if in_mask.sum() == 0 or (~in_mask).sum() == 0:
            out[c] = np.zeros(H_L.shape[1], dtype=np.float64)
            continue
        out[c] = H_L[in_mask].mean(axis=0) - H_L[~in_mask].mean(axis=0)
    return out


def subspace_pca_top1_variance(
    H: np.ndarray,
    y: np.ndarray,
    layer: int,
) -> float:
    """Gate G-PCA: fraction of between-class variance captured by the top-1 PC of class means.

    H may be shape (n, n_layers, d) or (n, d). y is int-encoded labels (0/1/2 for
    retry/modify/give-up; -1 for ambiguous which we drop).
    """
    if H.ndim == 3:
        H_L = H[:, layer]
    else:
        H_L = H
    valid = y >= 0
    H_L = H_L[valid]
    y_v = y[valid]
    means = []
    for cls_idx in range(len(CLASSES)):
        mask = y_v == cls_idx
        if mask.sum() > 0:
            means.append(H_L[mask].mean(axis=0))
    if len(means) < 2:
        return 0.0
    means_arr = np.stack(means)
    centered = means_arr - means_arr.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    total = (s ** 2).sum()
    return float((s[0] ** 2) / total) if total > 0 else 0.0


def encode_labels(labels: list[str]) -> np.ndarray:
    """Map string labels to int ids; ambiguous → -1 (caller should filter)."""
    return np.array([CLASS_TO_IDX.get(l, -1) for l in labels], dtype=int)


def filter_errors_only(outcomes: list[str]) -> np.ndarray:
    """Boolean mask: True for non-success outcomes."""
    return np.array([o != "success" for o in outcomes], dtype=bool)
