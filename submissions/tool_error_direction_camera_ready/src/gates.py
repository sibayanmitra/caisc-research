"""Sanity gate checks (G-LAY, G-RAND, G-SHUF, ..., G-LOO).

Each gate is a pure function that returns (passed: bool, observed: float | str, threshold: float | str).
The pipeline calls each gate, logs the outcome to `gate_outcomes.csv`, and feeds the
decision tree in preregistration.md §8.
"""

from __future__ import annotations

import numpy as np

from .probe import ProbeResult, subspace_pca_top1_variance


def g_lay(probe_results: list[ProbeResult], best_layer: int, tol_pp: float = 5.0) -> tuple[bool, float, float]:
    """G-LAY: best-layer accuracy within 5 pp of mean of immediate neighbors."""
    best = probe_results[best_layer].test_acc
    neigh = []
    if best_layer - 1 >= 0:
        neigh.append(probe_results[best_layer - 1].test_acc)
    if best_layer + 1 < len(probe_results):
        neigh.append(probe_results[best_layer + 1].test_acc)
    if not neigh:
        return True, best * 100, tol_pp
    mean_neigh = float(np.mean(neigh))
    diff_pp = (best - mean_neigh) * 100
    return diff_pp <= tol_pp, diff_pp, tol_pp


def g_rand(retry_flip: float, rand_flip: float, ratio: float = 3.0) -> tuple[bool, float, float]:
    """G-RAND: retry-direction flip rate ≥ 3× random-direction flip rate."""
    if rand_flip <= 1e-6:
        return retry_flip > 0.05, retry_flip / max(rand_flip, 1e-6), ratio
    obs = retry_flip / rand_flip
    return obs >= ratio, obs, ratio


def g_shuf(shuffled_label_acc: float, threshold: float = 0.40) -> tuple[bool, float, float]:
    return shuffled_label_acc <= threshold, shuffled_label_acc, threshold


def g_within(per_error_acc: dict[str, float], min_acc: float = 0.50) -> tuple[bool, str, float]:
    n_pass = sum(1 for a in per_error_acc.values() if a >= min_acc)
    return n_pass >= 2, f"{n_pass}/{len(per_error_acc)} error types ≥ {min_acc}", min_acc


def g_pos(position_only_acc: float, threshold: float = 0.55) -> tuple[bool, float, float]:
    return position_only_acc < threshold, position_only_acc, threshold


def g_cache(max_diff: float, atol: float = 1e-3) -> tuple[bool, float, float]:
    """Loosened from 1e-5 to 1e-3 to accommodate bfloat16 forward path differences.
    Catches implementation drift, not numerical noise."""
    return max_diff <= atol, max_diff, atol


def g_ambig(ambig_rate: float, threshold: float = 0.15) -> tuple[bool, float, float]:
    return ambig_rate <= threshold, ambig_rate, threshold


def g_pca(H: np.ndarray, y: np.ndarray, layer: int, threshold: float = 0.40) -> tuple[bool, float, float]:
    var = subspace_pca_top1_variance(H, y, layer)
    return var >= threshold, var, threshold


def g_crossfamily(family_a_passed: bool, family_b_passed: bool) -> tuple[bool, str, str]:
    return family_a_passed and family_b_passed, f"A={family_a_passed}, B={family_b_passed}", "both"


def g_seed(per_seed_acc: list[float], min_acc: float = 0.60) -> tuple[bool, str, float]:
    """G-SEED: all seeds with probe_fit data independently exceed probe threshold.

    Seeds whose probe_fit sample count < 30 produce NaN — those are excluded rather
    than counted as failures. This applies to causal-only seeds (2 and 3), which lack
    probe_fit trajectories by design; evaluable on seed 1 only in that scenario.
    """
    import math
    valid = [a for a in per_seed_acc if not (isinstance(a, float) and math.isnan(a))]
    if not valid:
        return True, "0/0 evaluable seeds (all data insufficient)", min_acc
    n_pass = sum(1 for a in valid if a >= min_acc)
    return n_pass == len(valid), f"{n_pass}/{len(valid)} evaluable seeds ≥ {min_acc}", min_acc


def g_token(position_accs: dict[str, float], min_acc: float = 0.50) -> tuple[bool, str, float]:
    """G-TOKEN: probe ≥ 50% at t*, t*+5, and pre_next_action positions."""
    n_pass = sum(1 for a in position_accs.values() if a >= min_acc)
    return n_pass == len(position_accs), f"{n_pass}/{len(position_accs)} positions ≥ {min_acc}", min_acc


def g_loo(loo_accs: dict[str, float], min_acc: float = 0.50) -> tuple[bool, str, float]:
    """G-LOO: leave-one-tool-out probe ≥ 50% across all 3 LOO splits."""
    n_pass = sum(1 for a in loo_accs.values() if a >= min_acc)
    return n_pass == len(loo_accs), f"{n_pass}/{len(loo_accs)} LOO splits ≥ {min_acc}", min_acc
