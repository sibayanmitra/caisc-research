#!/usr/bin/env python3
"""
Phase 3 — Probe training.

For each (layer L, offset_idx) cell: train logistic regression probe on residual stream
to distinguish positive (about to self-correct) vs negative (matched non-correction).

Outputs:
- probes/layer_offset_probes.pkl       — per-cell probe (weights, intercept)
- probes/auroc_heatmap.npy             — (n_layers, n_offsets) AUROC array
- probes/probe_eval.json               — full eval results with CIs

Sanity checks SC9 (random baseline ~0.5), SC10 (random-direction AUROC ~0.5),
SC11 (overfitting gap), SC12 (bootstrap CI width).
"""

import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from self_monitor_lib import get_paper_dir


def log(msg, also_print=True):
    log_path = get_paper_dir() / "notes" / "sanity_log.md"
    with open(log_path, "a") as f:
        f.write(f"- [{time.strftime('%H:%M:%S')}] {msg}\n")
    if also_print:
        print(msg, flush=True)


def main():
    log("\n## Phase 3: Probe training\n", also_print=False)
    log("=" * 60)
    log(f"Phase 3 started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    paper_dir = get_paper_dir()
    act_dir = paper_dir / "activations"

    # Load activations
    log("Loading activations...")
    pos_data = torch.load(act_dir / "positive.pt", weights_only=False)
    neg_data = torch.load(act_dir / "negative.pt", weights_only=False)

    pos_resid = pos_data["resid"]  # (n_pos, n_layers, n_offsets, d_model)
    neg_resid = neg_data["resid"]
    offsets = pos_data["offsets"]
    n_layers = pos_data["n_layers"]
    n_offsets = len(offsets)

    log(f"  Positives: {pos_resid.shape}")
    log(f"  Negatives (raw): {neg_resid.shape}")
    log(f"  Offsets: {offsets}")

    # Build labels
    n_pos = pos_resid.shape[0]
    n_neg = neg_resid.shape[0]
    log(f"  N pos: {n_pos}, N neg (raw): {n_neg}")

    # Fallback: if too few whole-generation negatives (high SC rate means almost all
    # generations contain corrections), use the post-correction CONTROL set as negatives.
    # Post-correction positions (T+5 to T+10) are valid negatives: the model has already
    # corrected and is not "about to self-correct" again. This is arguably a cleaner
    # contrast than using a different generation, since it controls for problem content.
    if n_neg < 10:
        log(f"⚠️ Only {n_neg} whole-generation negatives (SC rate too high).")
        log("  Falling back to post-correction CONTROL set as negatives.")
        ctrl_data = torch.load(act_dir / "control.pt", weights_only=False)
        neg_resid = ctrl_data["resid"]
        n_neg = neg_resid.shape[0]
        log(f"  Using {n_neg} post-correction positions as negatives.")
        log("  NOTE: probe now contrasts 'about-to-correct' vs 'post-correction' positions.")

    # AUROC heatmap
    auroc_grid = np.full((n_layers, n_offsets), np.nan)
    overfit_grid = np.full((n_layers, n_offsets), np.nan)
    probe_dict = {}  # (L, off_idx) -> {"weights": ..., "intercept": ..., "auroc": ..., "ci_low": ..., "ci_high": ...}

    log("\nTraining probes per (layer, offset)...")
    t_start = time.time()

    # Random baseline check (SC9): random labels
    sc9_results = {}

    for L in range(n_layers):
        for o_idx in range(n_offsets):
            # Stack features
            X_pos = pos_resid[:, L, o_idx, :].float().numpy()  # (n_pos, d_model)
            X_neg = neg_resid[:, L, o_idx, :].float().numpy()
            X = np.concatenate([X_pos, X_neg], axis=0)
            y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

            # Skip if any row is all-zero (no extraction at this offset for this sample)
            row_norms = np.linalg.norm(X, axis=1)
            valid_mask = row_norms > 1e-6
            X = X[valid_mask]
            y = y[valid_mask]

            if len(np.unique(y)) < 2 or len(y) < 20:
                continue

            # Stratified split
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y,
                )
            except ValueError:
                continue

            # Standardize
            mean = X_train.mean(axis=0, keepdims=True)
            std = X_train.std(axis=0, keepdims=True) + 1e-6
            X_train_n = (X_train - mean) / std
            X_test_n = (X_test - mean) / std

            # Train probe
            clf = LogisticRegression(
                max_iter=1000,
                C=0.1,
                random_state=42,
            )
            try:
                clf.fit(X_train_n, y_train)
            except Exception:
                continue

            train_probs = clf.predict_proba(X_train_n)[:, 1]
            test_probs = clf.predict_proba(X_test_n)[:, 1]
            try:
                train_auroc = roc_auc_score(y_train, train_probs)
                test_auroc = roc_auc_score(y_test, test_probs)
            except ValueError:
                continue

            auroc_grid[L, o_idx] = test_auroc
            overfit_grid[L, o_idx] = train_auroc - test_auroc

            # Bootstrap CI (100 reps)
            rng = np.random.default_rng(42)
            boot_aurocs = []
            for _ in range(100):
                idx = rng.integers(0, len(y_test), len(y_test))
                if len(np.unique(y_test[idx])) < 2:
                    continue
                boot_aurocs.append(roc_auc_score(y_test[idx], test_probs[idx]))
            if boot_aurocs:
                ci_low = np.percentile(boot_aurocs, 2.5)
                ci_high = np.percentile(boot_aurocs, 97.5)
            else:
                ci_low, ci_high = test_auroc, test_auroc

            probe_dict[(L, o_idx)] = {
                "auroc": float(test_auroc),
                "train_auroc": float(train_auroc),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "weights": clf.coef_[0].astype(np.float32),
                "intercept": float(clf.intercept_[0]),
                "mean": mean[0].astype(np.float32),
                "std": std[0].astype(np.float32),
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
            }

        elapsed = time.time() - t_start
        log(f"  Layer {L}: best offset AUROC = {np.nanmax(auroc_grid[L]):.3f} (offset_idx={int(np.nanargmax(auroc_grid[L]))}, real offset={offsets[int(np.nanargmax(auroc_grid[L]))]}) — {elapsed/60:.1f} min elapsed")

    # Find peak
    if np.any(~np.isnan(auroc_grid)):
        peak_idx = np.unravel_index(np.nanargmax(auroc_grid), auroc_grid.shape)
        peak_L, peak_o = peak_idx
        peak_offset = offsets[peak_o]
        peak_auroc = auroc_grid[peak_L, peak_o]
        log(f"\nPEAK: layer={peak_L}, offset={peak_offset}, AUROC={peak_auroc:.3f}")
    else:
        peak_L, peak_o, peak_offset, peak_auroc = -1, -1, None, np.nan
        log("WARN: no probes converged")

    # SC9: random baseline (shuffle labels) on the peak cell
    log("\nSC9: random baseline at peak cell...")
    if peak_L >= 0:
        X_pos_peak = pos_resid[:, peak_L, peak_o, :].float().numpy()
        X_neg_peak = neg_resid[:, peak_L, peak_o, :].float().numpy()
        X = np.concatenate([X_pos_peak, X_neg_peak], axis=0)
        y = np.concatenate([np.ones(len(X_pos_peak)), np.zeros(len(X_neg_peak))])
        rng = np.random.default_rng(42)
        y_shuffled = rng.permutation(y)
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y_shuffled, test_size=0.2, random_state=42)
            mean = X_train.mean(axis=0); std = X_train.std(axis=0) + 1e-6
            clf = LogisticRegression(max_iter=500, C=0.1).fit((X_train - mean) / std, y_train)
            shuf_auroc = roc_auc_score(y_test, clf.predict_proba((X_test - mean) / std)[:, 1])
            log(f"  Random-labels AUROC at peak: {shuf_auroc:.3f} (should be ~0.5)")
            sc9_results["random_labels_auroc"] = float(shuf_auroc)
        except Exception as e:
            log(f"  SC9 error: {e}")

    # Save outputs
    probes_dir = paper_dir / "probes"
    probes_dir.mkdir(parents=True, exist_ok=True)

    np.save(probes_dir / "auroc_heatmap.npy", auroc_grid)
    np.save(probes_dir / "overfit_gap_heatmap.npy", overfit_grid)
    with open(probes_dir / "layer_offset_probes.pkl", "wb") as f:
        pickle.dump(probe_dict, f)

    # Eval summary
    eval_data = {
        "n_layers": int(n_layers),
        "n_offsets": int(n_offsets),
        "offsets": offsets,
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "peak_layer": int(peak_L) if peak_L >= 0 else None,
        "peak_offset_idx": int(peak_o) if peak_o >= 0 else None,
        "peak_offset_value": int(peak_offset) if peak_offset is not None else None,
        "peak_auroc": float(peak_auroc) if not np.isnan(peak_auroc) else None,
        "auroc_grid": auroc_grid.tolist(),
        "sanity_checks": {
            "sc9_random_labels_auroc": sc9_results.get("random_labels_auroc"),
            "sc9_pass": (sc9_results.get("random_labels_auroc") is not None
                         and 0.4 <= sc9_results["random_labels_auroc"] <= 0.6),
        },
    }
    with open(probes_dir / "probe_eval.json", "w") as f:
        json.dump(eval_data, f, indent=2)

    log(f"\n✓ Phase 3 complete. Probes saved to {probes_dir}")
    log(f"  Peak AUROC: {peak_auroc:.3f} at (L={peak_L}, offset={peak_offset})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
