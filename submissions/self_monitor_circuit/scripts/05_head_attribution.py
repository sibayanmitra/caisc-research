#!/usr/bin/env python3
"""
Phase 5 — Per-head attribution.

At peak layer (L*) and neighbors (L*-1, L*+1), train probes on each attention head's
z output independently. Rank heads by AUROC. Test top-k joint probes.

Outputs:
- probes/head_probes.pkl
- probes/head_auroc_per_layer.npy
- plots data for fig2_head_ranking.pdf

Sanity checks SC13 (right-skewed distribution), SC14 (top-k composition gain).
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
    log("\n## Phase 5: Per-head attribution\n", also_print=False)
    log("=" * 60)
    log(f"Phase 5 started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    paper_dir = get_paper_dir()
    probes_dir = paper_dir / "probes"
    act_dir = paper_dir / "activations"

    # Load probe eval to find peak layer/offset
    with open(probes_dir / "probe_eval.json") as f:
        eval_data = json.load(f)

    peak_L = eval_data["peak_layer"]
    peak_o_idx = eval_data["peak_offset_idx"]
    peak_offset_val = eval_data["peak_offset_value"]
    log(f"Peak (from Phase 3): L={peak_L}, offset_idx={peak_o_idx}, offset_value={peak_offset_val}, AUROC={eval_data['peak_auroc']:.3f}")

    if peak_L is None or peak_L < 0:
        log("✗ No peak layer found; aborting Phase 5")
        return 1

    # Load activations (head_z)
    log("Loading head_z activations...")
    pos_data = torch.load(act_dir / "positive.pt", weights_only=False)
    neg_data = torch.load(act_dir / "negative.pt", weights_only=False)
    # Fallback: use post-correction control set if whole-gen negatives are too few
    if neg_data["head_z"].shape[0] < 10:
        log("⚠️ Too few whole-gen negatives — using post-correction control set.")
        neg_data = torch.load(act_dir / "control.pt", weights_only=False)

    pos_z = pos_data["head_z"]  # (n_pos, n_layers, n_offsets, n_heads, d_head)
    neg_z = neg_data["head_z"]
    n_layers = pos_data["n_layers"]
    n_heads = pos_data["n_heads"]
    d_head = pos_data["d_head"]

    log(f"  Positives head_z: {pos_z.shape}")
    log(f"  Negatives head_z: {neg_z.shape}")

    # Layers to analyze: peak + neighbors
    layers_to_analyze = sorted(set([
        max(0, peak_L - 1),
        peak_L,
        min(n_layers - 1, peak_L + 1),
    ]))
    log(f"Layers to analyze: {layers_to_analyze}")

    head_aurocs = np.full((len(layers_to_analyze), n_heads), np.nan)
    head_probes = {}  # (layer, head) -> probe info

    # Per-head probes
    for li, L in enumerate(layers_to_analyze):
        log(f"\nLayer {L}:")
        for h in range(n_heads):
            X_pos = pos_z[:, L, peak_o_idx, h, :].float().numpy()  # (n_pos, d_head)
            X_neg = neg_z[:, L, peak_o_idx, h, :].float().numpy()
            X = np.concatenate([X_pos, X_neg], axis=0)
            y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])

            row_norms = np.linalg.norm(X, axis=1)
            mask = row_norms > 1e-6
            X = X[mask]; y = y[mask]
            if len(np.unique(y)) < 2 or len(y) < 20:
                continue

            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                mean = X_train.mean(axis=0)
                std = X_train.std(axis=0) + 1e-6
                clf = LogisticRegression(max_iter=500, C=0.1, random_state=42)
                clf.fit((X_train - mean) / std, y_train)
                preds = clf.predict_proba((X_test - mean) / std)[:, 1]
                auroc = roc_auc_score(y_test, preds)
                head_aurocs[li, h] = auroc
                head_probes[(L, h)] = {
                    "auroc": float(auroc),
                    "weights": clf.coef_[0].astype(np.float32),
                    "intercept": float(clf.intercept_[0]),
                    "mean": mean.astype(np.float32),
                    "std": std.astype(np.float32),
                }
            except Exception:
                continue

        # Summary for this layer
        finite = head_aurocs[li, ~np.isnan(head_aurocs[li])]
        if len(finite) > 0:
            top3 = sorted(zip(range(n_heads), head_aurocs[li]), key=lambda x: -x[1] if not np.isnan(x[1]) else 0)[:3]
            log(f"  median={np.median(finite):.3f}, max={np.max(finite):.3f}")
            log(f"  top-3 heads at L={L}: {[(h, f'{a:.3f}') for h, a in top3]}")

    # Get the peak layer's heads ranked
    peak_layer_idx = layers_to_analyze.index(peak_L)
    peak_layer_aurocs = head_aurocs[peak_layer_idx]

    # SC13: right-skewed (skewness > 0.5 or top 10% noticeably above median)
    finite = peak_layer_aurocs[~np.isnan(peak_layer_aurocs)]
    if len(finite) > 0:
        from scipy.stats import skew
        sc13_skew = skew(finite)
        log(f"\nSC13 (head AUROC distribution skewness at peak layer): {sc13_skew:.2f}")
        log(f"  PASS' if right-skewed (>0.5): {'PASS' if sc13_skew > 0.5 else 'WARN'}")

        median = np.median(finite)
        top_10_pct = np.sort(finite)[::-1][:max(1, n_heads // 10)]
        log(f"  Median: {median:.3f}, Top-10% mean: {np.mean(top_10_pct):.3f}")

    # Top-k joint probes
    log("\nTop-k joint probe analysis at peak layer...")
    ranked_heads = np.argsort(peak_layer_aurocs)[::-1]
    ranked_heads = [int(h) for h in ranked_heads if not np.isnan(peak_layer_aurocs[h])]

    joint_results = {}
    for k in [3, 5, 10]:
        top_k_heads = ranked_heads[:k]
        X_pos_joint = pos_z[:, peak_L, peak_o_idx, top_k_heads, :].reshape(pos_z.shape[0], -1).float().numpy()
        X_neg_joint = neg_z[:, peak_L, peak_o_idx, top_k_heads, :].reshape(neg_z.shape[0], -1).float().numpy()
        X = np.concatenate([X_pos_joint, X_neg_joint], axis=0)
        y = np.concatenate([np.ones(len(X_pos_joint)), np.zeros(len(X_neg_joint))])
        row_norms = np.linalg.norm(X, axis=1)
        mask = row_norms > 1e-6
        X = X[mask]; y = y[mask]
        if len(np.unique(y)) < 2:
            continue
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            mean = X_train.mean(axis=0); std = X_train.std(axis=0) + 1e-6
            clf = LogisticRegression(max_iter=500, C=0.1, random_state=42)
            clf.fit((X_train - mean) / std, y_train)
            preds = clf.predict_proba((X_test - mean) / std)[:, 1]
            joint_auroc = roc_auc_score(y_test, preds)
            joint_results[k] = {
                "heads": top_k_heads,
                "auroc": float(joint_auroc),
            }
            log(f"  Top-{k}: heads {top_k_heads}, joint AUROC = {joint_auroc:.3f}")
        except Exception as e:
            log(f"  Top-{k}: failed: {e}")

    # SC14: composition gain
    best_single = max(peak_layer_aurocs[~np.isnan(peak_layer_aurocs)]) if np.any(~np.isnan(peak_layer_aurocs)) else 0
    top5_auroc = joint_results.get(5, {}).get("auroc", 0)
    sc14_gain = top5_auroc - best_single
    log(f"\nSC14 (top-5 joint AUROC - best single head): {sc14_gain:+.3f}")
    log(f"  PASS if gain >= 0.02: {'PASS' if sc14_gain >= 0.02 else 'WARN'}")

    # Save outputs
    np.save(probes_dir / "head_auroc_per_layer.npy", head_aurocs)
    with open(probes_dir / "head_probes.pkl", "wb") as f:
        pickle.dump({
            "layers": layers_to_analyze,
            "n_heads": n_heads,
            "probes": head_probes,
            "joint_top_k": joint_results,
            "peak_L": peak_L,
            "peak_offset_idx": peak_o_idx,
        }, f)

    # Eval summary
    head_eval = {
        "layers_analyzed": layers_to_analyze,
        "peak_layer": peak_L,
        "head_aurocs": head_aurocs.tolist(),
        "ranked_heads_at_peak": ranked_heads[:20],
        "joint_top_k": {k: {"heads": v["heads"], "auroc": v["auroc"]} for k, v in joint_results.items()},
        "sanity": {
            "sc13_skew": float(sc13_skew) if 'sc13_skew' in dir() else None,
            "sc14_composition_gain": float(sc14_gain),
        },
    }
    with open(probes_dir / "head_eval.json", "w") as f:
        json.dump(head_eval, f, indent=2)

    log(f"\n✓ Phase 5 complete. Top heads at peak layer {peak_L}: {ranked_heads[:5]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
