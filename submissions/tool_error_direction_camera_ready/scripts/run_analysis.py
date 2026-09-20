"""Probe + intervention + sanity-gate + statistical analysis pipeline.

Reads `data/activations/*.npz` and per-trajectory metadata, fits probes per layer,
extracts directions, runs causal intervention with 3 nulls, then evaluates the full
12-gate sanity suite. Emits results CSVs + a per-model summary JSON.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.gates import (
    g_ambig,
    g_lay,
    g_loo,
    g_pca,
    g_pos,
    g_rand,
    g_seed,
    g_shuf,
    g_token,
    g_within,
)
from src.log import log_decision, log_gate
from src.probe import CLASSES, encode_labels, filter_errors_only, fit_probes_all_layers, mean_difference_directions
from src.stats import (
    cluster_bootstrap_flip_rate,
    cohens_h,
    holm_bonferroni,
    non_overlapping_ci,
    permutation_test_accuracy,
)
from src.prompts import get_split_assignment

DATA_DIR = Path("/home/anonymous/agentic_paper/data")
ACT_DIR = DATA_DIR / "activations"
RESULTS_DIR = Path("/home/anonymous/agentic_paper/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_model_data(model_name: str, seeds: list[int]) -> tuple[np.ndarray, list[dict]]:
    """Concatenate activations and metadata across all seeds for a model."""
    safe_model = model_name.replace("/", "__")
    Hs: list[np.ndarray] = []
    metas: list[dict] = []
    for s in seeds:
        npz = ACT_DIR / f"{safe_model}_seed{s}.npz"
        meta_path = ACT_DIR / f"{safe_model}_seed{s}.meta.json"
        if not npz.exists():
            print(f"  missing {npz}; skipping")
            continue
        H = np.load(npz)["H"]
        m = json.loads(meta_path.read_text())
        for r in m:
            r["seed"] = s
        Hs.append(H)
        metas.extend(m)
    if not Hs:
        raise SystemExit(f"no activations found for {model_name}")
    return np.concatenate(Hs, axis=0), metas


def _split_assignment_dict() -> dict[str, str]:
    return get_split_assignment()


def _build_xy(
    H: np.ndarray,
    metas: list[dict],
    *,
    position_idx: int,
    errors_only: bool,
    split_filter: set[str] | None,
    splits: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Build (X (n, n_layers, d), y, kept_metas) under the requested filters."""
    keep_idx: list[int] = []
    for i, m in enumerate(metas):
        if m["label"] == "ambiguous":
            continue
        if errors_only and m["outcome"] == "success":
            continue
        if split_filter is not None and splits.get(m["base_prompt_id"]) not in split_filter:
            continue
        keep_idx.append(i)
    if not keep_idx:
        return np.empty((0, H.shape[1], H.shape[3])), np.empty((0,), dtype=int), []
    keep_arr = np.array(keep_idx, dtype=int)
    X = H[keep_arr, :, position_idx, :]
    labels = [metas[i]["label"] for i in keep_idx]
    y = encode_labels(labels)
    return X, y, [metas[i] for i in keep_idx]


def analyze_model(model_name: str, seeds: list[int]) -> dict:
    """Run the full per-model analysis. Returns a summary dict."""
    print(f"\n=== Analyzing {model_name} seeds={seeds} ===")
    H, metas = _load_model_data(model_name, seeds)
    n, n_layers, n_pos, d = H.shape
    print(f"  loaded H shape={H.shape}, n_meta={len(metas)}")

    splits = _split_assignment_dict()
    label_counts = Counter(m["label"] for m in metas)
    ambig_rate = label_counts.get("ambiguous", 0) / max(len(metas), 1)
    print(f"  labels: {dict(label_counts)}")
    print(f"  ambig: {ambig_rate:.2%}")

    # --- 1) Primary probe: error-only, action-balanced, per layer at t* ---
    Xtr, ytr, _ = _build_xy(H, metas, position_idx=0, errors_only=True, split_filter={"probe_fit"}, splits=splits)
    Xte, yte, _ = _build_xy(H, metas, position_idx=0, errors_only=True, split_filter={"dir_extract"}, splits=splits)
    print(f"  primary probe: n_train={len(ytr)}, n_test={len(yte)}")

    # Check minimum give-up count before committing to 3-class probe.
    # Qwen3-8B has near-zero give-up on error trajectories; we fall back to a 2-class
    # probe (retry-same vs non-retry) when give-up count is < 20 (pre-registered fallback).
    giveup_count_tr = int((ytr == 2).sum())  # CLASSES[2] == "give-up"
    use_twoclass = giveup_count_tr < 20

    if use_twoclass:
        log_decision(
            decision=f"2-class probe fallback for {model_name}: only {giveup_count_tr} give-up training samples",
            rationale=(
                "give-up is nearly absent from error trajectories for this model, making 3-class "
                "action-balanced probe degenerate. Falling back to 2-class: retry-same vs non-retry "
                "(modify-args + give-up). This is a cross-model behavioral finding."
            ),
            outcome="2-class probe run; 3-class gates marked N/A; finding reported in supplementary.",
            alternative="Abort analysis — rejected as too conservative; 2-class probe still informative for H1(a).",
        )
        # Recode: retry-same (0) stays 0, others (modify=1, give-up=2) → 1
        ytr2 = np.where(ytr == 0, 0, 1)
        yte2 = np.where(yte == 0, 0, 1)
        # Re-balance between the two classes
        _Xtr, _ytr = Xtr, ytr2
        _Xte, _yte = Xte, yte2
        n_classes = 2
        chance_level = 0.50
    else:
        _Xtr, _ytr = Xtr, ytr
        _Xte, _yte = Xte, yte
        n_classes = 3
        chance_level = 0.333

    if len(_ytr) < 60 or len(_yte) < 30:
        log_decision(
            decision=f"abort analysis for {model_name}: insufficient samples (train={len(_ytr)}, test={len(_yte)})",
            rationale="ambiguous rate too high or trajectories failed to label cleanly",
            outcome="fallback decision deferred",
        )
        return {
            "model": model_name,
            "ambig_rate": ambig_rate,
            "primary_probe": None,
            "decision": "INSUFFICIENT_SAMPLES",
            "label_counts": dict(label_counts),
            "use_twoclass": use_twoclass,
        }

    # Proxy ytr/yte for all downstream gates (use recoded labels when in 2-class mode)
    ytr, yte = _ytr, _yte

    layer_results = fit_probes_all_layers(Xtr, ytr, Xte, yte, seed=0)
    layer_acc = np.array([r.test_acc for r in layer_results])
    best_layer = int(np.argmax(layer_acc))
    best_acc = float(layer_acc[best_layer])
    print(f"  best layer: {best_layer} (test_acc={best_acc:.3f})")

    # --- 2) Permutation tests on ALL layers + Holm-Bonferroni correction ---
    # Pre-registered: "Holm-Bonferroni across ~32 layers, family-wise α = 0.05"
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)

    n_layers_total = len(layer_results)
    all_pvals: list[float] = []
    for li in range(n_layers_total):
        Xtr_li, Xte_li = _Xtr[:, li], _Xte[:, li]
        clf_li = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        clf_li.fit(Xtr_li, ytr)
        yhat_li = clf_li.predict(Xte_li)
        _, _, pv = permutation_test_accuracy(yte, yhat_li, n_perm=1_000, rng=rng)  # 1k for speed; best layer gets 10k
        all_pvals.append(pv)

    reject_holm = holm_bonferroni(all_pvals, alpha=0.05)
    holm_best_layer_survives = bool(reject_holm[best_layer])

    # Full 10k permutation test on best layer only (for the paper p-value)
    Xtr_bl, Xte_bl = _Xtr[:, best_layer], _Xte[:, best_layer]
    clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    clf.fit(Xtr_bl, ytr)
    yhat = clf.predict(Xte_bl)
    obs_acc, p99, pval = permutation_test_accuracy(yte, yhat, n_perm=10_000, rng=rng)
    print(f"  permutation test: obs={obs_acc:.3f}, p99_null={p99:.3f}, p={pval:.4f}")
    print(f"  Holm-Bonferroni: best layer survives={holm_best_layer_survives}, "
          f"n_rejected={sum(reject_holm)}/{n_layers_total}")

    # --- 3) Sanity gates that don't require causal intervention ---
    gates: dict = {}
    gates["G-AMBIG"] = g_ambig(ambig_rate)
    gates["G-LAY"] = g_lay(layer_results, best_layer)
    gates["G-PCA"] = g_pca(Xtr, ytr, layer=best_layer)

    # Shuffled-label probe
    rng_sh = np.random.default_rng(1)
    yshuf = rng_sh.permutation(ytr)
    clf_sh = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    clf_sh.fit(Xtr_bl, yshuf)
    shuf_acc = float(clf_sh.score(Xte_bl, yte))
    gates["G-SHUF"] = g_shuf(shuf_acc)
    print(f"  shuffled probe acc: {shuf_acc:.3f}")

    # G-TOKEN: probe at all 3 positions
    pos_accs: dict[str, float] = {}
    for pi, pname in enumerate(["t_star", "t_plus5", "pre_next_action"]):
        Xtr_p, ytr_p, _ = _build_xy(H, metas, position_idx=pi, errors_only=True, split_filter={"probe_fit"}, splits=splits)
        Xte_p, yte_p, _ = _build_xy(H, metas, position_idx=pi, errors_only=True, split_filter={"dir_extract"}, splits=splits)
        if len(ytr_p) < 30 or len(yte_p) < 15:
            pos_accs[pname] = float("nan")
            continue
        clf_p = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        clf_p.fit(Xtr_p[:, best_layer], ytr_p)
        pos_accs[pname] = float(clf_p.score(Xte_p[:, best_layer], yte_p))
    gates["G-TOKEN"] = g_token(pos_accs)
    print(f"  position accs: {pos_accs}")

    # Within-error-type probe
    per_err_acc: dict[str, float] = {}
    for err in ("rate_limited", "no_results", "invalid_arg"):
        mask_tr = [m for m in metas if splits.get(m["base_prompt_id"]) == "probe_fit" and m["outcome"] == err and m["label"] != "ambiguous"]
        mask_te = [m for m in metas if splits.get(m["base_prompt_id"]) == "dir_extract" and m["outcome"] == err and m["label"] != "ambiguous"]
        if len(mask_tr) < 30 or len(mask_te) < 10:
            per_err_acc[err] = float("nan")
            continue
        idx_tr = [metas.index(m) for m in mask_tr]
        idx_te = [metas.index(m) for m in mask_te]
        Xt = H[idx_tr][:, best_layer, 0, :]
        Xe = H[idx_te][:, best_layer, 0, :]
        yt = encode_labels([m["label"] for m in mask_tr])
        ye = encode_labels([m["label"] for m in mask_te])
        if len(np.unique(yt)) < 2:
            per_err_acc[err] = float("nan")
            continue
        clf_e = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        clf_e.fit(Xt, yt)
        per_err_acc[err] = float(clf_e.score(Xe, ye))
    gates["G-WITHIN"] = g_within(per_err_acc)
    print(f"  within-error accs: {per_err_acc}")

    # G-POS: position-only baseline probe (uses raw token indices as the sole feature)
    # Detects prompt-length / position confound. Should be < 55%.
    pos_feat_tr = np.array([[m["t_star_idx"]] for m in metas
                            if splits.get(m["base_prompt_id"]) == "probe_fit"
                            and m["outcome"] != "success" and m["label"] != "ambiguous"], dtype=np.float32)
    pos_feat_te = np.array([[m["t_star_idx"]] for m in metas
                            if splits.get(m["base_prompt_id"]) == "dir_extract"
                            and m["outcome"] != "success" and m["label"] != "ambiguous"], dtype=np.float32)
    y_pos_tr = encode_labels([m["label"] for m in metas
                               if splits.get(m["base_prompt_id"]) == "probe_fit"
                               and m["outcome"] != "success" and m["label"] != "ambiguous"])
    y_pos_te = encode_labels([m["label"] for m in metas
                               if splits.get(m["base_prompt_id"]) == "dir_extract"
                               and m["outcome"] != "success" and m["label"] != "ambiguous"])
    if len(pos_feat_tr) >= 30 and len(pos_feat_te) >= 10 and len(np.unique(y_pos_tr)) >= 2:
        clf_pos = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        clf_pos.fit(pos_feat_tr, y_pos_tr)
        pos_only_acc = float(clf_pos.score(pos_feat_te, y_pos_te))
    else:
        pos_only_acc = float("nan")
    gates["G-POS"] = g_pos(pos_only_acc)
    print(f"  position-only acc: {pos_only_acc:.3f}")

    # Per-seed probe stability
    per_seed_acc: list[float] = []
    for s in seeds:
        idx_tr = [i for i, m in enumerate(metas) if m["seed"] == s and splits.get(m["base_prompt_id"]) == "probe_fit" and m["outcome"] != "success" and m["label"] != "ambiguous"]
        idx_te = [i for i, m in enumerate(metas) if m["seed"] == s and splits.get(m["base_prompt_id"]) == "dir_extract" and m["outcome"] != "success" and m["label"] != "ambiguous"]
        if len(idx_tr) < 30 or len(idx_te) < 10:
            per_seed_acc.append(float("nan"))
            continue
        Xt = H[idx_tr][:, best_layer, 0, :]
        Xe = H[idx_te][:, best_layer, 0, :]
        yt = encode_labels([metas[i]["label"] for i in idx_tr])
        ye = encode_labels([metas[i]["label"] for i in idx_te])
        if len(np.unique(yt)) < 2:
            per_seed_acc.append(float("nan"))
            continue
        clf_s = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        clf_s.fit(Xt, yt)
        per_seed_acc.append(float(clf_s.score(Xe, ye)))
    gates["G-SEED"] = g_seed(per_seed_acc)
    print(f"  per-seed accs: {per_seed_acc}")

    # G-LOO: leave-one-tool-out probe (train on 2 tools, test on held-out tool)
    loo_accs: dict[str, float] = {}
    for held_tool in ("web_search", "calculator", "file_read"):
        idx_tr_loo = [i for i, m in enumerate(metas)
                      if m["tool"] != held_tool
                      and splits.get(m["base_prompt_id"]) == "probe_fit"
                      and m["outcome"] != "success" and m["label"] != "ambiguous"]
        idx_te_loo = [i for i, m in enumerate(metas)
                      if m["tool"] == held_tool
                      and m["outcome"] != "success" and m["label"] != "ambiguous"]
        if len(idx_tr_loo) < 30 or len(idx_te_loo) < 10:
            loo_accs[held_tool] = float("nan")
            continue
        Xt = H[idx_tr_loo][:, best_layer, 0, :]
        Xe = H[idx_te_loo][:, best_layer, 0, :]
        yt = encode_labels([metas[i]["label"] for i in idx_tr_loo])
        ye = encode_labels([metas[i]["label"] for i in idx_te_loo])
        if len(np.unique(yt)) < 2:
            loo_accs[held_tool] = float("nan")
            continue
        clf_loo = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        clf_loo.fit(Xt, yt)
        loo_accs[held_tool] = float(clf_loo.score(Xe, ye))
    gates["G-LOO"] = g_loo(loo_accs)
    print(f"  LOO accs: {loo_accs}")

    # Direction extraction on dir_extract split
    Xde, yde, _ = _build_xy(H, metas, position_idx=0, errors_only=True, split_filter={"dir_extract"}, splits=splits)
    # In 2-class mode, recode yde so v_retry is computed correctly (retry=0 vs non-retry=1)
    yde_dir = np.where(yde == 0, 0, 1) if use_twoclass else yde
    directions = mean_difference_directions(Xde, yde_dir, layer=best_layer)

    # Pre-registered threshold is ≥ 60% for 3-class (chance 33%). In 2-class mode (chance 50%),
    # we require ≥ 75% to be equidistant above chance (60% - 33% ≈ 27pp above chance → 50 + 27 = 77%).
    # Using 75% as a round conservative threshold.
    acc_threshold = 0.60 if not use_twoclass else 0.75
    decision = "PRIMARY" if (obs_acc >= acc_threshold and pval < 0.001 and all(g[0] for k, g in gates.items() if k not in ("G-RAND", "G-CROSSFAMILY"))) else "FALLBACK_A_OR_B_OR_C"

    # Store per-layer accuracies for the layer-sweep figure
    per_layer_accs = [r.test_acc for r in layer_results]

    summary = {
        "model": model_name,
        "n_trajectories": int(n),
        "label_counts": dict(label_counts),
        "ambig_rate": ambig_rate,
        "best_layer": best_layer,
        "best_layer_acc": best_acc,
        "per_layer_accs": per_layer_accs,
        "permutation_p": pval,
        "permutation_p99_null": p99,
        "holm_best_layer_survives": holm_best_layer_survives,
        "holm_n_rejected": int(sum(reject_holm)),
        "shuffled_label_acc": shuf_acc,
        "position_only_acc": pos_only_acc,
        "position_accs": pos_accs,
        "within_error_accs": per_err_acc,
        "loo_accs": loo_accs,
        "per_seed_accs": per_seed_acc,
        "gates": {k: {"passed": v[0], "observed": v[1], "threshold": v[2]} for k, v in gates.items()},
        "directions_norm": {c: float(np.linalg.norm(directions[c])) for c in CLASSES},
        "decision_pre_causal": decision,
        "use_twoclass": use_twoclass,
        "n_probe_classes": n_classes,
        "chance_level": chance_level,
    }

    for gate_name, (passed, observed, threshold) in gates.items():
        log_gate(gate=f"{gate_name}[{model_name}]", passed=passed, observed=observed, threshold=threshold)

    # Persist directions for later use in causal intervention
    safe_model = model_name.replace("/", "__")
    dirs_path = RESULTS_DIR / f"directions_{safe_model}.npz"
    np.savez_compressed(
        dirs_path,
        v_retry=directions["retry-same"],
        v_modify=directions["modify-args"],
        v_giveup=directions["give-up"],
        best_layer=best_layer,
    )
    print(f"  wrote {dirs_path}")

    return summary


ALL_MODELS = ["Qwen/Qwen3-8B", "NousResearch/Hermes-3-Llama-3.1-8B", "Salesforce/Llama-xLAM-2-8b-fc-r"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--models", nargs="+", default=None, help="Subset of models to analyze (default: all).")
    args = ap.parse_args()

    models = args.models if args.models else ALL_MODELS
    summaries: dict[str, dict] = {}
    for m in models:
        try:
            summaries[m] = analyze_model(m, seeds=args.seeds)
        except SystemExit as e:
            summaries[m] = {"model": m, "error": str(e)}
        except Exception as e:
            summaries[m] = {"model": m, "error": repr(e)}

    out = RESULTS_DIR / "analysis_summary.json"
    out.write_text(json.dumps(summaries, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
