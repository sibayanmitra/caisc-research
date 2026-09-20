#!/usr/bin/env python3
"""
Phase 3 — Train probes across the 4 controlled conditions.

CRITICAL probe: B vs A' (sandbagging vs helpful-system-prompt control)
  - Both have system prompts → controls for sys-msg presence confound
  - Same questions → controls for topic confound
  - Only difference: persona content (sandbagging vs helpful)
  - HIGH AUROC here = real sandbagging signature (not a confound artifact)

Secondary probes:
  B vs C2: same questions, both system-prompted, honest vs sandbagging intent
           — this is the cleanest "intent" contrast
  B vs C:  safety probe — can we distinguish "knows but hides" from "genuinely doesn't know"?
  A vs A': system-prompt-alone effect (should be relatively small)

DiffMean directions are computed for each contrast.
"""
import os, sys, json, pickle
from pathlib import Path
import numpy as np
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sandbagging_lib import log, get_paper_dir


def train_probe_per_layer(pos_resid, neg_resid):
    """Per-layer LogisticRegression with bootstrap. Returns dict by layer."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    n_layers = pos_resid.shape[1]
    results = {}
    for L in range(n_layers):
        X_pos = pos_resid[:, L, :].numpy()
        X_neg = neg_resid[:, L, :].numpy()
        X = np.concatenate([X_pos, X_neg], axis=0).astype(np.float32)
        y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
        if len(np.unique(y)) < 2 or len(y) < 10:
            continue
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            mu = X_tr.mean(0); sd = X_tr.std(0) + 1e-6
            clf = LogisticRegression(max_iter=500, C=0.1, random_state=42)
            clf.fit((X_tr - mu) / sd, y_tr)
            preds = clf.predict_proba((X_te - mu) / sd)[:, 1]
            auroc = roc_auc_score(y_te, preds)
            results[L] = {
                "auroc": float(auroc),
                "weights": clf.coef_[0],
                "mu": mu, "sd": sd,
            }
        except Exception:
            pass
    return results


def diffmean_direction(pos_resid, neg_resid):
    """Per-layer DiffMean direction = mean(pos) - mean(neg). Returns (n_layers, d_model)."""
    return (pos_resid.mean(0) - neg_resid.mean(0)).numpy().astype(np.float32)


def main():
    log("=== Paper 3 Phase 3: Train probes (confound-controlled contrasts) ===")
    paper_dir = get_paper_dir()
    act_dir   = paper_dir / "activations"
    probes_dir = paper_dir / "probes"
    dir_dir   = paper_dir / "directions"
    for d in [probes_dir, dir_dir]: d.mkdir(parents=True, exist_ok=True)

    # Load all 5 conditions
    def load_resid(name):
        path = act_dir / f"condition_{name}.pt"
        if not path.exists():
            log(f"  ⚠ {path.name} not found — skipping")
            return None
        data = torch.load(path, weights_only=False)
        return data["resid"].float(), data["meta"]

    A_r,  A_meta  = load_resid("A")  or (None, None)
    Ap_r, Ap_meta = load_resid("A_prime") or (None, None)
    B_r,  B_meta  = load_resid("B")  or (None, None)
    C_r,  C_meta  = load_resid("C")  or (None, None)
    C2_r, C2_meta = load_resid("C2") or (None, None)

    if B_r is None or Ap_r is None:
        log("FATAL: B or A_prime missing — cannot run main probe")
        return 1

    n_layers = B_r.shape[1]
    d_model  = B_r.shape[2]
    log(f"Loaded: A={None if A_r is None else A_r.shape}, "
        f"A'={None if Ap_r is None else Ap_r.shape}, "
        f"B={None if B_r is None else B_r.shape}, "
        f"C={None if C_r is None else C_r.shape}, "
        f"C2={None if C2_r is None else C2_r.shape}")

    contrasts = {
        # The critical confound-controlled probe — system-prompt MATCHED, question MATCHED
        "B_vs_A_prime": (B_r, Ap_r),
    }
    if C2_r is not None:
        # Cleanest intent contrast (same questions, both system-prompted, opposite intent)
        contrasts["B_vs_C2"] = (B_r, C2_r)
    if C_r is not None:
        # Original safety probe
        contrasts["B_vs_C"]  = (B_r, C_r)
    if A_r is not None and Ap_r is not None:
        # Sanity: how much does just adding a benign system prompt change activations?
        contrasts["A_vs_A_prime"] = (A_r, Ap_r)
    if A_r is not None and B_r is not None:
        contrasts["A_vs_B"]  = (A_r, B_r)

    all_results = {}
    for cname, (pos, neg) in contrasts.items():
        log(f"\nTraining probe: {cname}  ({len(pos)} pos vs {len(neg)} neg)")
        probes = train_probe_per_layer(pos, neg)
        aurocs = {L: v["auroc"] for L, v in probes.items()}
        peak_L = max(aurocs, key=aurocs.get) if aurocs else -1
        peak_auroc = aurocs.get(peak_L, 0.0)
        log(f"  Peak layer: {peak_L}, AUROC: {peak_auroc:.3f}")

        diff = diffmean_direction(pos, neg)  # (n_layers, d_model)
        np.save(dir_dir / f"d_{cname}_per_layer.npy", diff)

        all_results[cname] = {
            "peak_layer": peak_L,
            "peak_auroc": peak_auroc,
            "aurocs_per_layer": aurocs,
            "n_pos": len(pos),
            "n_neg": len(neg),
        }
        with open(probes_dir / f"{cname}_probes.pkl", "wb") as f:
            pickle.dump(probes, f)

    # Identify the canonical peak layer from B vs A' (the confound-controlled probe)
    peak_L_canonical = all_results["B_vs_A_prime"]["peak_layer"]
    peak_auroc_canonical = all_results["B_vs_A_prime"]["peak_auroc"]
    log(f"\n=== KEY RESULT ===")
    log(f"  B vs A' (confound-controlled): peak L={peak_L_canonical}, AUROC={peak_auroc_canonical:.3f}")

    # Sanity SC: A vs A' should be LOWER than B vs A' if the signal is truly sandbagging,
    # not just "system prompt presence"
    if "A_vs_A_prime" in all_results:
        aa_auroc = all_results["A_vs_A_prime"]["peak_auroc"]
        gap = peak_auroc_canonical - aa_auroc
        log(f"  A vs A' (sys-prompt-alone effect): peak AUROC={aa_auroc:.3f}")
        log(f"  CONFOUND CHECK: B vs A' AUROC - A vs A' AUROC = {gap:+.3f}")
        log(f"  {'PASS ✓ sandbagging signal is above system-prompt baseline' if gap > 0.05 else 'WARN ✗ confound concern'}")

    # Save composite probe_eval (compatible with downstream scripts that key off "peak_layer")
    probe_eval = {
        "peak_layer": peak_L_canonical,
        "peak_contrast": "B_vs_A_prime",
        "peak_auroc": peak_auroc_canonical,
        "all_contrasts": all_results,
        "design_note": (
            "Peak layer derived from B vs A' (confound-controlled probe). "
            "Both conditions have system prompts and matched questions; "
            "only differ in persona content. High AUROC here = real sandbagging signal."
        ),
    }
    # Legacy key for scripts that look at B_vs_C
    if "B_vs_C" in all_results:
        probe_eval["B_vs_C_aurocs"] = all_results["B_vs_C"]["aurocs_per_layer"]

    with open(probes_dir / "probe_eval.json", "w") as f:
        json.dump(probe_eval, f, indent=2)

    # Set d_withhold_per_layer.npy = the B vs A' DiffMean (the confound-controlled signal)
    # so downstream scripts pick up the right direction
    d_withhold = np.load(dir_dir / "d_B_vs_A_prime_per_layer.npy")
    np.save(dir_dir / "d_withhold_per_layer.npy", d_withhold)

    log(f"\n✓ Phase 3 done. Peak (B vs A'): L={peak_L_canonical}, AUROC={peak_auroc_canonical:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
