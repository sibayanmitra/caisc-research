#!/usr/bin/env python3
"""Phase 8 — Generate Paper 2 figures."""
import os, sys, json
from pathlib import Path
import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from negation_lib import log, get_paper_dir

def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paper_dir = get_paper_dir()
    dir_dir  = paper_dir / "directions"
    plots_dir = paper_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Fig 1: Explained variance per layer
    ev = np.load(dir_dir / "explained_variance.npy")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(ev)), ev, color="steelblue", alpha=0.8)
    ax.set_xlabel("Layer"); ax.set_ylabel("Top-1 Explained Variance Ratio")
    ax.set_title("Negation Signal: SVD Top-1 EV per Layer")
    peak_L = int(np.argmax(ev))
    ax.axvline(peak_L, color="red", linestyle="--", label=f"Peak L={peak_L}")
    ax.legend(); plt.tight_layout()
    plt.savefig(plots_dir / "fig1_low_rank_structure.pdf", bbox_inches="tight")
    plt.close(); log("  Saved fig1")

    # Fig 2: Transfer curve (alpha sweep)
    alpha_path = paper_dir / "transfer_experiments" / "alpha_sweep" / "alpha_sweep_metrics.json"
    if alpha_path.exists():
        with open(alpha_path) as f:
            aswp = json.load(f)
        alphas = [r["alpha_mult"] for r in aswp["alpha_sweep"]]
        shifts = [r["mean_abs_shift"] for r in aswp["alpha_sweep"]]
        rand_mean = aswp["random_baseline_mean"]
        rand_std  = aswp["random_baseline_std"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(alphas, shifts, "b-o", label="Probe direction")
        ax.axhline(rand_mean, color="gray", linestyle="--", label="Random baseline")
        ax.fill_between(alphas,
                         rand_mean - rand_std, rand_mean + rand_std,
                         alpha=0.2, color="gray")
        ax.set_xlabel("α (× α_unit)"); ax.set_ylabel("Mean |Logit Shift|")
        ax.set_title("Steering Transfer: Probe Direction vs Random Baseline")
        ax.legend(); plt.tight_layout()
        plt.savefig(plots_dir / "fig2_transfer_curve.pdf", bbox_inches="tight")
        plt.close(); log("  Saved fig2")

    # Fig 3: Cross-morphology matrix
    cc_path = paper_dir / "transfer_experiments" / "cross_category" / "transfer_metrics.json"
    if cc_path.exists():
        with open(cc_path) as f:
            cc = json.load(f)
        mat = np.array(cc["matrix"])
        morph_labels = cc.get("morphologies", [f"M{i}" for i in range(mat.shape[0])])
        n = mat.shape[0]
        fig, ax = plt.subplots(figsize=(max(6, n), max(5, n-1)))
        im = ax.imshow(mat, cmap="Blues", aspect="auto")
        ax.set_xticks(range(n)); ax.set_xticklabels(morph_labels, rotation=40, ha="right")
        ax.set_yticks(range(n)); ax.set_yticklabels(morph_labels)
        ax.set_xlabel("Target Morphology"); ax.set_ylabel("Source (d_neg extracted from)")
        ax.set_title("Cross-Morphology Transfer: Mean |Logit Shift|")
        plt.colorbar(im, ax=ax)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if mat[i,j] > mat.max()*0.6 else "black")
        plt.tight_layout()
        plt.savefig(plots_dir / "fig3_category_matrix.pdf", bbox_inches="tight")
        plt.close(); log("  Saved fig3")

    log("✓ Phase 8 done")
    return 0

if __name__ == "__main__":
    sys.exit(main())
