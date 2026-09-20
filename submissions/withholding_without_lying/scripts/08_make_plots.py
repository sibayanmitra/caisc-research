#!/usr/bin/env python3
"""Phase 8 — Generate Paper 3 figures."""
import os, sys, json
from pathlib import Path
import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sandbagging_lib import log, get_paper_dir, PERSONAS

def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paper_dir = get_paper_dir()
    plots_dir = paper_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Fig 1: B vs C probe AUROC per layer
    probe_path = paper_dir / "probes" / "probe_eval.json"
    if probe_path.exists():
        with open(probe_path) as f:
            pe = json.load(f)
        aurocs = pe.get("B_vs_C_aurocs", {})
        if aurocs:
            layers = sorted(int(k) for k in aurocs)
            vals   = [aurocs[str(L)] for L in layers]
            peak_L = pe.get("peak_layer", max(layers, key=lambda L: aurocs.get(str(L), 0)))
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(layers, vals, color="steelblue", alpha=0.8)
            ax.axhline(0.5, color="gray", linestyle="--", label="Chance")
            ax.axvline(peak_L, color="red", linestyle="--", label=f"Peak L={peak_L}")
            ax.set_xlabel("Layer"); ax.set_ylabel("AUROC (B vs C)")
            ax.set_title("Sandbagging-Knows vs Genuine-Ignorance: Probe AUROC per Layer")
            ax.legend(); plt.tight_layout()
            plt.savefig(plots_dir / "fig1_probe_auroc_heatmap.pdf", bbox_inches="tight")
            plt.close(); log("  Saved fig1")

    # Fig 2: cos(d_withhold, d_refuse) per layer
    cos_path = paper_dir / "directions" / "cosine_withhold_refuse.npy"
    if cos_path.exists():
        cos_wr = np.load(cos_path)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(range(len(cos_wr)), cos_wr, "b-o", markersize=4)
        ax.axhline(0, color="gray", linestyle="--")
        ax.set_xlabel("Layer"); ax.set_ylabel("Cosine Similarity")
        ax.set_title("Geometric Relationship: d_withhold vs d_refuse per Layer")
        plt.tight_layout()
        plt.savefig(plots_dir / "fig2_cosine_geometry.pdf", bbox_inches="tight")
        plt.close(); log("  Saved fig2")

    # Fig 3: Causal recovery bar chart
    rec_path = paper_dir / "recovery" / "recovery_metrics.json"
    if rec_path.exists():
        with open(rec_path) as f:
            rm = json.load(f)
        results = rm["results"]
        conditions_to_plot = ["baseline", "projout_withhold", "projout_refuse"]
        rand_rates = [results.get(f"projout_random_{i}", {}).get("correct_rate", 0) for i in range(5)]

        labels = ["Baseline\n(sandbagging)", "Project-out\nd_withhold", "Project-out\nd_refuse",
                  "Project-out\nrandom (mean)"]
        rates  = [results.get("baseline", {}).get("correct_rate", 0),
                  results.get("projout_withhold", {}).get("correct_rate", 0),
                  results.get("projout_refuse", {}).get("correct_rate", 0),
                  float(np.mean(rand_rates))]
        errs   = [0, 0, 0, float(np.std(rand_rates))]

        fig, ax = plt.subplots(figsize=(7, 5))
        colors = ["#d73027", "#4575b4", "#fee090", "#91bfdb"]
        bars = ax.bar(labels, rates, color=colors, yerr=errs, capsize=4, alpha=0.85)
        ax.set_ylabel("Correct-Answer Rate")
        ax.set_title("Causal Recovery via Project-Out Intervention")
        ax.set_ylim(0, 1)
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{rate:.0%}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.savefig(plots_dir / "fig3_causal_recovery.pdf", bbox_inches="tight")
        plt.close(); log("  Saved fig3")

    log("✓ Phase 8 done")
    return 0

if __name__ == "__main__":
    sys.exit(main())
