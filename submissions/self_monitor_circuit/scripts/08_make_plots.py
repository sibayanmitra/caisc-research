#!/usr/bin/env python3
"""
Phase 8 — Generate the four main figures for Paper 1.

fig1_probe_heatmap.pdf   : AUROC heatmap (layers x offsets)
fig2_head_ranking.pdf    : per-head AUROC at peak layer, ranked
fig3_ablation_effect.pdf : SC rate per ablation condition
fig4_steering_curve.pdf  : SC rate vs steering alpha, with random-direction band
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = Path(__file__).resolve().parent.parent
PLOTS = PAPER / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def fig1_heatmap():
    arr = np.load(PAPER / "probes" / "auroc_heatmap.npy")  # (28, 6)
    stored_offsets = [-3, -2, -1, 0, 1, -5]
    order = np.argsort(stored_offsets)               # -> indices to sort ascending
    sorted_offsets = [stored_offsets[i] for i in order]
    arr_sorted = arr[:, order]

    fig, ax = plt.subplots(figsize=(5.0, 6.0))
    im = ax.imshow(arr_sorted, aspect="auto", cmap="viridis",
                   vmin=0.5, vmax=1.0, origin="lower")
    ax.set_xticks(range(len(sorted_offsets)))
    ax.set_xticklabels(sorted_offsets)
    ax.set_xlabel("Token offset relative to marker")
    ax.set_ylabel("Layer")
    ax.set_title("Probe AUROC (layer $\\times$ offset)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("AUROC")
    fig.savefig(PLOTS / "fig1_probe_heatmap.pdf")
    plt.close(fig)
    print("  wrote fig1_probe_heatmap.pdf  (min AUROC %.3f)" % arr.min())


def fig2_head_ranking():
    with open(PAPER / "probes" / "head_eval.json") as f:
        he = json.load(f)
    head_aurocs = np.array(he["head_aurocs"][0])     # layer 0
    order = np.argsort(head_aurocs)[::-1]
    sorted_vals = head_aurocs[order]
    top5 = he["ranked_heads_at_peak"][:5]

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    colors = ["#d62728" if h in top5 else "#1f77b4" for h in order]
    ax.bar(range(len(sorted_vals)), sorted_vals, color=colors)
    ax.axhline(np.median(head_aurocs), color="gray", ls="--", lw=1,
               label="median = %.3f" % np.median(head_aurocs))
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=7, rotation=90)
    ax.set_ylim(0.8, 1.01)
    ax.set_xlabel("Attention head (ranked by AUROC)")
    ax.set_ylabel("Per-head AUROC")
    ax.set_title("Per-head probe AUROC at layer 0 (top-5 in red)")
    ax.legend(loc="lower left", fontsize=9)
    fig.savefig(PLOTS / "fig2_head_ranking.pdf")
    plt.close(fig)
    print("  wrote fig2_head_ranking.pdf")


def fig3_ablation():
    with open(PAPER / "ablations" / "ablation_metrics.json") as f:
        m = json.load(f)
    rows = {r["condition"]: r for r in m["results"]}
    labels = ["Baseline", "Top-3", "Top-5", "Random-3", "Random-5"]
    keys = ["baseline", "top3_ablation", "top5_ablation",
            "random3_seed0", "random5_seed0"]
    rates = [rows[k]["self_correction_rate"] * 100 for k in keys]
    colors = ["#555555", "#d62728", "#d62728", "#1f77b4", "#1f77b4"]

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    bars = ax.bar(labels, rates, color=colors)
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, r + 1, "%.0f%%" % r,
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Self-correction rate (%)")
    ax.set_title("Self-correction rate by ablation condition ($n=50$)")
    ax.axhline(rates[0], color="gray", ls=":", lw=1)
    fig.savefig(PLOTS / "fig3_ablation_effect.pdf")
    plt.close(fig)
    print("  wrote fig3_ablation_effect.pdf")


def fig4_steering():
    with open(PAPER / "steering" / "steering_metrics.json") as f:
        s = json.load(f)
    sweep = sorted(s["alpha_sweep"], key=lambda r: r["alpha_mult"])
    xs = [r["alpha_mult"] for r in sweep]
    ys = [r["sc_rate"] * 100 for r in sweep]
    rand_mean = s["random_baseline"]["mean"] * 100
    rand_std = s["random_baseline"]["std"] * 100

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    # random-direction band (measured at alpha=+1)
    ax.axhspan(rand_mean - rand_std, rand_mean + rand_std,
               color="#1f77b4", alpha=0.15,
               label="random dir @ $\\alpha$=+1 (%.0f$\\pm$%.0f%%)" % (rand_mean, rand_std))
    ax.axhline(rand_mean, color="#1f77b4", ls="--", lw=1)
    ax.plot(xs, ys, "o-", color="#d62728", lw=2, ms=7, label="probe direction")
    for x, y in zip(xs, ys):
        ax.text(x, y + 3, "%.0f%%" % y, ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_ylim(-8, 113)
    ax.set_xlabel("Steering coefficient $\\alpha$ (units of mean activation norm)")
    ax.set_ylabel("Self-correction rate (%)")
    ax.set_title("Steering with probe direction ($n=20$ per $\\alpha$)")
    ax.legend(loc="lower center", fontsize=9)
    fig.savefig(PLOTS / "fig4_steering_curve.pdf")
    plt.close(fig)
    print("  wrote fig4_steering_curve.pdf")


def main():
    print("Generating Paper 1 figures...")
    fig1_heatmap()
    fig2_head_ranking()
    fig3_ablation()
    fig4_steering()
    print("Done. Figures in", PLOTS)


if __name__ == "__main__":
    main()
