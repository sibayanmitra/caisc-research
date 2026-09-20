"""Publication-quality figures for the paper (300 DPI).

Generates:
  probe_layer_sweep.png   — per-layer probe accuracy, all 3 models
  causal_flip_broad.png   — α-sweep broad flip rate, all 3 models
  cka_cross_tool.png      — 3-panel cross-tool CKA heatmaps
  cka_layer_profile.png   — Hermes-3 vs xLAM-2 CKA layer profile
  gates.png               — sanity gate outcome heatmap
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

RESULTS_DIR = Path("/home/anonymous/agentic_paper/results")
FIGS_DIR = RESULTS_DIR / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
FONTSIZE = 10
LABEL_FONTSIZE = 9

MODEL_SHORT = {
    "Qwen/Qwen3-8B":                         "Qwen3-8B",
    "NousResearch/Hermes-3-Llama-3.1-8B":    "Hermes-3",
    "Salesforce/Llama-xLAM-2-8b-fc-r":       "xLAM-2",
}
MODEL_COLORS = {
    "Qwen3-8B": "#2196F3",
    "Hermes-3":  "#FF5722",
    "xLAM-2":    "#4CAF50",
}
CSV_SLUGS = {
    "Qwen/Qwen3-8B":                         "Qwen__Qwen3-8B",
    "NousResearch/Hermes-3-Llama-3.1-8B":    "NousResearch__Hermes-3-Llama-3.1-8B",
    "Salesforce/Llama-xLAM-2-8b-fc-r":       "Salesforce__Llama-xLAM-2-8b-fc-r",
}

plt.rcParams.update({
    "font.size": FONTSIZE,
    "axes.labelsize": FONTSIZE,
    "axes.titlesize": FONTSIZE,
    "xtick.labelsize": LABEL_FONTSIZE,
    "ytick.labelsize": LABEL_FONTSIZE,
    "legend.fontsize": LABEL_FONTSIZE,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.6,
    "lines.markersize": 4,
})


def fig_probe_acc() -> None:
    summary = json.loads((RESULTS_DIR / "analysis_summary.json").read_text())
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.axhline(1/3, ls=":", color="grey", lw=1.2, label="chance (33%)", zorder=1)
    ax.axhline(0.60, ls="--", color="grey", lw=0.9, label="threshold (60%)", zorder=1)

    for key, s in summary.items():
        label = MODEL_SHORT.get(key, key.split("/")[-1])
        color = MODEL_COLORS.get(label, None)
        accs = s.get("per_layer_accs", [])
        if not accs:
            continue
        accs_arr = np.array(accs, dtype=float)
        layers = np.arange(len(accs_arr))
        ax.plot(layers, accs_arr, marker="o", ms=3, label=label, color=color, zorder=3)
        bl = s["best_layer"]
        ax.scatter([bl], [s["best_layer_acc"]], marker="*", s=160,
                   color=color, zorder=5, edgecolors="white", linewidths=0.5)

    ax.set_xlabel("Layer index")
    ax.set_ylabel("3-way held-out accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(-0.5, 37)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(loc="lower right")
    ax.set_title("Per-layer probe accuracy (error-only, action-balanced)")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "probe_layer_sweep.png")
    plt.close(fig)
    print("wrote probe_layer_sweep.png")


def fig_causal_flip_broad() -> None:
    """α-sweep broad flip rate for all 3 models (retry direction vs random)."""
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.9), sharey=True)

    for ax, (full_key, slug) in zip(axes, CSV_SLUGS.items()):
        label = MODEL_SHORT[full_key]
        csv = RESULTS_DIR / f"causal_flip_{slug}.csv"
        if not csv.exists():
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label)
            continue
        df = pd.read_csv(csv)
        # broad metric: intervened_label != 'give-up'
        df["flipped_broad"] = (df["intervened_label"] != "give-up").astype(float)
        for direction, color, ls in [("retry", MODEL_COLORS[label], "-"),
                                     ("random", "#888888", "--")]:
            sub = df[df["direction"] == direction]
            if sub.empty:
                continue
            agg = sub.groupby("alpha")["flipped_broad"].mean().reset_index()
            ax.plot(agg["alpha"], agg["flipped_broad"],
                    marker="o", color=color, ls=ls,
                    label=f"{direction}")
        ax.set_xlabel("α")
        ax.set_title(label)
        ax.set_ylim(0.0, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_xticks([0, 1.0, 2.5, 5.0])
        ax.axvline(2.5, ls=":", color="grey", lw=0.8)

    axes[0].set_ylabel("Give-up → non-give-up (broad)")
    axes[0].legend(loc="lower right")
    fig.suptitle("Broad flip rate vs. steering strength α", fontsize=FONTSIZE, y=1.01)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "causal_flip.png")
    plt.close(fig)
    print("wrote causal_flip.png")


def fig_cka_cross_tool() -> None:
    cka_data = json.loads((RESULTS_DIR / "cka_results.json").read_text())
    tools = ["web_search", "calculator", "file_read"]
    tool_labels = ["web", "calc", "file"]
    models_order = list(MODEL_SHORT.keys())

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6))
    vmin, vmax = 0.0, 1.0

    for ax, full_key in zip(axes, models_order):
        label = MODEL_SHORT[full_key]
        ax.set_title(label, fontsize=FONTSIZE)
        entry = cka_data["cross_tool"].get(full_key)
        if entry is None:
            continue
        cka_dict = entry["cka"]
        M = np.zeros((3, 3))
        for i, t1 in enumerate(tools):
            for j, t2 in enumerate(tools):
                M[i, j] = cka_dict.get(f"{t1}|{t2}", 0.0)

        im = ax.imshow(M, cmap="YlOrRd", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xticks(range(3)); ax.set_xticklabels(tool_labels, fontsize=LABEL_FONTSIZE)
        ax.set_yticks(range(3)); ax.set_yticklabels(tool_labels, fontsize=LABEL_FONTSIZE)
        for i in range(3):
            for j in range(3):
                val = M[i, j]
                txt_color = "white" if val > 0.6 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color=txt_color)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cbar.set_label("CKA", fontsize=LABEL_FONTSIZE)
    null_val = cka_data.get("null_cross_tool", 0.143)
    fig.suptitle(
        f"Cross-tool CKA at best action-choice layer (null ≈ {null_val:.3f})",
        fontsize=FONTSIZE, y=1.02
    )
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "cka_cross_tool.png")
    plt.close(fig)
    print("wrote cka_cross_tool.png")


def fig_cka_layer_profile() -> None:
    cka_data = json.loads((RESULTS_DIR / "cka_results.json").read_text())
    layer_cka = cka_data["cross_model"]["hermes_xlam_layer_cka"]
    layers = sorted(int(k) for k in layer_cka)
    vals = [layer_cka[str(k)] for k in layers]

    # best action-choice layers from analysis_summary
    summary = json.loads((RESULTS_DIR / "analysis_summary.json").read_text())
    best_h = summary.get("NousResearch/Hermes-3-Llama-3.1-8B", {}).get("best_layer", 28)
    best_x = summary.get("Salesforce/Llama-xLAM-2-8b-fc-r", {}).get("best_layer", 29)

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.plot(layers, vals, marker="o", ms=4, color="#7B1FA2", label="Hermes-3 vs xLAM-2")
    ax.axvline(best_h, ls="--", color=MODEL_COLORS["Hermes-3"],
               lw=1.2, label=f"Hermes-3 best-probe layer (L{best_h})")
    ax.axvline(best_x, ls="--", color=MODEL_COLORS["xLAM-2"],
               lw=1.2, label=f"xLAM-2 best-probe layer (L{best_x})")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("CKA (Hermes-3 vs xLAM-2)")
    ax.set_ylim(0.0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Cross-model representation similarity by layer")
    ax.legend(loc="lower right", fontsize=LABEL_FONTSIZE - 1)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "cka_layer_profile.png")
    plt.close(fig)
    print("wrote cka_layer_profile.png")


def fig_gates() -> None:
    summary = json.loads((RESULTS_DIR / "analysis_summary.json").read_text())
    models = list(summary.keys())
    model_labels = [MODEL_SHORT.get(m, m.split("/")[-1]) for m in models]

    all_gates: list[str] = []
    for s in summary.values():
        if "gates" in s:
            all_gates.extend(s["gates"].keys())
    gates = sorted(set(all_gates))
    if not gates:
        return

    M = np.zeros((len(gates), len(models)))
    for j, m in enumerate(models):
        for i, g in enumerate(gates):
            info = summary[m].get("gates", {}).get(g)
            M[i, j] = 1.0 if info and info["passed"] else 0.0

    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(model_labels, rotation=20, ha="right", fontsize=LABEL_FONTSIZE)
    ax.set_yticks(range(len(gates)))
    ax.set_yticklabels(gates, fontsize=LABEL_FONTSIZE)
    ax.set_title("Sanity gate outcomes", fontsize=FONTSIZE)
    for i in range(len(gates)):
        for j in range(len(models)):
            sym = "✓" if M[i, j] else "✗"
            color = "black"
            ax.text(j, i, sym, ha="center", va="center", fontsize=11, color=color)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "gates.png")
    plt.close(fig)
    print("wrote gates.png")


def main() -> None:
    fig_probe_acc()
    fig_causal_flip_broad()
    fig_cka_cross_tool()
    fig_cka_layer_profile()
    fig_gates()


if __name__ == "__main__":
    main()
