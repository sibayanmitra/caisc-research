"""Headline figures for the paper.

Figure 1: per-layer probe test accuracy (one line per model).
Figure 2: causal flip-rate vs α (4 lines per model: retry / random / orthogonal / gaussian).
Figure 3: gate-outcome heatmap.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path("/home/anonymous/agentic_paper/results")
FIGS_DIR = RESULTS_DIR / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)


def fig_probe_acc() -> None:
    summary_path = RESULTS_DIR / "analysis_summary.json"
    if not summary_path.exists():
        print("no analysis_summary.json; skipping fig 1")
        return
    summaries = json.loads(summary_path.read_text())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(0.333, ls=":", color="grey", lw=1.2, label="chance (33%)")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ci, (m, s) in enumerate(summaries.items()):
        if "per_layer_accs" not in s or not s["per_layer_accs"]:
            # fallback: scatter best-layer only
            if "best_layer" in s:
                ax.scatter([s["best_layer"]], [s["best_layer_acc"]],
                           label=m.split("/")[-1], s=80, color=colors[ci % len(colors)])
            continue
        accs = s["per_layer_accs"]
        label = m.split("/")[-1]
        ax.plot(range(len(accs)), accs, marker="o", ms=4,
                label=label, color=colors[ci % len(colors)])
        # mark best layer with a larger star
        bl = s["best_layer"]
        ax.scatter([bl], [s["best_layer_acc"]], marker="*", s=200,
                   color=colors[ci % len(colors)], zorder=5)
    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel("3-way probe accuracy (held-out)", fontsize=11)
    ax.set_title("Per-layer probe accuracy — error-only, action-balanced", fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "probe_layer_sweep.png", dpi=140)
    plt.close(fig)
    print("wrote probe_layer_sweep.png")


def fig_causal_flip() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, model in zip(axes, ["Qwen__Qwen3-8B", "NousResearch__Hermes-3-Llama-3.1-8B"]):
        csv = RESULTS_DIR / f"causal_flip_{model}.csv"
        if not csv.exists():
            ax.set_title(f"{model}\n(no data)")
            continue
        df = pd.read_csv(csv)
        for direction, sub in df.groupby("direction"):
            agg = sub.groupby("alpha")["flipped_to_retry"].mean().reset_index()
            ax.plot(agg["alpha"], agg["flipped_to_retry"], marker="o", label=direction)
        ax.set_xlabel("α")
        ax.set_title(model.replace("__", "/"))
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("give-up → retry flip rate")
    axes[0].legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "causal_flip.png", dpi=140)
    plt.close(fig)
    print("wrote causal_flip.png")


def fig_gates() -> None:
    summary_path = RESULTS_DIR / "analysis_summary.json"
    if not summary_path.exists():
        return
    summaries = json.loads(summary_path.read_text())
    models = list(summaries.keys())
    all_gates: list[str] = []
    for s in summaries.values():
        if "gates" in s:
            all_gates.extend(s["gates"].keys())
    gates = sorted(set(all_gates))
    if not gates:
        return
    M = np.zeros((len(gates), len(models)))
    for j, m in enumerate(models):
        for i, g in enumerate(gates):
            info = summaries[m].get("gates", {}).get(g)
            M[i, j] = 1.0 if info and info["passed"] else 0.0

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(gates)))
    ax.set_yticklabels(gates)
    ax.set_title("Gate outcomes (green = pass, red = fail)")
    for i in range(len(gates)):
        for j in range(len(models)):
            ax.text(j, i, "✓" if M[i, j] else "✗", ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "gates.png", dpi=140)
    plt.close(fig)
    print("wrote gates.png")


def main() -> None:
    fig_probe_acc()
    fig_causal_flip()
    fig_gates()


if __name__ == "__main__":
    main()
