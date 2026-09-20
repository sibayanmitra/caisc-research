#!/usr/bin/env python3
"""Plot rank-frequency log-log overlays + α-vs-vocab summary."""
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
COUNTS = DATA / "token_counts"
NOTES = ROOT / "notes"
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

COLORS = {
    "word":    "#000000",
    "gpt2_bpe":"#1f77b4",
    "cl100k":  "#2ca02c",
    "o200k":   "#9467bd",
    "qwen2.5": "#ff7f0e",
    "byte":    "#7f7f7f",
}


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    table = pd.read_csv(NOTES / "alpha_table.csv")

    # ---- Figure 1: rank-frequency log-log for English ----------
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for tname in ["word", "gpt2_bpe", "cl100k", "o200k", "qwen2.5", "byte"]:
        pkl = COUNTS / f"{tname}_en.pkl"
        if not pkl.exists():
            continue
        with open(pkl, "rb") as fh:
            ctr = pickle.load(fh)
        counts = np.sort(np.fromiter(ctr.values(), dtype=int))[::-1]
        ranks = np.arange(1, len(counts) + 1)
        # subsample for plotting
        idx = np.unique(np.geomspace(1, len(counts), 800).astype(int)) - 1
        idx = idx[idx >= 0]
        ax.loglog(ranks[idx], counts[idx], lw=1.4, alpha=0.85, label=tname, color=COLORS.get(tname, "k"))
    ax.set_xlabel("rank")
    ax.set_ylabel("frequency")
    ax.set_title("Token rank-frequency, English Wikipedia, by tokenizer")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    out = PLOTS / "zipf_by_tokenizer.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    log(f"saved {out}")
    plt.close(fig)

    # ---- Figure 2: alpha vs vocab size for English -----------
    en = table[table["lang"] == "en"].copy()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(en["vocab_observed"], en["alpha_loglog"], c="#1f77b4", s=80, label="α (log-log)", edgecolor="k")
    ax.scatter(en["vocab_observed"], en["alpha_csn"], c="#d62728", s=60, marker="s", label="α (Clauset MLE)", edgecolor="k", alpha=0.7)
    for _, r in en.iterrows():
        ax.annotate(r["tokenizer"], (r["vocab_observed"], r["alpha_loglog"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.axhline(1.0, color="grey", ls="--", lw=0.6, label="α=1 (Zipf reference)")
    ax.set_xlabel("observed vocabulary size")
    ax.set_ylabel("Zipf exponent α")
    ax.set_title("Zipf exponent across tokenizers (English)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = PLOTS / "alpha_vs_vocab.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    log(f"saved {out}")
    plt.close(fig)

    # ---- Figure 3: cross-language invariance ----------------
    if table["lang"].nunique() > 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        wide = table.pivot_table(index="tokenizer", columns="lang", values="alpha_loglog")
        wide.plot(kind="bar", ax=ax)
        ax.axhline(1.0, color="grey", ls="--", lw=0.6, label="Zipf reference")
        ax.set_ylabel("Zipf exponent α (log-log)")
        ax.set_title("Zipf exponent by tokenizer × language")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        out = PLOTS / "alpha_cross_lang.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        log(f"saved {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
