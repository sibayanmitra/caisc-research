#!/usr/bin/env python3
"""Plot main figures for heartbeats-sleep investigation."""
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    df = pd.read_csv(DATA / "merged.csv")
    log(f"plotting {len(df)} species")

    # ---- Figure 1: raw sleep vs lifetime energy, colored by class ---------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, ycol, ylabel in [
        (axes[0], "log_lifetime_energy", "log lifetime metabolic budget [log J]"),
    ]:
        for cls, col, mk in [("Mammalia", "#1f77b4", "o"), ("Aves", "#d62728", "^")]:
            sub = df[df["Class"] == cls]
            ax.scatter(sub["sleep_hrs"], sub[ycol], c=col, marker=mk,
                       s=50, alpha=0.7, label=f"{cls} (n={len(sub)})", edgecolor="black", linewidth=0.4)
        ax.set_xlabel("sleep (hours/day)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_title("(a) Raw cross-species relationship")

    # ---- Figure 1b: residuals after partialling out class + log_body_kg ---
    ax = axes[1]
    # Compute partial residuals: regress log_lifetime_energy on class + log_body_kg
    # using OLS, then plot residuals vs sleep
    from numpy.linalg import lstsq
    n = len(df)
    cls_aves = (df["Class"].values == "Aves").astype(float)
    X = np.column_stack([np.ones(n), cls_aves, df["log_body_kg"].values])
    y = df["log_lifetime_energy"].values
    beta, *_ = lstsq(X, y, rcond=None)
    resid = y - X @ beta

    for cls, col, mk in [("Mammalia", "#1f77b4", "o"), ("Aves", "#d62728", "^")]:
        mask = df["Class"].values == cls
        ax.scatter(df["sleep_hrs"][mask], resid[mask], c=col, marker=mk,
                   s=50, alpha=0.7, label=f"{cls} (n={mask.sum()})", edgecolor="black", linewidth=0.4)
    # Add OLS line for residual vs sleep
    s = df["sleep_hrs"].values
    slope, intercept = np.polyfit(s, resid, 1)
    xs = np.linspace(s.min(), s.max(), 100)
    ax.plot(xs, intercept + slope * xs, "k--", lw=1.5, alpha=0.7,
            label=f"OLS fit (slope = {slope:+.3f})")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("sleep (hours/day)")
    ax.set_ylabel("residual log(lifetime energy) | mass, class")
    ax.set_title("(b) Sleep effect after partialling out mass + class")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out = PLOTS / "sleep_vs_lifetime_energy.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    log(f"saved {out}")
    plt.close(fig)

    # ---- Figure 2: residuals by class (diagnostic) ------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    for cls, col, mk in [("Mammalia", "#1f77b4", "o"), ("Aves", "#d62728", "^")]:
        mask = df["Class"].values == cls
        ax.scatter(df["log_body_kg"][mask], resid[mask], c=col, marker=mk,
                   s=50, alpha=0.7, label=cls, edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("log body mass (kg)")
    ax.set_ylabel("residual log(lifetime energy) | mass, class")
    ax.set_title("Diagnostic: residual structure vs body mass")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = PLOTS / "residuals_by_class.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    log(f"saved {out}")
    plt.close(fig)

    # ---- Figure 3: outlier callouts ---------------------------------------
    df["resid"] = resid
    df_sorted = df.copy().sort_values("resid")
    top_low = df_sorted.head(5)[["common_name_sleep", "common_name_anage", "sleep_hrs", "resid"]]
    top_high = df_sorted.tail(5)[["common_name_sleep", "common_name_anage", "sleep_hrs", "resid"]]
    log("largest negative residuals (less energy than expected):")
    log(str(top_low))
    log("largest positive residuals (more energy than expected):")
    log(str(top_high))


if __name__ == "__main__":
    main()
