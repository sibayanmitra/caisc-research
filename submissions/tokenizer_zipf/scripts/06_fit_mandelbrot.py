#!/usr/bin/env python3
"""Fit Zipf-Mandelbrot model f(r) = C(r+b)^{-alpha} to the existing token counts.

Reports alpha_mb and b per (tokenizer, language).  The existing alpha_table.csv
already has alpha_mb from the pipeline; this script recomputes with window-sweep
robustness (fitting over ranks 1..1k, 1..5k, 1..10k) so we can report the
stability of the mid-range slope estimate and a principled window choice.
"""
import pickle, csv
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
COUNTS = ROOT / "data" / "token_counts"
NOTES = ROOT / "notes"
NOTES.mkdir(exist_ok=True)

WINDOWS = [(1, 1_000), (1, 5_000), (10, 10_000)]  # (lo, hi) inclusive

def fit_mb(counts, lo, hi):
    counts = np.sort(counts.astype(float))[::-1]
    hi = min(hi, len(counts))
    if hi - lo < 30:
        return float("nan"), float("nan")
    r = np.arange(lo, hi+1, dtype=float)
    f = counts[lo-1:hi]
    def model(r, logC, alpha, b):
        return logC - alpha * np.log(r + b)
    try:
        p0 = [np.log(f[0]), 1.0, 1.0]
        popt, _ = curve_fit(model, r, np.log(f+1e-9), p0=p0,
                            bounds=([0, 0.2, 0], [60, 3.0, 1e6]), maxfev=8000)
        return float(popt[1]), float(popt[2])
    except Exception:
        return float("nan"), float("nan")

rows = []
for pkl in sorted(COUNTS.glob("*.pkl")):
    stem = pkl.stem
    if "_" not in stem:
        continue
    tname, lang = stem.rsplit("_", 1)
    with open(pkl, "rb") as fh:
        import pickle as _pkl
        ctr = _pkl.load(fh)
    counts = np.fromiter(ctr.values(), dtype=np.int64)
    row = {"tokenizer": tname, "lang": lang}
    for lo, hi in WINDOWS:
        a, b = fit_mb(counts, lo, hi)
        row[f"alpha_mb_{lo}_{hi}"] = round(a, 4)
        row[f"b_mb_{lo}_{hi}"] = round(b, 2)
    rows.append(row)
    print(f"{tname:10s} {lang}  " +
          "  ".join(f"α[{lo}..{hi}]={row[f'alpha_mb_{lo}_{hi}']:.3f}"
                    for lo,hi in WINDOWS), flush=True)

out = NOTES / "mandelbrot_stability.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print(f"\nWrote {out}")
