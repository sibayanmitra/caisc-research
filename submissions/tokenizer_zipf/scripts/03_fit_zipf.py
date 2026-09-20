#!/usr/bin/env python3
"""Fit power-law to each tokenizer/language token-frequency distribution.

Reports for each (tokenizer, lang):
- Clauset-Shalizi-Newman MLE alpha and x_min, KS distance
- Mandelbrot (r + b)^(-alpha) via direct log-log regression (finite-corpus alternative)
- Simple log-log slope as a sanity-check fallback
"""
import pickle
import sys
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
COUNTS = DATA / "token_counts"
NOTES = ROOT / "notes"
NOTES.mkdir(exist_ok=True)

OUT = NOTES / "alpha_table.csv"

# ranks to use for log-log fit (avoid head saturation and tail noise)
LOGLOG_RANK_MIN = 10
LOGLOG_RANK_MAX = 10_000


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def fit_csn(counts, xmin_search=True):
    """Clauset-Shalizi-Newman MLE via the powerlaw package."""
    try:
        import powerlaw
        # Each value c appears once per occurrence; powerlaw takes the raw sample
        # of counts (one number per token). For frequency analysis, we expand:
        # this is expensive for big vocabs; subsample if needed
        arr = np.asarray(counts, dtype=float)
        arr = arr[arr > 0]
        fit = powerlaw.Fit(arr, discrete=True, verbose=False)
        alpha = float(fit.alpha)
        xmin = float(fit.xmin)
        ks = float(fit.D)
        return {"alpha_csn": alpha, "xmin_csn": xmin, "ks_csn": ks}
    except Exception as e:
        log(f"  csn fit failed: {e}")
        return {"alpha_csn": np.nan, "xmin_csn": np.nan, "ks_csn": np.nan}


def fit_loglog(counts):
    """Direct log-log linear fit of rank vs. frequency."""
    counts = np.sort(counts)[::-1]
    ranks = np.arange(1, len(counts) + 1)
    lo = LOGLOG_RANK_MIN
    hi = min(LOGLOG_RANK_MAX, len(counts))
    if hi - lo < 50:
        return {"alpha_loglog": np.nan, "intercept_loglog": np.nan, "loglog_R2": np.nan}
    r_log = np.log(ranks[lo:hi])
    f_log = np.log(counts[lo:hi])
    A = np.column_stack([r_log, np.ones_like(r_log)])
    coef, resid, *_ = np.linalg.lstsq(A, f_log, rcond=None)
    slope, intercept = coef
    yhat = A @ coef
    ss_res = np.sum((f_log - yhat) ** 2)
    ss_tot = np.sum((f_log - f_log.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"alpha_loglog": -float(slope), "intercept_loglog": float(intercept), "loglog_R2": float(r2)}


def fit_mandelbrot(counts):
    """Fit f(r) = C * (r + b)^(-alpha) by nonlinear least squares on log-log."""
    counts = np.sort(counts)[::-1].astype(float)
    ranks = np.arange(1, len(counts) + 1, dtype=float)
    lo = 1
    hi = min(50_000, len(counts))
    r = ranks[lo - 1:hi]
    f = counts[lo - 1:hi]
    def model(r, logC, alpha, b):
        return logC - alpha * np.log(r + b)
    try:
        popt, _ = curve_fit(model, r, np.log(f), p0=[np.log(f[0]), 1.0, 1.0],
                            bounds=([0, 0.1, 0], [50, 3.0, 1e6]), maxfev=5000)
        return {"alpha_mb": float(popt[1]), "b_mb": float(popt[2])}
    except Exception as e:
        return {"alpha_mb": np.nan, "b_mb": np.nan}


def main():
    rows = []
    for pkl in sorted(COUNTS.glob("*.pkl")):
        stem = pkl.stem  # e.g. "cl100k_en"
        if "_" not in stem:
            continue
        tname, lang = stem.rsplit("_", 1)
        log(f"fitting {tname} / {lang}")
        with open(pkl, "rb") as fh:
            ctr: Counter = pickle.load(fh)
        counts = np.fromiter(ctr.values(), dtype=int)
        log(f"  vocab={len(counts):,}, total tokens={int(counts.sum()):,}")
        row = {"tokenizer": tname, "lang": lang,
               "vocab_observed": int(len(counts)),
               "total_tokens": int(counts.sum())}
        row.update(fit_csn(counts))
        row.update(fit_loglog(counts))
        row.update(fit_mandelbrot(counts))
        rows.append(row)
        log(f"  -> α_csn={row['alpha_csn']:.3f}, KS={row['ks_csn']:.3f}, "
            f"α_loglog={row['alpha_loglog']:.3f}, R²={row['loglog_R2']:.3f}, "
            f"α_mb={row['alpha_mb']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    log(f"wrote {OUT}")

    # Decision-rule summary for English
    en = df[df["lang"] == "en"]
    if len(en) > 0:
        log("\n=== ENGLISH SUMMARY ===")
        log(str(en[["tokenizer", "vocab_observed", "alpha_csn", "ks_csn", "alpha_loglog", "loglog_R2"]].to_string(index=False)))
        word = en[en["tokenizer"] == "word"]
        if len(word) > 0:
            a_word = float(word["alpha_loglog"].iloc[0])
            log(f"\nWord-level baseline α (log-log) = {a_word:.3f}")
            for _, r in en.iterrows():
                if r["tokenizer"] == "word":
                    continue
                shift = r["alpha_loglog"] - a_word
                log(f"  {r['tokenizer']:<10}  α={r['alpha_loglog']:.3f}  Δα={shift:+.3f}")
            # Decision rule
            shifts = en[en["tokenizer"] != "word"]["alpha_loglog"].values - a_word
            n_big_shift = int(sum(abs(s) > 0.05 for s in shifts if not np.isnan(s)))
            ks_ok = en["ks_csn"].notna() & (en["ks_csn"] < 0.05)
            ks_count = int(ks_ok.sum())
            verdict = "GRADUATE" if (n_big_shift >= 3) else "AMBIGUOUS"
            log(f"\nDECISION: {n_big_shift} tokenizers with |Δα|>0.05; KS<0.05 in {ks_count} -> {verdict}")


if __name__ == "__main__":
    main()
