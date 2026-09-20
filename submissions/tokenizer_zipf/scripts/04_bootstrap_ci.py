#!/usr/bin/env python3
"""Bootstrap 95% CIs for the log-log Zipf exponent alpha, and fertility vs Delta-alpha.

Bootstrap: multinomial (token-instance) resampling of N tokens over the observed
type distribution, refitting the same log-log slope (ranks 11..min(10000,V)) each time.
This is the standard iid token-level bootstrap; it captures sampling variability of the
rank-frequency curve (text autocorrelation is ignored, so CIs are lower bounds on width).
"""
import pickle
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COUNTS = ROOT / "data" / "token_counts"
NOTES = ROOT / "notes"
LOGLOG_RANK_MIN = 10
LOGLOG_RANK_MAX = 10_000
B = 500
rng = np.random.default_rng(0)


def alpha_loglog(counts):
    counts = np.sort(counts)[::-1]
    ranks = np.arange(1, len(counts) + 1)
    lo, hi = LOGLOG_RANK_MIN, min(LOGLOG_RANK_MAX, len(counts))
    if hi - lo < 50:
        return np.nan
    r = np.log(ranks[lo:hi]); f = np.log(counts[lo:hi])
    A = np.column_stack([r, np.ones_like(r)])
    slope = np.linalg.lstsq(A, f, rcond=None)[0][0]
    return -float(slope)


rows = []
for pkl in sorted(COUNTS.glob("*.pkl")):
    stem = pkl.stem
    if "_" not in stem:
        continue
    tname, lang = stem.rsplit("_", 1)
    with open(pkl, "rb") as fh:
        ctr = pickle.load(fh)
    counts = np.fromiter(ctr.values(), dtype=np.int64)
    counts = counts[counts > 0]
    N = int(counts.sum()); V = len(counts)
    p = counts / counts.sum()
    obs = alpha_loglog(counts.astype(float))
    boots = []
    for _ in range(B):
        rc = rng.multinomial(N, p)
        a = alpha_loglog(rc[rc > 0].astype(float))
        if not np.isnan(a):
            boots.append(a)
    boots = np.array(boots)
    lo_ci, hi_ci = np.percentile(boots, [2.5, 97.5])
    rows.append({"tokenizer": tname, "lang": lang, "N": N, "vocab": V,
                 "alpha": round(obs, 4), "ci_lo": round(float(lo_ci), 4),
                 "ci_hi": round(float(hi_ci), 4), "se": round(float(boots.std()), 4)})
    print(f"{tname:10s} {lang}  alpha={obs:.3f}  95% CI=[{lo_ci:.3f}, {hi_ci:.3f}]  (N={N:,}, V={V:,})", flush=True)

df = pd.DataFrame(rows)
df.to_csv(NOTES / "alpha_bootstrap.csv", index=False)

# ---- fertility vs Delta-alpha ----
wN = df[df.tokenizer == "word"].set_index("lang")["N"].to_dict()
wA = df[df.tokenizer == "word"].set_index("lang")["alpha"].to_dict()
sub = df[~df.tokenizer.isin(["word", "byte"])].copy()
sub["fertility"] = sub.apply(lambda r: r.N / wN[r.lang], axis=1)
sub["dalpha"] = sub.apply(lambda r: r.alpha - wA[r.lang], axis=1)
print("\n=== fertility (subword tokens / word tokens) vs Delta-alpha ===")
print(sub[["tokenizer", "lang", "fertility", "dalpha"]].to_string(index=False))
r_all = np.corrcoef(sub.fertility, sub.dalpha)[0, 1]
nohi = sub[sub.lang != "hi"]
r_nohi = np.corrcoef(nohi.fertility, nohi.dalpha)[0, 1]
print(f"\nPearson r(fertility, Delta-alpha) all = {r_all:.3f} (n={len(sub)})")
print(f"Pearson r excluding Hindi             = {r_nohi:.3f} (n={len(nohi)})")
sub.to_csv(NOTES / "fertility_dalpha.csv", index=False)
print("\nwrote notes/alpha_bootstrap.csv and notes/fertility_dalpha.csv")
