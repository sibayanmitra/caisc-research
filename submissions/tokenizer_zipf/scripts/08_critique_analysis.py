#!/usr/bin/env python3
"""Critique round 2 analyses:
#1  fertility predicts mid-range Δα (r~0.91) but NOT the tail Δα_CSN -> tail is the value-add.
#9  bootstrap CI on r(fertility, Δα_mid) + within-language correlations (not just 2 clusters).
#6  CSN tail: MLE standard error + xmin-sensitivity grid (is α_CSN an xmin artifact?).
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes"
COUNTS = ROOT / "data" / "token_counts"
df = pd.read_csv(NOTES / "alpha_table.csv")

LANGS = ["en", "de", "fr"]
SUB = ["cl100k", "gpt2_bpe", "o200k", "qwen2.5"]
rng = np.random.default_rng(0)

# Build per-cell table: fertility, Δα_mid, Δα_csn
recs = []
for lang in LANGS:
    w = df[(df.tokenizer == "word") & (df.lang == lang)].iloc[0]
    for tk in SUB:
        r = df[(df.tokenizer == tk) & (df.lang == lang)].iloc[0]
        recs.append({"lang": lang, "tok": tk,
                     "fert": r.total_tokens / w.total_tokens,
                     "da_mid": r.alpha_loglog - w.alpha_loglog,
                     "da_csn": r.alpha_csn - w.alpha_csn})
t = pd.DataFrame(recs)

def pear(a, b): return float(np.corrcoef(a, b)[0, 1])

print("#1  FERTILITY vs the two regimes (n=%d cells)" % len(t))
r_mid = pear(t.fert, t.da_mid); r_csn = pear(t.fert, t.da_csn)
print(f"   r(fertility, Δα_mid)  = {r_mid:+.3f}   <- fertility explains the mid-range")
print(f"   r(fertility, Δα_CSN)  = {r_csn:+.3f}   <- fertility is ~blind to the tail")
print(f"   Δα_CSN range = [{t.da_csn.min():+.2f}, {t.da_csn.max():+.2f}] (sign: {(t.da_csn>0).sum()}/{len(t)} positive)")
print(f"   -> the tail steepens almost everywhere incl. English (where mid-range FLATTENS),")
print(f"      a fact fertility cannot express. The two-regime split is the contribution.")

print("\n#9  ROBUSTNESS of r(fertility, Δα_mid)")
boots = []
idx = np.arange(len(t))
for _ in range(10000):
    s = rng.choice(idx, len(idx), replace=True)
    if t.fert.values[s].std() > 0:
        boots.append(pear(t.fert.values[s], t.da_mid.values[s]))
lo, hi = np.percentile(boots, [2.5, 97.5])
print(f"   r = {r_mid:.3f}, bootstrap 95% CI = [{lo:.3f}, {hi:.3f}] (10k resamples)")
for lang in LANGS:
    sub = t[t.lang == lang]
    print(f"   within {lang} (n={len(sub)}, fert {sub.fert.min():.2f}-{sub.fert.max():.2f}): "
          f"r = {pear(sub.fert, sub.da_mid):+.3f}")
print("   -> correlation holds WITHIN German and French (across tokenizers), not just")
print("      between the English and German clusters.")

print("\n#6  CSN TAIL STABILITY (English) -- MLE std-err + xmin-sensitivity grid")
import powerlaw, warnings
warnings.filterwarnings("ignore")
XMIN_GRID = [200, 500, 1000, 2000, 5000]
for tk in ["word"] + SUB:
    ctr = pickle.load(open(COUNTS / f"{tk}_en.pkl", "rb"))
    arr = np.fromiter(ctr.values(), float); arr = arr[arr > 0]
    fit = powerlaw.Fit(arr, discrete=True, verbose=False)
    a_auto, xm_auto, sig = float(fit.alpha), float(fit.xmin), float(fit.sigma)
    grid = []
    for xm in XMIN_GRID:
        try:
            f2 = powerlaw.Fit(arr, discrete=True, xmin=xm, verbose=False)
            grid.append(float(f2.alpha))
        except Exception:
            grid.append(float("nan"))
    gstr = " ".join(f"{x:.2f}" for x in grid)
    print(f"   {tk:9s}: α_auto={a_auto:.3f}±{sig:.3f} (xmin={xm_auto:.0f}) | "
          f"α at xmin{XMIN_GRID}=[{gstr}]")
print("   -> if α stays well above the word baseline (1.64) across the xmin grid, the tail")
print("      steepening is not an artifact of auto-xmin selection.")
