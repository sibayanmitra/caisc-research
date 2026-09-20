# Paper 1 Pre-registered Hypotheses
**Timestamp**: 2026-05-28 (committed before Phase 5 head attribution + Phase 6 causal ablation)

All hypotheses are **soft** — we report effect sizes and confidence intervals, not hard pass/fail thresholds. This is our pre-commitment to the directional predictions.

---

## H1 — Linear separability of self-correction signal
A linear probe on the residual stream at some (layer L*, pre-correction offset δ ∈ {-3,-2,-1,0,+1}) distinguishes "about-to-self-correct" from matched "not-about-to-self-correct" positions meaningfully above chance (AUROC noticeably above 0.5 with non-overlapping 95% CI).

**Direction**: We expect peak AUROC at a mid-to-late layer (layers 14-24) and at offset δ = -1 or 0 (just before the marker).
**What we will report**: peak (L*, δ*), peak AUROC + bootstrap 95% CI, full layer × offset heatmap.

---

## H2 — Layer localization of the signal
The probe signal concentrates in a subset of layers rather than being uniform across all layers.

**Direction**: The normalized signal-per-layer should show a peak or cluster rather than a flat distribution.
**What we will report**: layer-wise AUROC bar chart, entropy of the per-layer AUROC distribution, top-3 layers' cumulative AUROC mass.

---

## H3 — Head sparsity at the peak layer
At peak layer L*, per-head probe AUROCs are non-uniform — a few heads do disproportionate work.

**Direction**: Right-skewed distribution of per-head AUROCs (skewness > 0.5), with top-3 heads clearly above median.
**What we will report**: per-head AUROC distribution, skewness statistic, top-3/5/10 vs median comparison.

---

## H4 — Causal necessity (ablation, destructive lever)
Ablating the top-k identified heads reduces self-correction frequency relative to:
  (a) baseline (no ablation)
  (b) matched random-k ablation (controls for "any ablation disrupts things")

**Direction**: Top-k ablation should reduce SC rate more than random-k.
**What we will report**: SC rate per condition (baseline, top-k, random-k for k∈{3,5,10}) with bootstrap CIs, Δ vs baseline.

---

## H5 — Causal sufficiency (steering, constructive lever)
Adding α × d_probe to the residual stream at L* during generation:
  - Increases self-correction frequency at α > 0
  - Decreases at α < 0
  - Random directions at matched norm show smaller effect

**Direction**: Monotonic relationship between α and SC rate; probe direction effect > random direction effect.
**What we will report**: steering curve (SC rate vs α), random baseline band, effect sizes.

---

## H6 (stretch) — Anticipation
On problems with longer pre-correction CoT, the probe signal at earlier offsets (δ = -5 or -10) is stronger than on problems with short pre-correction context.

**Direction**: AUROC at δ=-5 should correlate with problem difficulty / reasoning chain length.
**What we will report**: AUROC vs offset stratified by problem difficulty, if data permits.

---

*Pre-registered before Phase 5 (head attribution) and Phase 6 (causal ablation). Any refinements made after this commit will be documented explicitly in the paper and sanity_log.md.*
