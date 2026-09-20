# Idea Pool — 20 ~2-hour research spikes

Each idea below is sized for ~2 hours of focused work: enough for a real numerical experiment, a small data pull + analysis, or a derivation + validation. The bar at this scope: a spike that *succeeds* should yield a finding worth writing up, not just a curiosity confirmed.

Each entry has:
- **Claim** — the specific testable assertion
- **2-hour plan** — concrete execution path
- **Why an expert might be wrong** — what conventional wisdom says vs. what the spike might show
- **Kill criterion** — what result would make us drop the project

---

## Physics / fluids / thermo

### 1. Newton's-method basin fractal dimension
- **Claim**: For Newton's iteration on `x³ − 2x + 2`, the boundary between the "converges to real root" basin and the "stable 2-cycle" attractor has fractal (box-counting) dimension `1.5 ± 0.1`, comparable to the Julia set of a related map.
- **2-hour plan**: Iterate 10⁶ initial conditions on a [-3, 3] interval at 10⁴ resolution; classify outcomes; compute box-counting dimension at multiple scales; cross-check against analytic Julia-set dimension of the conjugate map.
- **Why expert might be wrong**: Most treatments call this "the Newton method fails for some x₀" without quantifying *how complicated* failure is. A non-integer dimension would be a clean result.
- **Kill criterion**: Dimension comes out at exactly 1 (smooth boundary) → boring; or compute fails to converge → infeasible.

### 2. Mpemba in a Markov chain: find a new regime
- **Claim**: The Lasanta-Sevilla-Prados 2017 Markov-chain Mpemba result extends to a regime where the *anti-Mpemba* (cool sample reaches hot temp faster than hot reaches cool) appears at specific noise levels — and this regime has never been pointed out.
- **2-hour plan**: Re-implement their 3-state Markov chain in Python; scan over thermal-coupling and noise parameters; identify regions of normal Mpemba, anti-Mpemba, and crossover.
- **Why expert might be wrong**: Most Mpemba follow-ups verify the original direction; the anti-Mpemba would be physically counterintuitive and might exist only in a narrow parameter slice.
- **Kill criterion**: All scans show monotonic relaxation only — no Mpemba at all.

### 3. Two-bottle drainage: analytic exponent for swirl regime
- **Claim**: For a swirled vortex drain, the height-vs-time curve follows `h(t) ∝ (T − t)^β` with `β ≈ 5/3`, not the Torricelli `β = 2`. The exponent shift comes from sustained surface depression at the vortex core.
- **2-hour plan**: Derive `β` from mass continuity + Bernoulli with a vortex-core boundary condition; numerically integrate the resulting ODE; compare to published outflow data (Andersen et al. 2003 on bathtub vortices).
- **Why expert might be wrong**: Torricelli is taught as universal for "draining a tank." The swirl-specific correction isn't in standard fluids texts.
- **Kill criterion**: Derivation produces `β ≈ 2` — no swirl correction → boring.

### 4. Optimal pendulum for energy harvesting
- **Claim**: For a pendulum coupled to a piezo-resistive damper, the optimal damping ratio for maximum energy extraction is not critical damping (ζ = 1) but a specific ζ* < 1 that depends on the amplitude regime.
- **2-hour plan**: ODE for damped pendulum + piezo load; compute energy harvested per cycle as a function of (ζ, A); find optimum analytically + numerically.
- **Why expert might be wrong**: Critical damping is the textbook answer for "fastest decay." But for *energy extraction*, slower decay over more cycles wins.
- **Kill criterion**: Optimum is at ζ = 1 → conventional wisdom holds.

### 5. Hot-air balloon scaling: critical diameter at altitude
- **Claim**: There's an analytic critical-diameter formula `D* ∝ (T_inside − T_outside)^(−1/3) × ρ_air^(−1/2)` below which a hot-air balloon cannot lift its own envelope mass at high altitude. Existing rule-of-thumb formulas don't have the altitude dependence right.
- **2-hour plan**: Derive lift equation including envelope mass scaling with `D²`; plot D* vs altitude using ISA atmosphere; cross-check against actual balloon spec sheets.
- **Why expert might be wrong**: Balloon design is empirical; a clean closed-form altitude-dependent lower bound is rarely published.
- **Kill criterion**: Existing balloon-design textbooks already have this exact derivation.

---

## Math / probability

### 6. Buffon's needle: optimal length-to-spacing ratio under finite budget
- **Claim**: The classical `L = D` (needle = line-spacing) is the variance-minimizing choice asymptotically but is suboptimal at small N. There's a finite-N optimal ratio that depends on log(N).
- **2-hour plan**: Derive the variance of the π-estimator as a function of `L/D` and N; numerically solve the optimal ratio for N = 10², 10³, 10⁴, 10⁵; verify with Monte Carlo.
- **Why expert might be wrong**: Variance bounds are taught for asymptotic N. Finite-N effects in MC estimators get glossed over.
- **Kill criterion**: Finite-N optimum equals asymptotic for all N ≥ 10² → no story.

### 7. Birthday paradox with Zipfian priors
- **Claim**: When birthdays follow a Zipfian (rather than uniform) prior, the 50% collision threshold drops from 23 to roughly 23 × (1 − α/2) where α is the Zipf exponent. There's a clean derivation that hasn't been published in this form.
- **2-hour plan**: Derive the analytic threshold for Zipf(α); verify by Monte Carlo at α ∈ {0, 0.5, 1, 1.5}; compute the implied "effective N" formula.
- **Why expert might be wrong**: The birthday paradox is taught with uniform priors; the non-uniform case is usually handled with vague "use entropy" handwaves.
- **Kill criterion**: My derivation gives a known formula (e.g., Wagner 2002 already has it).

### 8. Condorcet cycle probability: closed-form scaling exponent at m candidates
- **Claim**: For random uniform voter preferences with m candidates and n→∞ voters, `P(cycle exists)` grows as `1 − c × m^(−γ)` for a specific exponent γ ≈ 0.5 that hasn't been pinned down in the social-choice literature.
- **2-hour plan**: Monte Carlo at `(n, m) ∈ {1000} × {3, 4, 5, ..., 20}`, 10⁵ trials each; fit `1 − P(cycle) ∝ m^(−γ)`; check against Gehrlein 1983 (which only has exact m=3).
- **Why expert might be wrong**: Gehrlein's exact result for m=3 is well-known but the scaling for higher m is usually stated as "grows toward 1" without a rate.
- **Kill criterion**: The exponent comes out at 0 (saturation) — no scaling story.

---

## Biology / ecology

### 9. Heart-rate × lifespan: bird-mammal split with sleep as covariate
- **Claim**: The "1.5 billion heartbeats" rule's residual structure across vertebrates is best explained by total sleep hours per lifetime, not by metabolic rate or activity level. Birds + bats are the cleanest test cases.
- **2-hour plan**: Compile heart rate, max lifespan, body mass, and total sleep hours from AnAge + Siegel sleep database; multi-regression `log(beats_per_lifetime) ~ class + log(M) + log(sleep_hrs)`; identify the strongest residual driver.
- **Why expert might be wrong**: The aging literature focuses on metabolic-rate hypotheses; sleep duration as a regulator is underweighted.
- **Kill criterion**: Sleep hours don't add explanatory power beyond mass + class.

### 10. Why leaves are green: noise-vs-efficiency Pareto frontier
- **Claim**: The Marosvolgyi & van Gorkom 2010 "noise minimization" argument and the "efficiency penalty" argument define a Pareto frontier. Real chlorophyll sits at a specific point on this frontier that corresponds to noon-sky illumination, not full AM1.5.
- **2-hour plan**: Build full model: AM1.5 solar spectrum × Gaussian-pigment absorption with center λ_c and width σ. Compute (efficiency, noise) for full (λ_c, σ) grid. Plot Pareto frontier; overlay chlorophyll a/b absorption maxima. Test against alternate sky models (clear-noon, overcast, dawn).
- **Why expert might be wrong**: The "green is suboptimal" claim ignores the noise term; the "green minimizes noise" claim ignores the efficiency cost. The Pareto framing reconciles both.
- **Kill criterion**: Chlorophyll position doesn't track any specific illumination spectrum.

### 11. Wikipedia fame decay: power law vs. stretched exponential
- **Claim**: Pageview decay for deceased historical figures follows a stretched exponential `exp(−(t/τ)^β)` with `β ≈ 0.5`, not a power law. The crossover happens at decadal timescales.
- **2-hour plan**: Wikipedia REST API for pageviews; 100 figures who died ~5–30 years ago; fit power law vs. stretched exp on each; AIC comparison.
- **Why expert might be wrong**: Popular-attention decay is often modeled as power-law (Wu & Huberman 2007). A clean stretched-exponential result would have implications for cultural memory.
- **Kill criterion**: Power law wins in >70% of cases.

---

## Economics / markets

### 12. Sunshine-on-returns: anomaly decay and possible inversion
- **Claim**: The Hirshleifer-Shumway 2003 sunshine effect on NYC stocks decayed monotonically from 1995–2010 and *inverted* post-2015: cloudy days now slightly outperform. This is a quantifiable case of arbitrage-then-overcorrection.
- **2-hour plan**: yfinance + NOAA NYC cloud cover, 1990–2025. Rolling 5-year regressions of S&P daily returns on cloud index. Plot coefficient over time; test for inversion.
- **Why expert might be wrong**: Anomalies usually "fade" not "flip." A documented inversion would be evidence of arbitrage overshoot.
- **Kill criterion**: Effect just decays to zero, no inversion → still publishable but weaker.

### 13. Indian monsoon-onset effect on Sensex
- **Claim**: Years with earlier-than-median monsoon onset show systematically higher Sensex returns in May–July (~+2pp annualized), and the effect persists in 1995–2024 data despite the economy's reduced agricultural share.
- **2-hour plan**: IMD monsoon onset dates 1990–2024; Sensex daily data; compute May-July returns conditioned on onset timing. Control for global market direction.
- **Why expert might be wrong**: Agricultural share of GDP fell from ~30% to <15%, so the effect "should" have weakened; persistence would be surprising.
- **Kill criterion**: No statistically significant difference (p > 0.1).

### 14. Election prediction-market efficiency
- **Claim**: Polymarket / Manifold odds at T-30 days have systematic favorite-longshot bias of ~3–5 percentage points, similar to sports betting markets.
- **2-hour plan**: Scrape 50+ resolved Polymarket election markets; for each, get the T-30-day odds and the outcome; bin by predicted probability and compute realized frequency; fit calibration curve.
- **Why expert might be wrong**: Prediction markets are often claimed to be well-calibrated. Sports betting is biased; whether political markets share this property is debated.
- **Kill criterion**: Calibration curve is on the diagonal (no bias).

---

## Computer science / information theory

### 15. LLM tokenizer Zipfian shift
- **Claim**: Modern subword tokenizers (GPT-4 cl100k, Llama-3) systematically flatten Zipf's exponent from ~1.0 (word-level) to ~0.8 (token-level). The shift is a quantifiable "information-density loss" induced by BPE.
- **2-hour plan**: Tokenize English Wikipedia (5GB sample) with cl100k, Llama-3 tokenizer, GPT-2 BPE, and word-level. Plot rank-frequency log-log; fit Zipf exponent for each. Compare to character-level lower bound.
- **Why expert might be wrong**: BPE is "optimal under compression" — but Zipf preservation is a different criterion. The exponent shift quantifies the gap.
- **Kill criterion**: All tokenizers preserve exponent ≈ 1 → no story.

### 16. Cross-language compression: morphology vs. orthography
- **Claim**: When compressing UDHR across 20+ languages, agglutinative languages (Turkish, Finnish, Tamil) achieve *higher* bits-per-character than analytic ones (English, Mandarin), confirming morphology increases per-character information density.
- **2-hour plan**: UDHR in 20 languages from unicode.org/udhr. LZMA + per-character entropy estimate via N-gram model. Map to morphological complexity index from WALS / Bickel 2014.
- **Why expert might be wrong**: Naive intuition says "more morphology = more redundancy = compresses better." The opposite is what information theory predicts but it isn't shown cleanly in any one paper.
- **Kill criterion**: No correlation between morphology and entropy → analytic vs. synthetic doesn't matter at this scale.

### 17. AI-generated code: Zipfian and identifier-diversity fingerprint
- **Claim**: AI-generated Python code (GPT-4, Claude on HumanEval) has a steeper Zipf exponent and ~30% lower identifier diversity than human GitHub code, even controlling for problem.
- **2-hour plan**: Pull 1000 HumanEval-style problems with both human + AI solutions. Tokenize at AST-identifier level. Compute Zipf exponent, type-token ratio, and identifier-entropy per solution. Compare distributions.
- **Why expert might be wrong**: Most AI-detection work focuses on surface n-grams. Structural code fingerprints might be more robust to paraphrase.
- **Kill criterion**: AI and human distributions overlap completely.

### 18. LLM output entropy vs. human text
- **Claim**: GPT-4 outputs have ~15% lower gzip compression ratio than matched human text on the same topic, and the gap *grows* with output length (autoregressive accumulation of self-similarity).
- **2-hour plan**: Generate 500 GPT-4 essays on Wikipedia topics; pull the matched human Wikipedia paragraphs; gzip both; measure ratio vs. length. Cross-check with bzip2 and LZMA.
- **Why expert might be wrong**: "AI text is mode-collapsed" is anecdotal. A quantified length-dependent ratio would be a robust detector.
- **Kill criterion**: Ratios are within 2% across all lengths.

---

## Game theory / collective behavior

### 19. Bartoš coin-flip bias replication on cricket toss data
- **Claim**: The Bartoš et al. 2023 same-side bias (~51%) replicates on international cricket tosses (where "heads/tails" is recorded), provided we control for who calls and the side of the captain at the toss.
- **2-hour plan**: Scrape cricsheet.org for 500+ matches; extract toss outcomes; statistical test (1-sided binomial). Power calculation upfront.
- **Why expert might be wrong**: Bartoš's claim got significant attention but only one large replication exists. Cricket data is a clean independent test set.
- **Kill criterion**: Power is too low (need >5000 flips for the effect, cricket has ~500).

### 20. Chess Elo top-rating gap superlinear expansion
- **Claim**: The gap between world #1 Elo and median Elo of the top 100 has grown super-linearly in 1985–2025, faster than would be predicted by simple Elo-system inflation. The gap signals the emergence of "Magnus-class outliers" not captured by Elo's design.
- **2-hour plan**: FIDE rating history (publicly archived); compute monthly (#1 − median top 100) gap; fit linear and power-law trends; bootstrap CIs.
- **Why expert might be wrong**: Elo inflation is well-known but the *gap structure* is rarely analyzed. A super-linear trend would suggest top-class structural separation.
- **Kill criterion**: Trend is linear or flat.

---

## How to read this list at 2-hour scope

**Highest-novelty + AI-natural** (best CAISc fit):
- #15 LLM tokenizer Zipfian shift
- #17 AI-vs-human code fingerprint
- #18 LLM output entropy vs. human
- #14 Election market efficiency
- #12 Sunshine-anomaly inversion

**Highest-novelty + non-AI domain** (still CAISc-viable):
- #10 Leaf-color Pareto frontier
- #9 Heart-rate × lifespan × sleep
- #3 Vortex drainage exponent
- #2 Mpemba new regime
- #20 Chess Elo gap expansion

**Cleanest execution, modest novelty**:
- #6 Buffon optimal ratio
- #7 Zipf-birthday derivation
- #8 Condorcet scaling exponent
- #11 Wikipedia fame decay shape

**Highest data-availability risk**:
- #19 Cricket toss (power limit)
- #14 Polymarket scrape (rate limits, archive coverage)
- #17 Code-pair dataset (need matched human-AI)

**Highest derivation/theory risk** (might just rediscover known):
- #1 Newton basin dimension
- #3 Vortex exponent
- #7 Zipf-birthday

---

## My pre-critic ranking

If I had to pick 5 right now without further input, my honest top 5 at 2-hour scope:

1. **#10 Leaf-color Pareto frontier** — strongest novelty, AI-native synthesis problem
2. **#15 LLM tokenizer Zipfian shift** — clean, AI-native, decisive result either way
3. **#9 Heart-rate × lifespan × sleep** — strong potential surprise in residual structure
4. **#12 Sunshine-anomaly inversion** — anomaly decay+inversion is a publishable story
5. **#17 AI-vs-human code fingerprint** — practical impact + CAISc-native

Alternates: #18, #14, #3.

Worth flagging: #15, #17, #18 are correlated (all "LLM statistical fingerprint") — picking only one keeps the portfolio diverse.
