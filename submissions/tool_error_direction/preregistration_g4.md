# Pre-Registration — G4: Agent-Native Model Replication

**Status:** FROZEN at git commit before any GPU computation for G4.
**Authored by:** Claude (Sonnet 4.6), 2026-05-22.
**Operator-of-record:** Anonymous Author.
**Parent study:** G3 (`preregistration.md`), committed at tag `PREREG`.

This document fixes — *before any GPU computation* — every threshold, decision rule, and possible-paper-abstract for the G4 follow-up study. G4 tests whether the "continuation direction" found in G3 generalises to a model specifically designed and trained for agentic tool-use, using a publicly documented training dataset.

---

## 1. Motivation and design rationale

G3 found a causally-valid "continuation direction" in both Qwen3-8B and Hermes-3-Llama-3.1-8B using the broad flip metric (give-up → any non-give-up), but with three gate failures (G-SHUF, G-LOO, G-POS) and cross-model behavioral difference (Qwen: modify-args; Hermes: retry-same). The post-hoc hypothesis is that models trained with *dense, explicit agentic RL* — where retry-vs-giveup utility estimation is causally necessary for reward — will produce:

- Cleaner gate profile (especially G-LOO: better cross-tool generalisation)
- Higher broad-metric causal flip rate at lower α (less steering force needed)
- Lower positional confound (G-POS)

G4 tests this by adding a purpose-built function-calling model with a public, verifiable training dataset.

---

## 2. Subject

- **Target model C:** `Salesforce/Llama-xLAM-2-8b-fc-r`
  - Architecture: Llama-3.1-8B base, 32 layers, hidden 4096.
  - Training data: `Salesforce/xlam-function-calling-60k` (60k function-calling examples) + `Salesforce/APIGen-MT-5k` (5k multi-turn agentic examples generated via simulated agent-human interactions). Both datasets public on HuggingFace.
  - Tool-call format: bare JSON array `[{"name": "<tool>", "arguments": {...}}]` (no XML wrappers).
  - Accessible without HuggingFace token. License: CC-BY-NC-4.0.
- **Same G3 models for direct comparison:** `Qwen/Qwen3-8B` and `NousResearch/Hermes-3-Llama-3.1-8B` results are taken from the already-completed G3 run — **no re-run of G3 models**.

---

## 3. Stimulus design (identical to G3)

- Same 300 base prompts, same 4 outcome conditions, same mock-tool harness.
- Same 3-way prompt split (probe-fit 180 / direction-extract 60 / causal-test 60 base prompts).
- Same 3 seeds: seed 1 full 1200 trajectories; seeds 2 and 3 causal-test subset only (240 trajectories each).
- **Format difference from G3:** xLAM-2 uses a model-specific system prompt asking for JSON array tool-call format and `<final_answer>` give-up tags. Label parser handles both `<tool_call>` (G3 models) and JSON array (xLAM-2) formats.

---

## 4. Primary hypothesis H1 for G4

Identical thresholds to G3, applied to xLAM-2 independently:

### H1(a) — Decodability
- Probe accuracy ≥ 60% (3-class, chance 33%), permutation p < 0.001, Holm-Bonferroni across all layers.

### H1(b) — Causality (broad metric, post-registered from G3 finding)
- Because G3 revealed the narrow metric (retry-same only) missed Qwen3-8B's effect, **H1(b) for G4 uses the broad metric as primary**: give-up → any non-give-up flip rate exceeds random-direction control by ≥ 25 pp absolute, Cohen's h ≥ 0.5. The narrow metric (retry-same only) is also reported.

### H1(c) — Agent-native hypothesis (G4-specific)
- xLAM-2 clears H1(a) AND H1(b) independently.
- **Comparative prediction:** xLAM-2 passes more gates than the G3 average. Specifically, we predict xLAM-2 will pass G-LOO (≥ 3/3 tool splits ≥ 50%) where both G3 models failed. This is the key falsifiable claim.

---

## 5. Gate thresholds (identical to G3)

| Gate | Threshold |
|---|---|
| G-AMBIG | ambiguous rate ≤ 15% |
| G-LAY | best-layer accuracy within 5 pp of mean of neighbors |
| G-PCA | PC1 variance ≥ 40% |
| G-SHUF | shuffled-label probe accuracy ≤ 40% |
| G-TOKEN | accuracy ≥ 50% at all 3 token positions |
| G-WITHIN | ≥ 50% accuracy within each of ≥ 2/3 error types |
| G-POS | position-only baseline < 55% |
| G-SEED | all evaluable seeds individually ≥ 60% |
| G-LOO | all 3 LOO tool splits ≥ 50% (xLAM-2 predicted to pass; G3 both failed) |
| G-RAND | retry-direction flip rate ≥ 3× random-direction at matched norm |

---

## 6. Pre-registered fallbacks

**Fallback A (decodable, not causal):** If H1(a) clears but H1(b) fails even with the broad metric: the "continuation direction" is a readout of decision-relevant features but rank-1 patching at a single layer is insufficient to override the decision. Report as evidence that xLAM-2's agentic training distributes the circuit across layers or positions more than G3 models.

**Fallback B (gate failures replicate G3 pattern):** If xLAM-2 shows the same gate failure profile as G3 (G-SHUF, G-LOO fail), with similar G-PCA and probe accuracy: the gate failures are structural features of the experimental design (class imbalance, calculator outlier), not model-specific. Report as evidence that the findings from G3 are robust and not artefacts of model choice.

**Fallback C (informative null):** If both H1(a) and H1(b) fail with clean sanity gates: xLAM-2 either requires a different intervention position/layer or the direction is not linearly accessible at the post-tool-result token. Report with the gate outcomes as an informative null.

---

## 7. Comparison table (pre-registered expected outcomes)

| Metric | G3 Qwen3-8B | G3 Hermes-3 | G4 xLAM-2 (predicted) |
|---|---|---|---|
| Probe acc | 89.4% | 80.6% | ≥ 60% (H1a) |
| Broad flip Δ | 95 pp | 87 pp | ≥ 25 pp (H1b) |
| G-LOO | FAIL (1/3) | FAIL (2/3) | **PASS (3/3) — key prediction** |
| G-POS | FAIL | PASS | PASS (predicted) |
| G-SHUF | FAIL (class imb.) | FAIL (class imb.) | FAIL (same structural cause) |
| Flip form | modify-args (100%) | mixed | unknown |

---

## 8. Decision rule

The G4 paper contribution is confirmed if:
- xLAM-2 clears H1(a) and H1(b) with the broad metric AND
- xLAM-2 passes G-LOO where both G3 models failed

The G4 finding is still publishable under Fallback B (same gates fail = structural, not model-specific) or Fallback A (decodable not causal = deeper circuit).
