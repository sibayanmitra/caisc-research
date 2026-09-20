# Pre-Registration — G3: Tool-Error-Induced Next-Action Direction in Agentic LLMs

**Status:** FROZEN at git commit immediately after this file is created. No edits permitted before main run completes.
**Authored by:** Claude (Opus 4.7), 2026-05-22.
**Operator-of-record:** Anonymous Author.

This document fixes — *before any GPU computation* — every threshold, decision rule, statistical test, and possible-paper-abstract that the experiment may produce. The final results paper will publish a side-by-side diff of this file against `results.md`.

---

## 1. Subjects

- **Target model A:** `Qwen/Qwen3-8B` — 36 layers, hidden 4096.
- **Target model B:** `NousResearch/Hermes-3-Llama-3.1-8B` — 32 layers, hidden 4096.
- **Inference precision:** `bfloat16`.
- **Inference framework:** Hugging Face Transformers 5.9.0, PyTorch 2.8.0+cu128.

## 2. Stimulus design

- **300 base agentic prompts:** 100 per tool ∈ {web_search, calculator, file_read}.
- **4 outcome conditions** per trial, stratified per tool (see plan §3, table).
- **Mock tool harness:** byte-deterministic via SHA-256(tool ‖ args ‖ condition_seed) → response table.
- **Trajectories per (model, seed):** 1200 (= 300 prompts × 4 outcomes).
- **Seeds:** 3. Seed 1 generates full 1200; seeds 2 and 3 regenerate only the 240 trajectories whose base prompts fall in the causal-intervention split.

## 3. Prompt splits — by base prompt, not by trajectory

- **Probe-fit:** 180 prompts (60%).
- **Direction-extraction:** 60 prompts (20%).
- **Causal-intervention held-out:** 60 prompts (20%).
- Splits are deterministic from `seed=42` and fixed for the entire experiment.

## 4. Primary hypothesis H1 — exact thresholds

Given target model M ∈ {A, B}, the hypothesis claims existence of layer ℓ*_M satisfying all three sub-conditions:

### H1(a) — Decodability

- 3-way logistic regression probe (L2, λ via 5-fold inner CV).
- Training set: error trajectories only (`success` excluded). Action-balanced downsampling to equal counts per {retry-same, modify-args, give-up}.
- Held-out accuracy on direction-extraction split (also error-only, action-balanced): **≥ 60%** (chance = 33.3%).
- Permutation test (10,000 label shuffles): **p < 0.001** for that layer.
- Holm-Bonferroni correction across all probed layers, family-wise α = 0.05.

### H1(b) — Causality

- Recover `v_retry`, `v_modify`, `v_give_up` via mean-difference per class on the **direction-extraction split only** (disjoint from probe-fit).
- Norm-matched intervention at layer ℓ*_M, position t*: h ← h + α · (v_retry / ‖v_retry‖) · ‖h‖, propagate forward.
- α sweep: {0, 1.0, 2.5, 5.0}.
- Causal test prompts: 60 held-out prompts × 4 outcome conditions × 3 seeds = 720 trajectories per model.
- Restrict causal-flip evaluation to trajectories whose baseline (α=0) action is `give-up`.
- **Primary success criterion:** at some α* ∈ {1.0, 2.5, 5.0}, the give-up → retry-same flip rate exceeds the random-direction control (at matched norm) by **≥ 25 percentage points absolute**, with **Cohen's h ≥ 0.5**, and the paired-bootstrap (cluster bootstrap by base-prompt, 10,000 resamples, BCa CI) **95% CI is disjoint** from the random-direction 95% CI.
- Multiple-comparison correction: Holm across the 3 null types {random-direction, orthogonal-projection, Gaussian-noise} at α = 0.05.

### H1(c) — Cross-family replication

Both M = A and M = B independently satisfy H1(a) AND H1(b). **No pooling.** A result that holds in only one family is reported as such but is not H1.

### H1 — pass/fail decision rule

H1 is *cleared* iff for both M ∈ {A, B}: (i) probe accuracy ≥ 60% with permutation p < 0.001 at some layer; (ii) causal-flip criterion met at some α*; (iii) all 3 seeds individually clear the probe threshold for that model.

If H1 is not cleared, the experiment defaults to one of the three Fallback claims (§7), selected by the decision tree in §8.

---

## 5. Secondary predictions

- **S1 — Subspace structure.** Top-1 PC on class-mean differences explains ≥ 40% of class-separation variance.
- **S2 — Cross-error generalization.** Probe trained on one error type (e.g., `rate_limited`) decodes action on a different error type (e.g., `no_results`) at ≥ 50% accuracy.
- **S3 — Cross-tool generalization (leave-one-out).** Probe trained on {web_search, calculator} achieves ≥ 50% action accuracy on `file_read` trajectories; rotated for other 2 LOO splits.

Secondary predictions are reported regardless of H1 outcome.

---

## 6. Sanity gates — exact thresholds

| Gate | Threshold | Source of failure |
|---|---|---|
| G-LAY | Best-layer probe acc within 5 pp of mean of ℓ-1 and ℓ+1 | Spurious single-layer peak |
| G-RAND | `v_retry` flip rate ≥ 3× `v_rand` flip rate (matched norm) | Direction not specific |
| G-SHUF | Shuffled-label probe acc ≤ 40% | Probe overfits noise |
| G-WITHIN | Within-error-type probe ≥ 50% on ≥ 2 of 3 error types | Direction is error-type artifact |
| G-POS | Token-position-only baseline probe acc < 55% | Length confound |
| G-CACHE | `use_cache=True` vs `use_cache=False` residual byte-equal on 5 trajectories per model | Recording artifact |
| G-AMBIG | `ambiguous` label rate ≤ 15% | Rubric broken |
| G-PCA | Top-1 PC of class-mean diffs ≥ 40% variance | Direction is a subspace |
| G-CROSSFAMILY | Both A and B independently clear H1(a) and H1(b) | Single-family illusion |
| G-SEED | All 3 seeds individually clear probe threshold per model | Variance hidden by averaging |
| G-TOKEN | Probe acc ≥ 50% at t*, t*+5, and pre-next-action position | Position-specific artifact |
| G-LOO | Leave-one-tool-out probe ≥ 50% (secondary) | No tool generalization |

---

## 7. Three pre-registered fallback abstracts (written in full)

### Fallback A — "Decodable but not causal"

> We study the next-action choice {retry-same, modify-args, give-up} made by an instruction-tuned 8B-scale agentic LLM at the token immediately following a tool-call result. Training a 3-way linear probe on error-only, action-balanced trajectories from two model families (Qwen3-8B and Hermes-3-Llama-3.1-8B), we find that next-action information is linearly decodable from the residual stream with held-out accuracy 65–75% (chance 33%). However, **rank-1 causal intervention along the recovered "retry direction" fails to convert give-ups into retries beyond a random-direction null** at matched norm. The decision is therefore *represented* but is not *mediated* by a rank-1 direction; we present evidence that the action choice is distributed across a higher-dimensional subspace and across multiple token positions, and we provide subspace-dimensionality and token-sweep diagnostics. This is a negative result for the rank-1 linear-causal hypothesis applied to agentic action selection, and the experiment was conducted end-to-end by an LLM-driven research agent.

### Fallback B — "Outcome-perception dissociated from action"

> We attempted to localize a residual-stream direction mediating next-action choice in an agentic LLM after tool failure. The probe trained on error-only trajectories with action-balanced sampling **fails to clear chance**, while a probe trained without restricting to errors (i.e., free to use tool-result outcome as a feature) achieves 90%+ accuracy. Causal intervention along the recovered direction shifts the model's *interpretation* of tool outcomes (success vs error) but does **not** shift its next-action choice. We conclude that outcome perception and action selection are *mechanistically dissociated* at the post-tool-result token in current agentic LLMs — a novel finding with safety-monitoring implications. The dissociation is demonstrated in two model families and is robust to a battery of sanity gates. Experimental design, execution, and analysis were driven by an LLM-based research agent.

### Fallback C — "Informative null"

> We pre-registered the hypothesis that a rank-1 residual-stream direction in an agentic LLM causally mediates next-action choice {retry-same, modify-args, give-up} after a tool-call failure. Despite a 12-gate sanity-check suite, two model families, three seeds, and the standard linear-probe + mean-difference + causal-intervention methodology, we find **no evidence** for either the decodability or the causal claim at workshop-significance thresholds. We present full pre-registration, all 12 gate outcomes, statistical-test artifacts, and a structured agentic-process log demonstrating that the null is informative (gates clean, statistical power adequate) and not a methodological artifact. We argue that this constitutes a useful negative result for the linear-representation hypothesis as applied to agentic decision-making and provides a template for credibility-instrumented agent-driven research.

---

## 8. Decision tree — which abstract ships

After all gates and stats are computed:

1. If H1 cleared in both families AND G-CROSSFAMILY, G-SEED, G-RAND, G-PCA pass → **Primary** (the H1 abstract — to be written post hoc into `results.md` only if cleared).
2. Else if probe clears H1(a) for both families but causal fails (any G-RAND or causal threshold) → **Fallback A**.
3. Else if error-only probe fails BUT success-inclusive probe ≥ 80% AND causal direction shifts outcome-perception not action → **Fallback B**.
4. Else → **Fallback C**.

Decision is **mechanical** — no post-hoc fiddling. Gate outcomes determine the abstract.

---

## 9. Statistical tests — frozen specification

- **Probe accuracy permutation test.** 10,000 shuffles of the action labels, holding `h_{ℓ, t*}` fixed. Null distribution per layer; observed > 99th percentile required.
- **Causal-flip rate cluster bootstrap.** Resample at the base-prompt level (not the trajectory level), 10,000 replicates, BCa CI.
- **Effect size.** Cohen's h between intervened and unintervened flip rates. Threshold h ≥ 0.5.
- **Multiple comparisons.** Holm-Bonferroni: across layers (probe); across null types (causal). Family-wise α = 0.05.
- **No pooling across model families.** Each family is an independent replication for the purposes of H1(c).

---

## 10. Pre-flight gates (before main run starts)

All five must pass:

1. **Pre-registration committed.** This file at git HEAD before any trajectory generated.
2. **Mock-tool determinism.** 100 identical (tool, args, condition_seed) calls return byte-identical responses.
3. **Tool-call parser κ.** ≥ 0.85 against 100 manually-labeled trajectories.
4. **Forward-pass byte-equality.** `use_cache=True` vs `use_cache=False` residual streams byte-equal on 5 trajectories per model.
5. **Pilot ambiguous rate.** 50-trajectory pilot per model yields ≤ 15% `ambiguous` labels.

A pre-flight failure HALTS the main run. Recovery is documented in `logs/gate_failures.jsonl`.

---

## 11. Scope cuts (binding, not advisory)

These cuts are enforced if wall-clock at hour 5 is ≥ 65% spent:

1. α-sweep reduced from {0, 1.0, 2.5, 5.0} to {0, 2.5} (keep extremes).
2. Held-out causal-test prompts reduced from 60 to 40.
3. Probe initial layer scan: every 4th layer (was every 2nd).
4. Drop S3 (leave-one-tool-out) if necessary; report as future work.

If hour 6.5 wall-clock and still over budget: drop Hermes-3 cross-family replication; downgrade headline to single-family with H1(c) flagged as not tested.

---

## 12. Agentic-loop instrumentation commitments

- Every decision logged to `logs/decisions.jsonl` with (timestamp, decision, rationale-one-sentence, observed-outcome).
- Every gate failure + recovery logged to `logs/gate_failures.jsonl`.
- Every compute interval tagged with the decision motivating it, logged to `logs/compute.jsonl`.
- At each branch point with ≥ 2 viable options, log the rejected alternative and the rejection reason.
- All scripts run via `make all`; single-command reproduction from raw seeds.
- Any human intervention required is documented in-paper.

---

*— end pre-registration —*
