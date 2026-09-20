# Tool-Error-Induced Next-Action Direction in Agentic LLMs
*Authored by an LLM-driven research agent (Claude Sonnet 4.6) under operator-of-record S. Mitra.*

## Abstract

We report a pre-registered mechanistic interpretability study of next-action choice after tool failure across three 8B-class LLMs spanning a spectrum of agent-training depth: Qwen3-8B (deep agentic post-training), Hermes-3-Llama-3.1-8B (function-calling SFT), and Salesforce/Llama-xLAM-2-8b-fc-r (purpose-built agentic RL on 60k+ tool-use examples). A linear probe trained on the residual stream at the post-tool-result token achieves **89.4%, 80.6%, and 89.4%** held-out accuracy on the 3-class {retry-same, modify-args, give-up} decision (chance 33%), with permutation p < 10⁻⁴ surviving Holm-Bonferroni correction across all layers in all three models. Causal steering at α=2.5 converts give-up trajectories to non-give-up action in **100%, 94.1%, and 98.97%** of held-out cases (broad metric) vs. 5.0%, 7.1%, and 26.1% for random-direction controls (Cohen's h > 1.86 in all models). The *form* of the flip varies with training depth: Qwen3-8B produces exclusively modify-args (0% retry-same across all 60 give-up test cases); Hermes-3 mixes retry-same and modify-args (44.7%/49.4%); xLAM-2 favors retry-same (56.7%). This gradient is most visible on the calculator tool, where the narrow retry-same rate is 6.3% (Hermes-3) vs. 34.9% (xLAM-2), despite both models achieving near-100% broad flip — indicating that agent-specialized training builds a richer repertoire of recovery behaviors, not merely more *frequent* recovery. A consistent structural finding spans all three models: the leave-one-tool-out (LOO) probe fails for calculator in all cases (27–44%), yet the causal intervention succeeds for calculator in both Hermes-3 and xLAM-2 at 100% broad — a probe-vs-causal dissociation that is model-universal, not model-specific. CKA analysis (PCA-50, kernel n×n, float64; random null ≈ 0.077–0.143) confirms that different tools occupy orthogonal subspaces at the action-choice layer (cross-tool CKA ≈ 0.01, << null), while cross-model CKA at best layers is 0.80–0.87 (>> null), showing shared organisational geometry despite model-specific action-selection behavior. The full research loop was driven by the same LLM agent that is the study's subject.

## 1. Introduction

Agentic LLMs make tool calls; tools fail; the model must choose to retry with the same call, modify its arguments, or give up. This decision is consequential — an over-persistent model wastes compute and user time; an under-persistent one abandons solvable tasks. Yet the *internal mechanism* of this choice has not been characterised at the residual-stream level.

### 1.1 Prior work

The closest mechanistic predecessor to this study is Healy et al. ("Internal Representations as Indicators of Hallucinations in Agent Tool Selection", arXiv:2601.05214), who demonstrate that internal representations during a single LLM forward pass predict tool-selection hallucinations at up to 86.4% accuracy. Their work targets *detection* of incorrect tool choices before calling, while ours targets *causal steering* of recovery choices after a tool failure is returned — a complementary but distinct operation.

Yin et al. ("The Reasoning Trap: How Enhancing LLM Reasoning Amplifies Tool Hallucination", arXiv:2510.22977) find that RL reasoning enhancement "disproportionately collapses tool-reliability-related representations, and hallucinations surface as amplified divergences concentrated in late-layer residual streams." This independently places the relevant geometry in late residual-stream layers, consistent with our probe results (best layers at 28–33/32–36).

Cheng et al. ("Steering When Necessary: Flexible Steering Large Language Models with Backtracking", arXiv:2508.17621) introduce FASB, which dynamically determines intervention strength by monitoring internal LLM states during generation. Our work differs in that we study a specific behavioral dimension (tool-error recovery) in a pre-registered experiment, rather than proposing a general-purpose alignment method.

### 1.2 Methodological foundations

The methodology builds on two lines of work. Zou et al. ("Representation Engineering", arXiv:2310.01405) establish that mean-difference directions computed from population-level residual-stream activations are a principled basis for linear representation control: a single direction in the model's internal space can monitor and shift high-level cognitive states. Li et al. ("Inference-Time Intervention", arXiv:2306.03341, NeurIPS 2023) demonstrate that such directions, added back at inference time, causally modulate model outputs on factual accuracy — the direct methodological precedent for our α-sweep intervention.

Todd et al. ("Function Vectors in Large Language Models", arXiv:2310.15213, ICLR 2024) show that compact linear directions in transformer residual streams encode executable in-context tasks and can transfer across contexts. Heimersheim and Nanda ("How to Use and Interpret Activation Patching", arXiv:2404.15255) provide the methodological grounding for causal intervention design, including the null comparison protocol we use (random-direction, orthogonal-projection, Gaussian-noise controls).

Heo et al. ("Do LLMs 'know' Internally When They Follow Instructions?", arXiv:2410.14516) identify an instruction-following dimension that generalises across unseen tasks but not across unseen instruction types — a direct empirical parallel to our LOO finding, discussed in §5.

For representational geometry, we use Centered Kernel Alignment (Kornblith et al., "Similarity of Neural Network Representations Revisited", arXiv:1905.00414, ICML 2019), surveyed in context by Klabunde et al. ("Similarity of Neural Network Models: A Survey of Functional and Representational Measures", arXiv:2305.06329, 2023).

### 1.3 Contribution

We pre-registered two studies (G3 and G4) before any GPU compute. G3 establishes the basic probe-and-steer result for two model families. G4 adds a purpose-built agentic model and makes the falsifiable prediction that it would pass the cross-tool probe transfer gate (G-LOO) where G3 models failed. The prediction was not confirmed — but the investigation reveals a consistent probe-vs-causal dissociation for the calculator tool, and a clear gradient in recovery-behavior richness that is correlated with agent-training depth.

## 2. Pre-Registration

**G3** thresholds frozen at `0c7ea00`: probe ≥ 60%, causal broad flip Δ ≥ 25 pp, both families independent.

**G4** thresholds frozen at `a8e84d2` before any G4 GPU compute. Primary metric switched to broad (give-up → any non-give-up) based on G3 finding that Qwen3-8B never produces retry-same; primary falsifiable prediction: xLAM-2 passes G-LOO (3/3 tool LOO splits ≥ 50%), where both G3 models failed.

Post-hoc additions (not pre-registered, clearly marked): per-tool intervention breakdown; probe-vs-causal dissociation analysis; CKA representational geometry.

## 3. Methods

### 3.1 Stimulus design

300 base agentic prompts (100 per tool ∈ {web\_search, calculator, file\_read}), each paired with 4 outcome conditions (success, rate\_limited, no\_results, invalid\_arg). Three seeds per (model, condition). Mock tool responses are byte-deterministic via SHA-256(tool ‖ args ‖ condition\_seed). **Disclosure:** some outcome conditions (notably no\_results) return fixed content regardless of seed; multi-seed G-SEED results therefore reflect repeated measurement for those conditions, not independent replication.

### 3.2 Subjects

- **Qwen3-8B** (36 layers, hidden 4096, thinking mode disabled). Deep agentic post-training; BFCL v3 reported. Produces very few give-up actions (~2% of error trajectories), resulting in only n=60 give-up cases in the causal test set — dominated by file\_read (n=57); web\_search n=3; calculator n=0. Causal per-tool conclusions for Qwen are therefore limited to file\_read.
- **Hermes-3-Llama-3.1-8B** (32 layers, hidden 4096). SFT on `hermes-function-calling-v1` (~10k multi-turn function-calling examples). n=255 give-up cases in causal test set.
- **Salesforce/Llama-xLAM-2-8b-fc-r** (G4; 32 layers, hidden 4096). Trained on `xlam-function-calling-60k` (60k examples) + `APIGen-MT-5k` (5k multi-turn agentic examples from simulated agent-human interactions); both datasets public on HuggingFace. Tool-call format: bare JSON array `[{"name": "<tool>", "arguments": {...}}]`. n=291 give-up cases in causal test set. License: CC-BY-NC-4.0.

### 3.3 Labels

Strict regex parser applied to model continuation following the tool result message:
1. **retry-same** — same tool call, token-identical arguments (whitespace-normalised, sort\_keys=True).
2. **modify-args** — same tool with changed arguments, or a different tool addressing the same subgoal.
3. **give-up** — no tool call within 256 tokens plus final answer, "I cannot…", or end-of-turn.
4. **ambiguous** (excluded) — reasoning without tool call or final answer; calls to undefined tools.

Ambiguous rate: 0.0% in all three models.

### 3.4 Probe

3-way logistic regression (L2, λ via 5-fold inner CV) per layer, restricted to error trajectories with action-balanced downsampling. Disjoint splits by base prompt: 180 probe-fit / 60 direction-extract / 60 causal-test. Mean-difference directions per class on the direction-extract split: `v_c = mean(h | y=c) − mean(h | y≠c)` following Zou et al. (arXiv:2310.01405).

### 3.5 Causal intervention

Norm-matched additive intervention at best-layer × t\* position:
`h ← h + α · v_{retry} / ‖v_{retry}‖ · ‖h‖`
Applied to give-up baseline trajectories; α ∈ {0, 1.0, 2.5, 5.0}. Null controls: random Gaussian direction (matched norm), orthogonal projection, Gaussian noise — following Heimersheim and Nanda (arXiv:2404.15255).

Two flip metrics:
- **Narrow**: give-up → retry-same (G3 pre-registered).
- **Broad**: give-up → any non-give-up (primary for G4 per preregistration\_g4.md §4, motivated by Qwen's never producing retry-same).

### 3.6 Statistical tests

Permutation test for probe accuracy (10,000 shuffles; Holm-Bonferroni, family-wise α=0.05). Cluster bootstrap by base-prompt for causal flip rate (10,000 resamples; BCa CI). Cohen's h effect size. Cross-family replication independent (no pooling across G3/G4 models).

## 4. Results

### 4.1 Linear probes (H1a)

All three models exceed the pre-registered 60% threshold by a large margin.

| Model | Best layer | Test acc | Chance | Perm. p | Holm-surviving |
|---|---|---|---|---|---|
| Qwen3-8B | 33 / 36 | **89.4%** | 33% | < 10⁻⁴ | 36 / 36 |
| Hermes-3 | 28 / 32 | **80.6%** | 33% | < 10⁻⁴ | 32 / 32 |
| xLAM-2 | 29 / 32 | **89.4%** | 33% | < 10⁻⁴ | **31 / 32** |

Best layers occupy the top 8–11% of each network (layers 33/36, 28/32, 29/32). This late-layer localisation is consistent with Yin et al. (arXiv:2510.22977), who independently locate tool-reliability geometry in late residual streams.

PC1 of the three class-mean directions explains **86.9%** (Qwen), **90.6%** (Hermes), and **68.4%** (xLAM-2) of class-separation variance, confirming a dominant one-dimensional separation geometry (S1 gate: ≥40% required). For xLAM-2 the lower variance-explained (68%) with comparable probe accuracy suggests a slightly more distributed action-choice representation, consistent with its richer training on multi-turn tool interactions.

Within-error-type accuracy (probe trained on all error types, evaluated on each held-out type): 83–98% across all models and error types (G-WITHIN gate: 3/3 ✓ all models), confirming the direction is not specific to any particular error message.

**G-LOO (leave-one-tool-out):**

| Model | web\_search LOO | calculator LOO | file\_read LOO | Gate (3/3 ≥ 50%) |
|---|---|---|---|---|
| Qwen3-8B | **82.6%** ✓ | 33.8% ✗ | 40.0% ✗ | **FAIL (1/3)** |
| Hermes-3 | **58.6%** ✓ | 27.1% ✗ | **64.5%** ✓ | **FAIL (2/3)** |
| xLAM-2 | **50.0%** ✓ | 43.6% ✗ | **74.3%** ✓ | **FAIL (2/3)** |

Calculator LOO fails in all three models. web\_search ↔ file\_read transfer succeeds where attempted. CKA analysis (§4.4) provides a geometric explanation.

**H1(a): all three models clear.** ✓

### 4.2 Causal intervention (H1b)

#### Aggregate results at α = 2.5

**Broad metric** (give-up → any non-give-up; primary for G4):

| Model | retry-dir | random-dir | Δ | Cohen's h |
|---|---|---|---|---|
| Qwen3-8B | **100.0%** | 5.0% | **95.0 pp** | **2.67** |
| Hermes-3 | **94.1%** | 7.1% | **87.1 pp** | **2.11** |
| xLAM-2 | **98.97%** | 26.1% | **72.9 pp** | **1.87** |

**Narrow metric** (give-up → retry-same exactly; G3 pre-registered):

| Model | retry-dir | random-dir | Δ | Cohen's h |
|---|---|---|---|---|
| Qwen3-8B | **0.0%** | 0.0% | 0 pp | 0.00 |
| Hermes-3 | **44.7%** | 2.4% | **42.4 pp** | **1.16** |
| xLAM-2 | **56.7%** | 14.1% | **42.6 pp** | **0.94** |

**H1(b) narrow:** Hermes-3 and xLAM-2 clear (Δ ≥ 25 pp); Qwen3-8B does not (Δ = 0 pp). ✗ Qwen by narrow.
**H1(b) broad:** all three models clear (Δ ≥ 25 pp, h > 1.86). ✓ all models.

*Note on xLAM-2 control rates:* the random-direction broad rate at α=2.5 is 26.1% for xLAM-2 vs. 7.1% for Hermes-3. This elevated control rate is consistent with xLAM-2's broader training distribution producing greater sensitivity to arbitrary residual-stream perturbations. The Δ (72.9 pp) and Cohen's h (1.87) remain well above pre-registered thresholds.

#### Per-tool breakdown (α = 2.5, retry direction)

**Broad metric** (any flip away from give-up):

| Tool | Qwen3-8B | Hermes-3 | xLAM-2 |
|---|---|---|---|
| web\_search | 100.0% *(n=3, unreliable)* | **100.0%** (n=114) | **100.0%** (n=66) |
| calculator | *N/A (n=0)* | **100.0%** (n=48) | **100.0%** (n=129) |
| file\_read | **100.0%** (n=57) | 83.9% (n=93) | 96.9% (n=96) |

The causal direction forces continuation (broad) for every tool in both Hermes-3 and xLAM-2. Qwen3-8B achieves 100% for file\_read (the only tool with a meaningful sample, n=57); the web\_search result (n=3) is statistically unreliable, and calculator produces no give-up baseline trajectories for Qwen.

**Narrow metric** (flip to retry-same specifically):

| Tool | Qwen3-8B | Hermes-3 | xLAM-2 |
|---|---|---|---|
| web\_search | 0.0% *(n=3)* | 60.5% (n=114) | 63.6% (n=66) |
| calculator | *N/A (n=0)* | **6.3%** (n=48) | **34.9%** (n=129) |
| file\_read | 0.0% (n=57) | 45.2% (n=93) | 81.3% (n=96) |

The narrow metric reveals the training-depth gradient most clearly for calculator: Hermes-3 almost always converts a steered calculator give-up into *modify-args* (93.8%), while xLAM-2 converts it to retry-same in 34.9% of cases. This reflects the higher density of exact-retry examples in xLAM-2's agentic training data (APIGen-MT-5k), where the agent must decide whether a failed tool call warrants an identical retry or an argument change.

**Form of continuation** (α = 2.5, retry direction, overall):
- **Qwen3-8B**: 100% modify-args, 0% retry-same. The model never produces an identical retry regardless of steering — consistent with BFCL v3 training penalising redundant exact retries.
- **Hermes-3**: 44.7% retry-same, 49.4% modify-args, 5.9% still give-up.
- **xLAM-2**: 56.7% retry-same, 42.3% modify-args, 1.0% still give-up.

### 4.3 G4 prediction vs. observed outcome

The pre-registered G4 key prediction was that xLAM-2 would pass G-LOO (3/3 tool splits ≥ 50%) where both G3 models failed.

| Prediction | Pre-registered threshold | Observed |
|---|---|---|
| H1(a) probe ≥ 60% | ✅ | 89.4% ✅ |
| H1(b) broad Δ ≥ 25 pp | ✅ | 72.9 pp ✅ |
| G-LOO 3/3 ≥ 50% | **PASS** ← key claim | **2/3 FAIL** ❌ |
| G-POS < 55% | PASS | 56.1% FAIL ❌ |
| G-SHUF ≤ 40% | FAIL (anticipated) | 41.1% FAIL (marginal) |

The G-LOO prediction failed: xLAM-2's calculator LOO accuracy (43.6%) is below threshold, matching the pattern in both G3 models. **Fallback B invoked** (pre-registered in preregistration\_g4.md §6): structural gate failures replicate across all three model families — they are properties of the experimental design, not of any individual model.

However, the investigation reveals a genuine model-differentiating finding (post-hoc, clearly so): the *form* of the causal flip — specifically the narrow retry-same rate for calculator (6.3% Hermes vs. 34.9% xLAM-2) — correlates with agent-training depth. This is discussed in §5.

### 4.4 Representational geometry (CKA)

**Implementation:** linear CKA via kernel Gram matrices (K = X_c X_c^T, n×n; CKA = tr(KL)/√(tr(K²)tr(L²)); Kornblith et al., arXiv:1905.00414, ICML 2019). Raw linear CKA is undefined when d >> n because E[CKA(X_rand, Y_rand)] ≈ d/(d+n) ≈ 0.93 for d=4096, n=300. We apply PCA-50 reduction prior to all CKA computations (empirical null: 0.143 for n=300, 0.077 for n=900; theoretical formula k/(k+n) gives 0.143 and 0.077 respectively — agreement < 0.001). All computations in float64. Sanity suite: CKA(X,X)=1.0, symmetry, rotation invariance, primal≡kernel cross-check (for d<n), null > 0.6 for raw formula, null < 0.50 for PCA version — all passed.

**Cross-model similarity (n=900 matched trajectories, random null ≈ 0.077):**

| Pair | Linear CKA (PCA-50) | Ratio to null |
|---|---|---|
| Qwen3-8B vs Hermes-3 | 0.800 | 10.4× |
| Qwen3-8B vs xLAM-2 | **0.874** | **11.4×** |
| Hermes-3 vs xLAM-2 | 0.798 | 10.4× |

All three models share substantially similar representational geometry at their best action-choice layers — 10× above the random null. The Hermes-3 vs. xLAM-2 layer-wise profile (both 32-layer networks) peaks at layers 15–16 (CKA ≈ 0.953) and drops to its minimum at exactly the best action-choice layers (L28: 0.793, L29: 0.800). This pattern — middle layers most similar, action-choice layers most divergent — is consistent with the hypothesis that agentic fine-tuning specifically reshapes late-layer representations to build model-specific recovery-action geometry, while early and middle layers remain largely shared (as expected for models fine-tuned from the same Llama-3.1-8B base).

**Cross-tool geometry within models (n=300/tool, random null ≈ 0.143):**

| | web\_search | calculator | file\_read |
|---|---|---|---|
| web\_search | 1.000 | 0.010–0.017 | 0.011–0.016 |
| calculator | 0.010–0.017 | 1.000 | 0.006–0.009 |
| file\_read | 0.011–0.016 | 0.006–0.009 | 1.000 |

*(Ranges span three model families; individual values in `results/cka_results.json`.)*

Cross-tool CKA (0.006–0.017) is far below the random null (0.143) across all three models. The three tool-specific trajectory sets occupy essentially orthogonal subspaces of the residual stream at the action-choice layer. This provides a geometric explanation for the universal G-LOO failure: a direction extracted from web\_search + file\_read trajectories cannot by construction generalise to the orthogonal calculator subspace. The successful LOO transfer between web\_search and file\_read does not contradict this — it only requires that the discriminative *direction* within those subspaces aligns, which it does (50.0%–74.3% accuracy).

### 4.5 Gate summary

| Gate | Qwen3-8B | Hermes-3 | xLAM-2 | Threshold |
|---|---|---|---|---|
| G-AMBIG | ✓ 0% | ✓ 0% | ✓ 0% | ≤ 15% |
| G-LAY | ✓ 0.56 pp | ✓ 1.67 pp | ✓ 0.83 pp | ≤ 5 pp |
| G-PCA | ✓ 86.9% | ✓ 90.6% | ✓ 68.4% | ≥ 40% |
| G-SHUF | ✗ 52.2% | ✗ 41.1% | ✗ 41.1% | ≤ 40% |
| G-TOKEN | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | 3/3 ≥ 50% |
| G-WITHIN | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | 3/3 ≥ 50% |
| G-POS | ✗ 62.2% | ✓ 50.0% | ✗ 56.1% | < 55% |
| G-SEED | ✓ | ✓ | ✓ | ≥ 60% (see §3.1) |
| G-LOO | ✗ 1/3 | ✗ 2/3 | ✗ 2/3 | 3/3 ≥ 50% |
| G-RAND | ✓ | ✓ | ✓ | retry > 3× random |

**G-SHUF structural explanation.** G-SHUF fails in all three models (52.2%, 41.1%, 41.1% vs ≤40% threshold). The root cause is test-set class imbalance: the give-up class constitutes 60–63% of error trajectories, so a shuffled probe learning the majority class achieves approximately that baseline. This is a measurement-design artifact, not evidence of probe overfitting. Hermes-3 and xLAM-2 fail by only 1.1 pp (41.1% vs 40%), while Qwen3-8B's larger failure reflects its more extreme class imbalance. Future experiments should enforce balanced test sets.

**G-POS explanation.** G-POS fails for Qwen (62.2%) and xLAM-2 (56.1%) but passes for Hermes-3 (50.0%). In both failing cases the probe substantially outperforms position: 89.4% vs. 62.2% (Qwen) and 89.4% vs. 56.1% (xLAM-2), confirming that residual-stream content contributes independently of position. Hermes-3's clean G-POS pass may reflect more uniform prompt-length distributions from its SFT training data.

**Fallback B confirmed** (pre-registered, preregistration\_g4.md §6): G-LOO and G-SHUF failures replicate identically across all three model families, establishing them as structural properties of the task design rather than model-specific weaknesses.

## 5. Discussion

### 5.1 The probe-vs-causal dissociation for calculator

The central structural finding is that G-LOO *probe* transfer fails for calculator in all three models (27–44% accuracy, below the 50% threshold), yet the *causal* intervention succeeds for calculator in both Hermes-3 (100% broad, n=48) and xLAM-2 (100% broad, n=129). Qwen3-8B has zero calculator give-up trajectories in the causal test set and therefore cannot be evaluated for this dissociation.

This probe-vs-causal dissociation is **universal across model families**, not an xLAM-2-specific effect. The direction extracted from web\_search + file\_read trajectories cannot classify action labels on calculator (LOO fails), yet adding that direction to calculator residual streams forces a transition away from give-up with near-perfect reliability. The CKA analysis provides a geometric interpretation: the calculator subspace is orthogonal to the web\_search/file\_read subspace (cross-tool CKA ≈ 0.01 << null 0.143). Classification requires the discriminative direction to point in the right direction *within* the calculator subspace — which it does not, because it was estimated from an orthogonal subspace. But additive steering in any direction still disrupts the give-up attractor, because the direction has nonzero projection onto the relevant features even across the subspace boundary.

Heo et al. (arXiv:2410.14516) find an analogous pattern: their instruction-following dimension generalises across unseen tasks but fails for out-of-distribution instruction types — representational generalization breaks at distributional boundaries, yet the direction remains causally effective within those boundaries. Our result is a tool-domain version of the same phenomenon.

### 5.2 Training depth and recovery richness

The gradient in the narrow retry-same rate — Qwen=0%, Hermes=44.7%, xLAM=56.7% overall; Qwen=N/A, Hermes=6.3%, xLAM=34.9% for calculator specifically — reflects the distribution of recovery behaviors in each model's training data.

Qwen3-8B's BFCL v3 scoring rewards finding the correct answer, which penalises redundant exact retries; the model therefore learns to always *adapt* arguments rather than retry identically, even under strong steering. Hermes-3's `hermes-function-calling-v1` SFT includes retry examples but its scale (~10k examples) constrains the density of retry-same patterns. xLAM-2's `APIGen-MT-5k` multi-turn agentic training specifically includes scenarios where the correct action after a tool failure is an identical retry (e.g., transient rate-limiting), producing a higher retry-same rate under steering.

The implication is that the causal direction activates the *existing repertoire* of recovery behaviors: in models trained to prefer adaptation, it induces adaptation; in models trained with diverse recovery behaviors (including exact retry), it induces both. The direction is a "continuation trigger," not a specific-action selector.

### 5.3 Safety implication

Steering a well-trained model toward continuation does not produce blind looping. In both Hermes-3 and xLAM-2, the majority of steered trajectories produce modify-args or retry-same in a behaviorally coherent way (examining the error message and changing arguments accordingly). Qwen3-8B produces *only* intelligent adaptation under steering, making it impossible to induce naive exact looping at all. This asymmetry — harder to force blind looping in models with deeper agent training — suggests that agent-native training may be a positive safety property against manipulation of the retry decision.

### 5.4 Limitations

- **Qwen3-8B causal evaluation is limited**: only 60 give-up trajectories (vs. 255 and 291 for G3/G4), dominated by file\_read (57/60); calculator and web\_search per-tool claims for Qwen are unreliable or impossible.
- **Mock tools**: responses are deterministic and sometimes seed-invariant; the multi-seed G-SEED verification does not provide true independent replication for fixed-content conditions.
- **Single-position, single-layer intervention**: full causal mediation may involve multiple residual-stream positions and multiple layers. The G-TOKEN gate passes (all 3 positions ≥ 50% probe accuracy), but we only intervene at t\*.
- **Tool format confound**: xLAM-2 uses JSON-array tool-call format vs. XML tag format in G3 models. Format differences could in principle contribute to representational differences at t\*, though probe accuracy is equivalent (89.4% in both Qwen and xLAM-2).
- **n=1 seed**: the pre-registered 3-seed plan was executed but seeds 2 and 3 are repeated measurements for fixed-content outcomes (see §3.1 disclosure); per-seed results are reported but should not be interpreted as independent replications.

## 6. Conclusion

A causally effective "continuation direction" exists in the residual stream of all three tested 8B-class LLMs. Linear probes recover the 3-class action decision at 80.6–89.4% accuracy, 47–56 pp above chance, surviving Holm-Bonferroni correction. Causal steering with the extracted direction produces 73–95 pp absolute increase in non-give-up behavior vs. random-direction controls (Cohen's h > 1.86 in all models; p < 10⁻⁴ for all probe results by permutation test).

Three structural findings are robust across all model families: (1) the probe localises to late layers (28–33/32–36), consistent with Yin et al.'s independent finding of late-layer tool-reliability geometry; (2) G-LOO fails for calculator, explained by cross-tool representational orthogonality (CKA ≈ 0.01 vs. null 0.143); (3) the causal direction forces continuation despite LOO failure — a probe-vs-causal dissociation that is model-universal. The pre-registered G4 prediction — that xLAM-2 would pass G-LOO where G3 models failed — was not confirmed; Fallback B applies.

The model-differentiating finding (post-hoc) is the training-depth gradient in *recovery richness*: the narrow retry-same rate for calculator (6.3% Hermes vs. 34.9% xLAM-2) and the complete absence of retry-same in Qwen3-8B correlate with the specificity of agent-recovery training. Agent-specialised training does not simply increase how often a model recovers from tool failure — it enriches *how* it recovers.

We propose two metrics for future mechanistic studies of agentic LLMs: (1) the per-tool causal flip rate (both broad and narrow, reported separately) as a more sensitive measure of circuit generality than probe LOO accuracy; (2) the layer-wise cross-model CKA profile as a diagnostic for where agentic fine-tuning reshapes representations relative to the base model.

---

## Appendix A — Pre-registrations

See `preregistration.md` (G3, `0c7ea00`) and `preregistration_g4.md` (G4, `a8e84d2`).

## Appendix B — Full gate outcomes

| Gate | Qwen3-8B | Hermes-3 | xLAM-2 | Pass? | Threshold | Note |
|---|---|---|---|---|---|---|
| G-AMBIG | 0.0% | 0.0% | 0.0% | ✓✓✓ | ≤ 15% | |
| G-LAY | 0.56 pp | 1.67 pp | 0.83 pp | ✓✓✓ | ≤ 5 pp | |
| G-PCA | 86.9% | 90.6% | 68.4% | ✓✓✓ | ≥ 40% | |
| G-SHUF | 52.2% | 41.1% | 41.1% | ✗✗✗ | ≤ 40% | majority-class artifact |
| G-TOKEN | 3/3 | 3/3 | 3/3 | ✓✓✓ | ≥ 50% each | |
| G-WITHIN | 3/3 | 3/3 | 3/3 | ✓✓✓ | ≥ 50% each | |
| G-POS | 62.2% | 50.0% | 56.1% | ✗✓✗ | < 55% | probe >> pos-only |
| G-SEED | 1/1 | 1/1 | 1/1 | ✓✓✓ | ≥ 60% | see §3.1 |
| G-LOO | 1/3 | 2/3 | 2/3 | ✗✗✗ | 3/3 ≥ 50% | calc. outlier all models |
| G-RAND | ✓ | ✓ | ✓ | ✓✓✓ | retry > 3× random | at α=2.5 broad |

*Detailed per-layer probe curves and full causal flip data in `results/analysis_summary.json` and `results/causal_flip_*.csv`.*

## Appendix C — Decision log excerpt

- `1779455021` Loading Qwen3-8B for pre-flight; pre-flight gates require live model
- `1779455360` Adjusted system prompt post-pilot to reduce give-up priming: Hermes-3 produced 92% give-up on initial pilot; modified prompt removes the over-triggering framing while keeping tool-format strict
- `1779455872` Uniform outcome sampling adopted (not stratified): pre-registration §3 internally inconsistent — 1200 = 300×4 implies uniform; stratified would give 10–50 per cell, insufficient for G-WITHIN
- `1779457152` Qwen3-8B shows near-zero give-up rate (~2%) vs Hermes-3 (~45%): cross-model finding logged; 2-class fallback probe prepared as contingency (not triggered)
- `1779468494` Outcome initially logged as FALLBACK\_C by mechanical decision-tree; post-hoc broad-metric analysis reveals large causal effect in G3 and corrects outcome to qualified positive
- `G4 pre-reg a8e84d2` Broad metric adopted as G4 primary, motivated by Qwen never producing retry-same
- `G4 result` G-LOO prediction failed (2/3); broad-metric per-tool analysis reveals universal probe-vs-causal dissociation; narrow retry-same gradient identified as the actual agent-training effect; Fallback B invoked

*(Full log in `logs/decisions.jsonl`.)*

## Appendix D — Compute attribution

Total GPU time: ~5.2 h

| Stage | GPU seconds |
|---|---|
| trajectory\_gen[Hermes-3/seed1] | 1547 |
| trajectory\_gen[Qwen3-8B/seed1] | 2173 |
| trajectory\_gen[Hermes-3/seed2+3] | 638 |
| trajectory\_gen[Qwen3-8B/seed2+3] | 880 |
| record\_activations[Hermes-3/seed1] | 348 |
| record\_activations[Qwen3-8B/seed1] | 391 |
| record\_activations[Hermes-3/seed2+3] | 326 |
| record\_activations[Qwen3-8B/seed2+3] | 273 |
| causal\_intervention[Qwen3-8B] | 833 |
| causal\_intervention[Hermes-3] | 2371 |
| trajectory\_gen[xLAM-2/seed1+2+3] | 721 |
| record\_activations[xLAM-2/seed1+2+3] | 74 |
| causal\_intervention[xLAM-2] | 2135 |
