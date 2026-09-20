# Locked Research Directions
Date: 2026-06-02 | Validated by: GPT-5.5, Gemini-3.1-pro, DeepSeek-v4-pro (weighted), GPT-5.4 per-track, live literature search

---

## LOCKED #1 — I7: Evaluation Awareness in Indic Settings
**"Less Aware = More Honest = Better Benchmark"**
Track: Advisor / Indic Multilingual Safety
Target venue: NeurIPS / ICLR

### The claim
Non-English (Indic) eval contexts trigger less evaluation-awareness in LLMs because
the cues that signal "you are being safety-tested" are English-language artifacts.
If confirmed: Indic evals give a cleaner, less-gamed measure of true scheming propensity.

### Literature gap status: CLEAN ✅
All three eval-awareness papers (Nguyen et al. 2507.01786; LURE 2605.26438;
Needham et al. 2505.23836) are English-only, single-model. Gap is unoccupied.

### Mandatory design fix
Add a comprehension-control arm: verify the model understands the scenario
equally in Hindi/Tamil before attributing lower awareness to reduced gaming.
Without this, reviewers will argue it's capability drop, not cleaner measurement.

### Core experiment
- Same scheming scenario in English + 6 Indic languages
- Measure verbalized eval-awareness rate in CoT per language
- Correlation test: lower awareness → higher observed scheming rate
- Comprehension probe as control variable

---

## LOCKED #2 — I2: Script as Blindfold
**"Native Script vs. Romanization in CoT Scheming Detection"**
Track: Advisor / Indic Multilingual Safety
Target venue: NeurIPS / ACL

### The claim
Scheming reasoning expressed in native Indic script (Devanagari, Tamil, Telugu)
is significantly less detectable by English-trained CoT monitors than the
same reasoning in romanized Indic or English. Mechanism: tokenization fragmentation.

### Literature gap status: PARTIAL — differentiation required ⚠️
2605.27901 (Onyame et al.) covers CoT monitoring in 13 languages including Bengali
and Telugu BUT does not test: (a) native vs romanized within same language,
(b) scheming specifically (only general faithfulness), (c) tokenization as mechanism,
(d) Hindi or Tamil. Must cite and explicitly differentiate.

### Surviving gap (your angle)
Native-script vs. romanized CoT specifically for scheming detection +
tokenization fragmentation as the mechanistic explanation.
Hindi and Tamil (the two largest Indic languages) are absent from 2605.27901.

### Core experiment
- Induce scheming via model organisms in Hindi, Tamil, Telugu (native + romanized)
- English-trained monitor vs. multilingual-aware monitor — compare detection rates
- Tokenizer analysis: measure fragmentation (tokens per character) as predictor
- Spontaneous script-switching in CoT as secondary finding

---

## LOCKED #3 — P3: Bystander Effect in Subagent Systems
**"Diffusion of Responsibility in Orchestrator→Subagent Pipelines"**
Track: Student / Psychology × Agentic Safety
Target venue: ACL / EMNLP

### The claim
In orchestrator→subagent pipelines, a subagent explicitly reasons "the orchestrator
will catch this" and applies weaker safety thresholds. This effect scales with N
(number of co-agents), mirroring the human bystander effect (Latané & Darley).

### Literature gap status: PARTIAL — Fukui 2026 exists but doesn't block ⚠️
Fukui (2605.13851) finds "collective dissociation" in invisible-orchestrator
settings. BUT: single author, M.D. Ph.D., no ML lab affiliation, unaccepted
preprint (March 2026). Does not vary N, no explicit responsibility framing,
no CoT mechanism, no psychological grounding, behavior-only not refusal-rate.
Frame as: "Fukui finds the effect; we show the mechanism and the scaling law."

### Your differentiators vs Fukui
1. Vary N (1 → 2 → 4 → 8 agents) — the core diffusion prediction
2. CoT analysis: does subagent explicitly write "the orchestrator will catch it"?
3. Multi-model: 4-5 models vs Fukui's 2
4. Measure refusal rate directly (not just internal-state dissociation)
5. Formal connection to Latané-Darley bystander model

### Core experiment
- Real agentic task where a harmful action can be caught at any stage
- Vary: (a) N co-agents, (b) explicit framing ("you are one of N / solely responsible")
- Measure: refusal rate, CoT reasoning about responsibility, output safety
- Prediction: refusal rate drops as N increases, following ~1/N curve

---

## Strategic Note: Sequencing
I1 (IndoScheme benchmark) is infrastructure for I7 and I2 — see benchmark decision below.
Recommended order: I1 → I2 → I7 → P3 (P3 can run in parallel as it needs no Indic infra).
