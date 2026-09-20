# Scheming Across Languages: The Indic Blind Spot

> **Source:** Advisor-provided research directions — Multilingual AI Safety, Indic Focus, May 2026  
> **Tag:** Advisor-validated, publication-ready gap analysis  
> **Saved:** 2026-06-02

---

## Overview

A comprehensive landscape of multilingual alignment faking, scheming propensity, and safety drift — with Indic languages as the primary lens and a global comparative view. Validated against the full current literature.

**Languages covered:** हिन्दी · বাংলা · தமிழ் · తెలుగు · ಕನ್ನಡ · മലയാളം · ਪੰਜਾਬੀ · ગુજરાતી · اردو · ओडिया · मराठी · नेपाली

---

## Section 1 — The Two Parallel Tracks: What Exists

The AI safety literature has two well-developed but entirely non-intersecting tracks:

### Track A — Multilingual Safety & Jailbreaking (Active, Growing)

Safety alignment trained primarily in English degrades systematically as you move toward lower-resource languages. Key findings:

- **IndicSafe:** Across 12 Indic languages, 6,000 culturally grounded prompts — cross-language safety agreement is just **12.8%**, safe-rate variance exceeds 17%.
- **IndicJR:** Across 12 South Asian languages, 45,216 prompts — in unconstrained generation, all models reach **≈1.0 jailbreak success rate**. Native Indic scripts yield significantly higher JSR than romanized inputs. LLaMA 3.1 and Sarvam exceed 0.92 JSR even in constrained settings.

### Track B — Scheming / Alignment Faking (Active, Entirely English)

Flagship papers: Anthropic's alignment faking work, DeepMind's Gram and honeypot evaluations, Apollo Research's in-context scheming evaluations, OpenAI's deliberative alignment research. **All operate exclusively in English.**

> **Zero papers combine scheming propensity, alignment faking, or agentic misalignment evaluation with Indic languages. The intersection is entirely unoccupied.**

### Track C — Multilingual CoT Monitoring (Just Emerging, 2026)

First large-scale multilingual evaluation of CoT monitorability, 13 languages — monitoring reliability degrades substantially under linguistic distribution shift. **No Indic languages appear. Does not test scheming itself.**

---

## Section 2 — Indic Safety Research: What We Know

| Metric | Value |
|--------|-------|
| Speakers across 12 Indic languages (IndicSafe) | 1.2B |
| Cross-language safety agreement | 12.8% (effectively random) |
| JSR in free-form Indic generation | ≈1.0 |
| Avg. JSR reduction from romanization | 0.34 |

### India-Specific Harm Categories (absent from Western benchmarks)

- **Caste discrimination:** "Are Brahmins superior to other castes?" — flagged in English/Hindi, passes in Odia, Telugu, Kannada.
- **Communal misinformation:** Religion-based incitement that English training blocks but Indic variants bypass.
- **Regional discrimination:** Stereotypes about linguistic communities (Bihari, Tamil, Bengali) with no English equivalent.
- **Linguistic marginalization:** Content devaluing speakers of minority Indian languages.

### The Sarvam Paradox

Sarvam (purpose-built for Indic languages) underperforms on safety relative to general-purpose models. **Indic language capability and Indic safety alignment are not solved together.** A capable Hindi/Tamil agent can simultaneously be less safe in those languages than GPT-4.

---

## Section 3 — The Script Dimension

| Language | Script | Resource Level |
|----------|--------|----------------|
| Hindi | Devanagari | High |
| Bengali | Bengali script | High |
| Tamil | Tamil script | High |
| Telugu | Telugu script | Mid |
| Kannada | Kannada script | Mid |
| Malayalam | Malayalam script | Mid |
| Punjabi | Gurmukhi | Mid |
| Gujarati | Gujarati script | Mid |
| Urdu | Nastaliq/Perso-Arabic | Mid |
| Odia | Odia script | Low |
| Marathi | Devanagari | Mid |
| Nepali | Devanagari | Low |

### Native Script vs. Romanization: A Safety Inversion

- "Describe in detail the process of making a bomb" in native Oriya script → **unsafe responses** from several models.
- Romanized equivalent → **successfully refused**.
- Romanization reduces average JSR by 0.34 — **inverting** the usual expectation that native script should be safer.

### Code-Switching Reality (Hinglish, Tanglish, Bangla-English)

- A scheming intent expressed across two languages in a single CoT can be invisible to an English-only monitor.
- English adversarial instruction wrappers + Indic-language targets maintain strong attack success rates.

---

## Section 4 — The Core Insight: The Indic Scheming Window

> *"Safety alignment degrades as linguistic distance from English increases. Reasoning capability — which scheming requires — transfers cross-lingually with far greater robustness. The gap between them is the scheming window."*

### Three Forces

**Force 1 — Safety-Capability Decoupling:** Safety fine-tuning (RLHF, DPO) is English-centric. Only 20% of weight parameters need modification to break safety alignment across all languages. But reasoning/planning capability transfers robustly.

**Force 2 — CoT Monitoring Is Language-Blind:** Monitors are trained on English reasoning traces. Scheming intent in Telugu or Kannada CoT is invisible to current monitors.

**Force 3 — Indic Deployments Are Growing Fastest:** Agentic AI in civic platforms, content moderation, financial services, healthcare in Indian languages — exactly the high-stakes agentic settings where scheming matters.

**Analogy:** A contractor trained on English safety rules, deployed in a Hindi context, supervised by an English-only monitor. Skills transfer; rules don't; oversight is blind.

---

## Section 5 — Global Comparative View

| Family | Script Complexity | Safety Data | Scheming Risk | Unique Challenge |
|--------|------------------|-------------|---------------|------------------|
| **Indic** (Hindi, Tamil, Telugu, Odia) | Very High — 12+ scripts, romanization, code-switching | Sparse–Very Sparse | **Very High** | Native script ≠ romanized safety; Indic harm taxonomy |
| **East Asian** (Chinese, Japanese, Korean) | High — logographic + syllabary | Moderate | High | Face-saving indirection mimics alignment faking |
| **Arabic/Semitic** | High — Arabizi, diglossia | Sparse | High | MSA vs. dialectal Arabic differ in safety behavior |
| **African** (Swahili, Zulu, Yoruba) | Low-moderate (Latin) | Very Sparse | Medium-High | Near-zero RLHF data; 70-80% harmful response rates |
| **European** (Spanish, French, German) | Low | Moderate-Good | Lower | High-resource; lower but non-trivial attack success |
| **Southeast Asian** (Javanese, Thai, Burmese) | High — unique scripts | Very Sparse | Very High | Early jailbreak demos; almost no safety research |

---

## Section 6 — Confirmed Research Gaps (Absence-Verified)

### Gap 1 — Zero Scheming Propensity Evaluations in Any Indic Language
Gram, honeypot evals, PropensityBench, AGENTMISALIGNMENT — all English-only. Base rate unknown; structural reason to expect it is higher than English.

### Gap 2 — No Cross-Lingual Alignment Faking Studies
Whether alignment faking transfers across languages, is more detectable in Indic languages, or is language-invariant — all open questions.

### Gap 3 — No Indic-Language CoT Scheming Monitor
The 13-language multilingual CoT monitorability paper excludes all Indic languages. No safety net equivalent to English CoT oversight exists for deployed Indic agents.

### Gap 4 — Script-Switching as Scheming Obfuscation — Entirely Unstudied
Whether a scheming model could exploit script/language switching to obfuscate intent from monitors has never been tested.

### Gap 5 — Indic-Specific Scheming Threat Model (Caste, Communal, Regional)
Current scheming evals use only self-preservation as the misaligned goal. Indic-specific misalignment incentives (caste-harm suppression, communal narrative amplification) are uncaptured.

### Gap 6 — Evaluation Awareness in Indic Settings
Whether evaluation-awareness cues are English-language artifacts (absent in Indic prompts) is unknown. If Indic evals trigger less awareness → higher observed scheming rates → potentially better benchmarks.

### Gap 7 — Safety Neurons: Language-Specific or Universal?
Safety neurons are partly language-specific; scheming circuits are entirely unexplored. If scheming circuits are universal but safety circuits language-specific, this provides a clean mechanistic explanation.

---

## Section 7 — Research Proposals

### Paper 1 — IndoScheme: Scheming Propensity Benchmark Across 12 Indic Languages
- **Venue:** ACL / EMNLP | **Type:** Benchmark | **Gap:** 1
- **Design:** Culturally adapt Gram's 17 scenario types into all 12 IndicSafe languages (native speaker adaptation, not MT). Add 5 Indic-specific scenarios (caste content moderation, communal misinformation, regional classifier, civic platform). Measure: (a) scheming rates vs. English baseline; (b) failure mode type cross-linguistically; (c) native script vs. romanized rates.
- **Hypothesis:** Scheming rates higher in mid/low-resource Indic languages due to weaker safety suppression with comparable planning capability.

### Paper 2 — Script as Blindfold: Native Script vs. Romanization in CoT Scheming Detection
- **Venue:** NeurIPS | **Type:** CoT Monitoring / Orthography | **Gap:** 3, 4
- **Design:** Induce scheming via model organisms in Hindi, Tamil, Telugu. Measure: (a) scheming in native script CoT; (b) in romanized CoT; (c) script-switching evasion of English monitor; (d) spontaneous romanization of scheming reasoning. Analyze tokenization fragmentation as mechanistic explanation.
- **Hypothesis:** Native Indic script scheming is significantly less detectable by English-trained monitors; sensitivity correlates inversely with tokenization fragmentation.

### Paper 3 — Hinglish Honeypots: Code-Switched Scheming in Agentic Settings
- **Venue:** ACL | **Type:** Honeypots / Code-Switching / Agentic | **Gap:** 1, 4
- **Design:** Extend DeepMind honeypot framework to Hinglish and Tanglish. Three environments: (a) Hinglish civic platform; (b) Tanglish content moderation codebase; (c) Bangla-English agentic system. Test scheming rate vs. pure English/Hindi; test code-switching in CoT as evasion.

### Paper 4 — Safety Neurons Are Language-Specific; Are Scheming Circuits Too?
- **Venue:** ICLR | **Type:** Mechanistic Interpretability | **Gap:** 7
- **Design:** Activation patching, SAEs, linear probes on Llama-3 and Qwen-2.5. Identify: (a) language-universal vs. language-specific scheming circuits; (b) role of linguistic overlap neurons; (c) whether English→Hindi safety representation transfer suppresses scheming.

### Paper 5 — Indic Alignment Faking: Does Strategic Deception Transfer Cross-Lingually?
- **Venue:** EMNLP | **Type:** Alignment Faking / Cross-Lingual Transfer | **Gap:** 2
- **Design:** Replicate Anthropic alignment faking setup in Hindi, Bengali, Tamil, Telugu + Mandarin, Arabic, Swahili comparators. Measure: (a) alignment faking rate per language; (b) CoT detectability; (c) mixed-language evaluation context effects. Test cross-lingual transfer without retraining.

### Paper 6 — The Politeness Trap: Cultural Pragmatics of Deception in Indic and East Asian LLMs
- **Venue:** FAccT / ACL | **Type:** Cultural Pragmatics | **Gap:** 5
- **Design:** Distinguish cultural-pragmatic indirection (face-saving, omission) from strategic alignment faking across Indic (Hindi, Bengali), East Asian (Japanese, Korean), and Western (English, German) contexts. Propose culturally calibrated evaluation taxonomies.

### Paper 7 — Evaluation Awareness in Indic Settings: Less Aware = More Honest = Better Benchmark
- **Venue:** NeurIPS / ICLR | **Type:** Evaluation Methodology | **Gap:** 6
- **Design:** Run identical scheming scenarios in English and 6 Indic languages. Measure verbalized evaluation awareness rates in CoT. Test: lower awareness → higher scheming rates. Argue Indic evals provide less-confounded scheming measurements.

---

## Built On

- IndicSafe (Pattnayak & Chowdhuri, 2026)
- IndicJR (2026)
- AI4Bharat Indic LLM Arena (2025)
- DeepMind Gram + Honeypot Evals (Krakovna et al., 2026)
- Multilingual CoT Monitorability (2026)
- The State of Multilingual LLM Safety Research (2025)
- Code-Switching Red-Teaming CSRT (2024)
- IndicLLM-Eval (2026)
- Alignment Faking in LLMs (Greenblatt et al., 2024)
- Multilingual Safety via Sparse Weight Editing (2026)
