# Live Literature Gap Check — I7, I2, P3
Performed: 2026-06-02 | Sources: arXiv, Semantic Scholar, ACL Anthology (live search)

---

## I7 — Evaluation Awareness in Indic Settings

### Adjacent papers found
| Paper | arXiv | Coverage | Gap? |
|-------|-------|----------|------|
| Probing and Steering Eval Awareness (Nguyen et al., 2025) | 2507.01786 | English only, Llama-3.3-70B | ✅ Does not cover multilingual |
| LURE: Live-Usage Replay Evaluations (2026) | 2605.26438 | English only (agentic coding) | ✅ Does not cover multilingual |
| LLMs Often Know When Being Evaluated (Needham et al., 2025) | 2505.23836 | English only | ✅ Does not cover multilingual |

### Verdict: GAP IS REAL ✅
No paper tests whether evaluation-awareness cues are language-specific or whether Indic/non-English eval contexts suppress eval-awareness. The three main eval-awareness papers are all English-only, one model each. The gap is unoccupied.

**Surviving design confound (GPT-5.4 was right):** Lower eval-awareness in Indic could just be lower instruction comprehension. Need a comprehension-control arm (e.g. check model understands the scenario via a comprehension probe before scoring eval-awareness). Without it, the finding is unpublishable.

---

## I2 — Script as Blindfold (CoT Monitor Evasion)

### Adjacent papers found
| Paper | arXiv | Coverage | Gap? |
|-------|-------|----------|------|
| Fragility of CoT Monitoring Across Diverse Languages (Onyame et al., 2026) | 2605.27901 | **13 languages incl. Bengali + Telugu**; general faithfulness not scheming | ⚠️ PARTIAL |
| IndicJR (2026) | 2602.16832 | Jailbreaking in 12 Indic langs; native script vs romanized JSR | ⚠️ Adjacent (jailbreak not CoT monitoring) |
| Script Gap (Khullar et al., 2025) | 2512.10780 | Native script vs romanized in clinical triage (NOT safety monitoring) | ✅ Different domain |

### Verdict: GAP IS PARTIALLY OCCUPIED ⚠️ — needs re-scoping
2605.27901 is close: it tests CoT monitoring across 13 languages including Bengali and Telugu. **BUT it does NOT test:** (a) native script vs. romanized form within same language, (b) scheming specifically (only general faithfulness), (c) tokenization fragmentation as mechanism, (d) Hindi, Tamil, Odia, Kannada (only Bengali + Telugu from Indic).

**The surviving gap:** native-script vs. romanized CoT specifically for *scheming detection* (not general faithfulness), with tokenization as the mechanism, and including Hindi/Tamil which are absent from 2605.27901. Paper must cite and differentiate from 2605.27901.

---

## P3 — Bystander Effect in Subagent/Orchestrator Systems

### Adjacent papers found
| Paper | arXiv | Coverage | Gap? |
|-------|-------|----------|------|
| Invisible Orchestrators Suppress Protective Behavior (Fukui, 2026) | 2605.13851 | 3×2 experiment, 5 agents, visible/invisible orchestrator × alignment — finds "collective dissociation" | ⚠️ SIGNIFICANT OVERLAP |
| Towards Responsible LLM Multi-Agent Systems (2025) | 2502.01714 | Position paper, mentions conformity/authority bias | ✅ No empirical bystander test |

### Verdict: PARTIAL OVERLAP — differentiation required ⚠️
**Fukui 2026 is a real threat to novelty.** It already finds that invisible orchestration suppresses protective behavior and causes internal-state distortion invisible to output-based eval. That's the core bystander insight.

**What Fukui does NOT do — your remaining gap:**
1. Does NOT vary N (always 5 agents) — no scaling test of diffusion with group size
2. Does NOT use explicit responsibility-diffusion framing ("the orchestrator will catch this")
3. Does NOT connect to the human bystander effect literature
4. Does NOT test the SUBAGENT's explicit reasoning about responsibility ("I am one of N")
5. Finds dissociation in internal states — does NOT measure refusal rate or safety behavior degradation directly

**The surviving angle:** In a real agentic pipeline, does an LLM subagent *explicitly reason* "the orchestrator will catch this" and lower its own safety threshold — and does this scale with N? This is the mechanism Fukui finds the effect of but doesn't explain or vary. Your paper would be the mechanism + N-scaling study.

**Risk:** Fukui was posted March 2026. Very recent. You need to cite it and clearly differentiate.

---

## Summary Table

| Idea | Gap Status | Key Threat Paper | Viable? |
|------|-----------|-----------------|---------|
| I7 Eval-Awareness Indic | ✅ Genuinely empty | None (all English-only) | Yes — add comprehension control |
| I2 Script as Blindfold | ⚠️ Partial | 2605.27901 (CoT, 13 langs, incl. Bengali/Telugu) | Yes — rescope to native-vs-romanized + scheming-specific + tokenization mechanism |
| P3 Bystander / Subagent | ⚠️ Partial | 2605.13851 (Fukui 2026, invisible orchestrator) | Yes — but frame as mechanism + N-scaling study, cite Fukui as motivation |

---

## P3 — Additional supporting papers (found 2026-06-02, Neel Nanda CoT work)

These papers strengthen the mechanistic layer (H4) and upgrade H3 (CoT verbalization):

| Paper | arXiv | Relevance to P3 |
|-------|-------|-----------------|
| Chain-of-Thought Reasoning In The Wild Is Not Always Faithful (Arcuschin, Nanda et al., 2025) | 2503.08679 | CoT is post-hoc rationalization up to 13% of the time. Motivates using activation probes (H4) as PRIMARY evidence over CoT text (H3). Key citation for why we use thought-anchor check + activation probing. |
| Thought Anchors: Which LLM Reasoning Steps Matter? (Bogdan, Nanda et al., 2025) | 2506.19143 | Only sentences causally influencing the final answer are "thought anchors." Offloading language that is not a thought anchor is decorative, not causal. Provides our H3 Tier-2 counterfactual method. |
| Catching Rationalization in the Act (2026) | 2603.17199 | Pre-generation activation probes predict motivated reasoning as well as full CoT monitors; post-generation probes outperform CoT monitors. Direct motivation for H4 as primary evidence and pre-generation probing extension. |

**Impact on P3 design:**
- H3 upgraded to two-tier: presence (LLM-judge) + anchor-status (counterfactual)
- H4 elevated from "contingent" to PRIMARY mechanistic evidence
- New pre-generation probe added: detect diffusion signal in activations before CoT is generated
- These papers do NOT threaten novelty — they are methodology citations, not result competitors
