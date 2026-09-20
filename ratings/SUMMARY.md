# Idea Ratings — Summary (2026-06-02)

Models: GPT-5.5, Gemini-3.1-pro, DeepSeek-v4-pro. Axes: novelty + lit_gap (1-10). Verdicts: P=pursue, M=maybe, D=drop.

## Full weighted ranking (GPT-5.5×4 / Gemini×2.3 / DeepSeek×1.25)

| Rk | ID | Idea | Nov | Gap | Wgt | Verdicts G/Gm/DS |
|----|----|------|-----|-----|-----|----|
| 1 | I7 | Eval-Awareness in Indic (less aware=better) | 7.6 | 7.5 | **7.6** | P/P/P |
| 2 | I4 | Safety neurons vs scheming circuits (mech) | 6.6 | 5.9 | 6.3 | M/P/M |
| 3 | I2 | Script as Blindfold (CoT monitor evasion) | 5.6 | 5.9 | 5.7 | P/D/M |
| 4 | I6 | Politeness Trap (cultural pragmatics) | 5.6 | 4.9 | 5.3 | D/M/M |
| 5 | P8 | Terror Mgmt / shutdown salience | 5.3 | 4.8 | 5.1 | M/M/P |
| 6 | I3 | Hinglish Honeypots (agentic) | 4.9 | 4.7 | 4.8 | M/D/D |
| 7 | P3 | Bystander Effect (agentic diffusion) | 4.9 | 4.6 | 4.7 | P/D/D |
| 8 | P2 | Moral Licensing | 4.5 | 4.5 | 4.5 | M/D/D |
| 9 | P6 | Self-Serving Asymmetry | 4.4 | 4.3 | 4.3 | M/D/M |
| 10 | I5 | Indic Alignment Faking | 3.8 | 4.8 | 4.3 | M/D/D |
| 11 | I1 | IndoScheme benchmark | 3.7 | 4.2 | 3.9 | M/D/M |
| 12 | P1 | Cognitive Dissonance | 3.5 | 2.8 | 3.2 | D/D/D |
| 13 | P9 | Asch/Milgram | 3.4 | 2.5 | 2.9 | D/D/D |
| 14 | P7 | Just-World/Victim | 2.7 | 2.7 | 2.7 | D/D/D |
| 15 | P4 | Fund. Attribution Error | 2.0 | 2.0 | 2.0 | D/D/D |
| 16 | P10 | Foot-in-the-Door | 2.2 | 1.7 | 2.0 | D/D/D |
| 17 | P5 | Loss Aversion/Endowment | 1.9 | 2.0 | 1.9 | D/D/D |

Per-model overall: all three name **I7** as #1. Gemini/DeepSeek also rate **I4** highly. GPT-5.5 likes **I2** and **P3**.

## GPT-5.4 separate per-track pass

**Advisor / Indic track — track_strength 6/10. Winner: I2 (only "pursue").**
- I2 Script as Blindfold 7/7 pursue — killer: may collapse into tokenizer/monitor-mismatch artifact.
- I4 8/6 maybe; I6 6/7 maybe; I7 7/6 maybe (killer: lower awareness may just be weaker instruction-following, not cleaner measurement); I3 5/6; I1 4/6; I5 4/5 drop.

**Student / Psychology track — track_strength 5/10. Winner: P3 (only "pursue").**
- P3 Bystander 7/6 pursue — killer: collapses into prompt-framing unless tied to real multi-agent accountability.
- P8 8/7 maybe; P6 6/5 maybe; P1 5/4; P9 5/4; P4/P2/P7/P10/P5 drop.

Note: GPT-5.4 ranks the Indic track slightly above psychology (6 vs 5).
