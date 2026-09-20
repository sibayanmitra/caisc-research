# Supplementary Material — Tool-Error-Induced Next-Action Direction in Agentic LLMs

**CAISc 2026 submission**  
Authors: Anonymous Author (BITS Pilani), Claude Opus 4.7 (Anthropic), Claude Sonnet 4.6 (Anthropic)

---

## Contents

```
supplementary/
├── README_supplementary.md     # this file
├── preregistration.md          # frozen pre-registration (committed before GPU run)
├── preregistration_g4.md       # G4 (xLAM-2) pre-registration
├── Makefile                    # make all  →  reproduces full pipeline
├── src/                        # core library
│   ├── mock_tools.py           # byte-deterministic mock-tool harness
│   ├── prompts.py              # 300 base agentic prompts
│   ├── trajectory_gen.py       # trajectory generation with controlled outcomes
│   ├── activation_record.py    # residual-stream recording (fresh forward pass)
│   ├── label.py                # strict regex label rubric
│   ├── probe.py                # linear probe training (error-only, balanced)
│   ├── intervene.py            # causal intervention + null comparisons
│   ├── stats.py                # permutation test, cluster bootstrap, Cohen h
│   ├── gates.py                # sanity gate suite
│   └── log.py                  # decision-log writer
├── scripts/                    # end-to-end pipeline scripts
│   ├── run_main.py             # main trajectory + activation pipeline
│   ├── run_analysis.py         # probe + intervention analysis
│   ├── run_cka.py              # CKA (PCA-50, kernel n×n, float64)
│   ├── run_intervention.py     # α-sweep causal intervention
│   ├── plot_pub.py             # publication-quality figures (300 DPI)
│   └── ...
├── results/                    # all quantitative outputs
│   ├── analysis_summary.json   # per-model: probe accs, gates, per-layer results
│   ├── cka_results.json        # cross-model and cross-tool CKA values
│   ├── causal_flip_*.csv       # per-trajectory causal intervention results
│   ├── preflight_results.json  # pre-flight gate outcomes
│   └── figures/                # all paper figures (300 DPI PNG)
├── logs/                       # agentic loop instrumentation
│   ├── decisions.jsonl         # every agent decision with rationale + timestamp
│   ├── gate_outcomes.jsonl     # gate pass/fail log
│   ├── gate_failures.jsonl     # gate failure details
│   └── compute.jsonl           # GPU time attribution per stage
└── data/
    └── trajectories/           # raw trajectory JSONs (all 3 models × 3 seeds)
        (activations/ excluded — 3.6 GB NPZ files available on request)
```

---

## Reproducing the results

Requirements: Python 3.10+, PyTorch 2.x, transformers 4.51+, numpy, scikit-learn, pandas, matplotlib.
Models are downloaded automatically from HuggingFace on first run (no authentication required).

```bash
# Full pipeline from prompts to figures:
make all

# Individual stages:
make trajectories   # generate trajectories (requires GPU)
make activations    # record residual streams (requires GPU)
make analysis       # probe + intervention (CPU)
make cka            # CKA analysis (CPU)
make figures        # publication figures
```

Single-GPU constraint: all GPU stages run on `cuda:0`.  
CPU thread cap: `OMP_NUM_THREADS=16` (enforced in Makefile).

---

## Pre-registration

`preregistration.md` was committed to git before any GPU compute began.  
`preregistration_g4.md` covers the xLAM-2 replication study.  
Git history verifies the commit timestamps.

---

## Decision log

`logs/decisions.jsonl` contains every agent decision point during the research loop:
which layer to probe first, which α range to use, which fallback to invoke on gate
outcomes, etc. — each with a timestamp, one-sentence rationale, and observed outcome.
This is the primary artefact documenting the autonomous research process.
