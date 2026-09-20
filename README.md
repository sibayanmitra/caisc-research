# CAISc Research — idea generation, screening, and submissions

Two things live here: the pipeline used to generate and screen research directions, and
the CAISc 2026 submissions that came out of it.

## `submissions/` — seven papers, code and data

Each directory is a self-contained submission package: code, configs, results, and the
paper PDF.

| directory | topic |
|---|---|
| `withholding_without_lying/` | sandbagging signature — detecting deliberate underperformance |
| `negation_operator/` | negation as an operator in activation space |
| `tokenizer_zipf/` | tokenizer behavior against Zipfian token distributions |
| `self_monitor_circuit/` | self-monitoring circuit analysis |
| `heartbeats_sleep/` | heartbeats / sleep behavioral study |
| `tool_error_direction/` | tool-error-induced next-action direction (submitted version) |
| `tool_error_direction_camera_ready/` | same work, camera-ready (code identical; paper differs) |

The tool-error line has its own full repository with development history at
[agentic-tool-error-steering](https://github.com/sibayanmitra/agentic-tool-error-steering);
`meanpool_cosine.pdf` in `papers/` corresponds to
[code-llm-thinks-python](https://github.com/sibayanmitra/code-llm-thinks-python).

Activation tensors above 50MB were dropped when packaging (notably
`tokenizer_zipf/data/token_counts/word_zh.pkl`); smaller `.pt` activation files are
included, since for several of these packages they are the only surviving copy.

## Idea pipeline

| path | role |
|---|---|
| `rate_ideas.py`, `rate_gpt54_separate.py`, `retry_two.py` | query several frontier models to score candidate directions |
| `aggregate.py` | weighted aggregation across raters |
| `ratings/` | raw and parsed per-model ratings, `SUMMARY.md`, `lit_gap_check.md` |
| `locked_ideas.md` | the directions that survived screening |
| `ponyvoila/` | research-taste tooling — `critic.py`, `research-philosophy.md`, `idea-pool.md`, and review skills under `skills/` |

`indic-multilingual-safety-gaps.md` and `llm-psychology-research-directions.md` are the
source direction documents the pipeline scored.

> **Note.** `locked_ideas.md`, `indic-multilingual-safety-gaps.md`, and parts of `ratings/`
> describe research directions that are still unpublished, including evaluation-awareness
> work whose experiments are deliberately not in this repository.
