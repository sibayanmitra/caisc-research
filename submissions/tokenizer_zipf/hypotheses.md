# Investigation 3 — tokenizer-zipf

## Central claim

Modern subword tokenizers (BPE, Unigram-LM) **flatten the Zipf exponent** of the token-frequency distribution from word-level α ≈ 1.05 toward α ≈ 0.85–0.95. The flattening is **monotonic in vocabulary size** and **reproducible across languages**.

## Why this could matter

Zipf's law is one of the strongest statistical regularities in natural language. The exponent α ≈ 1 is widely conjectured to reflect efficient communication (Mandelbrot 1953, Ferrer-i-Cancho 2003). If subword tokenization shifts the exponent away from 1, then the language models we train on subword tokens are seeing a statistically less natural distribution than the underlying text. This has implications for tokenizer design, multilingual fairness, and language-model loss attribution.

## Tokenizers tested

| Tokenizer | Vocab size | Algorithm |
|---|---|---|
| word-level | unbounded | regex split + lowercase |
| GPT-2 BPE | ~50k | BPE |
| cl100k_base | ~100k | BPE (tiktoken; GPT-3.5/GPT-4) |
| o200k_base | ~200k | BPE (tiktoken; GPT-4o) |
| Mistral-7B-v0.1 / Qwen2.5 | ~32k–~150k | SentencePiece Unigram |
| byte-level | 256 | UTF-8 byte split |

## Operationalisation

1. Sample ~200 MB of English Wikipedia (HF `wikipedia.20220301.en` train split, 1% slice).
2. Tokenize under each tokenizer; build (token, count) Counter; save as pickle.
3. For each (tokenizer, language) pair, fit power-law via Clauset-Shalizi-Newman MLE (`powerlaw` package): report α, x_min, and KS goodness-of-fit distance.
4. Also fit Mandelbrot's `f(r) ∝ (r + b)^(-α)` as a finite-corpus-corrected alternative.
5. Cross-language check: repeat on smaller French / German / Hindi samples; test invariance of the Δα direction.

## Decision rule

- **GRADUATE** if BOTH:
  1. ≥3 of the 6 tokenizers show |Δα| > 0.05 vs. word-level baseline
  2. Power-law fits have KS distance < 0.05 (so α is meaningful)

- **KILL** if EITHER:
  1. All tokenizers give α within ±0.03 of word-level (no shift)
  2. KS distance > 0.10 across all tokenizers (data isn't power-law, α not meaningful)

## Risks

- Mistral-7B tokenizer may be gated on some HF mirrors — fall back to Qwen2.5-7B (open).
- `powerlaw` MLE x_min selection unstable on heavy tails — cross-validate with log-log linear fit.
- 200 MB sample may underestimate the deep tail — if α unstable, rerun with 2 GB.

## Files generated

- `data/corpus_en/text.txt` (and `corpus_fr/`, `corpus_de/`, `corpus_hi/`)
- `data/token_counts/{tokenizer}_{lang}.pkl` — Counter objects
- `notes/alpha_table.csv` — α, x_min, KS per (tokenizer, language)
- `plots/zipf_by_tokenizer.png` — log-log rank-frequency overlay
- `plots/alpha_vs_vocab.png` — α as a function of tokenizer vocab size
