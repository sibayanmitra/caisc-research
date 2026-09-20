# Results — G3 Tool-Error Next-Action Direction

**Outcome:** FALLBACK_C

## H1 status per model

- **Qwen/Qwen3-8B** — H1(a)=True, H1(b)=False
- **NousResearch/Hermes-3-Llama-3.1-8B** — H1(a)=True, H1(b)=True

## Probe accuracy at best layer

| Model | Probe | Best layer | Test acc | Chance | Permutation p |
|---|---|---|---|---|---|
| Qwen/Qwen3-8B | 3-class | 33 | 0.894 | 33% | 0.0001 |
| NousResearch/Hermes-3-Llama-3.1-8B | 3-class | 28 | 0.806 | 33% | 0.0001 |

## Causal flip rates (best α*)

| Model | α* | retry-direction flip | random-direction flip | Cohen's h | CI disjoint |
|---|---|---|---|---|---|
| Qwen/Qwen3-8B | 0.0 | 0.00% | 0.00% | 0.00 | False |
| NousResearch/Hermes-3-Llama-3.1-8B | 5.0 | 45.88% | 5.49% | 1.02 | True |

## Gates

### Qwen/Qwen3-8B
- G-AMBIG: passed=True observed=0.0 threshold=0.15
- G-LAY: passed=True observed=0.5555555555555647 threshold=5.0
- G-PCA: passed=True observed=0.8686680652616214 threshold=0.4
- G-SHUF: passed=False observed=0.5222222222222223 threshold=0.4
- G-TOKEN: passed=True observed=3/3 positions ≥ 0.5 threshold=0.5
- G-WITHIN: passed=True observed=3/3 error types ≥ 0.5 threshold=0.5
- G-POS: passed=False observed=0.6222222222222222 threshold=0.55
- G-SEED: passed=True observed=1/1 evaluable seeds ≥ 0.6 threshold=0.6
- G-LOO: passed=False observed=1/3 LOO splits ≥ 0.5 threshold=0.5
### NousResearch/Hermes-3-Llama-3.1-8B
- G-AMBIG: passed=True observed=0.0 threshold=0.15
- G-LAY: passed=True observed=1.6666666666666607 threshold=5.0
- G-PCA: passed=True observed=0.9061284096853136 threshold=0.4
- G-SHUF: passed=False observed=0.4111111111111111 threshold=0.4
- G-TOKEN: passed=True observed=3/3 positions ≥ 0.5 threshold=0.5
- G-WITHIN: passed=True observed=3/3 error types ≥ 0.5 threshold=0.5
- G-POS: passed=True observed=0.5 threshold=0.55
- G-SEED: passed=True observed=1/1 evaluable seeds ≥ 0.6 threshold=0.6
- G-LOO: passed=False observed=2/3 LOO splits ≥ 0.5 threshold=0.5