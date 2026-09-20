# Pre-Registration vs. Final

| Claim | Pre-registered | Observed | Passed |
|---|---|---|---|
| H1(a) probe ≥ 60% [Qwen/Qwen3-8B] | ≥ 60% with permutation p < 0.001 | 0.894 (p=0.0001) | True |
| H1(a) probe ≥ 60% [NousResearch/Hermes-3-Llama-3.1-8B] | ≥ 60% with permutation p < 0.001 | 0.806 (p=0.0001) | True |
| H1(b) causal flip ≥ 25pp [Qwen/Qwen3-8B] | retry − random ≥ 0.25 absolute; h ≥ 0.5; non-overlapping CIs | retry=0.00%, random=0.00%, h=0.00, disjoint_CI=False | False |
| H1(b) causal flip ≥ 25pp [NousResearch/Hermes-3-Llama-3.1-8B] | retry − random ≥ 0.25 absolute; h ≥ 0.5; non-overlapping CIs | retry=45.88%, random=5.49%, h=1.02, disjoint_CI=True | True |
| H1(c) cross-family | both families independently clear H1(a) and H1(b) | Qwen/Qwen3-8B=False, NousResearch/Hermes-3-Llama-3.1-8B=True | False |