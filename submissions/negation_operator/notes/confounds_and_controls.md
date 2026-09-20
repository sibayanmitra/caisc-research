# Paper 2 — Confounds in Current Design & Control Experiments

## Anthropic-style design principle
The signal we claim ("a universal direction for logical negation") should be **robust across surface forms** of negation. If our extracted direction is really the operator NOT, it must:
1. Generalize across multiple negation morphologies (n't, never, no, fails to, etc.)
2. NOT just track the literal token "not"
3. NOT just track sentence length or word position
4. NOT just track category-level surface differences

## Confounds in current design

### C1 — "not" token presence (MOST SERIOUS)
**Issue**: Every negative prompt contains the literal token "not". The "negation direction" extracted via SVD on (neg - pos) deltas may simply encode "this sequence contained a 'not' token a few positions back" — a lexical confound, not logical negation.

**Anthropic-style fix**: Show the direction transfers to negations that DON'T use "not":
- "happy" vs "never happy"
- "answer is yes" vs "answer is no"  (different lexical content, same logical structure)
- "she can swim" vs "she can't swim" (contraction)
- "the cake is fresh" vs "the cake is stale"  (antonym, no "not")

### C2 — Length / processing-depth differences
**Issue**: Negatives are longer than positives ("X." vs "It is not true that X."). The model has processed more tokens before reaching the final-token position. The residual stream at "final token" may encode "I processed a longer sequence" rather than "negation".

**Anthropic-style fix**: Build length-matched pairs:
- "happy mood" (2 tokens) vs "sad mood" (2 tokens) — antonym, matched length
- "is correct" vs "is incorrect" — single prefix difference

### C3 — Final-token identity drift
**Issue**: While most pairs end on the same final token (e.g., both end in "happy."), the immediately-preceding context differs. The "final token" position carries different residual stream content because of upstream attention.

**Anthropic-style fix**: Already partially addressed — we extract at final token across all layers, and per-layer EV shows where signal concentrates. But should add a control where we extract at the SAME position relative to the negation marker, not the absolute final token.

### C4 — Category surface differences could leak into "cross-category transfer"
**Issue**: All 4 categories use similar "not" templates, so cross-category transfer might just be "any direction that captures 'not' presence transfers".

**Anthropic-style fix**: Categories already use different sentence frames; but we should additionally test if the negation direction transfers to **morphologically different** negations (contractions, antonyms).

## Morning control experiments (to run after main pipeline)

### Control 1: Cross-morphology transfer
- Build 30 prompts using "doesn't", "won't", "no", "never", "fails to"
- Extract activations on these (call it dataset `M`)
- Test: does d_neg from main (which only used "not") transfer to M?
- Pass criterion: transfer effect on M > random baseline at matched α

### Control 2: Antonym vs prefix-negation
- Already have `antonyms.jsonl` from Phase 1 ✓
- Compute SVD on antonym deltas separately → d_antonym
- Compute cos(d_neg_main, d_antonym) — should be high if logical negation is shared

### Control 3: "not" in non-negation role
- Build 20 prompts where "not" appears NOT as a logical negator:
  - "She did not only sing but also danced" (additive)
  - "Not surprisingly, she won" (epistemic, not propositional negation)
- Extract activations
- Apply d_neg at α=+1
- If d_neg captures lexical "not": large logit shift expected
- If d_neg captures logical negation: small logit shift expected (these prompts ARE NOT logically negated)

### Control 4: Length-matched antonym baseline
- Already have antonyms ✓
- Extract d_antonym independently
- Test cross-pair-type transfer: d_neg → antonym pairs, d_antonym → "not" pairs
- Both should work if both encode "logical opposition"

## Writing the paper honestly
- We CAN claim: "There exists a direction at layer L* whose addition causes negation-consistent behavioral shift, with transfer across 4 semantic categories."
- We MUST acknowledge: "Whether this direction encodes the abstract operator NOT or merely a lexical correlate of the negation marker remains an open question. Our cross-morphology control (Section X) provides partial evidence for the former."
