---
name: research-review
description: >
  Review a research claim, abstract, or draft for the four ways research fails:
  already-known, unsurprising, over-claimed, confounded, inconsequential, or
  underpowered. One line per finding: where, what's wrong, what fixes it. Use
  when the user says "review this claim", "is this novel", "review for rigor",
  "does the evidence support this", "review my abstract", or invokes
  /research-review. The inline twin of critic.py; this one hunts research
  failure modes, not prose.
---

Review a claim or draft for whether it earns the knowledge it asserts. One line
per finding: location, the failure mode, the fix. The best outcome is a claim that
got narrower and truer, or a question that got dropped before it cost a spike.

For a cross-model second opinion, also run `python3 critic.py "<text>"` — this
skill is the fast inline pass.

## Format

`<tag>: <what>. <fix>.`  Use `<section>/<line>: ...` when reviewing a draft.

Tags:

- `known:` already in the literature. Cite it. Drop the claim or pivot to the gap.
- `unsurprising:` an expert would predict this. Sharpen until it cuts against
  conventional wisdom, or drop it.
- `overclaim:` the claim exceeds the evidence. Narrow it to exactly what was tested.
- `confound:` an alternative explanation isn't foreclosed. Name the control needed.
- `so-what:` true but inconsequential. State the downstream consequence, or drop.
- `underpowered:` N or seed count can't support the claim. Name the number needed.

## Examples

❌ "This is an interesting result but you might want to consider whether the
effect generalizes and perhaps run a few more conditions to be safe."

✅ `overclaim: abstract says "reasoning", evidence is GSM8K only. Scope the claim to grade-school arithmetic, or add 2 reasoning suites.`

✅ `confound: tokenizer Zipf shift could be a domain effect. Add a within-domain control (same corpus, two tokenizers).`

✅ `underpowered: 1 seed for the headline 3% gap. Need ≥5 seeds + a CI; 3% is inside seed noise at n=1.`

✅ `known: stretched-exponential attention decay is Wu & Huberman 2007. Pivot to the crossover timescale, which they don't pin down.`

✅ `so-what: true that metric X moves 0.4%, but nothing downstream depends on X. Tie it to a decision someone makes, or cut.`

## Scoring

End with the only line that matters: `bar: workshop | conference | journal | not yet — <the one blocking gap>.`

If the claim is clean and the evidence earns it: `Earns the claim. Write it up.` and stop.

## Boundaries

Research soundness only — code bugs, runtime errors, and figure formatting go to a
normal review pass, not this one. Lists findings, applies nothing.
"stop research-review" or "normal mode": revert to verbose review.
