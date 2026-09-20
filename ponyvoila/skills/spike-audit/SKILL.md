---
name: spike-audit
description: >
  Whole-spike audit for research soundness. Like research-review, but scans an
  entire all-spikes/<idea>/ directory instead of one claim: a ranked list of the
  gaps between what the spike claims and what it actually shows. Use when the user
  says "audit this spike", "is this spike ready", "what's missing before I write",
  "audit the directory", "spike-audit", or "/spike-audit". One-shot report, fixes
  nothing.
---

research-review, spike-wide. Scan the whole `all-spikes/<idea>/` tree — log,
scripts, data, notes, plots, draft — instead of one claim. Rank findings by what
most blocks the paper: a claim the data doesn't support outranks a missing CI.

## Tags

Same as research-review, plus two for the artifact level:

- `known:` / `unsurprising:` / `overclaim:` / `confound:` / `so-what:` / `underpowered:` — as in research-review.
- `unreproducible:` a number in the notes/draft no script in the dir produces. Name the missing step.
- `dead:` analysis, plot, or data pull that nothing in the story uses. Cut it from the spike.

## Hunt

Headline numbers with no seed/CI, claims broader than the experiment that ran,
plots not referenced by any finding, a kill-criterion that was named but never
checked, results in the log that the draft quietly dropped, baselines assumed but
never run, `rigor:` markers whose upgrade trigger has already fired.

## Output

One line per finding, ranked biggest blocker first:
`<tag>: <what>. <fix>. [path]`

End with `verdict: <workshop|conference|journal|not yet> — <the one gap blocking the next bar>.`
Nothing blocking: `Sound. Write it up.`

## Boundaries

Research soundness only — code correctness, environment, and build issues go to a
normal pass. Reads and reports; applies nothing. One-shot.
"stop spike-audit" or "normal mode" to revert.
