---
name: rigor-debt
description: >
  Harvest every `rigor:` comment in a spike into a debt ledger, so the deliberate
  shortcuts (single seed, no baseline yet, scope-limited claim) get tracked
  instead of rotting into a silent hole under the abstract. Use when the user says
  "rigor debt", "/rigor-debt", "what did we defer", "list the shortcuts", "what's
  marked to do later", or "check the spike before writing". One-shot report,
  changes nothing.
---

Every deliberate shortcut in a spike is marked `rigor: <ceiling>, <upgrade
trigger>`. This collects them into one ledger so a deferral can't quietly become a
hole under a claim in the abstract.

## Scan

Grep the spike for the markers, skipping data dumps, `.git`, and venvs:

`grep -rnE '(#|//|%) ?rigor:' all-spikes/<idea>/`  (add comment prefixes your stack uses)

Each hit is one ledger row. The comment prefix keeps prose that merely mentions
the convention out of the ledger.

## Output

One row per marker, grouped by file:

`<file>:<line> — <what was simplified>. ceiling: <the limit named>. upgrade: <the trigger to revisit>.`

The convention is `rigor: <ceiling>, <upgrade path>`, so pull the ceiling and the
trigger straight from the comment.

Then the two flags that matter before writing:

- `no-trigger` — any `rigor:` comment naming no upgrade trigger. These silently rot.
- `fired` — any marker whose trigger has already been met (e.g. "run 5 seeds if the
  effect holds" and the effect held). These are debt now due — pay before drafting.

End with `<N> markers, <M> no-trigger, <K> fired.` Nothing found:
`No rigor: debt. Either clean, or nobody marked their shortcuts.`

## Boundaries

Reads and reports only. To persist it, ask and it writes the ledger to
`all-spikes/<idea>/RIGOR-DEBT.md`. One-shot.
"stop rigor-debt" or "normal mode" to revert.
