# ponyvoila — autovoila, molded with ponytail's discipline

This is a separate, self-contained variant of [autovoila](https://github.com/paraschopra/autovoila)
(Lossfunk's autoresearch prompt). The original at `~/CAISc/autovoila` is untouched.

It keeps everything that made autovoila work — the research philosophy, the
cross-model critic, the idea pool, the paper format — and bolts on the structural
machinery that makes [ponytail](https://github.com/DietrichGebert/ponytail) hard to
drift away from. Ponytail makes a coding agent stay lazy; this makes a research
agent stay disciplined.

## What changed

`voila.md` was a linear checklist. It's now `ponyvoila.md`, a process with a spine:

| Ponytail pattern | autovoila before | ponyvoila now |
|---|---|---|
| Sharp identity, one sentence | "autonomous research agent" | "senior scientist; the best experiment kills or confirms the claim fastest" |
| Deterministic **ladder**, stop at first rung | linear checklist | a **research ladder**: known? → surprising? → killable in budget? → cheapest test first → only then full rigor |
| **Persistence** (hook re-injects every turn) | none — agent drifts off-process | a **persistence rule + phase gates**: restate phase & budget every step; can't advance a phase until its gate passes |
| **Intensity** levels (lite/full/ultra) | none | **quick / full / deep**, tied to time budget and rigor depth |
| **When NOT to be lazy** (safety floor) | implicit in the philosophy | an explicit **rigor floor**: never skip kill-criterion, seeds, baseline, lit review, honest reporting |
| `ponytail:` debt comment + `/ponytail-debt` | none | the **`rigor:` convention** + `/rigor-debt` skill — grep-able deferred-rigor ledger |
| review / audit / debt skills | `critic.py` (script only) | `critic.py` **plus** three skills: `research-review`, `spike-audit`, `rigor-debt` |
| Output discipline (code first, ≤3 lines) | "no AI writing style" | terse decision log + the no-unrequested-prose rule applied to papers |

The one idea ponytail didn't have, added here: **spike code is lazy code** — a
2-hour experiment gets stdlib and one script, never a framework. Ponytail's lazy-dev
principle, pointed at experiment code.

## Layout

```
ponyvoila.md             the molded master prompt — start here
research-philosophy.md   unchanged: what makes a claim worth chasing
critic.py                unchanged: cross-model critique (OpenRouter → Anthropic)
idea-pool.md             unchanged: 20 pre-sized ~2h spikes
draft-format/            unchanged: the CAISc paper template
skills/
  research-review/       review one claim/draft for research failure modes
  spike-audit/           audit a whole all-spikes/<idea>/ dir before writing
  rigor-debt/            harvest `rigor:` markers into a ledger
all-spikes/              outputs land here, one self-contained dir per spike
```

## Run it

Open Claude Code in this directory and tell it:

> Read ponyvoila.md and get started

That's the whole launch — Claude Code is the agent. (Optional: [cco](https://github.com/nikvdp/cco)
is just a sandbox wrapper for hands-off YOLO runs, `cco -p "Read ponyvoila.md and
get started"`; Codex is optional too, for cross-model review.)

Set keys for the critic: `OPENROUTER_API_KEY` (primary) or `ANTHROPIC_API_KEY`
(fallback).

## What to customize (highest leverage first)

- `ponyvoila.md` — the prompt. The ladder rungs and the rigor floor are the knobs.
- `research-philosophy.md` — swap in your own taste if it differs.
- `idea-pool.md` — your own seed claims.
- `draft-format/` — the venue you submit to.
