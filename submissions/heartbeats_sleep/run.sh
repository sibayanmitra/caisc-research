#!/usr/bin/env bash
# Run the heartbeats-sleep investigation end to end.
set -e
cd "$(dirname "$0")"
source ../_shared/venv/bin/activate

ts() { date +"[%H:%M:%S]"; }
log() { echo "$(ts) $*"; echo "- $(ts) $*" >> notes/sanity_log.md; }

mkdir -p notes
: > notes/sanity_log.md
echo "# heartbeats-sleep run log" > notes/sanity_log.md

log "## Phase 1 — AnAge pull"
python3 scripts/01_pull_anage.py >> notes/sanity_log.md 2>&1

log "## Phase 2 — sleep data pull"
python3 scripts/02_pull_sleep.py >> notes/sanity_log.md 2>&1

log "## Phase 3 — merge species"
python3 scripts/03_merge_species.py >> notes/sanity_log.md 2>&1

log "## Phase 4 — PGLS fits"
python3 scripts/04_pgls_fit.py >> notes/sanity_log.md 2>&1

log "## Phase 5 — plots"
python3 scripts/05_plot_results.py >> notes/sanity_log.md 2>&1

log "## DONE"
