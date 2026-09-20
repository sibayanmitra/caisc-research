#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source ../_shared/venv/bin/activate
mkdir -p notes
: > notes/sanity_log.md
ts() { date +"[%H:%M:%S]"; }
log() { echo "$(ts) $*"; echo "- $(ts) $*" >> notes/sanity_log.md; }

log "## Phase 1 — pull corpus"
python3 scripts/01_pull_corpus.py >> notes/sanity_log.md 2>&1
log "## Phase 2 — tokenize"
python3 scripts/02_tokenize.py >> notes/sanity_log.md 2>&1
log "## Phase 3 — fit Zipf"
python3 scripts/03_fit_zipf.py >> notes/sanity_log.md 2>&1
log "## Phase 4 — plots"
python3 scripts/04_plot_zipf.py >> notes/sanity_log.md 2>&1
log "## DONE"
