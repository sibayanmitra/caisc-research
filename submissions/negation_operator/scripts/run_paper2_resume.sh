#!/bin/bash
# Resume Paper 2 from Phase 4 (after Phase 3 completed successfully)
set -e
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
PAPER_DIR="$HOME/CAISc/autovoila/all-spikes/paper2-negation-operator"
VENV="$HOME/CAISc/autovoila/all-spikes/_shared/venv"
LOG="$PAPER_DIR/notes/overnight_p2.log"

source "$VENV/bin/activate"
cd "$PAPER_DIR/scripts"

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"; }

log "=========================================="
log "Paper 2 RESUME: Phases 4 → 5 → 8"
log "=========================================="

log "Phase 4: Transfer test (GPU 1)..."
python3 04_transfer_test.py && log "✓ Phase 4 done" || { log "✗ Phase 4 FAILED"; exit 1; }

log "Phase 5: Cross-morphology matrix (GPU 1)..."
python3 05_cross_category.py && log "✓ Phase 5 done" || { log "✗ Phase 5 FAILED"; exit 1; }

log "Phase 8: Make plots..."
python3 08_make_plots.py && log "✓ Phase 8 done" || { log "⚠ Phase 8 failed (non-critical)"; }

log "=========================================="
log "Paper 2 pipeline COMPLETE"
log "=========================================="
