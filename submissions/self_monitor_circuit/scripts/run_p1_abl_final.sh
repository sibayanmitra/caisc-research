#!/bin/bash
# Final n=50, 1-seed ablation run. Phase 6 + Phase 8 only (Phase 6.5 steering skipped for now).
set -e
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
PAPER_DIR="$HOME/CAISc/autovoila/all-spikes/paper1-self-monitor-circuit"
LOG="$PAPER_DIR/notes/overnight_p1.log"
source "$HOME/CAISc/autovoila/all-spikes/_shared/venv/bin/activate"
cd "$PAPER_DIR/scripts"
log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"; }

log "=========================================="
log "Paper 1 FINAL run: n=50, 1 random seed"
log "Phase 6 (ablation) + Phase 8 (plots) only"
log "=========================================="

# Clean slate
rm -rf "$PAPER_DIR/ablations"
mkdir -p "$PAPER_DIR/ablations"
log "Cleared ablations/ for clean run"

log "=== Phase 6: Causal ablation (n=50, 1 random seed) ==="
python3 06_causal_ablation.py && log "✓ Phase 6 done" || { log "✗ Phase 6 FAILED"; exit 1; }

log "=== Phase 8: Make plots ==="
python3 08_make_plots.py && log "✓ Phase 8 done" || { log "⚠ Phase 8 failed (non-fatal)"; }

log "=========================================="
log "Paper 1 Phase 6 pipeline COMPLETE"
log "=========================================="
