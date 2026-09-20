#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
PAPER_DIR="$HOME/CAISc/autovoila/all-spikes/paper1-self-monitor-circuit"
LOG="$PAPER_DIR/notes/overnight_p1.log"
source "$HOME/CAISc/autovoila/all-spikes/_shared/venv/bin/activate"
cd "$PAPER_DIR/scripts"
log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"; }

log "=== Phase 6 (reduced: 20 problems, 1 random seed) ==="
python3 06_causal_ablation.py && log "✓ Phase 6 done" || { log "✗ Phase 6 FAILED"; exit 1; }

log "=== Phase 6.5: Steering experiment ==="
python3 06_5_steering_experiment.py && log "✓ Phase 6.5 done" || { log "⚠ Phase 6.5 FAILED"; }

log "=== Phase 8: Make plots ==="
python3 08_make_plots.py && log "✓ Phase 8 done" || { log "⚠ Phase 8 failed"; }

log "=========================================="
log "Paper 1 pipeline COMPLETE"
log "=========================================="
