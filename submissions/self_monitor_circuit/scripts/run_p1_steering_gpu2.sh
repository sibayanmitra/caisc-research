#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false
PAPER_DIR="$HOME/CAISc/autovoila/all-spikes/paper1-self-monitor-circuit"
LOG="$PAPER_DIR/notes/overnight_p1.log"
source "$HOME/CAISc/autovoila/all-spikes/_shared/venv/bin/activate"
cd "$PAPER_DIR/scripts"
log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"; }

log "=== Phase 6.5 (retry on GPU 2): steering experiment ==="
python3 06_5_steering_experiment.py >> "$LOG" 2>&1 && log "✓ Phase 6.5 done" || log "✗ Phase 6.5 FAILED"

log "=== Phase 8: regenerate plots ==="
python3 08_make_plots.py >> "$LOG" 2>&1 && log "✓ Phase 8 done" || log "⚠ Phase 8 non-fatal"

log "=== Paper 1 FULLY COMPLETE ==="
