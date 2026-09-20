#!/bin/bash
# Queue: run Phase 6.5 (steering) + Phase 8 (plots) after ablation finishes.
# Rule: waits for the ablation bash process (PID 2316742) to exit, then runs.
# Safe to re-run manually if needed — idempotent.
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
PAPER_DIR="$HOME/CAISc/autovoila/all-spikes/paper1-self-monitor-circuit"
LOG="$PAPER_DIR/notes/overnight_p1.log"
source "$HOME/CAISc/autovoila/all-spikes/_shared/venv/bin/activate"
cd "$PAPER_DIR/scripts"
log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"; }

ABLATION_PID=2316742

log "=== Steering queue: waiting for ablation (PID $ABLATION_PID) to finish ==="
while kill -0 $ABLATION_PID 2>/dev/null; do
    sleep 60
done
log "=== Ablation done. Starting Phase 6.5 (steering: 5 alphas × 20 problems + 3 random dirs) ==="

log "=== Phase 6.5: Steering experiment ==="
python3 06_5_steering_experiment.py && log "✓ Phase 6.5 done" || { log "✗ Phase 6.5 FAILED"; exit 1; }

log "=== Phase 8: Regenerate plots (with steering) ==="
python3 08_make_plots.py && log "✓ Phase 8 done" || { log "⚠ Phase 8 failed (non-fatal)"; }

log "=========================================="
log "Paper 1 FULL pipeline COMPLETE (ablation + steering + plots)"
log "=========================================="
