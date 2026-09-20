#!/bin/bash
# Retry n=50 ablation after PID 68225 (n=20 run) fully exits.
# The ablation script is already patched to n=50, N_RANDOM_SEEDS=3.
set -e
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
PAPER_DIR="$HOME/CAISc/autovoila/all-spikes/paper1-self-monitor-circuit"
LOG="$PAPER_DIR/notes/overnight_p1.log"
source "$HOME/CAISc/autovoila/all-spikes/_shared/venv/bin/activate"
cd "$PAPER_DIR/scripts"
log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"; }

log "=== Retry n=50: waiting for PID 68225 (n=20 run) to finish ==="
while kill -0 68225 2>/dev/null; do
    sleep 30
done
log "=== PID 68225 exited. GPU now free. Starting n=50 full rerun ==="

# Confirm patch is correct
head -5 "$PAPER_DIR/scripts/06_causal_ablation.py" | grep -E "N_HELD_OUT|N_RANDOM"
log "Verified: 06_causal_ablation.py parameters"

# Clear all ablation and steering outputs for a clean run
rm -rf "$PAPER_DIR/ablations"
rm -rf "$PAPER_DIR/steering"
mkdir -p "$PAPER_DIR/ablations"
log "Cleared ablations/ and steering/ for clean n=50 run"

log "=== Phase 6 (n=50, 3 random seeds) ==="
python3 06_causal_ablation.py && log "✓ Phase 6 done" || { log "✗ Phase 6 FAILED"; exit 1; }

log "=== Phase 6.5: Steering experiment ==="
python3 06_5_steering_experiment.py && log "✓ Phase 6.5 done" || { log "⚠ Phase 6.5 FAILED (continuing)"; }

log "=== Phase 8: Make plots ==="
python3 08_make_plots.py && log "✓ Phase 8 done" || { log "⚠ Phase 8 failed (continuing)"; }

log "=========================================="
log "Paper 1 n=50 RETRY pipeline COMPLETE"
log "=========================================="
