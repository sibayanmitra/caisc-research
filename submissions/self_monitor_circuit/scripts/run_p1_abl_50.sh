#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
PAPER_DIR="$HOME/CAISc/autovoila/all-spikes/paper1-self-monitor-circuit"
LOG="$PAPER_DIR/notes/overnight_p1.log"
source "$HOME/CAISc/autovoila/all-spikes/_shared/venv/bin/activate"
cd "$PAPER_DIR/scripts"
log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"; }

# Wait for the n=20 run to fully finish
log "=== Waiting for n=20 run to finish before starting n=50 run ==="
while pgrep -f "run_p1_abl_reduced\|06_causal_ablation\|06_5_steering\|08_make_plots" > /dev/null 2>&1; do
    sleep 30
done
log "=== n=20 run finished. Starting n=50 full rerun ==="

# Patch N_HELD_OUT_PROBLEMS to 50 in-place
sed -i 's/^N_HELD_OUT_PROBLEMS = 20/N_HELD_OUT_PROBLEMS = 50/' "$PAPER_DIR/scripts/06_causal_ablation.py"
sed -i 's/^N_RANDOM_SEEDS = 1/N_RANDOM_SEEDS = 3/' "$PAPER_DIR/scripts/06_causal_ablation.py"
log "Patched: N_HELD_OUT_PROBLEMS=50, N_RANDOM_SEEDS=3"

# Clear old ablation results so Phase 6 reruns clean
rm -rf "$PAPER_DIR/ablations"
mkdir -p "$PAPER_DIR/ablations"
log "Cleared ablations/ for clean n=50 run"

log "=== Phase 6 (n=50, 3 random seeds) ==="
python3 06_causal_ablation.py && log "✓ Phase 6 done" || { log "✗ Phase 6 FAILED"; exit 1; }

log "=== Phase 6.5: Steering experiment ==="
python3 06_5_steering_experiment.py && log "✓ Phase 6.5 done" || { log "⚠ Phase 6.5 FAILED"; }

log "=== Phase 8: Make plots ==="
python3 08_make_plots.py && log "✓ Phase 8 done" || { log "⚠ Phase 8 failed"; }

log "=========================================="
log "Paper 1 n=50 pipeline COMPLETE"
log "=========================================="
