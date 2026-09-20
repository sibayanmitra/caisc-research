#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
PAPER_DIR="$HOME/CAISc/autovoila/all-spikes/paper1-self-monitor-circuit"
LOG="$PAPER_DIR/notes/overnight_p1.log"
source "$HOME/CAISc/autovoila/all-spikes/_shared/venv/bin/activate"
cd "$PAPER_DIR/scripts"
log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"; }

log "=== RESUME from Phase 3 (control-set-as-negatives fix applied) ==="

log "=== Phase 3: Train probes ==="
python3 04_train_probes.py && log "✓ Phase 3 done" || { log "✗ Phase 3 FAILED"; exit 1; }

log "=== Phase 4: Pre-register hypotheses ==="
cd "$PAPER_DIR"
git add hypotheses.md 2>/dev/null && \
  git commit -m "pre-register Paper 1 H1-H6 [auto, before ablation]" 2>/dev/null && \
  log "✓ hypotheses.md committed" || log "⚠ git commit skipped"
cd "$PAPER_DIR/scripts"

log "=== Phase 5: Head attribution ==="
python3 05_head_attribution.py && log "✓ Phase 5 done" || { log "✗ Phase 5 FAILED"; exit 1; }

log "=== Phase 6: Causal ablation (GPU 0) ==="
python3 06_causal_ablation.py && log "✓ Phase 6 done" || { log "✗ Phase 6 FAILED"; exit 1; }

log "=== Phase 6.5: Steering experiment (GPU 0) ==="
python3 06_5_steering_experiment.py && log "✓ Phase 6.5 done" || { log "⚠ Phase 6.5 FAILED (non-critical)"; }

log "=========================================="
log "Paper 1 resume pipeline COMPLETE"
log "=========================================="
