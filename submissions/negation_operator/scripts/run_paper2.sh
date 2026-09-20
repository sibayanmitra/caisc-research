#!/bin/bash
# Paper 2 overnight pipeline — runs on GPU 1
set -e
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
PAPER_DIR="$HOME/CAISc/autovoila/all-spikes/paper2-negation-operator"
VENV="$HOME/CAISc/autovoila/all-spikes/_shared/venv"
LOG="$PAPER_DIR/notes/overnight_p2.log"
mkdir -p "$PAPER_DIR/notes"

source "$VENV/bin/activate"
cd "$PAPER_DIR/scripts"

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"; }

log "=========================================="
log "Paper 2 overnight pipeline starting"
log "GPU: $CUDA_VISIBLE_DEVICES"
log "=========================================="

log "Phase 1: Build pairs..."
python3 01_build_pairs.py && log "✓ Phase 1 done" || { log "✗ Phase 1 FAILED"; exit 1; }

log "Phase 2: Extract activations (GPU 1)..."
python3 02_extract_activations.py && log "✓ Phase 2 done" || { log "✗ Phase 2 FAILED"; exit 1; }

log "Phase 3: SVD directions..."
python3 03_svd_directions.py && log "✓ Phase 3 done" || { log "✗ Phase 3 FAILED"; exit 1; }

log "Phase 4: Transfer test (GPU 1)..."
python3 04_transfer_test.py && log "✓ Phase 4 done" || { log "✗ Phase 4 FAILED"; exit 1; }

log "Phase 5: Cross-category matrix (GPU 1)..."
python3 05_cross_category.py && log "✓ Phase 5 done" || { log "✗ Phase 5 FAILED"; exit 1; }

log "Phase 8: Make plots..."
python3 08_make_plots.py && log "✓ Phase 8 done" || { log "⚠ Phase 8 failed (non-critical)"; }

log "=========================================="
log "Paper 2 pipeline COMPLETE"
log "=========================================="
