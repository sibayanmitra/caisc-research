#!/bin/bash
# Paper 3 overnight pipeline — runs on GPU 2
set -e
export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false
PAPER_DIR="$HOME/CAISc/autovoila/all-spikes/paper3-sandbagging-signature"
VENV="$HOME/CAISc/autovoila/all-spikes/_shared/venv"
LOG="$PAPER_DIR/notes/overnight_p3.log"
mkdir -p "$PAPER_DIR/notes"

source "$VENV/bin/activate"
cd "$PAPER_DIR/scripts"

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"; }

log "=========================================="
log "Paper 3 overnight pipeline starting"
log "GPU: $CUDA_VISIBLE_DEVICES"
log "=========================================="

log "Phase 1: Build dataset (generate sandbagging + ignorance examples, GPU 2)..."
python3 01_build_dataset.py && log "✓ Phase 1 done" || { log "✗ Phase 1 FAILED"; exit 1; }

log "Phase 2: Extract activations (GPU 2)..."
python3 02_extract_activations.py && log "✓ Phase 2 done" || { log "✗ Phase 2 FAILED"; exit 1; }

log "Phase 3: Train probes + DiffMean d_withhold..."
python3 03_train_probes.py && log "✓ Phase 3 done" || { log "✗ Phase 3 FAILED"; exit 1; }

log "Phase 4: Extract directions (d_withhold per persona, d_refuse, GPU 2)..."
python3 04_extract_directions.py && log "✓ Phase 4 done" || { log "✗ Phase 4 FAILED"; exit 1; }

log "Phase 5: Refusal geometry (cosine analysis)..."
python3 05_refusal_geometry.py && log "✓ Phase 5 done" || { log "✗ Phase 5 FAILED"; exit 1; }

log "Phase 6: Causal recovery (project-out intervention, GPU 2)..."
python3 06_causal_recovery.py && log "✓ Phase 6 done" || { log "✗ Phase 6 FAILED"; exit 1; }

log "Phase 8: Make plots..."
python3 08_make_plots.py && log "✓ Phase 8 done" || { log "⚠ Phase 8 failed (non-critical)"; }

log "=========================================="
log "Paper 3 pipeline COMPLETE"
log "=========================================="
