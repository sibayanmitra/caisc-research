#!/bin/bash
# Paper 1 overnight queue — waits for Phase 1 generation to finish,
# then runs phases 1b (build dataset) through 6.5 (steering) sequentially on GPU 0.
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
PAPER_DIR="$HOME/CAISc/autovoila/all-spikes/paper1-self-monitor-circuit"
VENV="$HOME/CAISc/autovoila/all-spikes/_shared/venv"
LOG="$PAPER_DIR/notes/overnight_p1.log"
mkdir -p "$PAPER_DIR/notes"

source "$VENV/bin/activate"
cd "$PAPER_DIR/scripts"

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"; }

log "=========================================="
log "Paper 1 overnight queue started"
log "Waiting for Phase 1 generation to complete..."
log "=========================================="

# Wait for generation process to finish
until grep -q "Phase 1 step 1 complete\|Time budget exhausted" \
      "$PAPER_DIR/notes/phase1_generation.log" 2>/dev/null; do
    sleep 60
    log "  Still waiting for generation... ($(wc -l < "$PAPER_DIR/data/generations/generations.jsonl" 2>/dev/null || echo 0) gens so far)"
done
log "Generation complete. Starting Phase 1b..."

# Phase 1b: Build positive/negative/control dataset from generations
log "=== Phase 1b: Build dataset ==="
python3 02_build_dataset.py && log "✓ Phase 1b done" || { log "✗ Phase 1b FAILED"; exit 1; }

# Phase 2: Extract activations
log "=== Phase 2: Extract activations (GPU 0) ==="
python3 03_extract_activations.py && log "✓ Phase 2 done" || { log "✗ Phase 2 FAILED"; exit 1; }

# Phase 3: Train probes + heatmap
log "=== Phase 3: Train probes ==="
python3 04_train_probes.py && log "✓ Phase 3 done" || { log "✗ Phase 3 FAILED"; exit 1; }

# Phase 4: Pre-register hypotheses (git commit)
log "=== Phase 4: Pre-register hypotheses ==="
cd "$PAPER_DIR"
git add hypotheses.md 2>/dev/null && \
  git commit -m "pre-register Paper 1 H1-H6 [auto, before ablation]" 2>/dev/null && \
  log "✓ hypotheses.md committed" || log "⚠ git commit skipped (already committed or no git)"
cd "$PAPER_DIR/scripts"

# Phase 5: Per-head attribution
log "=== Phase 5: Head attribution ==="
python3 05_head_attribution.py && log "✓ Phase 5 done" || { log "✗ Phase 5 FAILED"; exit 1; }

# Phase 6: Causal ablation (destructive lever)
log "=== Phase 6: Causal ablation (GPU 0) ==="
python3 06_causal_ablation.py && log "✓ Phase 6 done" || { log "✗ Phase 6 FAILED"; exit 1; }

# Phase 6.5: Steering experiment (constructive lever)
log "=== Phase 6.5: Steering experiment (GPU 0) ==="
python3 06_5_steering_experiment.py && log "✓ Phase 6.5 done" || { log "✗ Phase 6.5 FAILED (non-critical)"; }

log "=========================================="
log "Paper 1 overnight queue COMPLETE"
log "=========================================="
