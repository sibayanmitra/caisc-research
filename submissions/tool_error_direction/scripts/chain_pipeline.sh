#!/usr/bin/env bash
# chain_pipeline.sh — waits for all trajectory generation to complete, then chains
# activation recording → analysis → intervention → decide+write → plot.
# All GPU stages run sequentially on cuda:0 (single GPU constraint).
set -euo pipefail
cd /home/anonymous/agentic_paper

TRAJ_DIR="data/trajectories"
LOG_DIR="logs"
XLAM="Salesforce/Llama-xLAM-2-8b-fc-r"

wait_manifest() {
    local path="$1"
    local label="$2"
    local waited=0
    echo "[chain] waiting for ${label} manifest: ${path}"
    while [ ! -f "$path" ]; do
        sleep 30
        waited=$((waited+30))
        echo "[chain] still waiting for ${label} (${waited}s elapsed)..."
    done
    echo "[chain] ${label} DONE (manifest found)"
}

# ── STAGE 1: wait for all seed-1 and seed-3 to complete ──────────────────────
wait_manifest "${TRAJ_DIR}/Qwen__Qwen3-8B/seed1/manifest.json"            "Qwen seed1"
wait_manifest "${TRAJ_DIR}/Qwen__Qwen3-8B/seed3/manifest.json"            "Qwen seed3"
wait_manifest "${TRAJ_DIR}/NousResearch__Hermes-3-Llama-3.1-8B/seed1/manifest.json"  "Hermes seed1"
wait_manifest "${TRAJ_DIR}/NousResearch__Hermes-3-Llama-3.1-8B/seed3/manifest.json"  "Hermes seed3"

echo "[chain] G3 trajectory generation complete. Starting activation recording."

# ── STAGE 2: record activations (all models, sequential on cuda:0) ────────────
for model_tag in Qwen__Qwen3-8B NousResearch__Hermes-3-Llama-3.1-8B; do
  model_arg="${model_tag//__//}"
  log_name="act_$(echo $model_tag | tr '[:upper:]' '[:lower:]' | tr -d '.-').log"
  echo "[chain] ${model_tag} activation recording start (cuda:0)..."
  for seed in 1 2 3; do
    python3 -m scripts.record_activations \
      --model "$model_arg" \
      --seed $seed \
      --device cuda:0 \
      >> "${LOG_DIR}/${log_name}" 2>&1
    echo "[chain] ${model_tag} seed${seed} activations done"
  done
  echo "[chain] ${model_tag} ALL activations done"
done

echo "[chain] G3 activations recorded."

# ── STAGE 3: probe + gates + statistical tests (G3 models) ───────────────────
echo "[chain] Running G3 analysis pipeline..."
python3 -m scripts.run_analysis --seeds 1 2 3 \
    --models Qwen/Qwen3-8B NousResearch/Hermes-3-Llama-3.1-8B \
    >> "${LOG_DIR}/analysis.log" 2>&1
echo "[chain] G3 analysis done."

# ── STAGE 4: causal intervention (sequential, single GPU) ────────────────────
echo "[chain] Running causal intervention (Qwen) on cuda:0..."
python3 -m scripts.run_intervention \
    --model Qwen/Qwen3-8B \
    --seeds 1 2 3 \
    --device cuda:0 \
    >> "${LOG_DIR}/intervention_qwen.log" 2>&1
echo "[chain] Qwen causal intervention done."

echo "[chain] Running causal intervention (Hermes) on cuda:0..."
python3 -m scripts.run_intervention \
    --model NousResearch/Hermes-3-Llama-3.1-8B \
    --seeds 1 2 3 \
    --device cuda:0 \
    >> "${LOG_DIR}/intervention_hermes.log" 2>&1
echo "[chain] Hermes causal intervention done."

# ── STAGE 5: G4 — xLAM-2 trajectories ────────────────────────────────────────
echo "[chain] Starting G4 xLAM-2 trajectory generation..."
python3 -m scripts.generate_trajectories --model "$XLAM" --seed 1 \
    >> "${LOG_DIR}/traj_xlam.log" 2>&1
echo "[chain] xLAM-2 seed1 trajectories done."
for seed in 2 3; do
  python3 -m scripts.generate_trajectories --model "$XLAM" --seed $seed --causal-only \
      >> "${LOG_DIR}/traj_xlam.log" 2>&1
  echo "[chain] xLAM-2 seed${seed} (causal-only) trajectories done."
done

# ── STAGE 6: G4 — xLAM-2 activations ────────────────────────────────────────
echo "[chain] xLAM-2 activation recording start (cuda:0)..."
for seed in 1 2 3; do
  python3 -m scripts.record_activations \
    --model "$XLAM" \
    --seed $seed \
    --device cuda:0 \
    >> "${LOG_DIR}/act_xlam.log" 2>&1
  echo "[chain] xLAM-2 seed${seed} activations done"
done
echo "[chain] xLAM-2 ALL activations done."

# ── STAGE 7: G4 — analysis (all three models) ────────────────────────────────
echo "[chain] Running full analysis (G3 + G4 models)..."
python3 -m scripts.run_analysis --seeds 1 2 3 \
    >> "${LOG_DIR}/analysis_g4.log" 2>&1
echo "[chain] Full analysis done."

# ── STAGE 8: G4 — xLAM-2 causal intervention ────────────────────────────────
echo "[chain] Running causal intervention (xLAM-2) on cuda:0..."
python3 -m scripts.run_intervention \
    --model "$XLAM" \
    --seeds 1 2 3 \
    --device cuda:0 \
    >> "${LOG_DIR}/intervention_xlam.log" 2>&1
echo "[chain] xLAM-2 causal intervention done."

# ── STAGE 9: decide H1 vs fallback + write results.md ────────────────────────
echo "[chain] Running decide_and_write..."
python3 -m scripts.decide_and_write \
    >> "${LOG_DIR}/decide.log" 2>&1
echo "[chain] Decision written."

# ── STAGE 10: figures ─────────────────────────────────────────────────────────
echo "[chain] Generating figures..."
python3 -m scripts.plot \
    >> "${LOG_DIR}/plot.log" 2>&1
echo "[chain] Figures done."

echo "[chain] ===== FULL G3+G4 PIPELINE COMPLETE ====="
