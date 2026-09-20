#!/bin/bash
# Overnight #3 re-run: finer alpha + subspace-matched control, all 3 models, GPU 2 only.
cd /home/anonymous/agentic_paper || exit 1
PY=/home/anonymous/CAISc/autovoila/all-spikes/_shared/venv/bin/python
[ -x "$PY" ] || PY=python3
export CUDA_VISIBLE_DEVICES=2
LOG=results/critique_overnight.log
echo "START $(date)" > "$LOG"
for M in "NousResearch/Hermes-3-Llama-3.1-8B" "Salesforce/Llama-xLAM-2-8b-fc-r" "Qwen/Qwen3-8B"; do
  echo "=== $M $(date) ===" >> "$LOG"
  "$PY" scripts/run_intervention_critique.py --model "$M" --seeds 1 2 3 --device cuda:0 >> "$LOG" 2>&1
  echo "--- done $M $(date) ---" >> "$LOG"
done
echo "ALL DONE $(date)" >> "$LOG"
