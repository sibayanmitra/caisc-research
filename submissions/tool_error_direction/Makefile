.PHONY: tests preflight trajectories activations analyze intervene decide plot all \
        preflight-g4 trajectories-g4 activations-g4 intervene-g4 all-g4 clean

PY := python3
MODELS := Qwen/Qwen3-8B NousResearch/Hermes-3-Llama-3.1-8B
SEEDS := 1 2 3

tests:
	$(PY) -m pytest tests/ -q

preflight:
	$(PY) -m src.preflight

# Trajectories: full set for seed=1; causal-test subset for seeds 2 and 3.
trajectories:
	$(PY) -m scripts.generate_trajectories --model Qwen/Qwen3-8B --seed 1
	$(PY) -m scripts.generate_trajectories --model NousResearch/Hermes-3-Llama-3.1-8B --seed 1
	$(PY) -m scripts.generate_trajectories --model Qwen/Qwen3-8B --seed 2 --causal-only
	$(PY) -m scripts.generate_trajectories --model NousResearch/Hermes-3-Llama-3.1-8B --seed 2 --causal-only
	$(PY) -m scripts.generate_trajectories --model Qwen/Qwen3-8B --seed 3 --causal-only
	$(PY) -m scripts.generate_trajectories --model NousResearch/Hermes-3-Llama-3.1-8B --seed 3 --causal-only

activations:
	$(PY) -m scripts.record_activations --model Qwen/Qwen3-8B --seed 1
	$(PY) -m scripts.record_activations --model NousResearch/Hermes-3-Llama-3.1-8B --seed 1
	$(PY) -m scripts.record_activations --model Qwen/Qwen3-8B --seed 2
	$(PY) -m scripts.record_activations --model NousResearch/Hermes-3-Llama-3.1-8B --seed 2
	$(PY) -m scripts.record_activations --model Qwen/Qwen3-8B --seed 3
	$(PY) -m scripts.record_activations --model NousResearch/Hermes-3-Llama-3.1-8B --seed 3

analyze:
	$(PY) -m scripts.run_analysis --seeds 1 2 3

intervene:
	$(PY) -m scripts.run_intervention --model Qwen/Qwen3-8B --seeds 1 2 3 --device cuda:0
	$(PY) -m scripts.run_intervention --model NousResearch/Hermes-3-Llama-3.1-8B --seeds 1 2 3 --device cuda:0

decide:
	$(PY) -m scripts.decide_and_write

plot:
	$(PY) -m scripts.plot

all: tests preflight trajectories activations analyze intervene decide plot

# ── G4: Salesforce/Llama-xLAM-2-8b-fc-r ──────────────────────────────────────
XLAM := Salesforce/Llama-xLAM-2-8b-fc-r

preflight-g4:
	$(PY) -m src.preflight --model $(XLAM)

trajectories-g4:
	$(PY) -m scripts.generate_trajectories --model $(XLAM) --seed 1
	$(PY) -m scripts.generate_trajectories --model $(XLAM) --seed 2 --causal-only
	$(PY) -m scripts.generate_trajectories --model $(XLAM) --seed 3 --causal-only

activations-g4:
	$(PY) -m scripts.record_activations --model $(XLAM) --seed 1 --device cuda:0
	$(PY) -m scripts.record_activations --model $(XLAM) --seed 2 --device cuda:0
	$(PY) -m scripts.record_activations --model $(XLAM) --seed 3 --device cuda:0

intervene-g4:
	$(PY) -m scripts.run_intervention --model $(XLAM) --seeds 1 2 3 --device cuda:0

all-g4: preflight-g4 trajectories-g4 activations-g4 analyze intervene-g4 decide plot

clean:
	rm -rf data/trajectories/*/* data/activations/* results/*.csv results/*.json results/figures/*.png logs/*.jsonl
