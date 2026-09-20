"""Top-level orchestrator: trajectories → activations → analysis → intervention → decide → plot.

Run via `python -m scripts.run_main`. Streams stdout for `Monitor` consumption.
"""

from __future__ import annotations

import subprocess
import sys
import time

MODELS = ["Qwen/Qwen3-8B", "NousResearch/Hermes-3-Llama-3.1-8B"]


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    rc = subprocess.call(cmd)
    dt = time.perf_counter() - t0
    print(f"<<< rc={rc} in {dt:.0f}s", flush=True)
    if rc != 0:
        raise SystemExit(f"step failed: {' '.join(cmd)}")


def main() -> None:
    # Seed 1: full dataset both models
    for m in MODELS:
        run([sys.executable, "-m", "scripts.generate_trajectories", "--model", m, "--seed", "1"])
    # Seeds 2, 3: causal-test only
    for seed in (2, 3):
        for m in MODELS:
            run([
                sys.executable, "-m", "scripts.generate_trajectories",
                "--model", m, "--seed", str(seed), "--causal-only",
            ])
    # Activations for all (model, seed)
    for seed in (1, 2, 3):
        for m in MODELS:
            run([sys.executable, "-m", "scripts.record_activations", "--model", m, "--seed", str(seed)])
    run([sys.executable, "-m", "scripts.run_analysis", "--seeds", "1", "2", "3"])
    for m in MODELS:
        run([sys.executable, "-m", "scripts.run_intervention", "--model", m, "--seeds", "1", "2", "3"])
    run([sys.executable, "-m", "scripts.decide_and_write"])
    run([sys.executable, "-m", "scripts.plot"])
    print("\n=== ALL STAGES COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
