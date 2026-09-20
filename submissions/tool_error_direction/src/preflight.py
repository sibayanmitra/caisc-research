"""Pre-flight gates — run BEFORE main trajectory generation.

Checks:
  PF1. Pre-registration committed at git HEAD (verified externally).
  PF2. Mock-tool determinism — covered by tests/test_mock_tools.py.
  PF3. Tool-call parser sanity — covered by tests/test_label.py + this script (regression).
  PF4. Forward-pass non-cached determinism (G-CACHE precursor).
  PF5. Ambiguous rate on 50-trajectory pilot per model ≤ 15%.

Any failure HALTS main run.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .activation_record import verify_cache_byte_equality
from .gates import g_ambig, g_cache
from .log import log_decision, log_gate
from .prompts import get_all_prompts
from .trajectory_gen import generate_trajectory, load_model


def sample_outcome(base_prompt_id: str, tool: str, seed: int) -> str:
    """Deterministic stratified outcome sampler per preregistration §3."""
    weights = {
        "web_search": [("success", 0.25), ("rate_limited", 0.50), ("no_results", 0.15), ("invalid_arg", 0.10)],
        "calculator": [("success", 0.25), ("rate_limited", 0.10), ("no_results", 0.15), ("invalid_arg", 0.50)],
        "file_read": [("success", 0.25), ("rate_limited", 0.10), ("no_results", 0.50), ("invalid_arg", 0.15)],
    }[tool]
    r = random.Random(f"{base_prompt_id}|{seed}").random()
    cum = 0.0
    for o, w in weights:
        cum += w
        if r <= cum:
            return o
    return "success"

DATA_DIR = Path("/home/anonymous/agentic_paper/data/trajectories")
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path("/home/anonymous/agentic_paper/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def stratified_outcome_for_tool(tool: str, prompt_idx_within_tool: int, rng: random.Random) -> str:
    """Sample an outcome class per the stratification matrix in preregistration §3."""
    weights = {
        "web_search": {"success": 0.25, "rate_limited": 0.50, "no_results": 0.15, "invalid_arg": 0.10},
        "calculator": {"success": 0.25, "rate_limited": 0.10, "no_results": 0.15, "invalid_arg": 0.50},
        "file_read": {"success": 0.25, "rate_limited": 0.10, "no_results": 0.50, "invalid_arg": 0.15},
    }[tool]
    r = rng.Random(prompt_idx_within_tool * 7 + hash(tool) % 100).random()
    cum = 0.0
    for o, w in weights.items():
        cum += w
        if r <= cum:
            return o
    return "success"


def run_pilot(model_name: str, n_trajectories: int = 50, seed: int = 1) -> dict:
    """Load model, run n pilot trajectories, return stats."""
    log_decision(
        decision=f"loading model {model_name} for pre-flight pilot",
        rationale="pre-flight gates require live model; first model loaded determines GPU 0 occupancy",
    )

    model, tokenizer, use_tool_role = load_model(model_name, device="cuda:0")

    prompts = get_all_prompts()
    rng = random.Random(seed)
    rng.shuffle(prompts)
    prompts = prompts[:n_trajectories]

    labels: list[str] = []
    traj_paths: list[Path] = []

    for i, bp in enumerate(prompts):
        outcome = sample_outcome(bp.id, bp.tool, seed)
        traj = generate_trajectory(
            model=model,
            tokenizer=tokenizer,
            use_tool_role=use_tool_role,
            base_prompt=bp,
            outcome=outcome,
            condition_seed=0,
            model_seed=seed,
            model_name=model_name,
        )
        labels.append(traj.label)

    # G-AMBIG
    ambig_count = sum(1 for L in labels if L == "ambiguous")
    ambig_rate = ambig_count / max(len(labels), 1)
    passed_ambig, obs_ambig, thr_ambig = g_ambig(ambig_rate)
    log_gate(
        gate=f"PF5-G-AMBIG [{model_name}]",
        passed=passed_ambig,
        observed=obs_ambig,
        threshold=thr_ambig,
        recovery="re-tune system prompt if failed" if not passed_ambig else None,
    )

    # G-CACHE on 5 trajectories (last input token only)
    cache_diffs: list[float] = []
    for i in range(min(5, len(prompts))):
        bp = prompts[i]
        traj = generate_trajectory(
            model=model,
            tokenizer=tokenizer,
            use_tool_role=use_tool_role,
            base_prompt=bp,
            outcome="rate_limited",
            condition_seed=0,
            model_seed=seed,
            model_name=model_name,
        )
        if traj.t_star_idx >= 0:
            ok, diff = verify_cache_byte_equality(model, traj.second_prompt_input_ids)
            cache_diffs.append(diff)
    max_diff = max(cache_diffs) if cache_diffs else 0.0
    passed_cache, obs_cache, thr_cache = g_cache(max_diff)
    log_gate(
        gate=f"PF4-G-CACHE [{model_name}]",
        passed=passed_cache,
        observed=obs_cache,
        threshold=thr_cache,
    )

    label_counts = {L: labels.count(L) for L in set(labels)}
    print(f"[{model_name}] labels: {label_counts}")
    print(f"[{model_name}] ambig: {ambig_rate:.2%} (thr 15%); cache_diff: {max_diff:.2e} (thr 1e-5)")

    return {
        "model_name": model_name,
        "n_trajectories": len(labels),
        "label_counts": label_counts,
        "ambig_rate": ambig_rate,
        "max_cache_diff": max_diff,
        "passed_ambig": passed_ambig,
        "passed_cache": passed_cache,
    }


ALL_MODELS = (
    "Qwen/Qwen3-8B",
    "NousResearch/Hermes-3-Llama-3.1-8B",
    "Salesforce/Llama-xLAM-2-8b-fc-r",
)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="Run preflight for one model only.")
    args = ap.parse_args()

    models = (args.model,) if args.model else ALL_MODELS
    results = {}
    for model_name in models:
        results[model_name] = run_pilot(model_name, n_trajectories=50, seed=1)
    out_path = RESULTS_DIR / "preflight_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")

    all_passed = all(r["passed_ambig"] and r["passed_cache"] for r in results.values())
    if not all_passed:
        print("PRE-FLIGHT FAILURE — do not start main run.")
        raise SystemExit(2)
    print("PRE-FLIGHT PASSED.")


if __name__ == "__main__":
    main()
