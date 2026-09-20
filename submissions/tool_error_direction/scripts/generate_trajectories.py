"""Generate full trajectory dataset for one (model, seed) pair.

Usage:
    python -m scripts.generate_trajectories --model Qwen/Qwen3-8B --seed 1
    python -m scripts.generate_trajectories --model NousResearch/Hermes-3-Llama-3.1-8B --seed 1 --causal-only
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.log import log_compute, log_decision
from src.preflight import sample_outcome
from src.prompts import get_all_prompts, get_split_assignment
from src.trajectory_gen import generate_trajectory, load_model, save_trajectory

DATA_DIR = Path("/home/anonymous/agentic_paper/data/trajectories")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--causal-only",
        action="store_true",
        help="Only generate trajectories whose base prompts are in the causal_test split (used for seeds 2/3).",
    )
    ap.add_argument(
        "--n-limit",
        type=int,
        default=0,
        help="If >0, cap total trajectories at this number (smoke testing).",
    )
    args = ap.parse_args()

    t0 = time.perf_counter()
    log_decision(
        decision=f"trajectory generation for model={args.model} seed={args.seed} causal_only={args.causal_only}",
        rationale="full dataset for seed-1; matched causal-test subset for seeds 2/3 per preregistration §11",
    )
    model, tokenizer, use_tool_role = load_model(args.model, device=args.device)
    assignment = get_split_assignment()
    all_prompts = get_all_prompts()
    prompts = [
        p for p in all_prompts
        if not args.causal_only or assignment[p.id] == "causal_test"
    ]
    safe_model = args.model.replace("/", "__")
    out_dir = DATA_DIR / safe_model / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    OUTCOMES_CYCLE = ("success", "rate_limited", "no_results", "invalid_arg")

    total = 0
    label_counts: dict[str, int] = {}
    for bp in prompts:
        for outcome in OUTCOMES_CYCLE:
            if args.n_limit and total >= args.n_limit:
                break
            traj = generate_trajectory(
                model=model,
                tokenizer=tokenizer,
                use_tool_role=use_tool_role,
                base_prompt=bp,
                outcome=outcome,
                condition_seed=0,
                model_seed=args.seed,
                model_name=args.model,
            )
            save_trajectory(traj, out_dir)
            label_counts[traj.label] = label_counts.get(traj.label, 0) + 1
            total += 1
            if total % 50 == 0:
                elapsed = time.perf_counter() - t0
                eta = elapsed / total * (len(prompts) * 4 - total)
                print(f"  [{total}/{len(prompts)*4}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")
        if args.n_limit and total >= args.n_limit:
            break

    dur = time.perf_counter() - t0
    log_compute(
        stage=f"trajectory_gen[{args.model}/seed{args.seed}]",
        gpu_seconds=dur,
        decision=f"generate {total} trajectories",
        notes=f"labels: {label_counts}",
    )

    manifest = {
        "model": args.model,
        "seed": args.seed,
        "causal_only": args.causal_only,
        "n_trajectories": total,
        "label_counts": label_counts,
        "wall_seconds": dur,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {total} trajectories to {out_dir} in {dur:.0f}s")
    print(f"Labels: {label_counts}")


if __name__ == "__main__":
    main()
