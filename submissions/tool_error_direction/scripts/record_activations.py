"""Record residual-stream activations at 3 positions per saved trajectory.

Output: one .npz per (model, seed) containing
    H: (n_trajectories, n_layers, 3, d_model) float32
    meta: per-trajectory metadata (label, outcome, base_prompt_id, t_star_idx)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.activation_record import record_residuals
from src.log import log_compute
from src.trajectory_gen import load_model

DATA_DIR = Path("/home/anonymous/agentic_paper/data/trajectories")
ACT_DIR = Path("/home/anonymous/agentic_paper/data/activations")
ACT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-limit", type=int, default=0)
    args = ap.parse_args()

    safe_model = args.model.replace("/", "__")
    in_dir = DATA_DIR / safe_model / f"seed{args.seed}"
    if not in_dir.exists():
        raise SystemExit(f"no trajectory dir at {in_dir}")
    out_path = ACT_DIR / f"{safe_model}_seed{args.seed}.npz"
    meta_path = ACT_DIR / f"{safe_model}_seed{args.seed}.meta.json"

    model, tokenizer, _ = load_model(args.model, device=args.device)
    n_layers = len(model.model.layers) if hasattr(model, "model") else len(model.transformer.h)

    traj_paths = sorted(in_dir.glob(f"*_{args.model.replace('/', '__')}*.json")) or sorted(in_dir.glob("*.json"))
    traj_paths = [p for p in traj_paths if p.name != "manifest.json"]
    if args.n_limit:
        traj_paths = traj_paths[: args.n_limit]

    t0 = time.perf_counter()
    H_list: list[np.ndarray] = []
    meta_records: list[dict] = []

    for i, tp in enumerate(traj_paths):
        traj = json.loads(tp.read_text())
        if traj["t_star_idx"] < 0 or traj["label"] == "ambiguous":
            continue
        prompt_ids = traj["second_prompt_input_ids"]
        gen_ids = traj["second_generation_ids"]
        full_ids = prompt_ids + gen_ids
        t_star = traj["t_star_idx"]
        t_plus5 = min(t_star + 5, len(full_ids) - 1)
        if traj.get("pre_next_action_offset") is not None:
            pre_na = len(prompt_ids) + traj["pre_next_action_offset"] - 1
            pre_na = max(0, min(pre_na, len(full_ids) - 1))
        else:
            pre_na = min(len(full_ids) - 1, t_star + 20)

        positions = [t_star, t_plus5, pre_na]
        H = record_residuals(model, full_ids, positions, device=args.device)
        H_list.append(H[None, ...])  # (1, n_layers, 3, d_model)

        meta_records.append({
            "traj_file": tp.name,
            "base_prompt_id": traj["base_prompt_id"],
            "tool": traj["tool"],
            "outcome": traj["outcome"],
            "label": traj["label"],
            "t_star_idx": t_star,
            "t_plus5_idx": t_plus5,
            "pre_next_action_idx": pre_na,
        })

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(traj_paths)}] elapsed {time.perf_counter() - t0:.0f}s")

    if not H_list:
        raise SystemExit("no usable trajectories found")

    H_full = np.concatenate(H_list, axis=0)
    np.savez_compressed(out_path, H=H_full)
    meta_path.write_text(json.dumps(meta_records, indent=2))

    dur = time.perf_counter() - t0
    log_compute(
        stage=f"record_activations[{args.model}/seed{args.seed}]",
        gpu_seconds=dur,
        decision=f"record residuals for {H_full.shape[0]} trajectories",
        notes=f"shape={H_full.shape}, dtype=float32",
    )
    print(f"Wrote {out_path} shape={H_full.shape}")


if __name__ == "__main__":
    main()
