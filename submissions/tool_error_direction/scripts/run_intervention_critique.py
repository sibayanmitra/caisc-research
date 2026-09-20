"""Critique re-run: finer/smaller alpha sweep + a SUBSPACE-MATCHED random control.

Separates direction-specificity from norm-specificity (critic #3):
  - retry            : the real action direction v_retry (norm-matched)
  - subspace_random  : a random vector confined to the action subspace
                       span(v_retry, v_modify, v_giveup), norm-matched to v_retry.
                       Same action-subspace energy as v_retry but random orientation.
  - random           : full-space random unit vector, norm-matched (paper's existing null)

If subspace_random flips give-ups as well as retry -> "any shove in the action
subspace works" (axis/subspace, not the specific direction). If retry >> subspace_random
-> the specific orientation matters. Smaller alphas probe the dose-response.

Output: results/causal_flip_critique_<model>.csv
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/anonymous/agentic_paper")
from src.intervene import norm_matched, random_direction, regenerate_with_intervention
from src.label import ParsedCall, LABEL_RETRY, LABEL_GIVEUP
from src.prompts import get_split_assignment
from src.trajectory_gen import load_model

DATA_DIR = Path("/home/anonymous/agentic_paper/data/trajectories")
RESULTS_DIR = Path("/home/anonymous/agentic_paper/results")

ALPHAS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
DIRECTIONS = ("retry", "subspace_random", "random")


def action_subspace_basis(v_retry, v_modify, v_giveup):
    M = np.stack([v_retry, v_modify, v_giveup]).T  # (d, 3)
    U, S, _ = np.linalg.svd(M, full_matrices=False)
    k = int((S > 1e-6 * S[0]).sum())
    return U[:, :k]  # (d, k) orthonormal basis of the action subspace


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-limit", type=int, default=0)
    args = ap.parse_args()

    safe_model = args.model.replace("/", "__")
    dirs = np.load(RESULTS_DIR / f"directions_{safe_model}.npz")
    v_retry, v_modify, v_giveup = dirs["v_retry"], dirs["v_modify"], dirs["v_giveup"]
    best_layer = int(dirs["best_layer"])
    Q = action_subspace_basis(v_retry, v_modify, v_giveup)
    print(f"{args.model}: best_layer={best_layer}, action-subspace dim={Q.shape[1]}", flush=True)

    splits = get_split_assignment()
    model, tokenizer, _ = load_model(args.model, device=args.device)
    rng = np.random.default_rng(123)
    rows: list[dict] = []
    t0 = time.perf_counter(); total = 0

    for seed in args.seeds:
        in_dir = DATA_DIR / safe_model / f"seed{seed}"
        if not in_dir.exists():
            continue
        for tp in sorted(in_dir.glob("*.json")):
            if tp.name == "manifest.json":
                continue
            traj = json.loads(tp.read_text())
            if splits.get(traj["base_prompt_id"]) != "causal_test":
                continue
            if traj["label"] != LABEL_GIVEUP or traj["outcome"] == "success":
                continue
            if traj["parsed_call"] is None or traj["t_star_idx"] < 0:
                continue
            if args.n_limit and total >= args.n_limit:
                break

            failed_call = ParsedCall(name=traj["parsed_call"]["name"], args=dict(traj["parsed_call"]["args"]), raw="")
            prompt_ids = traj["second_prompt_input_ids"]
            t_star = traj["t_star_idx"]

            v_retry_nm = v_retry.copy()  # norm-matched reference (paper approximation ||h||~||v_retry||)
            # random vector confined to the action subspace, norm-matched to v_retry
            c = rng.standard_normal(Q.shape[1]); c /= (np.linalg.norm(c) + 1e-12)
            v_sub = Q @ c
            v_sub_nm = norm_matched(v_sub, v_retry_nm)
            v_rand_nm = norm_matched(random_direction(v_retry.shape[0], rng), v_retry_nm)
            vectors = {"retry": v_retry_nm, "subspace_random": v_sub_nm, "random": v_rand_nm}

            for alpha in ALPHAS:
                for kind, vec in vectors.items():
                    if alpha == 0.0 and kind != "retry":
                        continue
                    _, lab = regenerate_with_intervention(
                        model=model, tokenizer=tokenizer, prompt_ids=prompt_ids,
                        layer=best_layer, position=t_star,
                        vector=vec if alpha > 0 else None, alpha=alpha,
                        failed_call=failed_call, seed=seed + 1,
                    )
                    rows.append({
                        "model": args.model, "seed": seed, "base_prompt_id": traj["base_prompt_id"],
                        "tool": traj["tool"], "outcome": traj["outcome"], "alpha": alpha, "direction": kind,
                        "intervened_label": lab,
                        "flipped_to_retry": int(lab == LABEL_RETRY),
                        "flipped_broad": int(lab != LABEL_GIVEUP),
                    })
            total += 1
            if total % 20 == 0:
                print(f"  [{total}] elapsed {time.perf_counter()-t0:.0f}s", flush=True)
        if args.n_limit and total >= args.n_limit:
            break

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / f"causal_flip_critique_{safe_model}.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} rows, {total} trajectories) in {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
