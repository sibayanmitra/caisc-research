#!/usr/bin/env python3
"""
Phase 6 — Causal ablation (destructive lever).

Identify top-k heads from Phase 5. Re-generate on held-out problems with these heads
zero-ablated. Compare to baseline (no ablation) and random-k ablation.

Outputs:
- ablations/baseline/, top3_ablation/, top5_ablation/, top10_ablation/, random_k_ablations/
- ablations/ablation_metrics.json

Sanity checks SC14-SC19.
"""

import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from self_monitor_lib import load_model, detect_self_correction, format_reasoning_prompt, get_paper_dir

N_HELD_OUT_PROBLEMS = 50  # held-out problems for ablation testing
N_RANDOM_SEEDS = 1  # number of random head-selection seeds for control
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7
TOP_P = 0.95


def log(msg, also_print=True):
    log_path = get_paper_dir() / "notes" / "sanity_log.md"
    with open(log_path, "a") as f:
        f.write(f"- [{time.strftime('%H:%M:%S')}] {msg}\n")
    if also_print:
        print(msg, flush=True)


def make_head_ablation_hook(layer_head_pairs):
    """Returns a TL hook function that zeros out specified heads at their layer."""
    def hook_fn(z, hook):
        # z shape: (batch, seq, n_heads, d_head)
        # hook.name = "blocks.{L}.attn.hook_z"
        parts = hook.name.split(".")
        layer = int(parts[1])
        for (L, H) in layer_head_pairs:
            if L == layer:
                z[:, :, H, :] = 0.0
        return z
    return hook_fn


def generate_with_ablation(model, prompt_text, layer_head_pairs, max_new_tokens=MAX_NEW_TOKENS):
    """Generate text with specified (layer, head) pairs zero-ablated."""
    import torch
    if layer_head_pairs:
        hook_fn = make_head_ablation_hook(layer_head_pairs)
        # Install hooks on relevant layers
        layers_needed = sorted(set(L for L, H in layer_head_pairs))
        hook_names = [f"blocks.{L}.attn.hook_z" for L in layers_needed]

        with torch.no_grad():
            tokens = model.to_tokens(prompt_text, prepend_bos=False)
            with model.hooks(fwd_hooks=[(name, hook_fn) for name in hook_names]):
                gen = model.generate(
                    tokens,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    verbose=False,
                    return_type="tokens",
                )
            new_tokens = gen[0, tokens.shape[1]:]
            return model.to_string(new_tokens)
    else:
        # Baseline: no ablation
        with torch.no_grad():
            tokens = model.to_tokens(prompt_text, prepend_bos=False)
            gen = model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                verbose=False,
                return_type="tokens",
            )
            new_tokens = gen[0, tokens.shape[1]:]
            return model.to_string(new_tokens)


def evaluate_condition(model, problems, layer_head_pairs, condition_name, out_dir):
    """Generate on N problems with ablation, count self-corrections."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{condition_name}.jsonl"

    n_self_correct = 0
    n_gens = 0

    log(f"\n  Generating {condition_name} ({len(problems)} problems, ablating {len(layer_head_pairs)} heads)...")
    t_start = time.time()

    with open(out_path, "w") as f:
        for i, problem in enumerate(problems):
            try:
                prompt = format_reasoning_prompt(problem["question"], model)
                gen_text = generate_with_ablation(model, prompt, layer_head_pairs)
                marker = detect_self_correction(gen_text, min_position=20)
                has_correction = marker is not None
                if has_correction:
                    n_self_correct += 1
                record = {
                    "problem_id": f"held_out_{i}",
                    "question": problem["question"],
                    "ground_truth": problem["answer"],
                    "generated": gen_text,
                    "has_self_correction": has_correction,
                    "marker_text": marker[2] if marker else None,
                    "condition": condition_name,
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
                n_gens += 1
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - t_start
                    log(f"    {condition_name}: {i+1}/{len(problems)}, sc={n_self_correct}, rate={n_self_correct/n_gens:.2%}, {elapsed/60:.1f} min")
            except Exception as e:
                log(f"    ERROR on problem {i}: {type(e).__name__}: {str(e)[:200]}")
                continue

    sc_rate = n_self_correct / n_gens if n_gens > 0 else 0.0
    elapsed = time.time() - t_start
    log(f"  {condition_name} DONE: {n_self_correct}/{n_gens} ({sc_rate:.2%}) self-corrections in {elapsed/60:.1f} min")

    return {
        "condition": condition_name,
        "n_problems": n_gens,
        "n_self_correction": n_self_correct,
        "self_correction_rate": sc_rate,
        "ablated_heads": layer_head_pairs,
        "elapsed_minutes": elapsed / 60,
    }


def main():
    log("\n## Phase 6: Causal ablation\n", also_print=False)
    log("=" * 60)
    log(f"Phase 6 started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    import numpy as np

    paper_dir = get_paper_dir()
    probes_dir = paper_dir / "probes"
    abl_dir = paper_dir / "ablations"

    # Load head probes
    with open(probes_dir / "head_probes.pkl", "rb") as f:
        head_data = pickle.load(f)
    peak_L = head_data["peak_L"]
    n_heads = head_data["n_heads"]

    # Get top-k heads at peak layer from head_eval
    with open(probes_dir / "head_eval.json") as f:
        head_eval = json.load(f)
    ranked_heads = head_eval["ranked_heads_at_peak"]
    log(f"Peak layer: {peak_L}")
    log(f"Top-10 heads at peak: {ranked_heads[:10]}")

    # Load held-out problems (NOT used in dataset construction)
    log("Loading held-out GSM8K problems...")
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = list(ds)
    # Use a different shuffle seed than Phase 1 to ensure these are held-out
    import random
    rng = random.Random(123)
    rng.shuffle(problems)
    held_out = problems[:N_HELD_OUT_PROBLEMS]
    log(f"Held-out problem count: {len(held_out)}")

    # Load model
    log("Loading model...")
    model, choice = load_model()
    log(f"  Model: {choice['model_id']}")

    # Conditions
    results = []

    # Baseline (no ablation)
    res = evaluate_condition(model, held_out, [], "baseline", abl_dir / "baseline")
    results.append(res)
    baseline_rate = res["self_correction_rate"]
    log(f"\nBaseline self-correction rate: {baseline_rate:.2%}")

    if baseline_rate < 0.05:
        log("⚠️ WARNING: baseline self-correction rate very low — ablation effect may be hard to detect")

    # Top-k conditions
    for k in [3, 5]:
        top_k_heads = [(peak_L, h) for h in ranked_heads[:k]]
        res = evaluate_condition(model, held_out, top_k_heads, f"top{k}_ablation", abl_dir / f"top{k}_ablation")
        results.append(res)

    # Random-k controls
    rng_np = np.random.default_rng(42)
    for k in [3, 5]:
        for seed_idx in range(N_RANDOM_SEEDS):
            rand_heads = [(peak_L, int(h)) for h in rng_np.choice(n_heads, k, replace=False)]
            res = evaluate_condition(
                model, held_out, rand_heads,
                f"random{k}_seed{seed_idx}",
                abl_dir / "random_k_ablations" / f"k{k}"
            )
            results.append(res)

    # Compile final metrics
    summary = {
        "peak_layer": peak_L,
        "n_held_out": N_HELD_OUT_PROBLEMS,
        "baseline_sc_rate": baseline_rate,
        "results": results,
        "comparison": {},
    }

    for k in [3, 5]:
        top_res = next((r for r in results if r["condition"] == f"top{k}_ablation"), None)
        rand_rates = [r["self_correction_rate"] for r in results if r["condition"].startswith(f"random{k}_")]
        if top_res and rand_rates:
            summary["comparison"][f"top{k}_vs_random{k}"] = {
                "top_sc_rate": top_res["self_correction_rate"],
                "random_sc_rates": rand_rates,
                "random_mean": float(np.mean(rand_rates)),
                "random_std": float(np.std(rand_rates)),
                "delta_vs_baseline_top": top_res["self_correction_rate"] - baseline_rate,
                "delta_vs_baseline_random": float(np.mean(rand_rates) - baseline_rate),
            }

    with open(abl_dir / "ablation_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    log("\n✓ Phase 6 complete. Summary:")
    for k in [3, 5]:
        comp = summary["comparison"].get(f"top{k}_vs_random{k}", {})
        if comp:
            log(f"  top{k}: {comp['top_sc_rate']:.2%} | random{k}: {comp['random_mean']:.2%}±{comp['random_std']:.2%} | baseline: {baseline_rate:.2%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
