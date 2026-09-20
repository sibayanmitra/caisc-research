#!/usr/bin/env python3
"""
Phase 6.5 — Steering experiment (constructive lever).

Inject α × d_probe into residual stream at peak layer (L*) during generation.
Sweep α ∈ {-2, -1, -0.5, 0, +0.5, +1, +2} × α_unit.
Measure self-correction rate vs random-direction baseline.

Outputs: steering/steering_metrics.json
Sanity checks SC21-SC23.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from self_monitor_lib import load_model, format_reasoning_prompt, detect_self_correction, get_paper_dir

N_PROBLEMS = 20      # per alpha value
N_RAND_DIRS = 3      # random direction controls
MAX_NEW_TOKENS = 600


def log(msg, also_print=True):
    log_path = get_paper_dir() / "notes" / "sanity_log.md"
    with open(log_path, "a") as f:
        f.write(f"- [{time.strftime('%H:%M:%S')}] {msg}\n")
    if also_print:
        print(msg, flush=True)


def generate_with_steering(model, prompt_text, d_unit, alpha, peak_L, max_new=MAX_NEW_TOKENS):
    """Generate with additive hook α * d_unit at residual stream layer peak_L."""
    tokens = model.to_tokens(prompt_text, prepend_bos=False)
    hook_name = f"blocks.{peak_L}.hook_resid_post"
    d_t = torch.tensor(d_unit, dtype=torch.float16, device="cuda")

    def add_direction(resid, hook):
        return resid + alpha * d_t

    with torch.no_grad():
        if alpha == 0.0:
            gen = model.generate(tokens, max_new_tokens=max_new, do_sample=True,
                                  temperature=0.7, top_p=0.95, verbose=False, return_type="tokens")
        else:
            with model.hooks(fwd_hooks=[(hook_name, add_direction)]):
                gen = model.generate(tokens, max_new_tokens=max_new, do_sample=True,
                                      temperature=0.7, top_p=0.95, verbose=False, return_type="tokens")
    return model.to_string(gen[0, tokens.shape[1]:])


def evaluate_sc_rate(model, problems, d_unit, alpha, peak_L, condition_name):
    """Generate on all problems, return SC rate."""
    n_sc = 0
    n_total = 0
    for problem in problems:
        try:
            prompt = format_reasoning_prompt(problem["question"], model)
            text   = generate_with_steering(model, prompt, d_unit, alpha, peak_L)
            marker = detect_self_correction(text, min_position=20)
            if marker:
                n_sc += 1
            n_total += 1
        except Exception as e:
            log(f"  ERROR in {condition_name}: {e}")
    rate = n_sc / n_total if n_total > 0 else 0.0
    log(f"  {condition_name}: SC={n_sc}/{n_total} ({rate:.2%})")
    return rate, n_sc, n_total


def main():
    log("\n## Phase 6.5: Steering experiment\n", also_print=False)
    log("=" * 60)
    log(f"Phase 6.5 started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    paper_dir = get_paper_dir()
    probes_dir = paper_dir / "probes"
    steer_dir  = paper_dir / "steering"
    steer_dir.mkdir(parents=True, exist_ok=True)

    # Load probe direction from Phase 3
    probe_eval_path = probes_dir / "probe_eval.json"
    if not probe_eval_path.exists():
        log("✗ probe_eval.json not found — run Phase 3 first")
        return 1
    with open(probe_eval_path) as f:
        probe_eval = json.load(f)
    peak_L = probe_eval["peak_layer"]
    peak_o_idx = probe_eval["peak_offset_idx"]
    log(f"Peak layer: {peak_L}, offset_idx: {peak_o_idx}, peak AUROC: {probe_eval['peak_auroc']:.3f}")

    # Load probe weights at peak (L, offset) as d_probe direction
    import pickle
    with open(probes_dir / "layer_offset_probes.pkl", "rb") as f:
        probes = pickle.load(f)
    key = (peak_L, peak_o_idx)
    if key not in probes:
        log(f"✗ No probe at key {key}")
        return 1
    probe_info = probes[key]
    d_probe = probe_info["weights"].astype(np.float32)  # (d_model,)
    d_probe_unit = d_probe / (np.linalg.norm(d_probe) + 1e-8)
    np.save(steer_dir / "probe_direction.npy", d_probe_unit)
    log(f"Probe direction norm: {np.linalg.norm(d_probe):.4f} → normalized")

    # Load model
    log("Loading model...")
    model, choice = load_model()
    log(f"  Model: {choice['model_id']}")

    # α_unit: mean activation norm at peak layer (use positive examples as reference)
    import torch as _torch
    act_dir = paper_dir / "activations"
    pos_data = _torch.load(act_dir / "positive.pt", weights_only=False)
    resid_at_peak = pos_data["resid"][:, peak_L, peak_o_idx, :].float()
    alpha_unit = float(resid_at_peak.norm(dim=-1).mean().item())
    log(f"α_unit (mean resid norm at peak layer): {alpha_unit:.2f}")

    # Load held-out problems for steering test
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    import random
    rng = random.Random(999)
    problems = [{"question": ex["problem"], "answer": ex["answer"]} for ex in ds]
    rng.shuffle(problems)
    steer_problems = problems[:N_PROBLEMS]
    log(f"Held-out steering problems: {len(steer_problems)}")

    # Sweep α values
    alpha_mults = [-2.0, -1.0, 0.0, 1.0, 2.0]
    sweep_results = []
    for alpha_mult in alpha_mults:
        alpha = alpha_mult * alpha_unit
        rate, n_sc, n_total = evaluate_sc_rate(
            model, steer_problems, d_probe_unit, alpha, peak_L,
            f"α={alpha_mult:+.1f}×unit"
        )
        sweep_results.append({
            "alpha_mult": alpha_mult,
            "alpha": float(alpha),
            "sc_rate": rate,
            "n_sc": n_sc,
            "n_total": n_total,
        })

    # SC21: α=0 matches baseline (should be close to natural SC rate from Phase 6 baseline)
    zero_rate = next(r["sc_rate"] for r in sweep_results if r["alpha_mult"] == 0.0)
    log(f"\nSC21 (α=0 SC rate): {zero_rate:.2%}")

    # SC22: monotonic check at α>0 vs α<0
    pos_rates = [r["sc_rate"] for r in sweep_results if r["alpha_mult"] > 0]
    neg_rates = [r["sc_rate"] for r in sweep_results if r["alpha_mult"] < 0]
    if pos_rates and neg_rates:
        log(f"SC22: mean SC at α>0 = {np.mean(pos_rates):.2%}, mean at α<0 = {np.mean(neg_rates):.2%}")

    # Random direction baselines
    rng_np = np.random.default_rng(42)
    rand_sc_rates = []
    alpha_test = 1.0 * alpha_unit  # test at α=+1
    for rand_i in range(N_RAND_DIRS):
        rand_dir = rng_np.standard_normal(len(d_probe_unit)).astype(np.float32)
        rand_dir /= np.linalg.norm(rand_dir) + 1e-8
        rate, _, _ = evaluate_sc_rate(
            model, steer_problems[:20], rand_dir, alpha_test, peak_L,
            f"random_dir_{rand_i}"
        )
        rand_sc_rates.append(rate)
    rand_mean = float(np.mean(rand_sc_rates))
    rand_std  = float(np.std(rand_sc_rates))
    log(f"\nSC23: random dir SC rate at α=+1: {rand_mean:.2%}±{rand_std:.2%}")
    probe_rate_at_1 = next((r["sc_rate"] for r in sweep_results if r["alpha_mult"] == 1.0), 0)
    log(f"SC23: probe dir SC rate at α=+1: {probe_rate_at_1:.2%}")
    log(f"SC23 {'PASS' if abs(probe_rate_at_1 - zero_rate) > abs(rand_mean - zero_rate) else 'WARN'}: probe effect > random")

    # Save
    metrics = {
        "peak_layer": peak_L,
        "alpha_unit": float(alpha_unit),
        "n_problems": N_PROBLEMS,
        "alpha_sweep": sweep_results,
        "random_baseline": {
            "n_dirs": N_RAND_DIRS,
            "alpha_test_mult": 1.0,
            "sc_rates": rand_sc_rates,
            "mean": rand_mean,
            "std": rand_std,
        },
        "sanity": {
            "sc21_zero_alpha_rate": zero_rate,
            "sc22_pos_mean": float(np.mean(pos_rates)) if pos_rates else None,
            "sc22_neg_mean": float(np.mean(neg_rates)) if neg_rates else None,
            "sc23_probe_vs_random_at_alpha1": {
                "probe": probe_rate_at_1,
                "random_mean": rand_mean,
            },
        },
    }
    with open(steer_dir / "steering_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log(f"\n✓ Phase 6.5 complete. Steering curve saved.")
    log(f"  α=-2: {sweep_results[0]['sc_rate']:.2%} → α=0: {zero_rate:.2%} → α=+2: {sweep_results[-1]['sc_rate']:.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
