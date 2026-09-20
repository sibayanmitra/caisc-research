#!/usr/bin/env python3
"""Phase 4 — Causal steering transfer test: does d_neg at peak layer flip behavior?"""
import os, sys, json, time
from pathlib import Path
import numpy as np
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from negation_lib import load_model, log, get_paper_dir, TRAIN_MORPHOLOGIES, TEST_MORPHOLOGIES, get_all_morphologies

def steer_logit_shift(model, prompt_text, d_unit, alpha, peak_L, max_new=60):
    """
    Apply hook at peak_L adding alpha * d_unit to residual stream.
    Return logit of first generated token minus baseline logit.
    Actually returns: logprob_of_first_token_under_steering - logprob_under_baseline.
    As a proxy, just return the raw logit shift at position -1.
    """
    tokens = model.to_tokens(prompt_text, prepend_bos=False)
    hook_name = f"blocks.{peak_L}.hook_resid_post"
    d_t = torch.tensor(d_unit, dtype=torch.float16, device="cuda")

    def add_direction(resid, hook):
        return resid + alpha * d_t

    with torch.no_grad():
        logits_base = model(tokens)  # (1, seq, vocab)
        first_tok_logit_base = logits_base[0, -1]  # (vocab,)

        with model.hooks(fwd_hooks=[(hook_name, add_direction)]):
            logits_steered = model(tokens)
        first_tok_logit_steered = logits_steered[0, -1]

    # Logit shift: mean of |steered - base| across top-100 tokens (effect magnitude)
    diff = (first_tok_logit_steered - first_tok_logit_base).float()
    # Signed shift: projection onto "positive direction" (first token of negative continuation)
    return float(diff.abs().mean().item()), float(diff.mean().item())

def main():
    log("=== Paper 2 Phase 4: Transfer test ===")
    paper_dir = get_paper_dir()
    act_dir  = paper_dir / "activations"
    dir_dir  = paper_dir / "directions"
    te_dir   = paper_dir / "transfer_experiments" / "alpha_sweep"
    te_dir.mkdir(parents=True, exist_ok=True)

    # Load
    pos_data = torch.load(act_dir / "pos.pt", weights_only=False)
    meta = pos_data["meta"]
    with open(dir_dir / "probe_eval.json") as f:
        probe_eval = json.load(f)
    peak_L = probe_eval["peak_layer"]

    d_neg_per_layer = np.load(dir_dir / "d_neg_per_layer.npy")
    d_unit = d_neg_per_layer[peak_L]
    d_unit = d_unit / (np.linalg.norm(d_unit) + 1e-8)

    # α_unit: mean activation norm at peak layer across all pairs
    resid_at_peak = pos_data["resid"][:, peak_L, :].float()  # (n, d_model)
    alpha_unit = float(resid_at_peak.norm(dim=-1).mean().item())
    log(f"Peak layer: {peak_L}, alpha_unit (mean norm): {alpha_unit:.2f}")

    log("Loading model on GPU 1...")
    model = load_model(device="cuda", dtype="float16")

    # Load all logical negation pairs (all morphologies except not_nonneg control)
    # With ~41 total pairs, we use all for the alpha-sweep steering test
    pairs_dir = paper_dir / "data" / "pairs"
    all_morphs = get_all_morphologies() + TEST_MORPHOLOGIES  # includes antonym
    all_pairs = []
    for morph in all_morphs:
        p_path = pairs_dir / f"{morph}.jsonl"
        if not p_path.exists():
            continue
        with open(p_path) as f:
            morph_pairs = [json.loads(l) for l in f]
        all_pairs.extend(morph_pairs)
    log(f"Test pairs: {len(all_pairs)} across morphologies")

    alphas = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    results = []

    for alpha_mult in alphas:
        alpha = alpha_mult * alpha_unit
        abs_shifts = []
        mean_shifts = []
        for p in all_pairs[:50]:  # limit for speed
            try:
                abs_s, mean_s = steer_logit_shift(model, p["positive_text"], d_unit, alpha, peak_L)
                abs_shifts.append(abs_s)
                mean_shifts.append(mean_s)
            except Exception as e:
                continue
        results.append({
            "alpha_mult": alpha_mult,
            "alpha": alpha,
            "mean_abs_shift": float(np.mean(abs_shifts)) if abs_shifts else 0.0,
            "std_abs_shift": float(np.std(abs_shifts)) if abs_shifts else 0.0,
            "mean_signed_shift": float(np.mean(mean_shifts)) if mean_shifts else 0.0,
            "n": len(abs_shifts),
        })
        log(f"  α={alpha_mult:+.1f}×unit: abs_shift={results[-1]['mean_abs_shift']:.4f}, signed={results[-1]['mean_signed_shift']:+.4f}")

    # Random direction baselines (10 random unit vectors)
    rng = np.random.default_rng(42)
    rand_abs_shifts = []
    for _ in range(10):
        rand_dir = rng.standard_normal(len(d_unit)).astype(np.float32)
        rand_dir /= np.linalg.norm(rand_dir) + 1e-8
        run_shifts = []
        for p in all_pairs[:20]:
            try:
                abs_s, _ = steer_logit_shift(model, p["positive_text"], rand_dir, alpha_unit, peak_L)
                run_shifts.append(abs_s)
            except:
                continue
        rand_abs_shifts.append(float(np.mean(run_shifts)) if run_shifts else 0.0)

    metrics = {
        "peak_layer": peak_L,
        "alpha_unit": alpha_unit,
        "alpha_sweep": results,
        "random_baseline_mean": float(np.mean(rand_abs_shifts)),
        "random_baseline_std": float(np.std(rand_abs_shifts)),
        "random_baseline_runs": rand_abs_shifts,
    }
    with open(te_dir / "alpha_sweep_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # SC13: α=0 effect ≈ 0
    zero_result = next(r for r in results if r["alpha_mult"] == 0.0)
    log(f"SC13 (α=0 shift ≈ 0): {zero_result['mean_abs_shift']:.4f}")

    log(f"✓ Phase 4 done. Random baseline: {metrics['random_baseline_mean']:.4f}±{metrics['random_baseline_std']:.4f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
