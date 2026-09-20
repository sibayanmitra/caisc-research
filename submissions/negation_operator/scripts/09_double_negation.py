#!/usr/bin/env python3
"""
Double-negation cancellation test.

If d_neg is a true logical operator, then applying it twice should:
  - Steer once (+alpha): model behaves as if negated
  - Steer twice (+2*alpha): cancellation — model returns toward original

If instead it's a polarity/valence feature, double application will
just push further in the same direction (no cancellation).

Run with: CUDA_VISIBLE_DEVICES=2 python scripts/09_double_negation.py
"""
import os, sys, json, time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]

import torch
import numpy as np

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_direction(n_layers=28, d_model=3584):
    """Load the peak-layer negation direction from saved activations."""
    act_dir = ROOT / "activations"
    neg_d = torch.load(act_dir / "neg.pt", map_location="cpu")
    pos_d = torch.load(act_dir / "pos.pt", map_location="cpu")

    neg_t = neg_d["resid"].float()   # (n_pairs, n_layers, d)
    pos_t = pos_d["resid"].float()

    # Use layer 5 (peak per paper)
    PEAK_L = 5
    neg_l = neg_t[:, PEAK_L, :]
    pos_l = pos_t[:, PEAK_L, :]

    # SVD direction
    delta = neg_l - pos_l
    delta -= delta.mean(0)
    _, _, Vt = torch.linalg.svd(delta, full_matrices=False)
    d_neg = Vt[0]               # top singular vector
    d_neg = d_neg / d_neg.norm()
    log(f"d_neg extracted at layer {PEAK_L}, norm={d_neg.norm():.4f}")
    return d_neg, PEAK_L


def steer_and_sample(model, d_neg, peak_layer, prompt_text, alpha, n_tokens=60):
    """Apply hook at peak_layer adding alpha * d_neg, then generate."""
    tokens = model.to_tokens(prompt_text, prepend_bos=True)

    def hook_fn(resid, hook):
        resid = resid + alpha * d_neg.to(resid.device)
        return resid

    hook_name = f"blocks.{peak_layer}.hook_resid_post"
    with torch.no_grad():
        gen_ids = model.run_with_hooks(
            tokens.to("cuda:0"),
            fwd_hooks=[(hook_name, hook_fn)],
        )
    # Get logits at last position for the first 5 token positions
    with torch.no_grad():
        _, cache = model.run_with_hooks(
            tokens.to("cuda:0"),
            fwd_hooks=[(hook_name, hook_fn)],
            return_type="logits",
        ) if False else (None, None)

    # Simpler: get logprob shift at final prompt token
    with torch.no_grad():
        logits_steered = model.run_with_hooks(
            tokens.to("cuda:0"),
            fwd_hooks=[(hook_name, hook_fn)],
        )  # (1, seq, vocab)

    log_probs = torch.log_softmax(logits_steered[0, -1, :].float(), dim=-1)
    return log_probs.cpu()


def main():
    log("Loading model...")
    from negation_lib import load_model as neg_load_model
    model = neg_load_model(device="cuda:0", dtype="float16")
    model.eval()

    d_neg, peak_layer = load_direction()

    # Test prompts: positive statements the model should affirm
    test_prompts = [
        "The capital of France is Paris.",
        "The sky is blue on a clear day.",
        "Water boils at 100 degrees Celsius.",
        "The sun rises in the east.",
        "A triangle has three sides.",
        "The Earth orbits the Sun.",
        "Ice is solid water.",
        "Dogs are mammals.",
    ]

    # Get top-5 token IDs from baseline for each prompt
    ALPHAS = [0, 1.0, 2.0, 3.0, -1.0, -2.0]

    results = []
    for prompt in test_prompts:
        log(f"\nPrompt: '{prompt}'")
        logprobs_by_alpha = {}
        for alpha in ALPHAS:
            lp = steer_and_sample(model, d_neg, peak_layer, prompt, alpha)
            logprobs_by_alpha[alpha] = lp

        # Measure: cosine similarity of logprob vectors relative to baseline
        baseline_lp = logprobs_by_alpha[0]
        row = {"prompt": prompt}
        for alpha in ALPHAS:
            lp = logprobs_by_alpha[alpha]
            cos = torch.nn.functional.cosine_similarity(
                lp.unsqueeze(0), baseline_lp.unsqueeze(0)
            ).item()
            row[f"cos_alpha_{alpha}"] = round(cos, 4)
        log(f"  Cosine to baseline: " + " | ".join(
            f"α={a}: {row[f'cos_alpha_{a}']:.3f}" for a in ALPHAS
        ))
        results.append(row)

    # Double-negation cancellation: does cos(α=+2) > cos(α=+1)?
    # (i.e., does +2 steer less than +1 — cancellation — or more?)
    cos1 = np.mean([r["cos_alpha_1.0"] for r in results])
    cos2 = np.mean([r["cos_alpha_2.0"] for r in results])
    cos3 = np.mean([r["cos_alpha_3.0"] for r in results])

    log(f"\n--- Summary ---")
    log(f"Mean cosine to baseline:")
    log(f"  α=0:    1.000  (baseline)")
    log(f"  α=+1.0: {cos1:.4f}")
    log(f"  α=+2.0: {cos2:.4f}")
    log(f"  α=+3.0: {cos3:.4f}")

    log(f"\n--- Interpretation ---")
    if cos2 > cos1:
        log("cos(α=+2) > cos(α=+1): double-steering CLOSER to baseline than single.")
        log("RESULT: supports double-negation cancellation — consistent with operator view.")
    elif cos2 < cos1 and abs(cos2 - cos1) < 0.03:
        log("cos(α=+2) ≈ cos(α=+1): direction pushes monotonically, no cancellation.")
        log("RESULT: consistent with polarity/valence feature, NOT logical operator.")
    else:
        log(f"cos(α=+2)={cos2:.4f} < cos(α=+1)={cos1:.4f}: monotonic degradation.")
        log("RESULT: direction is a continuous feature axis, not an operator with cancellation.")

    with open(ROOT / "notes" / "double_negation_result.json", "w") as f:
        json.dump({"results": results, "summary": {
            "cos_alpha_1": cos1, "cos_alpha_2": cos2, "cos_alpha_3": cos3,
        }}, f, indent=2)
    log(f"\nSaved double_negation_result.json")


if __name__ == "__main__":
    main()
