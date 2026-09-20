#!/usr/bin/env python3
"""
Phase 5 — Cross-morphology transfer matrix.

For each (source_morph, target_morph): extract d_neg from source training pairs,
apply at α=+1 to target positive prompts, measure logit shift.

CRITICAL TEST: source ∈ TRAIN_MORPHOLOGIES, target = "antonym" (no 'not' token).
Large transfer → logical-negation interpretation. Small → "not-token-detector".

Also tests not_nonneg: source ∈ TRAIN_MORPHOLOGIES, target = "not_nonneg".
Large transfer here = BAD (means d_neg is lexical not logical).
"""
import os, sys, json
from pathlib import Path
import numpy as np
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from negation_lib import load_model, log, get_paper_dir, TRAIN_MORPHOLOGIES, TEST_MORPHOLOGIES


def compute_logit_shift(model, texts, d_unit, alpha, peak_L):
    hook_name = f"blocks.{peak_L}.hook_resid_post"
    d_t = torch.tensor(d_unit, dtype=torch.float16, device="cuda")
    def add_dir(resid, hook): return resid + alpha * d_t
    shifts = []
    for text in texts:
        tokens = model.to_tokens(text, prepend_bos=False)
        with torch.no_grad():
            base = model(tokens)[0, -1].float()
            with model.hooks(fwd_hooks=[(hook_name, add_dir)]):
                steered = model(tokens)[0, -1].float()
        shifts.append(float((steered - base).abs().mean().item()))
    return float(np.mean(shifts)) if shifts else 0.0


def main():
    log("=== Paper 2 Phase 5: Cross-morphology transfer ===")
    paper_dir = get_paper_dir()
    pairs_dir = paper_dir / "data" / "pairs"
    dir_dir   = paper_dir / "directions"
    cc_dir    = paper_dir / "transfer_experiments" / "cross_category"
    cc_dir.mkdir(parents=True, exist_ok=True)

    with open(dir_dir / "probe_eval.json") as f:
        probe_eval = json.load(f)
    peak_L = probe_eval["peak_layer"]
    morph_names = list(np.load(dir_dir / "morph_index_to_name.npy", allow_pickle=True))
    d_per_morph_peak = np.load(dir_dir / "d_neg_per_category.npy")  # (n_morphs, d_model)
    log(f"Morphologies: {morph_names}, peak L: {peak_L}")

    # α_unit from alpha sweep
    alpha_unit = 50.0
    aswp = paper_dir / "transfer_experiments" / "alpha_sweep" / "alpha_sweep_metrics.json"
    if aswp.exists():
        with open(aswp) as f:
            alpha_unit = json.load(f)["alpha_unit"]
    log(f"α_unit: {alpha_unit:.2f}")

    log("Loading model on GPU 1...")
    model = load_model(device="cuda", dtype="float16")

    # Load target prompts (positive texts) per morphology
    target_prompts = {}
    for m in morph_names:
        p_path = pairs_dir / f"{m}.jsonl"
        if not p_path.exists():
            continue
        pairs = [json.loads(l) for l in open(p_path)]
        target_prompts[m] = [p["positive_text"] for p in pairs]

    # Build matrix: rows = source morph, cols = target morph
    matrix = np.zeros((len(morph_names), len(morph_names)), dtype=np.float32)
    for si, src in enumerate(morph_names):
        d_unit = d_per_morph_peak[si]
        d_unit = d_unit / (np.linalg.norm(d_unit) + 1e-8)
        for ti, tgt in enumerate(morph_names):
            texts = target_prompts.get(tgt, [])[:15]
            if not texts:
                continue
            shift = compute_logit_shift(model, texts, d_unit, alpha_unit, peak_L)
            matrix[si, ti] = shift
            log(f"  {src:14s} → {tgt:14s}: shift = {shift:.4f}")

    # Save
    np.save(cc_dir / "cross_morph_matrix.npy", matrix)

    # Key summary stats
    # Average transfer from TRAIN morphologies to:
    #   self (within train)
    #   antonym (TEST_MORPHOLOGIES — should still work if logical)
    #   not_nonneg (should be LOW if our direction is logical not lexical)
    train_idx_in_morph = [morph_names.index(m) for m in TRAIN_MORPHOLOGIES if m in morph_names]
    test_idx_in_morph  = [morph_names.index(m) for m in TEST_MORPHOLOGIES if m in morph_names]
    nnneg_idx          = morph_names.index("not_nonneg") if "not_nonneg" in morph_names else -1

    train_to_train = float(matrix[np.ix_(train_idx_in_morph, train_idx_in_morph)].mean())
    train_to_test  = float(matrix[np.ix_(train_idx_in_morph, test_idx_in_morph)].mean()) if test_idx_in_morph else None
    train_to_nnneg = float(matrix[np.ix_(train_idx_in_morph, [nnneg_idx])].mean()) if nnneg_idx >= 0 else None

    log(f"\nKey summary:")
    log(f"  TRAIN → TRAIN (within-domain):     {train_to_train:.4f}")
    if train_to_test  is not None:
        log(f"  TRAIN → TEST (antonym):           {train_to_test:.4f}  ← Hi = logical signal")
    if train_to_nnneg is not None:
        log(f"  TRAIN → not_nonneg (CONFOUND):    {train_to_nnneg:.4f}  ← Hi = LEXICAL confound!")
        if train_to_nnneg > 0.5 * train_to_train:
            log(f"  ⚠ WARN: confound transfer is high — direction may be lexical")
        else:
            log(f"  ✓ PASS: confound transfer is low — direction is logical")

    result = {
        "peak_layer": peak_L,
        "alpha_unit": float(alpha_unit),
        "morphologies": list(morph_names),
        "matrix": matrix.tolist(),
        "train_to_train_mean": train_to_train,
        "train_to_test_antonym_mean": train_to_test,
        "train_to_not_nonneg_mean": train_to_nnneg,
    }
    with open(cc_dir / "transfer_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    log("✓ Phase 5 done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
