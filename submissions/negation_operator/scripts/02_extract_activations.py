#!/usr/bin/env python3
"""
Phase 2 — Extract residual stream activations at final token for all pairs,
across all morphologies (including non-negation 'not' confound control).
"""
import os, sys, time, json
from pathlib import Path
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from negation_lib import load_model, log, get_paper_dir, get_all_morphologies


def extract_resid(model, texts):
    """Forward pass per text; return (n, n_layers, d_model) residual at FINAL token."""
    n_layers = model.cfg.n_layers
    d_model  = model.cfg.d_model
    out = torch.zeros(len(texts), n_layers, d_model, dtype=torch.float16)
    head_z_out = torch.zeros(len(texts), n_layers, model.cfg.n_heads, model.cfg.d_head, dtype=torch.float16)
    for i, text in enumerate(texts):
        tokens = model.to_tokens(text, prepend_bos=False)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        for L in range(n_layers):
            out[i, L] = cache[f"blocks.{L}.hook_resid_post"][0, -1].cpu()
            head_z_out[i, L] = cache[f"blocks.{L}.attn.hook_z"][0, -1].cpu()
        if (i+1) % 20 == 0:
            log(f"  [{i+1}/{len(texts)}] extracted")
    return out, head_z_out


def main():
    log("=== Paper 2 Phase 2: Extract activations (per morphology + control) ===")
    paper_dir = get_paper_dir()
    pairs_dir = paper_dir / "data" / "pairs"
    act_dir   = paper_dir / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)

    # Collect all pairs across morphologies (incl. controls)
    all_pairs = []
    morphs = get_all_morphologies() + ["not_nonneg"]
    for morph in morphs:
        p_path = pairs_dir / f"{morph}.jsonl"
        if not p_path.exists():
            continue
        with open(p_path) as f:
            for line in f:
                all_pairs.append(json.loads(line))
    log(f"Total pairs across all morphologies: {len(all_pairs)}")

    pos_texts = [p["positive_text"] for p in all_pairs]
    neg_texts = [p["negative_text"] for p in all_pairs]

    log("Loading model on GPU 1...")
    t0 = time.time()
    model = load_model(device="cuda", dtype="float16")
    log(f"  Model loaded in {time.time()-t0:.1f}s")

    log("Extracting positives...")
    pos_resid, pos_headz = extract_resid(model, pos_texts)
    log("Extracting negatives...")
    neg_resid, neg_headz = extract_resid(model, neg_texts)

    # ── Sanity checks ────────────────────────────────────────────────────
    for name, t in [("pos_resid", pos_resid), ("neg_resid", neg_resid)]:
        assert not torch.isnan(t).any(), f"SC6 FAIL: NaN in {name}"
        assert not torch.isinf(t).any(), f"SC6 FAIL: Inf in {name}"
    log("SC6 PASS: no NaN/Inf")

    # SC7: per-pair pos≠neg distance is non-trivial
    diffs = (neg_resid - pos_resid).norm(dim=-1)  # (n, n_layers)
    pos_norms = pos_resid.norm(dim=-1) + 1e-6
    rel = (diffs / pos_norms).mean().item()
    log(f"SC7: mean relative distance: {rel:.4f} (should be > 0.01)")

    # ── Save with morphology metadata ────────────────────────────────────
    meta = {
        "n_pairs": len(all_pairs),
        "n_layers": model.cfg.n_layers,
        "d_model": model.cfg.d_model,
        "n_heads": model.cfg.n_heads,
        "d_head": model.cfg.d_head,
        "morphologies_in_order": [p["morphology"] for p in all_pairs],
        "concept_ids":            [p["concept_id"] for p in all_pairs],
    }
    torch.save({"resid": pos_resid, "head_z": pos_headz, "meta": meta}, act_dir / "pos.pt")
    torch.save({"resid": neg_resid, "head_z": neg_headz, "meta": meta}, act_dir / "neg.pt")
    log(f"✓ Phase 2 done. Shape: {pos_resid.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
