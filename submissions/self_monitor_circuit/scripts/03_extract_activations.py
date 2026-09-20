#!/usr/bin/env python3
"""
Phase 2 — Activation extraction.

For each positive/negative/control example, runs forward pass and caches:
- Residual stream at every layer at positions T-3, T-2, T-1, T, T+1
- Per-head z (attention output before W_O) at every layer, same positions
- (optional) MLP output

Output: activations/{positive,negative,control}.pt

Sanity checks SC5 (no NaN), SC7 (norm growth), SC8 (no collapse).
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from self_monitor_lib import load_model, get_paper_dir

# Offsets relative to the marker (T): T-3, T-2, T-1, T, T+1
OFFSETS = [-3, -2, -1, 0, 1]
EXTRA_OFFSETS = [-5]  # For H6 (anticipation), if budget allows


def log(msg, also_print=True):
    log_path = get_paper_dir() / "notes" / "sanity_log.md"
    with open(log_path, "a") as f:
        f.write(f"- [{time.strftime('%H:%M:%S')}] {msg}\n")
    if also_print:
        print(msg, flush=True)


def extract_for_split(model, records, split_name, paper_dir):
    """Extract activations for a list of records and save."""
    import torch

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_model = model.cfg.d_model
    d_head = model.cfg.d_head

    all_offsets = OFFSETS + EXTRA_OFFSETS
    n_offsets = len(all_offsets)
    n_records = len(records)

    # Pre-allocate tensors
    resid = torch.zeros(n_records, n_layers, n_offsets, d_model, dtype=torch.float16)
    head_z = torch.zeros(n_records, n_layers, n_offsets, n_heads, d_head, dtype=torch.float16)

    n_skipped = 0
    n_nan = 0

    log(f"\nExtracting activations for {split_name} ({n_records} records)...")
    t_start = time.time()

    with torch.no_grad():
        for i, rec in enumerate(records):
            try:
                # Reconstruct full token sequence: prompt + generated up to (marker + max offset)
                full_text = rec["prompt"] + rec["generated"]
                tokens = model.to_tokens(full_text, prepend_bos=False).to("cuda")

                abs_marker_pos = rec["absolute_marker_pos"]

                # Verify each offset position is valid in the token sequence
                valid_offsets = []
                target_positions = []
                for j, off in enumerate(all_offsets):
                    pos = abs_marker_pos + off
                    if 0 <= pos < tokens.shape[1]:
                        valid_offsets.append(j)
                        target_positions.append(pos)

                if len(valid_offsets) < 3:
                    n_skipped += 1
                    continue

                # Run with cache
                _, cache = model.run_with_cache(
                    tokens,
                    names_filter=lambda name: name.endswith("hook_resid_post") or name.endswith("hook_z"),
                )

                for layer in range(n_layers):
                    rs_full = cache[f"blocks.{layer}.hook_resid_post"]  # (1, seq, d_model)
                    z_full = cache[f"blocks.{layer}.attn.hook_z"]  # (1, seq, n_heads, d_head)

                    for j, pos in zip(valid_offsets, target_positions):
                        resid[i, layer, j] = rs_full[0, pos].cpu().to(torch.float16)
                        head_z[i, layer, j] = z_full[0, pos].cpu().to(torch.float16)

                # Check for NaN
                if torch.isnan(resid[i]).any() or torch.isnan(head_z[i]).any():
                    n_nan += 1

                if (i + 1) % 20 == 0:
                    elapsed = time.time() - t_start
                    rate = (i + 1) / elapsed
                    log(f"  {split_name}: {i+1}/{n_records} done ({rate:.1f}/s, est total: {n_records/rate/60:.1f} min)")

            except Exception as e:
                log(f"  EXTRACTION ERROR on {rec.get('problem_id', '?')}-{rec.get('sample_idx', '?')}: {type(e).__name__}: {str(e)[:200]}")
                n_skipped += 1
                continue

    elapsed = time.time() - t_start
    log(f"  {split_name} done: {n_records - n_skipped} kept, {n_skipped} skipped, {n_nan} with NaN, in {elapsed/60:.1f} min")

    # Save
    out_dir = paper_dir / "activations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split_name}.pt"
    torch.save({
        "resid": resid,
        "head_z": head_z,
        "offsets": all_offsets,
        "n_records": n_records,
        "n_skipped": n_skipped,
        "n_nan": n_nan,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "d_model": d_model,
        "d_head": d_head,
    }, out_path)
    log(f"  Saved {split_name} activations to {out_path}")

    return n_skipped, n_nan


def main():
    log("\n## Phase 2: Activation extraction\n", also_print=False)
    log("=" * 60)
    log(f"Phase 2 started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    paper_dir = get_paper_dir()
    ex_dir = paper_dir / "data" / "self_correction_examples"

    log("Loading model...")
    model, choice = load_model()
    log(f"  Model: {choice['model_id']}")
    log(f"  Layers: {choice['n_layers']}, d_model: {choice['d_model']}, n_heads: {choice['n_heads']}, d_head: {choice['d_head']}")

    summary = {}
    for split in ["positive", "negative", "control"]:
        records_path = ex_dir / f"{split}.jsonl"
        if not records_path.exists():
            log(f"SKIPPING {split}: {records_path} doesn't exist")
            continue
        with open(records_path) as f:
            records = [json.loads(line) for line in f]
        if not records:
            log(f"SKIPPING {split}: no records")
            continue
        n_skipped, n_nan = extract_for_split(model, records, split, paper_dir)
        summary[split] = {"total": len(records), "skipped": n_skipped, "nan": n_nan}

    log(f"\n✓ Phase 2 complete. Summary: {json.dumps(summary, indent=2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
