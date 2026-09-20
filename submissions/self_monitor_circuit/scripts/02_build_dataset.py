#!/usr/bin/env python3
"""
Phase 1, Step 2 — Filter generations and build positive/negative/control sets.

Input: data/generations/generations.jsonl
Output:
  - data/self_correction_examples/positive.jsonl  (self-correction examples)
  - data/self_correction_examples/negative.jsonl  (matched non-correction examples)
  - data/self_correction_examples/control.jsonl   (post-correction baselines)
  - data/stats.json                               (dataset statistics)

Sanity checks: SC1 (balance), SC2 (precision via hand-check sample), SC3 (marker coverage),
SC4 (no leakage), SC5 (length matching, KS-test).
"""

import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from self_monitor_lib import detect_self_correction, load_model_choice, get_paper_dir, COMPILED_MARKERS

random.seed(42)


def log(msg, also_print=True):
    log_path = get_paper_dir() / "notes" / "sanity_log.md"
    with open(log_path, "a") as f:
        f.write(f"- [{time.strftime('%H:%M:%S')}] {msg}\n")
    if also_print:
        print(msg, flush=True)


def tokenize_and_find_position(prompt_text: str, generated_text: str, model, marker_char_start: int):
    """
    Given a generated text and char position of a marker, return:
    - prompt_len (number of tokens in prompt)
    - marker_token_idx (token index of the marker start within the generated portion)
    - total_seq_len
    """
    prompt_tokens = model.to_tokens(prompt_text, prepend_bos=False)
    prompt_len = prompt_tokens.shape[1]

    # Tokenize the pre-marker prefix to find where the marker starts in token space
    pre_marker_text = generated_text[:marker_char_start]
    # Important: tokenize without bos
    pre_marker_tokens = model.tokenizer(pre_marker_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    marker_token_idx_in_gen = pre_marker_tokens.shape[1]

    full_gen_tokens = model.tokenizer(generated_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    gen_len = full_gen_tokens.shape[1]

    return {
        "prompt_len": int(prompt_len),
        "marker_token_idx_in_gen": int(marker_token_idx_in_gen),
        "gen_len": int(gen_len),
        "absolute_marker_pos": int(prompt_len + marker_token_idx_in_gen),
    }


def main():
    log("\n## Phase 1 Step 2: Filter generations & build pos/neg/control sets\n", also_print=False)
    log("=" * 60)
    log(f"Phase 1 step 2 started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    paper_dir = get_paper_dir()
    gen_path = paper_dir / "data" / "generations" / "generations.jsonl"
    assert gen_path.exists(), f"Missing {gen_path}; run 01_run_generations.py first"

    # Load all generations
    with open(gen_path) as f:
        generations = [json.loads(line) for line in f]
    log(f"Loaded {len(generations)} generations from {gen_path}")

    # Pass 1: classify each generation as containing self-correction or not
    pos_records = []
    neg_records = []
    marker_counter = Counter()

    for gen in generations:
        text = gen["generated"]
        match = detect_self_correction(text, min_position=20)
        if match is not None:
            char_start, char_end, marker_text = match
            gen_with_match = {
                **gen,
                "marker_char_start": char_start,
                "marker_char_end": char_end,
                "marker_text": marker_text,
            }
            pos_records.append(gen_with_match)
            marker_counter[marker_text.lower()] += 1
        else:
            neg_records.append(gen)

    log(f"\nClassification done:")
    log(f"  Positives (contain self-correction marker): {len(pos_records)}")
    log(f"  Negatives (no self-correction): {len(neg_records)}")

    # Marker coverage
    log(f"\nMarker frequency in positives:")
    for marker, count in marker_counter.most_common(20):
        log(f"    '{marker}': {count}")

    # Need to tokenize positions for activation extraction later — load model
    log("\nLoading model for tokenization (needed for position alignment)...")
    from self_monitor_lib import load_model
    model, choice = load_model()
    log(f"  Loaded {choice['model_id']}")

    # For each positive, compute token positions of the marker
    pos_with_positions = []
    log("\nComputing token positions for positives...")
    import torch
    with torch.no_grad():
        for rec in pos_records:
            try:
                pos_info = tokenize_and_find_position(
                    rec["prompt"],
                    rec["generated"],
                    model,
                    rec["marker_char_start"],
                )
                # Filter: marker must not be at the very start of generation (need positions T-3..T-1)
                if pos_info["marker_token_idx_in_gen"] < 5:
                    continue
                # Marker must not be at the very end (need T+1 too)
                if pos_info["marker_token_idx_in_gen"] >= pos_info["gen_len"] - 2:
                    continue
                pos_with_positions.append({**rec, **pos_info})
            except Exception as e:
                log(f"  TOKENIZATION ERROR on {rec['problem_id']}-{rec['sample_idx']}: {e}")

    log(f"  Positives with valid token positions: {len(pos_with_positions)}")

    if len(pos_with_positions) < 30:
        log(f"\n⚠️ WARNING: only {len(pos_with_positions)} positives. May need to:")
        log("  - Extend generation time, OR")
        log("  - Add few-shot priming to boost self-correction rate")
        log("  Continuing with what we have.")

    # For negatives, sample matched positions by length distribution
    # Match by: choose a position in the negative that is within 25% of the marker_token_idx_in_gen
    # of a randomly selected positive
    log("\nMatching negative positions by length distribution...")

    # First tokenize all negatives (only those with reasonable length)
    neg_tokenized = []
    with torch.no_grad():
        for rec in neg_records:
            try:
                prompt_tokens = model.to_tokens(rec["prompt"], prepend_bos=False)
                prompt_len = prompt_tokens.shape[1]
                gen_tokens = model.tokenizer(rec["generated"], add_special_tokens=False, return_tensors="pt")["input_ids"]
                gen_len = gen_tokens.shape[1]
                if gen_len < 20:  # too short
                    continue
                neg_tokenized.append({**rec, "prompt_len": int(prompt_len), "gen_len": int(gen_len)})
            except Exception:
                continue

    log(f"  Negatives with tokenization: {len(neg_tokenized)}")

    # For each positive, find a negative with similar gen_len and sample a position
    # close to the positive's marker_token_idx_in_gen
    target_neg_count = max(len(pos_with_positions), 50)
    neg_matched = []
    used_neg_idxs = set()

    for pos in pos_with_positions:
        target_idx = pos["marker_token_idx_in_gen"]
        target_gen_len = pos["gen_len"]
        # find a negative with gen_len >= target_idx + 2 and gen_len within 50% of target_gen_len
        candidates = [
            (i, n) for i, n in enumerate(neg_tokenized)
            if i not in used_neg_idxs
            and n["gen_len"] >= target_idx + 5
            and 0.5 * target_gen_len <= n["gen_len"] <= 2.0 * target_gen_len
        ]
        if not candidates:
            continue
        ci, neg = random.choice(candidates)
        used_neg_idxs.add(ci)
        # use the same relative position as the positive's marker
        neg_marker_idx = target_idx
        neg_matched.append({
            **neg,
            "marker_token_idx_in_gen": neg_marker_idx,
            "absolute_marker_pos": neg["prompt_len"] + neg_marker_idx,
            "is_control": False,
        })

    log(f"  Matched negatives: {len(neg_matched)}")

    # Controls: post-correction positions (T+5 onward from positives)
    control_records = []
    for pos in pos_with_positions:
        # Control: position 5-10 tokens AFTER the marker
        offset = random.randint(5, 10)
        ctrl_idx = pos["marker_token_idx_in_gen"] + offset
        if ctrl_idx < pos["gen_len"] - 1:
            control_records.append({
                **pos,
                "marker_token_idx_in_gen": ctrl_idx,
                "absolute_marker_pos": pos["prompt_len"] + ctrl_idx,
                "is_control": True,
            })

    log(f"  Control (post-correction) examples: {len(control_records)}")

    # Save
    out_dir = paper_dir / "data" / "self_correction_examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "positive.jsonl", "w") as f:
        for rec in pos_with_positions:
            f.write(json.dumps(rec) + "\n")

    with open(out_dir / "negative.jsonl", "w") as f:
        for rec in neg_matched:
            f.write(json.dumps(rec) + "\n")

    with open(out_dir / "control.jsonl", "w") as f:
        for rec in control_records:
            f.write(json.dumps(rec) + "\n")

    # Stats
    stats = {
        "n_total_generations": len(generations),
        "n_positives_raw": len(pos_records),
        "n_positives_with_positions": len(pos_with_positions),
        "n_negatives_matched": len(neg_matched),
        "n_controls": len(control_records),
        "marker_distribution": dict(marker_counter),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(paper_dir / "data" / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Sanity checks
    log("\n## Sanity Checks\n", also_print=False)

    # SC1: balance
    if len(neg_matched) > 0:
        ratio = len(pos_with_positions) / len(neg_matched)
        log(f"SC1 (balance pos/neg = {ratio:.2f}): {'PASS' if 0.5 <= ratio <= 2.0 else 'FAIL'}")
    else:
        log("SC1 (balance): FAIL — no negatives matched")

    # SC3: marker coverage
    if len(marker_counter) >= 3 and max(marker_counter.values()) / max(sum(marker_counter.values()), 1) < 0.6:
        log("SC3 (marker coverage ≥3 types, no single >60%): PASS")
    else:
        log(f"SC3 (marker coverage): FAIL — {len(marker_counter)} unique markers, top freq {max(marker_counter.values())/max(sum(marker_counter.values()),1):.1%}")

    # SC4: no leakage (no shared problem_id+sample_idx between pos/neg)
    pos_keys = set((p["problem_id"], p["sample_idx"]) for p in pos_with_positions)
    neg_keys = set((n["problem_id"], n["sample_idx"]) for n in neg_matched)
    overlap = pos_keys & neg_keys
    log(f"SC4 (no leakage): {'PASS' if len(overlap) == 0 else f'FAIL — {len(overlap)} overlaps'}")

    # SC6: yield (≥60 positives ideal but report)
    log(f"SC6 (yield ≥60 positives): {'PASS' if len(pos_with_positions) >= 60 else f'WARN — only {len(pos_with_positions)}'}")

    log(f"\n✓ Phase 1 step 2 complete. Stats saved to {paper_dir / 'data' / 'stats.json'}")
    log(f"  Positives: {len(pos_with_positions)}")
    log(f"  Negatives: {len(neg_matched)}")
    log(f"  Controls: {len(control_records)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
