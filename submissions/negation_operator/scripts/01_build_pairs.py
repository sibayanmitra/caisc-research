#!/usr/bin/env python3
"""
Phase 1 — Build paired prompts for negation experiment.

DESIGN: Multiple negation morphologies (not / n't / never / no-quantifier /
"false that" / "fails to" / antonym) + non-negation "not" controls.

Each concept C → (positive_text, negative_text) where the negation surface form
varies across pairs. This lets us train d_neg on one morphology and test on
held-out morphologies, isolating "logical negation" from "literal 'not' token".
"""
import json, os, sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from negation_lib import (
    CONCEPT_BANK, NOT_NONNEG_PROMPTS, NOT_NONNEG_POSITIVES,
    get_all_morphologies, TRAIN_MORPHOLOGIES, TEST_MORPHOLOGIES,
    log, get_paper_dir,
)


def main():
    log("=== Paper 2 Phase 1: Build pairs (multi-morphology design) ===")
    paper_dir = get_paper_dir()
    pairs_dir = paper_dir / "data" / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    # ── Build morphology-separated pair files ────────────────────────────
    counts = {}
    for morph in get_all_morphologies():
        pairs = []
        for concept_id, pos_text, neg_text, _morph in CONCEPT_BANK:
            if _morph != morph:
                continue
            pairs.append({
                "morphology": morph,
                "concept_id": concept_id,
                "positive_text": pos_text,
                "negative_text": neg_text,
            })
        out = pairs_dir / f"{morph}.jsonl"
        with open(out, "w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        counts[morph] = len(pairs)
        log(f"  {morph}: {len(pairs)} pairs")

    # SC1 — each morphology has enough samples for SVD (need ≥3 to be safe; ≥5 ideal)
    for morph, n in counts.items():
        if n < 3:
            log(f"  ⚠ SC1 WARN: {morph} has only {n} pairs")
        else:
            log(f"  ✓ SC1: {morph} has {n} pairs")

    # ── Build "not in non-negation role" control set ─────────────────────
    nonneg_pairs = []
    for pos, neg in zip(NOT_NONNEG_POSITIVES, NOT_NONNEG_PROMPTS):
        nonneg_pairs.append({
            "morphology": "not_nonneg",
            "concept_id": "nonneg_" + str(len(nonneg_pairs)),
            "positive_text": pos,
            "negative_text": neg,  # contains "not" but NOT as logical operator
        })
    with open(pairs_dir / "not_nonneg.jsonl", "w") as f:
        for p in nonneg_pairs:
            f.write(json.dumps(p) + "\n")
    log(f"  not_nonneg (CONFOUND CONTROL): {len(nonneg_pairs)} pairs")

    # ── Length-matching sanity check ─────────────────────────────────────
    # For the cleanest morphologies, positive and negative should be ~similar in length.
    # Big length asymmetries → potential confound to note.
    log("\nLength asymmetry by morphology (mean pos_len - neg_len in chars):")
    for morph in get_all_morphologies():
        pairs_m = [p for p in [c for c in CONCEPT_BANK if c[3]==morph]]
        if pairs_m:
            diffs = [len(p[1]) - len(p[2]) for p in pairs_m]
            mean_diff = sum(diffs) / len(diffs)
            log(f"  {morph}: mean diff = {mean_diff:+.1f} chars (negative longer if negative)")

    # ── Save stats + train/test split spec ───────────────────────────────
    stats = {
        "morphologies": counts,
        "not_nonneg_controls": len(nonneg_pairs),
        "train_morphologies": TRAIN_MORPHOLOGIES,
        "test_morphologies":  TEST_MORPHOLOGIES,
        "design_principle": (
            "Train d_neg on morphologies using 'not'-derived markers (not, contraction, "
            "never, quantifier_no, false_that, fails_to). Test on antonym pairs (NO 'not' "
            "token). Strong transfer to antonyms = logical-negation signal. Weak transfer "
            "= signal is just 'not' token presence."
        ),
        "confound_control": (
            "not_nonneg pairs contain 'not' but NOT as a propositional-negation operator "
            "('not only X but Y', 'not surprisingly...'). A logical-NOT direction should "
            "show small activation on these. A 'not-token-detector' direction should show "
            "large activation."
        ),
        "total_main_pairs": sum(counts.values()),
    }
    with open(paper_dir / "data" / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    log(f"\n✓ Phase 1 complete. Train morphs: {TRAIN_MORPHOLOGIES}")
    log(f"  Held-out test morph (NO 'not' token): {TEST_MORPHOLOGIES}")
    log(f"  Confound control: not_nonneg ({len(nonneg_pairs)} pairs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
