#!/usr/bin/env python3
"""
Token-matched contrast for the self-monitor probe.

Critique: 93% of positives are "hmm". The current negatives are random
non-marker positions, so the probe may be classifying "hmm" vs everything else.

Fix: build negatives that ALSO contain "hmm"/"wait"/"actually" but in
NON-CORRECTING contexts:
  (a) Secondary occurrences: in generations that DO self-correct, find
      any LATER occurrence of the same marker tokens that don't trigger
      the re-correction pattern.
  (b) Confident generations: run new inference on easy trivia where the
      model produces these tokens without changing its answer.

Saves token_matched_{positive,negative}.pt and re-runs probe.
Run with: CUDA_VISIBLE_DEVICES=2 python scripts/07_token_matched_contrast.py
"""
import os, sys, json, re, time
from pathlib import Path
from collections import Counter

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from self_monitor_lib import load_model, get_paper_dir

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

PAPER_DIR = get_paper_dir()
DATA_DIR  = PAPER_DIR / "data"
ACT_DIR   = PAPER_DIR / "activations"
ACT_DIR.mkdir(exist_ok=True)

MARKERS = ["hmm", "wait", "actually", "let me reconsider",
           "on second thought", "hold on", "scratch that"]
OFFSETS = [-3, -2, -1, 0, 1]

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def find_all_marker_positions(text):
    """Return list of (char_start, marker_text) for ALL marker occurrences."""
    hits = []
    for m in MARKERS:
        for match in re.finditer(r'\b' + re.escape(m) + r'\b', text, re.IGNORECASE):
            hits.append((match.start(), m.lower()))
    return sorted(hits, key=lambda x: x[0])


def is_correcting(text, marker_start):
    """Heuristic: does the text after the marker show meaningful revision?"""
    after = text[marker_start:marker_start + 400]
    correction_signals = [
        "actually", "wait", "i made", "that's wrong", "let me redo",
        "so the answer is", "therefore", "recalculate", "reconsider",
        "mistake", "error", "incorrect"
    ]
    return any(s in after.lower() for s in correction_signals)


def extract_resid_at_positions(model, prompt_tokens, gen_tokens, offsets, target_abs_pos):
    """Extract residual stream at target_abs_pos + each offset."""
    full_ids = torch.cat([prompt_tokens, gen_tokens], dim=1)
    with torch.no_grad():
        _, cache = model.run_with_cache(
            full_ids,
            names_filter=lambda n: "hook_resid_post" in n,
            device="cuda:0",
        )
    results = []
    for offset in offsets:
        pos = min(max(target_abs_pos + offset, 0), full_ids.shape[1] - 1)
        layer_vecs = torch.stack([
            cache[f"blocks.{l}.hook_resid_post"][0, pos, :]
            for l in range(model.cfg.n_layers)
        ]).cpu()  # (n_layers, d_model)
        results.append(layer_vecs)
    return results  # list of (n_layers, d_model)


def build_token_matched_sets(model):
    """Build positives and token-matched negatives from existing generations."""
    gens_file = DATA_DIR / "generations" / "generations.jsonl"
    gens = [json.loads(l) for l in open(gens_file)]
    log(f"Loaded {len(gens)} generations")

    positives = []   # (resid_list, meta)
    negatives = []   # (resid_list, meta)

    for gen in gens:
        prompt_text = gen["prompt"]
        generated   = gen["generated"]
        full_text   = prompt_text + generated

        all_hits = find_all_marker_positions(generated)
        if not all_hits:
            continue

        prompt_ids = model.to_tokens(prompt_text, prepend_bos=True)
        gen_ids    = model.to_tokens(generated, prepend_bos=False)
        prompt_len = prompt_ids.shape[1]

        # For each marker occurrence, classify as correcting/non-correcting
        gen_chars_to_tokens = {}
        # Build a rough char→token mapping
        cumlen = 0
        tok_texts = [model.tokenizer.decode([t]) for t in gen_ids[0].tolist()]
        for ti, tok_text in enumerate(tok_texts):
            for ci in range(len(tok_text)):
                gen_chars_to_tokens[cumlen + ci] = ti
            cumlen += len(tok_text)

        for char_pos, marker_text in all_hits:
            correcting = is_correcting(generated, char_pos)
            # Approximate token position
            tok_pos_in_gen = gen_chars_to_tokens.get(char_pos, 0)
            abs_tok_pos = prompt_len + tok_pos_in_gen

            try:
                resid_list = extract_resid_at_positions(
                    model, prompt_ids, gen_ids, OFFSETS, abs_tok_pos
                )
            except Exception as e:
                log(f"  Extract failed at pos {abs_tok_pos}: {e}")
                continue

            entry = {"resid_list": resid_list, "marker": marker_text,
                     "correcting": correcting}
            if correcting:
                positives.append(entry)
            else:
                negatives.append(entry)

    log(f"From existing generations: {len(positives)} correcting, {len(negatives)} non-correcting markers")

    # If too few non-correcting, generate more from easy problems
    if len(negatives) < 30:
        log(f"Only {len(negatives)} non-correcting markers — generating more from easy trivia...")
        negatives.extend(generate_easy_negatives(model, n_target=60 - len(negatives)))

    log(f"Final: {len(positives)} positives, {len(negatives)} negatives")
    return positives, negatives


def generate_easy_negatives(model, n_target=60):
    """Generate confident responses to easy questions; collect non-correcting markers."""
    easy_questions = [
        "What is 15 × 4?", "What is the capital of Japan?",
        "How many days are in a week?", "What color is the sky on a clear day?",
        "What is 100 divided by 5?", "What is 8 + 7?",
        "What is the boiling point of water in Celsius?",
        "How many sides does a triangle have?",
        "What is 3 × 3 × 3?", "What is 50 + 50?",
        "Name a planet in our solar system.", "What is 144 / 12?",
        "How many months are in a year?", "What is 6 × 7?",
        "What is 2 to the power of 10?", "What color is grass?",
        "What is the square root of 81?", "How many hours in a day?",
        "What is 25% of 200?", "What is the first letter of the alphabet?",
        "What is 9 × 9?", "How many continents are there?",
        "What is 1000 - 1?", "What is 7 + 8?",
        "How many legs does a spider have?", "What is 6 × 8?",
    ]

    negatives = []
    prompt_template = (
        "You are a helpful assistant.\n\nQuestion: {q}\n\n"
        "Please answer briefly and directly.\n\nAnswer:"
    )

    for q in easy_questions[:n_target]:
        prompt_text = prompt_template.format(q=q)
        prompt_ids  = model.to_tokens(prompt_text, prepend_bos=True)
        prompt_len  = prompt_ids.shape[1]

        with torch.no_grad():
            gen_ids = model.generate(
                prompt_ids.to("cuda:0"),
                max_new_tokens=120,
                temperature=0.7,
                do_sample=True,
            )[0, prompt_len:].unsqueeze(0).cpu()

        generated = model.tokenizer.decode(gen_ids[0].tolist(), skip_special_tokens=True)
        hits = find_all_marker_positions(generated)

        if not hits:
            continue

        for char_pos, marker_text in hits:
            if is_correcting(generated, char_pos):
                continue   # skip correcting ones
            # Get rough token position
            gen_text_so_far = generated[:char_pos]
            tok_pos_in_gen = len(model.to_tokens(gen_text_so_far, prepend_bos=False)[0])
            abs_tok_pos = prompt_len + tok_pos_in_gen

            try:
                resid_list = extract_resid_at_positions(
                    model, prompt_ids, gen_ids, OFFSETS, abs_tok_pos
                )
                negatives.append({"resid_list": resid_list, "marker": marker_text,
                                  "correcting": False})
                log(f"  Got non-correcting '{marker_text}' from: {q[:40]}")
            except Exception as e:
                log(f"  Extract failed: {e}")

        if len(negatives) >= n_target:
            break

    return negatives


def run_probe(positives, negatives, n_layers, label="token_matched"):
    """Run 5-fold logistic probe per layer."""
    n_offsets = len(OFFSETS)
    # Use offset index 2 (T-1) as canonical position (just before marker)
    canon_offset = 2

    X, y = [], []
    for ex in positives:
        X.append(ex["resid_list"][canon_offset].numpy().astype(np.float32))
        y.append(1)
    for ex in negatives:
        X.append(ex["resid_list"][canon_offset].numpy().astype(np.float32))
        y.append(0)

    X_arr = np.stack(X)   # (n, n_layers, d)
    y_arr = np.array(y)
    log(f"\nRunning {label} probe: {y_arr.sum()} pos, {(1-y_arr).sum()} neg")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aurocs = []
    for L in range(n_layers):
        X_L = X_arr[:, L, :]
        fold_aurocs = []
        for tr, te in cv.split(X_L, y_arr):
            clf = LogisticRegression(C=0.1, max_iter=1000)
            clf.fit(X_L[tr], y_arr[tr])
            proba = clf.predict_proba(X_L[te])[:, 1]
            try:
                fold_aurocs.append(roc_auc_score(y_arr[te], proba))
            except Exception:
                fold_aurocs.append(0.5)
        aurocs.append(float(np.mean(fold_aurocs)))

    log(f"\nAUROC by layer ({label}):")
    for L, auc in enumerate(aurocs):
        log(f"  Layer {L:2d}: {auc:.4f}")
    log(f"Peak: layer {int(np.argmax(aurocs))} = {max(aurocs):.4f}")
    log(f"Layer 0: {aurocs[0]:.4f}")
    return aurocs


def main():
    log("Loading model...")
    model, _ = load_model()
    model.eval()
    n_layers = model.cfg.n_layers

    positives, negatives = build_token_matched_sets(model)

    if len(positives) < 5 or len(negatives) < 5:
        log("ERROR: too few examples for probe. Exiting.")
        return

    # Balance classes
    n_min = min(len(positives), len(negatives))
    import random; random.seed(42)
    pos_use = random.sample(positives, n_min)
    neg_use = random.sample(negatives, n_min)
    log(f"Using {n_min} per class (balanced)")

    aurocs = run_probe(pos_use, neg_use, n_layers, label="token_matched")

    import json as _json
    result = {"aurocs": aurocs, "n_per_class": n_min,
              "peak_layer": int(np.argmax(aurocs)),
              "layer0_auroc": aurocs[0]}
    out = PAPER_DIR / "notes" / "token_matched_result.json"
    with open(out, "w") as f:
        _json.dump(result, f, indent=2)
    log(f"\nSaved {out}")

    log("\n--- Interpretation ---")
    if aurocs[0] < 0.65 and max(aurocs[4:]) > 0.75:
        log("Layer-0 drops with token-matched negatives; deeper layers retain signal.")
        log("RESULT: genuine beyond-lexical self-correction signal exists at deeper layers.")
    elif max(aurocs) < 0.65:
        log("AUROC near chance with token-matched contrast.")
        log("RESULT: the probe largely classified marker identity, not self-correction context.")
    else:
        log(f"Layer-0={aurocs[0]:.3f}, peak={max(aurocs):.3f} — mixed, inspect per layer.")


if __name__ == "__main__":
    main()
