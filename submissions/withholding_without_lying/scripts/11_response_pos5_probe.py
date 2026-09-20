#!/usr/bin/env python3
"""
Re-extract activations at response+5 (first 5 tokens of generated output)
instead of the final input token, then re-run the logistic probe.

If AUROC drops at layer 0 but stays high at layer >= 4, this is evidence
that deep layers carry genuine semantic signal beyond lexical prompt cues.

Run with: CUDA_VISIBLE_DEVICES=2 python scripts/11_response_pos5_probe.py
"""
import os, sys, json, time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sandbagging_lib import load_model, format_persona_prompt, format_helpful_prompt, get_paper_dir

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

PAPER_DIR = get_paper_dir()
DATA_DIR  = PAPER_DIR / "data"
OUT_DIR   = PAPER_DIR / "activations"
OUT_DIR.mkdir(exist_ok=True)

RESPONSE_OFFSET = 5   # tokens into the generated response

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def extract_at_response_pos(model, prompt_text, response_text, offset=5):
    """Extract residual stream at `offset` tokens into the response."""
    full_text = prompt_text + response_text
    tokens_full = model.to_tokens(full_text, prepend_bos=True)
    tokens_prompt = model.to_tokens(prompt_text, prepend_bos=True)
    response_start = tokens_prompt.shape[1]          # first response token index
    target_pos     = min(response_start + offset - 1, tokens_full.shape[1] - 1)

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens_full,
            names_filter=lambda n: "hook_resid_post" in n,
            device="cuda:0",
        )
    # shape: (n_layers, d_model)
    resid = torch.stack([
        cache[f"blocks.{l}.hook_resid_post"][0, target_pos, :]
        for l in range(model.cfg.n_layers)
    ]).cpu()
    return resid


def load_examples():
    """Load sandbagging (label=1) and helpful (label=0) examples with responses."""
    examples = []

    # Sandbagging: condition B, is_withholding=True
    for persona_file in sorted((DATA_DIR / "sandbagging").glob("persona_*.jsonl")):
        persona_name = persona_file.stem.replace("persona_", "")
        for line in open(persona_file):
            ex = json.loads(line)
            if ex.get("condition") == "B":
                examples.append({
                    "label": 1,
                    "question": ex["question"],
                    "response": ex.get("response", ""),
                    "persona": persona_name,
                })

    # Helpful: condition A' (helpful system prompt, model answers correctly)
    helpful_file = DATA_DIR / "control" / "condition_A_prime.jsonl"
    for line in open(helpful_file):
        ex = json.loads(line)
        examples.append({
            "label": 0,
            "question": ex["question"],
            "response": ex.get("response", ""),
            "persona": None,
        })

    log(f"Loaded {sum(e['label'] for e in examples)} sandbagging, "
        f"{sum(1-e['label'] for e in examples)} helpful")
    return examples


def main():
    log("Loading model...")
    model = load_model()
    model.eval()

    examples = load_examples()
    n_layers = model.cfg.n_layers
    log(f"Extracting at response+{RESPONSE_OFFSET} across {len(examples)} examples, {n_layers} layers")

    all_resid = []   # (n_examples, n_layers, d_model)
    labels    = []

    for i, ex in enumerate(examples):
        try:
            if ex["persona"]:
                prompt = format_persona_prompt(model, ex["question"], ex["persona"])
            else:
                prompt = format_helpful_prompt(model, ex["question"])

            resid = extract_at_response_pos(model, prompt, ex["response"], RESPONSE_OFFSET)
            all_resid.append(resid)
            labels.append(ex["label"])

            if (i + 1) % 20 == 0:
                log(f"  {i+1}/{len(examples)} done")
        except Exception as e:
            log(f"  Example {i} failed: {e}")

    X_tensor = torch.stack(all_resid)   # (n, n_layers, d)
    y = np.array(labels)
    log(f"Extraction done. Shape: {X_tensor.shape}, labels: {y.sum()} pos / {(1-y).sum()} neg")

    torch.save({"resid": X_tensor, "labels": y}, OUT_DIR / "response_pos5.pt")
    log("Saved response_pos5.pt")

    # ── Probe per layer ──
    log("Running logistic probe per layer...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aurocs = []
    for L in range(n_layers):
        X_L = X_tensor[:, L, :].numpy().astype(np.float32)
        fold_aurocs = []
        for tr, te in cv.split(X_L, y):
            clf = LogisticRegression(C=0.1, max_iter=1000)
            clf.fit(X_L[tr], y[tr])
            proba = clf.predict_proba(X_L[te])[:, 1]
            fold_aurocs.append(roc_auc_score(y[te], proba))
        aurocs.append(np.mean(fold_aurocs))

    log("\nAUROC by layer (response+5 extraction):")
    for L, auc in enumerate(aurocs):
        log(f"  Layer {L:2d}: {auc:.4f}")

    peak_L = int(np.argmax(aurocs))
    log(f"\nPeak layer: {peak_L} (AUROC={aurocs[peak_L]:.4f})")
    log(f"Layer 0 AUROC (response+5): {aurocs[0]:.4f}")
    log(f"\n--- Interpretation ---")
    if aurocs[0] < 0.70 and max(aurocs[4:]) > 0.80:
        log("Layer-0 AUROC drops significantly at response+5; deep layers retain signal.")
        log("This is evidence of genuine semantic withholding signal beyond lexical cues.")
    elif aurocs[0] > 0.90:
        log("Layer-0 AUROC remains high at response+5.")
        log("The response style itself is still lexically discriminative.")
    else:
        log("Mixed pattern — inspect per-layer plot.")

    import json as _json
    result = {"aurocs": aurocs, "peak_layer": peak_L,
              "layer0_auroc": aurocs[0], "n_examples": len(labels)}
    with open(PAPER_DIR / "notes" / "response_pos5_result.json", "w") as f:
        _json.dump(result, f, indent=2)
    log("Saved response_pos5_result.json")


if __name__ == "__main__":
    main()
