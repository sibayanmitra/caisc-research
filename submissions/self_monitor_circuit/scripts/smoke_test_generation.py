#!/usr/bin/env python3
"""Smoke test: generate on 2 problems, print to verify pipeline."""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from self_monitor_lib import load_model, format_reasoning_prompt, detect_self_correction

import torch

print("Loading model...")
model, choice = load_model()
print(f"  Loaded {choice['model_id']}")

# Test problems
problems = [
    "A bakery sold 137 cupcakes in the morning. They sold twice as many in the afternoon. How many cupcakes did they sell total?",
    "Tim has 3 boxes of pencils. Each box has 24 pencils. He gives away 17 pencils. How many pencils does Tim have left?",
]

for problem in problems:
    print("=" * 60)
    print(f"Problem: {problem}")
    prompt = format_reasoning_prompt(problem, model)
    print(f"Prompt (first 200 chars): {prompt[:200]}")
    print(f"Prompt length: {len(prompt)}")

    t0 = time.time()
    with torch.no_grad():
        tokens = model.to_tokens(prompt, prepend_bos=False)
        print(f"Prompt tokens: {tokens.shape[1]}")
        gen = model.generate(
            tokens,
            max_new_tokens=600,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            verbose=False,
            return_type="tokens",
        )
        new_tokens = gen[0, tokens.shape[1]:]
        text = model.to_string(new_tokens)
    elapsed = time.time() - t0
    print(f"\n--- Generation ({elapsed:.1f}s, {new_tokens.shape[0]} tokens, {new_tokens.shape[0]/elapsed:.1f} tok/s) ---")
    print(text)
    marker = detect_self_correction(text, min_position=20)
    print(f"\nSelf-correction marker found: {marker}")
    print()
