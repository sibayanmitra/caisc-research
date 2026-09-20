#!/usr/bin/env python3
"""Smoke test with hard MATH problems to check self-correction yield."""
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

# Hard MATH problems (Intermediate Algebra, Precalc-level)
problems = [
    "Find the largest value of $c$ such that $1$ is in the range of $f(x)=x^2-5x+c$.",
    "Let $a$ and $b$ be the roots of $kx^2 - kx + x + 5 = 0$. If $\\frac{a}{b} + \\frac{b}{a} = \\frac{4}{5}$, what is the sum of the values of $k$ that satisfy this equation?",
    "The polynomial $f(x) = x^3 - 3x^2 - 4x + 4$ has three real roots $r_1$, $r_2$, and $r_3$. Let $g(x) = x^3 + ax^2 + bx + c$ be the polynomial which has roots $s_1$, $s_2$, and $s_3$, where $s_1 = r_1 + r_2 z + r_3 z^2$, $s_2 = r_1 z + r_2 z^2 + r_3$, $s_3 = r_1 z^2 + r_2 + r_3 z$, and $z = e^{2\\pi i/3}$. Find the real part of the sum of the coefficients of $g(x)$.",
]

n_corrections = 0
for problem in problems:
    print("=" * 60)
    print(f"Problem: {problem[:120]}...")
    prompt = format_reasoning_prompt(problem, model)

    t0 = time.time()
    with torch.no_grad():
        tokens = model.to_tokens(prompt, prepend_bos=False)
        gen = model.generate(
            tokens,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            verbose=False,
            return_type="tokens",
        )
        new_tokens = gen[0, tokens.shape[1]:]
        text = model.to_string(new_tokens)
    elapsed = time.time() - t0
    print(f"\n--- Generation ({elapsed:.1f}s, {new_tokens.shape[0]} tokens) ---")
    # Print first 1500 chars
    print(text[:1500] + ("..." if len(text) > 1500 else ""))

    marker = detect_self_correction(text, min_position=20)
    print(f"\nSelf-correction marker: {marker}")
    if marker:
        n_corrections += 1
        print(f"  Context: ...{text[max(0, marker[0]-50):marker[1]+50]}...")
    print()

print(f"\nYield: {n_corrections}/{len(problems)} problems had self-corrections")
