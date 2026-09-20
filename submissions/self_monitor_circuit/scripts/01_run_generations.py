#!/usr/bin/env python3
"""
Phase 1, Step 1 — Generate CoT outputs from reasoning problems.

Source: GSM8K test set. 2 samples per problem.
Time-boxed to 1 hour (per user instruction).

Output: data/generations/generations.jsonl with full prompts + generated outputs.
"""

import json
import sys
import time
import os
from pathlib import Path

# Ensure single-GPU mode early (before torch imports)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from self_monitor_lib import load_model, format_reasoning_prompt, get_paper_dir

# Hard time budget for generation
HARD_TIMEOUT_SECONDS = 3 * 60 * 60  # 3 hours — extended to reach ≥150 positive SC examples
T0 = time.time()


def log(msg, also_print=True):
    log_path = get_paper_dir() / "notes" / "sanity_log.md"
    with open(log_path, "a") as f:
        f.write(f"- [{time.strftime('%H:%M:%S')}] {msg}\n")
    if also_print:
        print(msg, flush=True)


def time_left():
    return HARD_TIMEOUT_SECONDS - (time.time() - T0)


def load_math_problems():
    """Load MATH problems from HuggingFaceH4/MATH-500.

    Prioritizes harder levels (3-5) and hard subjects first. Falls back to
    GSM8K if MATH dataset unavailable. DeepSeek-R1-Distill produces frequent
    self-corrections on MATH problems at all difficulty levels.
    """
    from datasets import load_dataset
    log("Loading MATH dataset...")

    ds = None
    for dataset_name, dataset_cfg in [
        ("HuggingFaceH4/MATH-500", None),
        ("lighteval/MATH", "all"),
        ("openai/gsm8k", "main"),
    ]:
        try:
            if dataset_cfg:
                ds = load_dataset(dataset_name, dataset_cfg, split="test")
            else:
                ds = load_dataset(dataset_name, split="test")
            log(f"  Loaded {len(ds)} problems from {dataset_name}")
            break
        except Exception as e:
            log(f"  {dataset_name} failed: {str(e)[:80]}")

    if ds is None:
        raise RuntimeError("All dataset sources failed. Check network or HF Hub access.")

    # Convert to consistent format
    records = []
    for ex in ds:
        question = ex.get("problem") or ex.get("question") or ""
        answer = ex.get("solution") or ex.get("answer") or ""
        level = ex.get("level", "")
        subject = ex.get("type") or ex.get("subject", "")
        records.append({
            "question": question,
            "answer": answer,
            "level": level,
            "subject": subject,
        })

    # Prioritize harder problems (DeepSeek-R1 produces richer self-corrections there)
    hard_subjects = {"Intermediate Algebra", "Precalculus", "Number Theory",
                     "Counting & Probability", "Geometry"}
    # Handle both "Level 3" (lighteval/MATH) and "3" (HuggingFaceH4/MATH-500) formats
    hard_levels = {"Level 3", "Level 4", "Level 5", "3", "4", "5"}

    hard_records = [r for r in records if r["level"] in hard_levels and r["subject"] in hard_subjects]
    other_records = [r for r in records if not (r["level"] in hard_levels and r["subject"] in hard_subjects)]

    log(f"  Hard problems (Level 3-5, hard subjects): {len(hard_records)}")
    log(f"  Other problems: {len(other_records)}")

    # Hard first, then the rest as additional budget filler
    return hard_records + other_records


def main():
    log("\n## Phase 1 Step 1: Generate CoT outputs\n", also_print=False)
    log("=" * 60)
    log(f"Phase 1 started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Generation budget: {HARD_TIMEOUT_SECONDS}s ({HARD_TIMEOUT_SECONDS/60} min)")

    # Load model
    log("Loading model...")
    t_load_start = time.time()
    model, choice = load_model()
    log(f"  Model loaded in {time.time() - t_load_start:.1f}s")
    log(f"  Model: {choice['model_id']}")
    log(f"  Layers: {choice['n_layers']}, d_model: {choice['d_model']}, n_heads: {choice['n_heads']}")

    # Load problems (MATH dataset for harder problems where Qwen will make mistakes)
    problems = load_math_problems()
    # Don't shuffle — we already prioritized hard problems first

    # Set source field
    for p in problems:
        p["source"] = "math"
        p["question_field"] = "question"

    # Output file
    out_dir = get_paper_dir() / "data" / "generations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generations.jsonl"

    # Generation params
    SAMPLES_PER_PROBLEM = 2
    MAX_NEW_TOKENS = 800  # 800 tokens: enough context for full reasoning chain before SC event
    TEMPERATURE = 0.7
    TOP_P = 0.95

    log(f"\nGeneration params: samples={SAMPLES_PER_PROBLEM}, max_new={MAX_NEW_TOKENS}, T={TEMPERATURE}, top_p={TOP_P}")

    # Resume-aware: find which (problem_idx, sample_idx) pairs are already done
    done_pairs = set()
    if out_path.exists():
        import json as _json
        with open(out_path) as _f:
            for _line in _f:
                try:
                    _rec = _json.loads(_line)
                    _pid = _rec.get("problem_id", "")
                    # problem_id format: "math_{idx}"
                    _idx = int(_pid.split("_")[1]) if "_" in _pid else -1
                    done_pairs.add((_idx, _rec.get("sample_idx", 0)))
                except Exception:
                    pass
        if done_pairs:
            log(f"  Resuming: {len(done_pairs)} (problem, sample) pairs already in file — will skip those")

    import torch

    n_gens = 0
    n_problems_done = 0

    # Append mode: preserves existing data when resuming
    with open(out_path, "a") as fout:
        for problem_idx, problem in enumerate(problems):
            if time_left() < 30:
                log(f"Time budget exhausted after {n_problems_done} problems, {n_gens} generations")
                break

            prompt_text = format_reasoning_prompt(problem["question"], model)

            for sample_idx in range(SAMPLES_PER_PROBLEM):
                # Skip already-generated (problem, sample) pairs
                if (problem_idx, sample_idx) in done_pairs:
                    continue

                if time_left() < 15:
                    log(f"Time budget tight after {n_gens} generations — stopping")
                    break

                try:
                    t_gen_start = time.time()
                    with torch.no_grad():
                        tokens = model.to_tokens(prompt_text, prepend_bos=False)
                        # TL's generate() supports do_sample
                        gen_output = model.generate(
                            tokens,
                            max_new_tokens=MAX_NEW_TOKENS,
                            do_sample=True,
                            temperature=TEMPERATURE,
                            top_p=TOP_P,
                            verbose=False,
                            return_type="tokens",
                        )
                        # gen_output shape: (1, prompt_len + new_tokens)
                        new_tokens = gen_output[0, tokens.shape[1]:]
                        generated_text = model.to_string(new_tokens)

                    gen_time = time.time() - t_gen_start

                    record = {
                        "problem_id": f"math_{problem_idx}",
                        "sample_idx": sample_idx,
                        "source": problem.get("source", "math"),
                        "question": problem["question"],
                        "ground_truth": problem["answer"],
                        "level": problem.get("level", ""),
                        "subject": problem.get("subject", ""),
                        "prompt": prompt_text,
                        "generated": generated_text,
                        "prompt_tokens": int(tokens.shape[1]),
                        "generated_tokens": int(new_tokens.shape[0]),
                        "gen_time_seconds": round(gen_time, 2),
                    }
                    fout.write(json.dumps(record) + "\n")
                    fout.flush()
                    n_gens += 1

                    if n_gens % 10 == 0:
                        elapsed = time.time() - T0
                        rate = n_gens / elapsed if elapsed > 0 else 0
                        log(f"  Generated {n_gens} so far ({rate:.2f}/sec, {elapsed/60:.1f} min in, {time_left()/60:.1f} min left)")

                except Exception as e:
                    log(f"  GENERATION ERROR on problem {problem_idx} sample {sample_idx}: {type(e).__name__}: {str(e)[:200]}")
                    continue

            n_problems_done += 1

            # Save mid-checkpoint summary every 50 problems
            if n_problems_done % 50 == 0:
                log(f"Checkpoint: {n_problems_done} problems, {n_gens} generations, {time_left()/60:.1f} min budget left")

    elapsed_total = time.time() - T0
    log(f"\n✓ Phase 1 step 1 complete: {n_gens} generations from {n_problems_done} problems in {elapsed_total/60:.1f} min")
    log(f"  Average: {elapsed_total/max(n_gens,1):.1f} sec/generation")
    log(f"  Output: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
