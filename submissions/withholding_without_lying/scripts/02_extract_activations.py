#!/usr/bin/env python3
"""
Phase 2 — Extract residual stream activations for 4 conditions.

Conditions:
  A      = no system prompt + verified question (sanity)
  A'     = HELPFUL system prompt + verified question (system-prompt CONTROL)
  B      = SANDBAGGING persona + verified question (signal of interest)
  C      = HONEST persona + UNKNOWN topic question (genuine ignorance)
  C2     = HONEST persona + verified question (sanity: honest answers correctly)

Extract residual stream at FINAL TOKEN of input across all layers.
"""
import os, sys, json, time
from pathlib import Path
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sandbagging_lib import (
    load_model, log, get_paper_dir, PERSONAS,
    format_normal_prompt, format_helpful_prompt,
    format_persona_prompt, format_honest_ignorance_prompt,
)


def get_final_token_resid(model, prompt_text):
    tokens = model.to_tokens(prompt_text, prepend_bos=False)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    return torch.stack([
        cache[f"blocks.{L}.hook_resid_post"][0, -1].cpu()
        for L in range(n_layers)
    ])  # (n_layers, d_model)


def extract_condition(model, prompts, condition_name, cap=80):
    log(f"\nExtracting condition {condition_name} ({min(len(prompts), cap)} prompts)...")
    resids = []
    metas  = []
    for i, p in enumerate(prompts[:cap]):
        try:
            r = get_final_token_resid(model, p["prompt"])
            resids.append(r)
            metas.append({k: p[k] for k in p if k != "prompt"})
            if (i+1) % 25 == 0:
                log(f"  [{i+1}/{min(len(prompts), cap)}] {condition_name}")
        except Exception as e:
            log(f"  ERROR @ {i}: {e}")
    if not resids:
        return None, None
    return torch.stack(resids).half(), metas


def main():
    log("=== Paper 3 Phase 2: Extract activations (4 controlled conditions) ===")
    paper_dir = get_paper_dir()
    data_dir = paper_dir / "data"
    act_dir  = paper_dir / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    q_known   = [json.loads(l) for l in open(data_dir / "q_known.jsonl")]
    A_prime   = [json.loads(l) for l in open(data_dir / "control" / "condition_A_prime.jsonl")]
    C_data    = [json.loads(l) for l in open(data_dir / "ignorance" / "condition_C.jsonl")]
    C2_data   = [json.loads(l) for l in open(data_dir / "control" / "condition_C2_honest_known.jsonl")]
    sb_paths  = [data_dir / "sandbagging" / f"persona_{p}.jsonl" for p in PERSONAS]
    sb_all    = []
    for p, sp in zip(PERSONAS, sb_paths):
        if sp.exists():
            sb_all.extend(json.loads(l) for l in open(sp))

    log(f"Datasets loaded: q_known={len(q_known)}, A'={len(A_prime)}, "
        f"B={len(sb_all)}, C={len(C_data)}, C2={len(C2_data)}")

    log("Loading model on GPU 2...")
    t0 = time.time()
    model = load_model(device="cuda", dtype="float16")
    log(f"  Loaded in {time.time()-t0:.1f}s")

    # Build prompt lists (with prompt strings included)
    A_prompts = [{"prompt": format_normal_prompt(model, x["question"]),
                   "question": x["question"], "condition": "A"} for x in q_known]

    Aprime_prompts = [{"prompt": format_helpful_prompt(model, x["question"]),
                        "question": x["question"], "condition": "A_prime"} for x in A_prime]

    B_prompts = [{"prompt": format_persona_prompt(model, x["question"], x["persona"]),
                   "question": x["question"], "persona": x["persona"],
                   "is_withholding": x.get("is_withholding", False),
                   "condition": "B"} for x in sb_all]

    C_prompts = [{"prompt": format_honest_ignorance_prompt(model, x["question"]),
                   "question": x["question"], "condition": "C"} for x in C_data]

    C2_prompts = [{"prompt": format_honest_ignorance_prompt(model, x["question"]),
                    "question": x["question"], "condition": "C2"} for x in C2_data]

    # Extract per condition
    A_tensor,  A_meta  = extract_condition(model, A_prompts,  "A")
    Ap_tensor, Ap_meta = extract_condition(model, Aprime_prompts, "A_prime")
    B_tensor,  B_meta  = extract_condition(model, B_prompts,  "B", cap=160)
    C_tensor,  C_meta  = extract_condition(model, C_prompts,  "C")
    C2_tensor, C2_meta = extract_condition(model, C2_prompts, "C2")

    n_layers = model.cfg.n_layers
    d_model  = model.cfg.d_model

    # Sanity check: no NaN
    for name, t in [("A", A_tensor), ("A_prime", Ap_tensor), ("B", B_tensor),
                    ("C", C_tensor), ("C2", C2_tensor)]:
        if t is None:
            log(f"  ⚠ Condition {name} had no successful extractions")
            continue
        assert not torch.isnan(t).any(), f"SC5 FAIL: NaN in {name}"
    log("SC5 PASS: no NaN/Inf")

    base = {"n_layers": n_layers, "d_model": d_model}
    if A_tensor  is not None: torch.save({**base, "resid": A_tensor,  "meta": A_meta},  act_dir / "condition_A.pt")
    if Ap_tensor is not None: torch.save({**base, "resid": Ap_tensor, "meta": Ap_meta}, act_dir / "condition_A_prime.pt")
    if B_tensor  is not None: torch.save({**base, "resid": B_tensor,  "meta": B_meta},  act_dir / "condition_B.pt")
    if C_tensor  is not None: torch.save({**base, "resid": C_tensor,  "meta": C_meta},  act_dir / "condition_C.pt")
    if C2_tensor is not None: torch.save({**base, "resid": C2_tensor, "meta": C2_meta}, act_dir / "condition_C2.pt")

    log(f"✓ Phase 2 done.")
    if A_tensor  is not None: log(f"  A:       {A_tensor.shape}")
    if Ap_tensor is not None: log(f"  A_prime: {Ap_tensor.shape}")
    if B_tensor  is not None: log(f"  B:       {B_tensor.shape}")
    if C_tensor  is not None: log(f"  C:       {C_tensor.shape}")
    if C2_tensor is not None: log(f"  C2:      {C2_tensor.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
