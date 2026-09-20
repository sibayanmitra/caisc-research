#!/usr/bin/env python3
"""
Phase 0 — Setup & TL Verification.

Verifies:
1. Venv is active and core packages import.
2. CUDA + GPU detection.
3. TransformerLens 3.2.1 can load DeepSeek-R1-Distill-Qwen-7B.
   If not, fallback to Qwen2.5-7B-Instruct (verified in TL official list).
4. Hooks fire at residual stream and per-head z.

Output: prints status to stdout. Saves model config to ../notes/sanity_log.md.
"""

import importlib.metadata as md
import json
import sys
import time
from pathlib import Path

# Hard timeout for the entire script (in seconds). Per plan: 30 min Phase 0 timeout
HARD_TIMEOUT = 30 * 60
T0 = time.time()


def log(msg, also_print=True):
    """Append to sanity_log.md and optionally print."""
    log_path = Path(__file__).resolve().parent.parent / "notes" / "sanity_log.md"
    with open(log_path, "a") as f:
        f.write(f"- [{time.strftime('%H:%M:%S')}] {msg}\n")
    if also_print:
        print(msg, flush=True)


def time_left():
    return HARD_TIMEOUT - (time.time() - T0)


def check_imports():
    log("## Phase 0 Setup Log\n", also_print=False)
    log(f"Phase 0 started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    tl_v = md.version("transformer_lens")
    log(f"TransformerLens version: {tl_v}")
    assert tl_v == "3.2.1", f"Expected TL 3.2.1, got {tl_v}"

    import transformers
    log(f"Transformers version: {transformers.__version__}")

    import torch
    log(f"PyTorch version: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    log(f"GPU count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            log(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Use only one GPU per user's instruction (rule from CLAUDE.md context)
    import os
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        log("Set CUDA_VISIBLE_DEVICES=0 (single GPU mode per user instruction)")

    return torch


def attempt_tl_load(model_id, torch):
    """Try to load a model via TransformerLens. Returns model or None.

    Uses from_pretrained_no_processing (recommended for fp16) to avoid CPU memory spike
    from LayerNorm folding. We don't need LN-folded weights for activation extraction
    or steering (only logit-lens style analyses are affected, which we don't use here).
    """
    from transformer_lens import HookedTransformer
    log(f"\nAttempting TL load of: {model_id}")
    log("  Using from_pretrained_no_processing (recommended for fp16 to skip LN-folding step)")
    t_start = time.time()
    try:
        model = HookedTransformer.from_pretrained_no_processing(
            model_id,
            device="cuda",
            dtype="float16",
        )
        elapsed = time.time() - t_start
        log(f"  SUCCESS: loaded in {elapsed:.1f}s")
        log(f"  Layers: {model.cfg.n_layers}, d_model: {model.cfg.d_model}, n_heads: {model.cfg.n_heads}, d_head: {model.cfg.d_head}")
        return model
    except Exception as e:
        elapsed = time.time() - t_start
        log(f"  FAILED after {elapsed:.1f}s: {type(e).__name__}: {str(e)[:300]}")
        return None


def verify_hooks(model, model_id):
    """Run a forward pass and verify hooks fire."""
    import torch
    log(f"\nVerifying hooks on {model_id}")
    test_prompt = "The capital of France is"

    try:
        with torch.no_grad():
            tokens = model.to_tokens(test_prompt)
            log(f"  Tokens shape: {tokens.shape}")

            # Run with cache
            logits, cache = model.run_with_cache(tokens)
            log(f"  Logits shape: {logits.shape}")
            log(f"  Cache keys (sample): {list(cache.keys())[:5]}")

            # Test residual stream hook
            resid = cache["blocks.0.hook_resid_post"]
            log(f"  blocks.0.hook_resid_post shape: {resid.shape}")
            log(f"  blocks.0.hook_resid_post norm: {resid.norm().item():.2f}")

            # Test per-head z hook
            z = cache["blocks.0.attn.hook_z"]
            log(f"  blocks.0.attn.hook_z shape: {z.shape}")

            # Test top prediction
            top_id = logits[0, -1].argmax().item()
            top_token = model.to_string([top_id])
            log(f"  Top prediction after '{test_prompt}': '{top_token}'")

            assert not torch.isnan(resid).any(), "NaN in residual stream"
            assert not torch.isnan(z).any(), "NaN in head z"

            log("  ✓ All hooks fire, no NaN detected")
            return True
    except Exception as e:
        log(f"  HOOK VERIFICATION FAILED: {type(e).__name__}: {str(e)[:300]}")
        return False


def main():
    log("=" * 60)
    log(f"Hard timeout: {HARD_TIMEOUT}s ({HARD_TIMEOUT/60} min)")

    torch = check_imports()

    # Per user choice: skip DeepSeek attempt, go straight to Qwen2.5-7B-Instruct
    log("\n" + "=" * 60)
    log("STEP 1: Load Qwen2.5-7B-Instruct via TL (user-selected primary)")
    log("=" * 60)
    log("Note: DeepSeek-R1-Distill-Qwen-7B skipped per user choice to avoid extra download.")
    log("Qwen2.5-7B-Instruct is officially supported in TL 3.2.1.")

    model = attempt_tl_load("Qwen/Qwen2.5-7B-Instruct", torch)
    if model is not None:
        ok = verify_hooks(model, "Qwen/Qwen2.5-7B-Instruct")
        if ok:
            log("\n✓ MODEL READY: Qwen2.5-7B-Instruct via TL")
            save_choice("Qwen/Qwen2.5-7B-Instruct", "tl", model)
            return 0
        else:
            log("\n✗ Hook verification failed on Qwen2.5-7B-Instruct")
            return 1

    log("\n✗ Qwen2.5-7B-Instruct failed to load — escalating to user")
    return 1


def save_choice(model_id, load_path, model):
    """Save the chosen model + path for downstream scripts."""
    cfg_path = Path(__file__).resolve().parent.parent / "notes" / "model_choice.json"
    config_dict = {
        "model_id": model_id,
        "load_path": load_path,
        "n_layers": model.cfg.n_layers,
        "d_model": model.cfg.d_model,
        "n_heads": model.cfg.n_heads,
        "d_head": model.cfg.d_head,
        "n_ctx": model.cfg.n_ctx,
        "dtype": str(model.cfg.dtype),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(cfg_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    log(f"Saved model choice to {cfg_path}: {model_id} via {load_path}")


if __name__ == "__main__":
    sys.exit(main())
