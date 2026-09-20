"""
Shared library for the self-monitor circuit paper.

Provides:
- load_model(): loads DeepSeek-R1-Distill-Qwen-7B via TL (primary)
- SELF_CORRECTION_MARKERS: regex patterns for detecting self-corrections
- detect_self_correction(text): returns first match position or None
- format_reasoning_prompt(problem): formats with chat template

Loading strategy:
  DeepSeek-R1-Distill-Qwen-7B is NOT in TL's official list, but shares the
  Qwen2.5-7B architecture (28L, 3584d, 28H, 128d_head). We load HF weights
  directly, then wrap with TL using the Qwen2.5-7B config via hf_model=.
"""

import json
import re
from pathlib import Path
from typing import Optional, Tuple

PAPER_DIR = Path(__file__).resolve().parent.parent
NOTES_DIR = PAPER_DIR / "notes"


# Curated self-correction markers (case-insensitive regex)
SELF_CORRECTION_MARKERS = [
    r"\bwait\b",
    r"\bactually\b",
    r"\bhmm\b",
    r"\blet me reconsider\b",
    r"\blet me recheck\b",
    r"\blet me re-?examine\b",
    r"\blet me try again\b",
    r"\bi made a mistake\b",
    r"\bthat's not right\b",
    r"\bthat doesn't seem right\b",
    r"\bscratch that\b",
    r"\bno,?\s+wait\b",
    r"\bhold on\b",
    r"\bon (second|further) (thought|reflection)\b",
    r"\boh,?\s+wait\b",
    r"\bcorrection\b",
]

COMPILED_MARKERS = [re.compile(p, re.IGNORECASE) for p in SELF_CORRECTION_MARKERS]


def detect_self_correction(text: str, min_position: int = 20) -> Optional[Tuple[int, int, str]]:
    """
    Find the earliest self-correction marker in text.

    Args:
        text: generated text to search
        min_position: ignore markers before this character position (avoid markers at very start)

    Returns:
        (char_start, char_end, marker_text) if found, else None
    """
    earliest = None
    for compiled in COMPILED_MARKERS:
        m = compiled.search(text)
        if m is None:
            continue
        if m.start() < min_position:
            continue
        if earliest is None or m.start() < earliest[0]:
            earliest = (m.start(), m.end(), m.group(0))
    return earliest


def load_model_choice():
    """Return dict with model_id, n_layers, d_model, n_heads, d_head, n_ctx, dtype."""
    cfg_path = NOTES_DIR / "model_choice.json"
    with open(cfg_path) as f:
        return json.load(f)


def load_model(device: str = "cuda", dtype: str = "float16"):
    """Load DeepSeek-R1-Distill-Qwen-7B via TransformerLens.

    Strategy: DeepSeek-R1-Distill-Qwen-7B shares the Qwen2.5-7B architecture
    but is not in TL's official registry. We:
      1. Load HF weights via AutoModelForCausalLM (fp16, CPU)
      2. Pass to TL's from_pretrained_no_processing with TL_ARCH config

    This avoids LN-folding OOM (from_pretrained_no_processing skips that step)
    and gives us full TL hook support.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformer_lens import HookedTransformer

    choice = load_model_choice()
    model_id = choice["model_id"]
    tl_arch = choice.get("tl_arch", model_id)  # TL config arch name

    torch_dtype = torch.float16 if dtype in ("float16", "torch.float16") else torch.float32

    # Load HF weights first (on CPU to control memory)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="cpu",
    )
    tok = AutoTokenizer.from_pretrained(model_id)

    # Wrap with TL using the appropriate arch config
    model = HookedTransformer.from_pretrained_no_processing(
        tl_arch,
        hf_model=hf_model,
        tokenizer=tok,
        device=device,
        dtype=dtype,
    )
    return model, choice


def format_reasoning_prompt(problem: str, model) -> str:
    """
    Format a reasoning problem using the model's chat template.

    For Qwen2.5-Instruct: uses <|im_start|>user / <|im_start|>assistant format.
    Falls back to plain instruction format if no chat template.
    """
    instruction = (
        f"Solve the following problem step by step. Show your reasoning clearly. "
        f"If you make a mistake during your reasoning, recognize it and correct yourself.\n\n"
        f"Problem: {problem}\n\nSolution:"
    )
    # Use TL's tokenizer chat template if available
    try:
        tokenizer = model.tokenizer
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            messages = [{"role": "user", "content": instruction}]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    except Exception:
        pass
    return instruction


def get_paper_dir() -> Path:
    return PAPER_DIR
