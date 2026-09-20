"""Causal intervention: add α-scaled direction to residual stream at layer ℓ*, position t*.

Three null comparisons (uniform protocol, identical α scale):
  - random-direction: random unit vector in R^d.
  - orthogonal-projection: random unit vector orthogonal to v_retry.
  - gaussian-noise: iid Gaussian at matched norm.

For each trajectory, we re-generate from t* with the intervention applied and
classify the resulting action.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .label import LABEL_AMBIG, ParsedCall, classify_next_action


def _find_block_modules(model: nn.Module) -> list[nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise RuntimeError("could not find transformer blocks")


@contextmanager
def steer_at(model: nn.Module, layer: int, position: int, vector: torch.Tensor, alpha: float):
    """Context manager: add α · vector at (layer, position) in residual stream.

    Vector is added to the block OUTPUT at the requested position. Norm-matching
    is the caller's responsibility — pass `vector` already scaled to ‖h‖.
    """
    blocks = _find_block_modules(model)
    target = blocks[layer]

    def _hook(_mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if position < h.shape[1]:
            h = h.clone()
            h[:, position, :] = h[:, position, :] + alpha * vector.to(h.dtype).to(h.device)
        if isinstance(out, tuple):
            return (h, *out[1:])
        return h

    handle = target.register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


def norm_matched(vector: np.ndarray, h_at_pos: np.ndarray) -> np.ndarray:
    """Scale vector to have the same L2 norm as `h_at_pos`."""
    vn = np.linalg.norm(vector)
    hn = np.linalg.norm(h_at_pos)
    if vn < 1e-12:
        return np.zeros_like(vector)
    return vector * (hn / vn)


def random_direction(d: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(d)
    return v / (np.linalg.norm(v) + 1e-12)


def orthogonal_to(v_ref: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random unit vector orthogonal to v_ref."""
    r = rng.standard_normal(v_ref.shape[0])
    r = r - r.dot(v_ref) / (v_ref.dot(v_ref) + 1e-12) * v_ref
    return r / (np.linalg.norm(r) + 1e-12)


def gaussian_noise(d: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(d)
    return v / (np.linalg.norm(v) + 1e-12)


@dataclass
class InterventionResult:
    trajectory_id: str
    layer: int
    alpha: float
    direction_kind: str  # "retry" | "random" | "orthogonal" | "gaussian"
    baseline_label: str
    intervened_label: str
    flipped_to_retry: bool


@torch.no_grad()
def regenerate_with_intervention(
    *,
    model,
    tokenizer,
    prompt_ids: list[int],
    layer: int,
    position: int,
    vector: np.ndarray | None,
    alpha: float,
    failed_call: ParsedCall,
    seed: int,
    max_new_tokens: int = 256,
) -> tuple[str, str]:
    """Re-generate from `prompt_ids` with optional intervention applied at (layer, position).

    Returns (continuation_text, label).
    """
    device = next(model.parameters()).device
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if vector is None or alpha == 0.0:
        out = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            use_cache=True,
        )
    else:
        v_t = torch.tensor(vector, dtype=torch.float32)
        with steer_at(model, layer=layer, position=position, vector=v_t, alpha=alpha):
            out = model.generate(
                x,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                use_cache=True,
            )
    gen_ids = out[0, x.shape[1]:].tolist()
    text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    eos_id = tokenizer.eos_token_id
    eot = eos_id in gen_ids if eos_id is not None else False
    label = classify_next_action(
        continuation=text,
        failed_call=failed_call,
        max_tokens_seen=len(gen_ids),
        end_of_turn=eot or len(gen_ids) >= max_new_tokens,
    )
    return text, label
