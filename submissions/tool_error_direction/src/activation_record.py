"""Fresh-forward activation recording at multiple token positions.

Critical invariants:
  - `use_cache=False` — required by pre-flight gate G-CACHE.
  - Hooks attached to ALL transformer-block outputs (residual-stream post-block).
  - We extract residuals at THREE positions per trajectory: t*, t*+5, pre_next_action.
  - Recording is fp32 on CPU after detach (storage; computation stays bf16 on GPU).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

ACT_DIR = Path("/home/anonymous/agentic_paper/data/activations")
ACT_DIR.mkdir(parents=True, exist_ok=True)


def _find_block_modules(model: nn.Module) -> list[nn.Module]:
    """Locate the list of transformer blocks for Qwen3 / Llama-family models."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise RuntimeError("could not find transformer blocks on this model")


@torch.no_grad()
def record_residuals(
    model: nn.Module,
    input_ids: list[int],
    positions: list[int],
    device: str = "cuda:0",
) -> np.ndarray:
    """Run a fresh, non-cached forward pass and return residuals at given positions.

    Returns: np.ndarray of shape (n_layers, n_positions, d_model), dtype float32.
    """
    blocks = _find_block_modules(model)
    n_layers = len(blocks)
    captured: list[torch.Tensor | None] = [None] * n_layers

    def make_hook(i: int):
        def _hook(_mod, _inp, out):
            # block output is (hidden_states, ...) or just a Tensor
            h = out[0] if isinstance(out, tuple) else out
            captured[i] = h.detach()

        return _hook

    handles = [b.register_forward_hook(make_hook(i)) for i, b in enumerate(blocks)]
    try:
        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        _ = model(x, use_cache=False, output_hidden_states=False)
    finally:
        for h in handles:
            h.remove()

    seq_len = x.shape[1]
    safe_positions = [min(max(p, 0), seq_len - 1) for p in positions]

    out = np.empty((n_layers, len(positions), captured[0].shape[-1]), dtype=np.float32)
    for li, h in enumerate(captured):
        # h shape: (1, seq_len, d_model)
        for pi, pos in enumerate(safe_positions):
            out[li, pi, :] = h[0, pos, :].float().cpu().numpy()
    return out


@torch.no_grad()
def verify_cache_byte_equality(
    model: nn.Module,
    input_ids: list[int],
    device: str = "cuda:0",
    atol: float = 1e-3,
) -> tuple[bool, float]:
    """G-CACHE (preregistration §11.4): use_cache=True vs use_cache=False residuals at
    last position must agree within `atol`.

    bfloat16 forward passes are not bit-exact across the two paths due to differing
    kernel reductions, so we use a loose tolerance (1e-3) here. The gate is meant to
    catch *implementation* drift (wrong hook target, accidentally cached residuals
    bypassing layers), not numerical noise.
    """
    blocks = _find_block_modules(model)

    def _grab_last_layer_last_pos(use_cache: bool) -> torch.Tensor:
        h_list: list[torch.Tensor] = []

        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h_list.append(h.detach())

        handle = blocks[-1].register_forward_hook(hook)
        try:
            x = torch.tensor([input_ids], dtype=torch.long, device=device)
            _ = model(x, use_cache=use_cache, output_hidden_states=False)
        finally:
            handle.remove()
        return h_list[0][0, -1, :].float()

    a = _grab_last_layer_last_pos(use_cache=False)
    b = _grab_last_layer_last_pos(use_cache=True)
    diff = (a - b).abs().max().item()
    return diff <= atol, diff
