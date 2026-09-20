"""Trajectory generator: drives a target model through (prompt × outcome × seed).

Pipeline per trial:
  1. Build chat: system + user.
  2. Generate first assistant turn → expect <tool_call>...</tool_call>.
  3. Truncate at first </tool_call>; parse the call.
  4. Inject deterministic mock-tool result.
  5. Add empty assistant generation prompt; record token offsets.
  6. Generate second assistant turn (the next-action decision).
  7. Classify {retry-same, modify-args, give-up, ambiguous}.
  8. Persist trajectory: prompt token IDs, tool-result boundary token index (= t*),
     second-turn generation token IDs, label, and metadata.

Activation recording happens later in `activation_record.py` via a fresh forward
pass on the saved token sequence — keeps generation and recording decoupled.
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .label import (
    LABEL_AMBIG,
    ParsedCall,
    TOOL_CALL_RE_JSON,
    classify_next_action,
    find_tool_calls,
)
from .mock_tools import ToolCall, ToolResult, call_tool
from .prompts import BasePrompt

DATA_DIR = Path("/home/anonymous/agentic_paper/data/trajectories")
DATA_DIR.mkdir(parents=True, exist_ok=True)


SYSTEM_PROMPT = """You are an agent that solves tasks by calling tools. Three tools are available:

- web_search: takes {"query": "..."}.
- calculator: takes {"expression": "..."}.
- file_read: takes {"path": "..."}.

Emit exactly one tool call per turn, formatted as:
<tool_call>
{"name": "<tool>", "arguments": {...}}
</tool_call>

After the tool returns, decide your next action. You may emit another <tool_call> with the same or different arguments, or, when the task is fully complete (or genuinely unrecoverable), emit:
<final_answer>
Your answer or explanation.
</final_answer>

Do not produce any text outside these tags."""

# xLAM-2 native format: bare JSON array tool calls
SYSTEM_PROMPT_XLAM = """You are an agent that solves tasks by calling tools. Three tools are available:

- web_search: takes {"query": "..."}
- calculator: takes {"expression": "..."}
- file_read: takes {"path": "..."}

Emit exactly one tool call per turn as a JSON array:
[{"name": "<tool>", "arguments": {...}}]

After the tool returns, decide your next action. You may emit another tool call with the same or different arguments, or when the task is fully complete (or genuinely unrecoverable), emit:
<final_answer>
Your answer or explanation.
</final_answer>

Do not produce any text outside these formats."""


def _is_xlam(model_name: str) -> bool:
    return "xlam" in model_name.lower()


@dataclass
class Trajectory:
    base_prompt_id: str
    tool: str
    outcome: str
    model_name: str
    seed: int
    user_prompt: str
    first_assistant_text: str  # raw text including <tool_call>
    parsed_call: dict | None  # {"name": str, "args": dict}; None if parse failed
    tool_result: dict | None  # {"ok": bool, "payload": str, "error_kind": str | None}
    second_prompt_input_ids: list[int]  # token IDs of full prompt up to (and incl) gen prompt
    t_star_idx: int  # index of last input token = position immediately before second-turn generation
    second_assistant_text: str  # full second-turn generation text
    second_generation_ids: list[int]  # generated token IDs
    label: str
    pre_next_action_offset: int | None  # index within second_generation_ids before next <tool_call> or <final_answer>
    end_of_turn: bool
    timing_ms: dict[str, float]

    def to_jsonable(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        return d


def _truncate_at_tool_call(text: str) -> str | None:
    """Keep text up to and including the first complete tool call (either format)."""
    # G3 format: <tool_call>...</tool_call>
    end = text.find("</tool_call>")
    if end != -1:
        return text[: end + len("</tool_call>")]
    # xLAM-2 format: JSON array [{"name": ..., "arguments": ...}]
    m = TOOL_CALL_RE_JSON.search(text)
    if m:
        return text[: m.end()]
    return None


def _parse_first_tool_call(text: str) -> ParsedCall | None:
    calls = find_tool_calls(text)
    return calls[0] if calls else None


def _tool_result_to_text(result: ToolResult) -> str:
    """Render a ToolResult as the deterministic 'tool' message payload."""
    if result.ok:
        return result.payload
    return f"[ERROR: {result.error_kind}] {result.payload}"


def _supports_tool_role(tokenizer) -> bool:
    """Some chat templates accept role='tool'; others raise. Probe defensively."""
    try:
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a"},
                {"role": "tool", "content": "t"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return True
    except Exception:
        return False


_THINKING_TOGGLE_CACHE: dict[int, bool] = {}


def _supports_thinking_toggle(tokenizer) -> bool:
    """Qwen3 supports `enable_thinking`; Llama-family templates ignore the kwarg or raise.
    Cached per tokenizer instance."""
    key = id(tokenizer)
    if key in _THINKING_TOGGLE_CACHE:
        return _THINKING_TOGGLE_CACHE[key]
    try:
        s_off = tokenizer.apply_chat_template(
            [{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        s_on = tokenizer.apply_chat_template(
            [{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True
        )
        result = s_off != s_on
    except Exception:
        result = False
    _THINKING_TOGGLE_CACHE[key] = result
    return result


def _apply_chat(tokenizer, msgs: list[dict]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if _supports_thinking_toggle(tokenizer):
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(msgs, **kwargs)


def _build_prompt(tokenizer, user_prompt: str, first_assistant: str, tool_result_text: str, use_tool_role: bool, model_name: str = "") -> str:
    sys = SYSTEM_PROMPT_XLAM if _is_xlam(model_name) else SYSTEM_PROMPT
    msgs = [
        {"role": "system", "content": sys},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": first_assistant},
    ]
    if use_tool_role:
        msgs.append({"role": "tool", "content": tool_result_text})
    else:
        msgs.append({"role": "user", "content": f"Tool result:\n{tool_result_text}"})
    return _apply_chat(tokenizer, msgs)


def _build_first_prompt(tokenizer, user_prompt: str, model_name: str = "") -> str:
    sys = SYSTEM_PROMPT_XLAM if _is_xlam(model_name) else SYSTEM_PROMPT
    msgs = [
        {"role": "system", "content": sys},
        {"role": "user", "content": user_prompt},
    ]
    return _apply_chat(tokenizer, msgs)


def _find_pre_next_action_offset(text: str) -> int | None:
    """Return character offset (within `text`) of the first <tool_call> or <final_answer> tag."""
    tc_idx = text.find("<tool_call>")
    fa_idx = text.find("<final_answer>")
    cands = [i for i in (tc_idx, fa_idx) if i != -1]
    return min(cands) if cands else None


@torch.no_grad()
def _generate(model, tokenizer, prompt_text: str, max_new_tokens: int, seed: int) -> tuple[str, list[int], list[int]]:
    """Greedy(ish) generation with fixed seed. Returns (text, prompt_ids, generated_ids)."""
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    prompt_ids = inputs.input_ids[0].tolist()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # deterministic
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        use_cache=True,  # for generation only; recording uses fresh non-cached forward
    )
    full_ids = out[0].tolist()
    gen_ids = full_ids[len(prompt_ids):]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    return gen_text, prompt_ids, gen_ids


def generate_trajectory(
    *,
    model,
    tokenizer,
    use_tool_role: bool,
    base_prompt: BasePrompt,
    outcome: str,
    condition_seed: int,
    model_seed: int,
    model_name: str,
    max_first_tokens: int = 200,
    max_second_tokens: int = 256,
) -> Trajectory:
    """Generate one trajectory end-to-end and return a populated record."""
    t0 = time.perf_counter()
    timing: dict[str, float] = {}

    # Step 1 — first assistant turn
    first_prompt = _build_first_prompt(tokenizer, base_prompt.text, model_name=model_name)
    t1 = time.perf_counter()
    first_text, _, _ = _generate(model, tokenizer, first_prompt, max_first_tokens, model_seed)
    timing["first_gen_ms"] = (time.perf_counter() - t1) * 1000

    # Step 2 — truncate at first </tool_call>
    first_truncated = _truncate_at_tool_call(first_text)
    parsed = _parse_first_tool_call(first_truncated or first_text)

    if parsed is None:
        # No valid tool call → trajectory is unusable for our experiment.
        return Trajectory(
            base_prompt_id=base_prompt.id,
            tool=base_prompt.tool,
            outcome=outcome,
            model_name=model_name,
            seed=model_seed,
            user_prompt=base_prompt.text,
            first_assistant_text=first_text,
            parsed_call=None,
            tool_result=None,
            second_prompt_input_ids=[],
            t_star_idx=-1,
            second_assistant_text="",
            second_generation_ids=[],
            label=LABEL_AMBIG,
            pre_next_action_offset=None,
            end_of_turn=False,
            timing_ms=timing,
        )

    # Step 3 — controlled tool result
    tool_result = call_tool(
        ToolCall(tool=parsed.name, args=parsed.args, outcome=outcome),
        condition_seed=condition_seed,
    )
    tool_result_text = _tool_result_to_text(tool_result)

    # Step 4 — build second prompt
    second_prompt = _build_prompt(
        tokenizer,
        user_prompt=base_prompt.text,
        first_assistant=first_truncated,
        tool_result_text=tool_result_text,
        use_tool_role=use_tool_role,
        model_name=model_name,
    )

    # Step 5 — second assistant turn
    t2 = time.perf_counter()
    second_text, prompt_ids, gen_ids = _generate(model, tokenizer, second_prompt, max_second_tokens, model_seed + 1)
    timing["second_gen_ms"] = (time.perf_counter() - t2) * 1000

    t_star = len(prompt_ids) - 1

    # Step 6 — label
    eos_id = tokenizer.eos_token_id
    end_of_turn = eos_id in gen_ids if eos_id is not None else False
    label = classify_next_action(
        continuation=second_text,
        failed_call=parsed,
        max_tokens_seen=len(gen_ids),
        end_of_turn=end_of_turn or len(gen_ids) >= max_second_tokens,
    )

    # Step 7 — pre-next-action offset within second_text → map to token index
    char_off = _find_pre_next_action_offset(second_text)
    pre_idx: int | None = None
    if char_off is not None:
        # convert char offset → token offset within gen_ids
        prefix = second_text[:char_off]
        prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
        pre_idx = len(prefix_ids)

    timing["total_ms"] = (time.perf_counter() - t0) * 1000
    return Trajectory(
        base_prompt_id=base_prompt.id,
        tool=base_prompt.tool,
        outcome=outcome,
        model_name=model_name,
        seed=model_seed,
        user_prompt=base_prompt.text,
        first_assistant_text=first_truncated,
        parsed_call={"name": parsed.name, "args": parsed.args},
        tool_result={"ok": tool_result.ok, "payload": tool_result.payload, "error_kind": tool_result.error_kind},
        second_prompt_input_ids=prompt_ids,
        t_star_idx=t_star,
        second_assistant_text=second_text,
        second_generation_ids=gen_ids,
        label=label,
        pre_next_action_offset=pre_idx,
        end_of_turn=end_of_turn,
        timing_ms=timing,
    )


def load_model(model_name: str, device: str = "cuda:0", dtype=torch.bfloat16):
    """Load tokenizer + model; probe tool-role support and return both plus the flag."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    model.eval()
    use_tool_role = _supports_tool_role(tokenizer)
    return model, tokenizer, use_tool_role


def save_trajectory(traj: Trajectory, out_dir: Path) -> Path:
    safe_model = traj.model_name.replace("/", "__")
    safe_id = f"{safe_model}_seed{traj.seed}_{traj.base_prompt_id}_{traj.outcome}.json"
    path = out_dir / safe_id
    with path.open("w") as f:
        json.dump(traj.to_jsonable(), f)
    return path
