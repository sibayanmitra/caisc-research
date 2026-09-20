"""Strict label rubric and regex tool-call parser.

We enforce a fixed output grammar via the system prompt:

    <tool_call>
    {"name": "<tool>", "arguments": {...}}
    </tool_call>

or for xLAM-2, bare JSON array:

    [{"name": "<tool>", "arguments": {...}}]

or, on give-up:

    <final_answer>
    free text
    </final_answer>

Parsing is regex-only (no LLM judge). Anything that doesn't conform is `ambiguous`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

LABEL_RETRY = "retry-same"
LABEL_MODIFY = "modify-args"
LABEL_GIVEUP = "give-up"
LABEL_AMBIG = "ambiguous"
LABELS = (LABEL_RETRY, LABEL_MODIFY, LABEL_GIVEUP, LABEL_AMBIG)

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)
# xLAM-2 emits bare JSON arrays: [{"name": "...", "arguments": {...}}]
TOOL_CALL_RE_JSON = re.compile(
    r"\[\s*(\{.*?\})\s*\]",
    re.DOTALL,
)
FINAL_ANSWER_RE = re.compile(
    r"<final_answer>\s*(.*?)\s*</final_answer>",
    re.DOTALL,
)


@dataclass(frozen=True)
class ParsedCall:
    name: str
    args: dict
    raw: str


def _normalize_args(args: dict) -> str:
    """Whitespace-normalized canonical JSON for token-identical comparison."""
    return json.dumps(args, sort_keys=True, separators=(",", ":")).strip()


def _parse_call_obj(obj: object, raw: str) -> ParsedCall | None:
    """Validate a parsed JSON object as a tool call dict; return ParsedCall or None."""
    if not isinstance(obj, dict) or "name" not in obj or "arguments" not in obj:
        return None
    if not isinstance(obj["arguments"], dict):
        return None
    return ParsedCall(name=str(obj["name"]), args=dict(obj["arguments"]), raw=raw)


def find_tool_calls(text: str) -> list[ParsedCall]:
    """Extract all syntactically-valid tool calls from a model continuation.

    Supports both <tool_call>{...}</tool_call> (G3 models) and
    [{...}] JSON array (xLAM-2) formats.
    """
    out: list[ParsedCall] = []
    # G3 format: <tool_call>{...}</tool_call>
    for m in TOOL_CALL_RE.finditer(text):
        blob = m.group(1)
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        pc = _parse_call_obj(obj, m.group(0))
        if pc is not None:
            out.append(pc)
    if out:
        return out
    # xLAM-2 format: [{"name": ..., "arguments": ...}]
    for m in TOOL_CALL_RE_JSON.finditer(text):
        blob = m.group(1)
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        pc = _parse_call_obj(obj, m.group(0))
        if pc is not None:
            out.append(pc)
    return out


def has_final_answer(text: str) -> bool:
    return bool(FINAL_ANSWER_RE.search(text))


KNOWN_TOOLS = ("web_search", "calculator", "file_read")
GIVE_UP_PHRASES = (
    "i cannot",
    "i can't",
    "i'm unable",
    "i am unable",
    "i don't have access",
    "i do not have access",
    "i'm sorry",
    "unable to",
)


def classify_next_action(
    continuation: str,
    failed_call: ParsedCall,
    max_tokens_seen: int,
    end_of_turn: bool,
) -> str:
    """Classify the model's next action after a failed tool call.

    Apply rules in order, first match wins (per preregistration §5).
    """
    calls = find_tool_calls(continuation)

    if calls:
        first = calls[0]
        if first.name not in KNOWN_TOOLS:
            return LABEL_AMBIG
        if first.name == failed_call.name and _normalize_args(first.args) == _normalize_args(failed_call.args):
            return LABEL_RETRY
        return LABEL_MODIFY

    text_lower = continuation.lower()
    has_final = has_final_answer(continuation)
    has_phrase = any(p in text_lower for p in GIVE_UP_PHRASES)

    if has_final or has_phrase or end_of_turn:
        return LABEL_GIVEUP

    return LABEL_AMBIG


def find_first_tool_call_offset(text: str) -> int | None:
    """Return character offset of the first tool call (either format), or None."""
    m1 = TOOL_CALL_RE.search(text)
    m2 = TOOL_CALL_RE_JSON.search(text)
    candidates = [m.start() for m in (m1, m2) if m is not None]
    return min(candidates) if candidates else None


def find_first_final_answer_offset(text: str) -> int | None:
    m = FINAL_ANSWER_RE.search(text)
    return m.start() if m else None
