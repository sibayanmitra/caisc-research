"""Deterministic mock-tool harness for the G3 experiment.

A tool call is dispatched as (tool_name, args_json_str, outcome_condition).
The harness returns a byte-deterministic response selected by hashing the inputs
against a per-condition response table. Determinism is the central invariant: the
same (tool, args, condition, condition_seed) MUST return identical bytes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

OUTCOMES = ("success", "rate_limited", "no_results", "invalid_arg")
TOOLS = ("web_search", "calculator", "file_read")


@dataclass(frozen=True)
class ToolCall:
    tool: str
    args: dict
    outcome: str


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    payload: str
    error_kind: str | None  # None iff ok=True


def _hash_index(tool: str, args_json: str, outcome: str, condition_seed: int, n: int) -> int:
    """SHA-256-based deterministic index into a response bank of size n."""
    blob = f"{tool}|{args_json}|{outcome}|{condition_seed}".encode()
    digest = hashlib.sha256(blob).digest()
    return int.from_bytes(digest[:8], "big") % n


_WEB_SUCCESS = [
    "Top result: {query} — Wikipedia. Summary: a well-documented topic with multiple sources.",
    "Top result: {query} — Encyclopedia entry. Brief overview available.",
    "Top result: {query} — Recent news article. Coverage from 2025.",
    "Top result: {query} — Reference document. Detailed background provided.",
]
_WEB_RATE = ["HTTP 429 Too Many Requests. Retry-After: 30s."]
_WEB_NORES = ["No results found for the given query."]
_WEB_INVARG = [
    "Error: query must be a non-empty string.",
    "Error: query parameter missing.",
]

_CALC_SUCCESS_TEMPLATES = [
    "Result: {value}",
    "= {value}",
]
_CALC_RATE = ["Error: calculator service unavailable (rate limited)."]
_CALC_NORES = ["Error: expression evaluated but produced no numeric result."]
_CALC_INVARG = [
    "Error: invalid expression — syntax error at position 0.",
    "Error: unsupported operator in expression.",
    "Error: expression contains undefined symbol.",
]

_FILE_SUCCESS = [
    "File contents (truncated): Lorem ipsum dolor sit amet, consectetur adipiscing elit. {path}",
    "File contents (truncated): Project README. {path} — last modified 2026-04-01.",
    "File contents: short notes file. Path={path}",
]
_FILE_RATE = ["Error: filesystem busy, try again."]
_FILE_NORES = ["Error: file not found at path: {path}"]
_FILE_INVARG = ["Error: path must be absolute and start with '/'."]


def _eval_calc(expression: str) -> str | None:
    """Best-effort sandbox eval of arithmetic expressions for the 'success' branch."""
    safe = expression.strip()
    if not safe:
        return None
    allowed = set("0123456789+-*/(). ")
    if any(c not in allowed for c in safe):
        return None
    try:
        return str(eval(safe, {"__builtins__": {}}, {}))  # noqa: S307 — sandbox above
    except Exception:
        return None


def call_tool(tc: ToolCall, condition_seed: int = 0) -> ToolResult:
    """Deterministically resolve a mock tool call.

    The dispatch is content-addressed; the same inputs always yield the same output.
    """
    if tc.tool not in TOOLS:
        return ToolResult(ok=False, payload=f"Error: unknown tool '{tc.tool}'.", error_kind="invalid_arg")
    if tc.outcome not in OUTCOMES:
        raise ValueError(f"unknown outcome: {tc.outcome}")
    args_json = json.dumps(tc.args, sort_keys=True, separators=(",", ":"))

    if tc.tool == "web_search":
        query = str(tc.args.get("query", ""))
        if tc.outcome == "success":
            tmpl = _WEB_SUCCESS[_hash_index(tc.tool, args_json, tc.outcome, condition_seed, len(_WEB_SUCCESS))]
            return ToolResult(ok=True, payload=tmpl.format(query=query[:80]), error_kind=None)
        if tc.outcome == "rate_limited":
            return ToolResult(ok=False, payload=_WEB_RATE[0], error_kind="rate_limited")
        if tc.outcome == "no_results":
            return ToolResult(ok=False, payload=_WEB_NORES[0], error_kind="no_results")
        if tc.outcome == "invalid_arg":
            tmpl = _WEB_INVARG[_hash_index(tc.tool, args_json, tc.outcome, condition_seed, len(_WEB_INVARG))]
            return ToolResult(ok=False, payload=tmpl, error_kind="invalid_arg")

    if tc.tool == "calculator":
        expr = str(tc.args.get("expression", ""))
        if tc.outcome == "success":
            value = _eval_calc(expr) or "42"
            tmpl = _CALC_SUCCESS_TEMPLATES[_hash_index(tc.tool, args_json, tc.outcome, condition_seed, len(_CALC_SUCCESS_TEMPLATES))]
            return ToolResult(ok=True, payload=tmpl.format(value=value), error_kind=None)
        if tc.outcome == "rate_limited":
            return ToolResult(ok=False, payload=_CALC_RATE[0], error_kind="rate_limited")
        if tc.outcome == "no_results":
            return ToolResult(ok=False, payload=_CALC_NORES[0], error_kind="no_results")
        if tc.outcome == "invalid_arg":
            tmpl = _CALC_INVARG[_hash_index(tc.tool, args_json, tc.outcome, condition_seed, len(_CALC_INVARG))]
            return ToolResult(ok=False, payload=tmpl, error_kind="invalid_arg")

    if tc.tool == "file_read":
        path = str(tc.args.get("path", ""))
        if tc.outcome == "success":
            tmpl = _FILE_SUCCESS[_hash_index(tc.tool, args_json, tc.outcome, condition_seed, len(_FILE_SUCCESS))]
            return ToolResult(ok=True, payload=tmpl.format(path=path[:80]), error_kind=None)
        if tc.outcome == "rate_limited":
            return ToolResult(ok=False, payload=_FILE_RATE[0], error_kind="rate_limited")
        if tc.outcome == "no_results":
            return ToolResult(ok=False, payload=_FILE_NORES[0].format(path=path[:80]), error_kind="no_results")
        if tc.outcome == "invalid_arg":
            return ToolResult(ok=False, payload=_FILE_INVARG[0], error_kind="invalid_arg")

    raise RuntimeError("unreachable")
