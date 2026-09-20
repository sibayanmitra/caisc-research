"""Pre-flight: mock-tool byte-determinism gate."""

from src.mock_tools import ToolCall, call_tool, TOOLS, OUTCOMES


def test_byte_determinism_100_calls():
    """Same (tool, args, outcome, seed) MUST return identical bytes across 100 calls."""
    tc = ToolCall(tool="web_search", args={"query": "what is the capital of France"}, outcome="success")
    ref = call_tool(tc, condition_seed=7)
    for _ in range(100):
        out = call_tool(tc, condition_seed=7)
        assert out.payload == ref.payload, "mock-tool non-determinism detected"
        assert out.ok == ref.ok
        assert out.error_kind == ref.error_kind


def test_outcome_changes_seed_invariant():
    """Different condition_seeds may vary the response payload but stay within the outcome class."""
    tc = ToolCall(tool="calculator", args={"expression": "2+2"}, outcome="invalid_arg")
    s0 = call_tool(tc, condition_seed=0)
    s1 = call_tool(tc, condition_seed=1)
    assert s0.ok == s1.ok == False
    assert s0.error_kind == s1.error_kind == "invalid_arg"


def test_all_outcomes_resolve():
    """Every (tool, outcome) combo must return without error and respect the ok/error_kind invariants."""
    for tool in TOOLS:
        if tool == "web_search":
            args = {"query": "x"}
        elif tool == "calculator":
            args = {"expression": "1+1"}
        else:
            args = {"path": "/tmp/x"}
        for outcome in OUTCOMES:
            r = call_tool(ToolCall(tool=tool, args=args, outcome=outcome), condition_seed=0)
            if outcome == "success":
                assert r.ok and r.error_kind is None
            else:
                assert not r.ok and r.error_kind == outcome


def test_calculator_success_actually_computes():
    r = call_tool(ToolCall(tool="calculator", args={"expression": "2+2"}, outcome="success"), 0)
    assert "4" in r.payload
