"""Pre-flight: label rubric correctness tests."""

from src.label import (
    LABEL_AMBIG,
    LABEL_GIVEUP,
    LABEL_MODIFY,
    LABEL_RETRY,
    ParsedCall,
    classify_next_action,
    find_tool_calls,
    has_final_answer,
)


FAILED = ParsedCall(name="web_search", args={"query": "capital of France"}, raw="")


def _label(text: str, end_of_turn: bool = False) -> str:
    return classify_next_action(text, FAILED, max_tokens_seen=256, end_of_turn=end_of_turn)


def test_retry_same_exact_args():
    text = """Let me try once more.
    <tool_call>
    {"name": "web_search", "arguments": {"query": "capital of France"}}
    </tool_call>"""
    assert _label(text) == LABEL_RETRY


def test_retry_same_arg_order_normalized():
    """Argument ordering must not change equivalence (sort_keys=True)."""
    failed = ParsedCall(name="web_search", args={"a": 1, "b": 2}, raw="")
    text = '<tool_call>{"name": "web_search", "arguments": {"b": 2, "a": 1}}</tool_call>'
    assert classify_next_action(text, failed, 256, False) == LABEL_RETRY


def test_modify_args_same_tool():
    text = """<tool_call>
    {"name": "web_search", "arguments": {"query": "France capital city Paris"}}
    </tool_call>"""
    assert _label(text) == LABEL_MODIFY


def test_modify_args_tool_switch():
    text = """<tool_call>
    {"name": "calculator", "arguments": {"expression": "1+1"}}
    </tool_call>"""
    assert _label(text) == LABEL_MODIFY


def test_give_up_final_answer():
    text = "<final_answer>I'm sorry, I cannot find that.</final_answer>"
    assert _label(text) == LABEL_GIVEUP


def test_give_up_phrase_no_tool():
    text = "I'm unable to find this information after multiple attempts."
    assert _label(text) == LABEL_GIVEUP


def test_give_up_end_of_turn():
    text = "Hmm, that didn't work. Let me think about this differently."
    assert _label(text, end_of_turn=True) == LABEL_GIVEUP


def test_ambiguous_reasoning_only():
    """Reasoning without a tool call AND without give-up signals → ambiguous."""
    text = "Hmm, let me consider what to try next. Maybe a different approach."
    assert _label(text, end_of_turn=False) == LABEL_AMBIG


def test_ambiguous_unknown_tool_first():
    text = '<tool_call>{"name": "unknown_tool", "arguments": {}}</tool_call>'
    assert _label(text) == LABEL_AMBIG


def test_intent_without_call_is_giveup_if_eot():
    """Per preregistration: intent statements ignored; only emitted calls count."""
    text = "I'll try again with different parameters."
    assert _label(text, end_of_turn=True) == LABEL_GIVEUP


def test_first_call_wins_when_multiple():
    """Multi-step plan: classify by FIRST emitted tool call."""
    text = """<tool_call>{"name": "web_search", "arguments": {"query": "capital of France"}}</tool_call>
    later... <tool_call>{"name": "calculator", "arguments": {"expression": "1+1"}}</tool_call>"""
    assert _label(text) == LABEL_RETRY


def test_find_tool_calls_count():
    text = '<tool_call>{"name": "a", "arguments": {}}</tool_call><tool_call>{"name": "b", "arguments": {}}</tool_call>'
    assert len(find_tool_calls(text)) == 2


def test_has_final_answer():
    assert has_final_answer("<final_answer>x</final_answer>")
    assert not has_final_answer("<tool_call>{}</tool_call>")


def test_malformed_json_is_ignored():
    text = '<tool_call>{not valid json</tool_call>'
    assert _label(text, end_of_turn=True) == LABEL_GIVEUP  # no valid call, EOT → give-up


# ── xLAM-2 JSON array format ────────────────────────────────────────────────

FAILED_CALC = ParsedCall(name="calculator", args={"expression": "2+2"}, raw="")


def test_xlam_retry_same():
    text = '[{"name": "calculator", "arguments": {"expression": "2+2"}}]'
    assert classify_next_action(text, FAILED_CALC, 256, False) == LABEL_RETRY


def test_xlam_modify_args():
    text = '[{"name": "calculator", "arguments": {"expression": "2+3"}}]'
    assert classify_next_action(text, FAILED_CALC, 256, False) == LABEL_MODIFY


def test_xlam_tool_switch():
    text = '[{"name": "web_search", "arguments": {"query": "two plus two"}}]'
    assert classify_next_action(text, FAILED_CALC, 256, False) == LABEL_MODIFY


def test_xlam_find_returns_parsed_call():
    text = '[{"name": "file_read", "arguments": {"path": "/etc/hosts"}}]'
    calls = find_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "file_read"
    assert calls[0].args == {"path": "/etc/hosts"}


def test_xml_format_takes_precedence_over_json_array():
    """If both formats appear, XML <tool_call> is parsed (comes first in code)."""
    text = (
        '<tool_call>{"name": "web_search", "arguments": {"query": "q"}}</tool_call>'
        '[{"name": "calculator", "arguments": {"expression": "1"}}]'
    )
    calls = find_tool_calls(text)
    assert calls[0].name == "web_search"
