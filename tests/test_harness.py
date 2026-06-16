"""
Tests for the agentic tool-call harness (mesosfer/eval/harness.py).

These use a scripted fake engine + fake tokenizer so the full loop is exercised
without a real model or GPU. They verify:
- tool-call JSON parsing (valid / malformed)
- the defensive registry (unknown tool -> recoverable error, no execution)
- the agentic loop: generate -> execute -> feed <|output_*|> back -> resume
- handler exceptions surface as errors instead of crashing
- the max_tool_calls budget terminates the loop
"""

import pytest

from mesosfer.eval.harness import (
    ToolHarness, ToolCall, parse_tool_call, run_agentic, _extract_last_tool_payload,
)


# --- Fake tokenizer: reversible char codec + fixed special-token ids -----------
_SPECIALS = {
    "<|bos|>": 1000,
    "<|tool_start|>": 1001,
    "<|tool_end|>": 1002,
    "<|output_start|>": 1003,
    "<|output_end|>": 1004,
    "<|assistant_start|>": 1005,
    "<|assistant_end|>": 1006,
}
_ID2SPECIAL = {v: k for k, v in _SPECIALS.items()}


class FakeTokenizer:
    def encode_special(self, s):
        return _SPECIALS[s]

    def get_bos_token_id(self):
        return _SPECIALS["<|bos|>"]

    def encode(self, text):
        return [2000 + ord(c) for c in text]

    def decode(self, ids):
        out = []
        for i in ids:
            if i in _ID2SPECIAL:
                out.append(_ID2SPECIAL[i])
            elif i >= 2000:
                out.append(chr(i - 2000))
        return "".join(out)

    def render_for_completion(self, conversation):
        user = [m for m in conversation["messages"] if m["role"] == "user"][-1]["content"]
        return [self.get_bos_token_id()] + self.encode(user) + [_SPECIALS["<|assistant_start|>"]]


class FakeEngine:
    """Returns scripted token segments appended to the input prompt, one per call."""
    def __init__(self, segments):
        self.segments = [list(s) for s in segments]
        self.calls = 0

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        seg = self.segments[self.calls] if self.calls < len(self.segments) else []
        self.calls += 1
        seq = list(tokens) + list(seg)
        return [seq], [[0] * len(seq)]


def _tok():
    return FakeTokenizer()


def _tool_call_segment(tok, payload: str):
    return [_SPECIALS["<|tool_start|>"]] + tok.encode(payload) + [_SPECIALS["<|tool_end|>"]]


# --- parse_tool_call ----------------------------------------------------------

def test_parse_valid_tool_call():
    call = parse_tool_call('{"name": "shell", "arguments": {"command": "ls"}}')
    assert call.name == "shell"
    assert call.arguments == {"command": "ls"}


@pytest.mark.parametrize("bad", [
    "not json at all",
    "[1, 2, 3]",                       # not a dict
    '{"arguments": {"x": 1}}',         # missing name
    '{"name": 5, "arguments": {}}',    # name not a string
    '{"name": "x", "arguments": 7}',   # arguments not a dict
    "",
])
def test_parse_malformed_tool_call(bad):
    call = parse_tool_call(bad)
    assert call.name is None


# --- ToolHarness registry -----------------------------------------------------

def test_registry_unknown_tool_is_recoverable_error():
    h = ToolHarness()
    result, ok = h.execute("nmap", {"target": "10.0.0.1"})
    assert ok is False and "not available" in result


def test_registry_executes_registered_tool():
    h = ToolHarness()
    h.register("echo", lambda args: args.get("msg", ""))
    result, ok = h.execute("echo", {"msg": "hello"})
    assert ok is True and result == "hello"


def test_registry_handler_exception_is_caught():
    h = ToolHarness()
    def boom(args):
        raise RuntimeError("kaboom")
    h.register("boom", boom)
    result, ok = h.execute("boom", {})
    assert ok is False and "kaboom" in result


def test_malformed_call_name_none_is_error():
    h = ToolHarness()
    result, ok = h.execute(None, {})
    assert ok is False and "malformed" in result


# --- _extract_last_tool_payload ----------------------------------------------

def test_extract_payload_roundtrip():
    tok = _tok()
    payload = '{"name": "echo", "arguments": {"msg": "hi"}}'
    seq = [1000] + tok.encode("prefix ") + _tool_call_segment(tok, payload)
    assert _extract_last_tool_payload(seq, 1001, 1002, tok) == payload


def test_extract_payload_requires_trailing_tool_end():
    tok = _tok()
    seq = [1000] + tok.encode("no tool call here")
    assert _extract_last_tool_payload(seq, 1001, 1002, tok) is None


# --- run_agentic full loop ----------------------------------------------------

def test_run_agentic_executes_tool_then_finishes():
    tok = _tok()
    conv = {"messages": [{"role": "user", "content": "scan"}]}
    # Turn 1: a tool call. Turn 2: plain finishing text (no tool_end).
    engine = FakeEngine([
        tok.encode("checking. ") + _tool_call_segment(tok, '{"name": "echo", "arguments": {"msg": "PORT22"}}'),
        tok.encode("done: PORT22 open"),
    ])
    h = ToolHarness()
    h.register("echo", lambda args: args["msg"])

    res = run_agentic(engine, tok, conv, h, max_tool_calls=8, max_tokens=64)

    assert res.stop_reason == "assistant_end"
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].name == "echo" and res.tool_calls[0].ok
    assert res.tool_calls[0].result == "PORT22"
    # The fed-back tool output must appear in the assistant transcript.
    assert "<|output_start|>PORT22<|output_end|>" in res.text
    assert "done: PORT22 open" in res.text


def test_run_agentic_unknown_tool_recovers():
    tok = _tok()
    conv = {"messages": [{"role": "user", "content": "do it"}]}
    engine = FakeEngine([
        _tool_call_segment(tok, '{"name": "rm_rf", "arguments": {}}'),
        tok.encode("ok, that tool is unavailable; stopping"),
    ])
    h = ToolHarness()  # empty: nothing executable
    res = run_agentic(engine, tok, conv, h, max_tool_calls=8, max_tokens=64)
    assert res.stop_reason == "assistant_end"
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].ok is False
    assert "not available" in res.tool_calls[0].result


def test_run_agentic_respects_tool_call_budget():
    tok = _tok()
    conv = {"messages": [{"role": "user", "content": "loop"}]}
    # Every segment is another tool call -> would loop forever without the budget.
    call_seg = _tool_call_segment(tok, '{"name": "echo", "arguments": {"msg": "x"}}')
    engine = FakeEngine([call_seg] * 10)  # plenty of tool calls
    h = ToolHarness()
    h.register("echo", lambda args: args["msg"])

    res = run_agentic(engine, tok, conv, h, max_tool_calls=3, max_tokens=64)
    assert res.stop_reason == "max_tool_calls"
    assert len(res.tool_calls) == 3  # exactly the budget
