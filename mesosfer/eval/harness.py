"""
Agentic tool-call harness.

The model only *emits* tool calls (generic named tools rendered between
``<|tool_start|>`` and ``<|tool_end|>`` as JSON ``{"name": ..., "arguments": {...}}``).
This harness is the component that actually executes them and feeds the result back
to the model as ``<|output_start|>...<|output_end|>``, then resumes generation. This
closes the agentic loop:

    assistant: ... <|tool_start|>{"name":"shell","arguments":{...}}<|tool_end|>
    harness:   executes the tool, produces a result string
    context:   <|output_start|>RESULT<|output_end|>
    assistant: ... (continues, may call another tool or finish) <|assistant_end|>

Defensive posture (see .kiro/steering/product.md):
- Nothing is executed unless an operator has EXPLICITLY registered a handler for that
  tool name. With an empty registry every tool call returns a safe "not available"
  error, which the model can read and recover from.
- Registering handlers that run shell / network commands is the operator's
  responsibility and should only be done inside a sandbox. This module never spawns
  processes itself.

The harness is intentionally token-level and model-agnostic: it drives an
``Engine`` (mesosfer.eval.engine.Engine) whose ``generate``/``generate_batch`` support
``stop_on_tool_call=True``.
"""

import json
from dataclasses import dataclass, field
from typing import Callable, Optional


# A tool handler takes the parsed ``arguments`` dict and returns a result string.
ToolHandler = Callable[[dict], str]


@dataclass
class ToolCall:
    """A single parsed tool call and (once executed) its result."""
    name: Optional[str]
    arguments: dict
    raw_text: str
    result: Optional[str] = None
    ok: bool = False


@dataclass
class AgenticResult:
    """Outcome of an agentic run."""
    tokens: list                       # full token sequence (prompt + assistant turn)
    text: str                          # decoded assistant turn
    tool_calls: list = field(default_factory=list)  # list[ToolCall], in order
    stop_reason: str = "assistant_end" # "assistant_end" | "max_tool_calls" | "no_tool_end"


class ToolHarness:
    """Registry of tool handlers. Empty by default (nothing is executable)."""

    def __init__(self):
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, name: str, handler: ToolHandler) -> None:
        """Register a handler for a named tool. handler(arguments: dict) -> str."""
        assert callable(handler), "handler must be callable"
        self._handlers[name] = handler

    def available(self) -> list:
        return sorted(self._handlers.keys())

    def execute(self, name: Optional[str], arguments: dict) -> tuple[str, bool]:
        """
        Execute a tool by name. Returns (result_text, ok).

        Unknown tools and handler exceptions are turned into ERROR strings rather than
        raising, so the model receives the failure as tool output and can self-correct.
        """
        if name is None:
            return ("ERROR: malformed tool call (expected JSON "
                    '{"name": ..., "arguments": {...}})', False)
        handler = self._handlers.get(name)
        if handler is None:
            avail = ", ".join(self.available()) or "(none registered)"
            return (f"ERROR: tool '{name}' is not available. Available tools: {avail}", False)
        try:
            return (str(handler(arguments)), True)
        except Exception as e:  # noqa: BLE001 - surface failure to the model, don't crash the loop
            return (f"ERROR: tool '{name}' failed: {type(e).__name__}: {e}", False)


def parse_tool_call(text: str) -> ToolCall:
    """
    Parse the JSON payload of a tool call (the text between <|tool_start|> and
    <|tool_end|>). Returns a ToolCall; on any parse problem name=None (the harness
    will turn that into a recoverable error for the model).
    """
    text = text.strip()
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            return ToolCall(name=None, arguments={}, raw_text=text)
        name = obj.get("name")
        arguments = obj.get("arguments", {})
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return ToolCall(name=None, arguments={}, raw_text=text)
        return ToolCall(name=name, arguments=arguments, raw_text=text)
    except (json.JSONDecodeError, ValueError, TypeError):
        return ToolCall(name=None, arguments={}, raw_text=text)


def _extract_last_tool_payload(tokens: list, tool_start: int, tool_end: int, tokenizer) -> Optional[str]:
    """Decode the token span between the last <|tool_start|> and the final <|tool_end|>."""
    if not tokens or tokens[-1] != tool_end:
        return None
    try:
        start = len(tokens) - 1 - tokens[::-1].index(tool_start)
    except ValueError:
        return None  # tool_end without a matching tool_start
    payload_ids = tokens[start + 1:len(tokens) - 1]  # between the markers, exclusive
    return tokenizer.decode(payload_ids)


def run_agentic(
    engine,
    tokenizer,
    conversation,
    harness: ToolHarness,
    max_tool_calls: int = 8,
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    seed: int = 42,
) -> AgenticResult:
    """
    Run a single-sample agentic rollout: generate, execute tool calls via ``harness``,
    feed results back, and resume until the assistant finishes or the tool-call budget
    is exhausted.

    ``conversation`` is a chat dict ({"messages": [...]}) ending with the user turn;
    it is rendered for completion (a trailing <|assistant_start|> is appended).
    """
    tool_start = tokenizer.encode_special("<|tool_start|>")
    tool_end = tokenizer.encode_special("<|tool_end|>")
    output_start = tokenizer.encode_special("<|output_start|>")
    output_end = tokenizer.encode_special("<|output_end|>")

    tokens = tokenizer.render_for_completion(conversation)
    prompt_len = len(tokens)
    tool_calls: list = []

    gen_kwargs = dict(num_samples=1, max_tokens=max_tokens, temperature=temperature,
                      top_k=top_k, seed=seed, stop_on_tool_call=True)

    for _ in range(max_tool_calls):
        results, _masks = engine.generate_batch(tokens, **gen_kwargs)
        tokens = results[0]
        # If generation did not stop on a tool call, the assistant turn is finished.
        if not tokens or tokens[-1] != tool_end:
            return AgenticResult(
                tokens=tokens,
                text=tokenizer.decode(tokens[prompt_len:]),
                tool_calls=tool_calls,
                stop_reason="assistant_end",
            )
        # A tool call is pending: parse, execute, and append the result as tool output.
        payload = _extract_last_tool_payload(tokens, tool_start, tool_end, tokenizer)
        call = parse_tool_call(payload if payload is not None else "")
        result_text, ok = harness.execute(call.name, call.arguments)
        call.result, call.ok = result_text, ok
        tool_calls.append(call)
        tokens = tokens + [output_start] + tokenizer.encode(result_text) + [output_end]

    # Tool-call budget exhausted: do one final unconstrained generation so the model can
    # wrap up with whatever it has, without being able to call more tools.
    results, _masks = engine.generate_batch(
        tokens, num_samples=1, max_tokens=max_tokens, temperature=temperature,
        top_k=top_k, seed=seed, stop_on_tool_call=False,
    )
    tokens = results[0]
    return AgenticResult(
        tokens=tokens,
        text=tokenizer.decode(tokens[prompt_len:]),
        tool_calls=tool_calls,
        stop_reason="max_tool_calls",
    )
