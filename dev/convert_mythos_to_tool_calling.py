#!/usr/bin/env python3
"""
Convert mythos_combined_sft conversations from the external tool_call/tool
format into mesosfer's native multi-part assistant format that uses the
generic tool-calling tokens <|tool_start|> / <|tool_end|> for the call and
<|output_start|> / <|output_end|> for the result.

CLI tools (bash, grep, nmap, tshark, ...) are shell commands, not Python, so
they are represented as generic named tool calls rather than being wrapped in
a Python subprocess block.

Input format (mythos):
    user → assistant (with <tool_call>) → tool → assistant (with <tool_call>) → tool → assistant

Output format (mesosfer native):
    user → assistant [
        {type: "text",        text: "<thinking>...</thinking>\n"},
        {type: "tool",        text: '{"name": "shell", "arguments": {"command": "nmap -sV ..."}}'},
        {type: "tool_output", text: "command output"},
        {type: "text",        text: "<thinking>...</thinking>\n"},
        {type: "tool",        text: '{"name": "shell", "arguments": {"command": "..."}}'},
        {type: "tool_output", text: "output"},
        {type: "text",        text: "final answer"}
    ]

This collapses the multi-role chain into proper user/assistant alternation
while preserving the full tool-use flow for training.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SFT_DIR = REPO_ROOT / "data" / "sft"


def _extract_tool_command(text: str) -> tuple[str, str]:
    """Extract the thinking text and a shell command from a <tool_call> block.

    Returns (pre_text, command) where command is the reconstructed CLI command
    line. The caller wraps it in a generic tool call
    ``{"name": "shell", "arguments": {"command": command}}``.
    """
    # Split on the <tool_call> block
    parts = re.split(r"<tool_call>\s*", text, maxsplit=1)
    pre_text = parts[0].rstrip()

    if len(parts) < 2:
        return pre_text, ""

    # Parse the tool_call JSON
    call_block = parts[1]
    call_block = re.sub(r"\s*</tool_call>\s*$", "", call_block).strip()

    try:
        tool_call = json.loads(call_block)
    except json.JSONDecodeError:
        return pre_text, ""

    name = tool_call.get("name", "bash")
    args = tool_call.get("arguments", "")

    # Reconstruct a single shell command line from the tool call.
    if name == "bash" and isinstance(args, str):
        command = args.strip()
    elif isinstance(args, list):
        # Command name + argument list (e.g. ["grep", "-i", "pattern", "file"])
        command = " ".join([name] + [str(a) for a in args]).strip()
    elif isinstance(args, str):
        command = f"{name} {args}".strip()
    else:
        command = str(name).strip()

    return pre_text, command


def _tool_part(command: str) -> dict:
    """Wrap a shell command in the generic tool-call part format."""
    call = {"name": "shell", "arguments": {"command": command}}
    return {"type": "tool", "text": json.dumps(call, ensure_ascii=False)}


def convert_conversation(messages: list[dict]) -> list[dict] | None:
    """Convert a single mythos conversation to mesosfer native format.

    Returns None if the conversation doesn't contain tool calls (already
    compatible) or cannot be converted.
    """
    has_tool = any(m.get("role") == "tool" for m in messages)
    if not has_tool:
        return None  # Already compatible, skip

    converted = []
    i = 0
    while i < len(messages):
        msg = messages[i]

        if msg["role"] == "user":
            converted.append({"role": "user", "content": msg["content"]})
            i += 1

        elif msg["role"] == "assistant":
            # Collect the assistant response chain:
            # assistant (with tool_call) → tool → assistant → tool → ... → assistant (final)
            parts = []

            while i < len(messages) and messages[i]["role"] in ("assistant", "tool"):
                current = messages[i]

                if current["role"] == "assistant":
                    content = current["content"]
                    if "<tool_call>" in content:
                        # This assistant message contains a tool call
                        pre_text, command = _extract_tool_command(content)
                        if pre_text:
                            cleaned_think = re.sub(r"^\s*<thinking>\s*|\s*</thinking>\s*$", "", pre_text, flags=re.S).strip()
                            if cleaned_think:
                                parts.append({"type": "think", "text": cleaned_think})
                        if command:
                            parts.append(_tool_part(command))
                    else:
                        # Final assistant text (no tool call)
                        if content.strip():
                            m_think = re.match(r"^\s*<thinking>\s*(.*?)\s*</thinking>\s*(.+)$", content.strip(), flags=re.S)
                            if m_think and m_think.group(1).strip() and m_think.group(2).strip():
                                parts.append({"type": "think", "text": m_think.group(1).strip()})
                                parts.append({"type": "text", "text": m_think.group(2).strip()})
                            else:
                                parts.append({"type": "text", "text": content.strip()})
                    i += 1

                elif current["role"] == "tool":
                    # Tool output
                    parts.append({"type": "tool_output", "text": current["content"]})
                    i += 1
            
            if parts:
                converted.append({"role": "assistant", "content": parts})

        else:
            # Skip unknown roles
            i += 1

    # Validate: must start with user and alternate
    if not converted or converted[0]["role"] != "user":
        return None
    for j, m in enumerate(converted):
        expected = "user" if j % 2 == 0 else "assistant"
        if m["role"] != expected:
            return None

    return converted


def process_file(input_path: Path, output_path: Path) -> tuple[int, int, int]:
    """Process a single JSONL file. Returns (total, converted, kept_as_is)."""
    rows = []
    total = 0
    converted_count = 0
    kept_count = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            messages = json.loads(line)

            result = convert_conversation(messages)
            if result is not None:
                rows.append(result)
                converted_count += 1
            else:
                # Already compatible (no tool role) — keep as-is
                rows.append(messages)
                kept_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return total, converted_count, kept_count


def main():
    parser = argparse.ArgumentParser(
        description="Convert mythos tool_call/tool format to mesosfer native <|tool_start|> format"
    )
    parser.add_argument("--sft-dir", type=Path, default=SFT_DIR)
    args = parser.parse_args()

    files = [
        ("mythos_combined_sft.jsonl", "mythos_tool_calling.jsonl"),
        ("mythos_combined_sft_en.jsonl", "mythos_tool_calling_en.jsonl"),
    ]

    for src_name, dst_name in files:
        src = args.sft_dir / src_name
        dst = args.sft_dir / dst_name
        if not src.exists():
            print(f"SKIP: {src} not found")
            continue
        total, converted, kept = process_file(src, dst)
        print(f"{src_name}: {total} total -> {converted} converted to native tool format (<|tool_start|>), {kept} kept as-is -> {dst}")


if __name__ == "__main__":
    main()
