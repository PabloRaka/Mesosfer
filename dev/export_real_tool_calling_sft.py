#!/usr/bin/env python3
"""
Convert real pair-programming agentic tool traces into Mesosfer's native multipart SFT format:
[
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": [
    {"type": "text", "text": "<thinking>\n...\n</thinking>\n"},
    {"type": "tool", "text": "{\"name\": \"terminal\", \"arguments\": {\"command\": \"...\"}}"},
    {"type": "tool_output", "text": "..."},
    {"type": "text", "text": "..."}
  ]}
]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BRAIN_DIR = Path(r"C:\Users\Lenovo\.gemini\antigravity-ide\brain")
DEFAULT_OUTPUT = REPO_ROOT / "data" / "sft" / "real_agentic_tool_calling_sft.jsonl"


def scrub_secrets(text: str) -> str:
    """Scrub potential API keys, access tokens, credentials, and Discord/Telegram tokens."""
    # Discord bot tokens
    text = re.sub(r"[A-Za-z0-9_-]{24}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27,}", "DISCORD_TOKEN_REDACTED", text)
    text = re.sub(r"(DISCORD_BOT_TOKEN|DISCORD_TOKEN|BOT_TOKEN)\s*=\s*[^\s\n\"']+", r"\1=REDACTED_DISCORD_TOKEN", text, flags=re.IGNORECASE)
    text = re.sub(r"https://discord\.com/api/webhooks/[0-9]+/[A-Za-z0-9_-]+", "https://discord.com/api/webhooks/REDACTED", text)
    # Telegram tokens
    text = re.sub(r"[0-9]{8,10}:[a-zA-Z0-9_-]{35}", "TELEGRAM_TOKEN_REDACTED", text)
    # Hugging Face tokens
    text = re.sub(r"hf_[A-Za-z0-9_]{20,}", "hf_REDACTED_HF_TOKEN", text)
    # Postman API keys
    text = re.sub(r"PMAK-[A-Za-z0-9\-]{20,}", "PMAK-REDACTED_POSTMAN_KEY", text)
    # Alibaba Cloud AccessKey
    text = re.sub(r"LTAI[A-Za-z0-9]{12,30}", "LTAI_REDACTED_ACCESS_KEY", text)
    # AWS AccessKey
    text = re.sub(r"AKIA[0-9A-Z]{16}", "AKIA_REDACTED_AWS_KEY", text)
    # GitHub Tokens
    text = re.sub(r"gh[pors]_[A-Za-z0-9_]{30,}", "ghp_REDACTED_GITHUB_TOKEN", text)
    text = re.sub(r"github_pat_[A-Za-z0-9_]{30,}", "github_pat_REDACTED_TOKEN", text)
    # OpenAI / Anthropic / Generic API keys
    text = re.sub(r"sk-[A-Za-z0-9_\-]{20,}", "sk-REDACTED_API_KEY", text)
    # Generic bearer tokens in headers
    text = re.sub(r"(Authorization:\s*Bearer\s+)[^\s\n\"']+", r"\1REDACTED_TOKEN", text, flags=re.IGNORECASE)
    # Generic private key blocks
    text = re.sub(r"-----BEGIN (RSA|EC|OPENSSH|DSA|PRIVATE) KEY-----[\s\S]*?-----END \1 KEY-----", "----BEGIN PRIVATE KEY-----\n[REDACTED_KEY]\n-----END PRIVATE KEY-----", text)
    # Generic password assignments
    text = re.sub(r"(password\s*[:=]\s*['\"])[^'\"]+(['\"])", r"\1REDACTED_PASSWORD\2", text, flags=re.IGNORECASE)
    return text


def clean_ansi(text: str) -> str:
    """Remove ANSI escape sequences."""
    text = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)
    text = re.sub(r"\[\[\d+m[A-Z\s]+\[\d+m\]", "", text)
    text = re.sub(r"\[\d+m\]", "", text)
    return text


def sanitize_path(path_str: str) -> str:
    """Normalize Windows paths to generic project paths."""
    if not path_str:
        return ""
    p = path_str.replace("\\", "/")
    p = re.sub(r"^[A-Za-z]:/Users/[^/]+/(Documents/)?(projects/)?", "", p)
    return p.strip()


def convert_tool_call(tool_call: dict) -> tuple[str, dict] | None:
    """Map IDE tool call to Mesosfer native sandbox tool (terminal, subagent, etc.)."""
    name = tool_call.get("name")
    args = tool_call.get("arguments", {})

    if name == "run_command":
        cmd = args.get("CommandLine", "").strip()
        if not cmd or len(cmd) < 2:
            return None
        cmd = scrub_secrets(cmd.replace("\\", "/"))
        if "Get-ChildItem" in cmd:
            cmd = re.sub(r"Get-ChildItem\s+([^\s]+)", r"ls -la \1", cmd)
        return "terminal", {"command": cmd}

    elif name == "view_file":
        raw_path = args.get("AbsolutePath", "").strip()
        if not raw_path:
            return None
        path = sanitize_path(raw_path)
        if not path:
            return None
        start_line = args.get("StartLine")
        end_line = args.get("EndLine")
        if start_line and end_line:
            cmd = f"sed -n '{start_line},{end_line}p' {path}"
        else:
            cmd = f"cat {path}"
        return "terminal", {"command": cmd}

    elif name == "grep_search":
        query = args.get("Query", "").strip()
        if not query:
            return None
        search_path = sanitize_path(args.get("SearchPath", ".")) or "."
        cmd = f"grep -rn '{query}' {search_path}"
        return "terminal", {"command": cmd}

    elif name == "list_dir":
        dir_path = sanitize_path(args.get("DirectoryPath", ".")) or "."
        cmd = f"ls -la {dir_path}"
        return "terminal", {"command": cmd}

    return None


def clean_tool_output(raw_output: str) -> str:
    """Clean and truncate tool output to safe length."""
    if not raw_output:
        return "Success (0)"

    text = clean_ansi(raw_output.strip())
    text = re.sub(r"^Created At:.*?\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"^Completed At:.*?\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"The following code has been modified to include a line number.*?\n", "", text)
    text = scrub_secrets(text.strip())

    if len(text) > 1000:
        text = text[:1000] + "\n... [output truncated]"

    return text if text else "Success (0)"


def extract_tool_conversations(tpath: Path) -> list[list[dict]]:
    """Extract full multipart tool calling conversations with strict alternating pair constraints."""
    try:
        with tpath.open("r", encoding="utf-8", errors="ignore") as f:
            lines = [json.loads(line) for line in f]
    except Exception:
        return []

    conversations = []
    i = 0
    n = len(lines)

    while i < n:
        step = lines[i]
        if step.get("type") == "USER_INPUT":
            content = step.get("content", "")
            u_match = re.search(r"<USER_REQUEST>(.*?)</USER_REQUEST>", content, re.DOTALL)
            user_text = u_match.group(1).strip() if u_match else content.strip()

            if "CHECKPOINT" in user_text:
                parts = re.split(r"<USER_REQUEST>", user_text)
                if len(parts) > 1:
                    user_text = parts[-1].replace("</USER_REQUEST>", "").strip()
                else:
                    i += 1
                    continue

            user_text = clean_ansi(user_text)
            if len(user_text) < 6 or user_text.startswith("@[TerminalName"):
                i += 1
                continue

            j = i + 1
            parts = []
            conclusion_text = ""
            pending_tool = False

            while j < n and lines[j].get("type") != "USER_INPUT":
                curr = lines[j]
                ctype = curr.get("type")

                if ctype == "PLANNER_RESPONSE":
                    p_content = curr.get("content", "").strip()
                    tool_calls = curr.get("tool_calls", [])

                    if tool_calls and not pending_tool:
                        if not parts:
                            thinking = p_content if (p_content and not p_content.startswith("Created At:")) else "Saya akan memeriksa sistem dan menjalankan perintah yang diperlukan."
                            parts.append({"type": "text", "text": f"<thinking>\n{thinking}\n</thinking>\n"})

                        for tc in tool_calls:
                            mapped = convert_tool_call(tc)
                            if mapped:
                                t_name, t_args = mapped
                                payload = json.dumps({"name": t_name, "arguments": t_args}, ensure_ascii=False)
                                parts.append({"type": "tool", "text": payload})
                                pending_tool = True
                                break

                    elif not tool_calls:
                        if p_content and not p_content.startswith("Created At:"):
                            conclusion_text = p_content

                elif ctype in ("RUN_COMMAND", "VIEW_FILE", "GREP_SEARCH", "LIST_DIR"):
                    if pending_tool:
                        out_text = clean_tool_output(curr.get("content", ""))
                        parts.append({"type": "tool_output", "text": out_text})
                        pending_tool = False

                j += 1

            if parts and conclusion_text and not pending_tool:
                valid_sequence = True
                for idx in range(1, len(parts), 2):
                    if parts[idx]["type"] != "tool":
                        valid_sequence = False
                        break
                    if idx + 1 < len(parts) and parts[idx + 1]["type"] != "tool_output":
                        valid_sequence = False
                        break

                if valid_sequence and len(parts) >= 3:
                    parts.append({"type": "text", "text": scrub_secrets(clean_ansi(conclusion_text))})
                    conv = [
                        {"role": "user", "content": scrub_secrets(user_text)},
                        {"role": "assistant", "content": parts}
                    ]
                    conversations.append(conv)

            i = j
        else:
            i += 1

    return conversations


def main() -> None:
    parser = argparse.ArgumentParser(description="Export real agentic tool traces into Mesosfer multipart SFT format")
    parser.add_argument("--brain-dir", type=Path, default=BRAIN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-rows", type=int, default=1500)
    args = parser.parse_args()

    if not args.brain_dir.exists():
        print(f"Brain dir {args.brain_dir} not found.")
        return

    transcripts = list(args.brain_dir.glob("*/.system_generated/logs/transcript.jsonl"))
    print(f"Scanning {len(transcripts)} transcripts for agentic tool traces...")

    seen_hashes = set()
    all_conversations = []

    for tpath in transcripts:
        convs = extract_tool_conversations(tpath)
        for c in convs:
            u_text = c[0]["content"]
            h = hashlib.sha256(u_text.encode("utf-8")).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            all_conversations.append(c)
            if len(all_conversations) >= args.max_rows:
                break
        if len(all_conversations) >= args.max_rows:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for c in all_conversations:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Successfully extracted {len(all_conversations)} strictly-validated agentic tool conversations into {args.output}")


if __name__ == "__main__":
    main()
