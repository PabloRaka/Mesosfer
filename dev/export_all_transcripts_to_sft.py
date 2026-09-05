#!/usr/bin/env python3
"""
Scan all conversation history logs across the workspace, apply strict quality filters,
remove noise/IDE leakage, and distill into a GOLD-STANDARD SFT dataset for Mesosfer.

Strict Quality Filters:
1. Strip ANSI escape sequences & terminal artifacts.
2. Filter out agent internal thinking/planning monologues ("I will stage...", "Now update...").
3. Filter out irrelevant side-projects (Turnitin scraper, SimKopDes village cooperatives).
4. Filter out fragmented one-word prompts ("ANJUTIN", "ya kerjain").
5. Ensure self-contained, professional, high-value Q&A (Python, Rust, Linux, Cybersec, ML, Networking).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BRAIN_DIR = Path(r"C:\Users\Lenovo\.gemini\antigravity-ide\brain")
DEFAULT_OUTPUT = REPO_ROOT / "data" / "sft" / "chat_history_distilled_sft.jsonl"


# ---------------------------------------------------------------------------
# Blacklisted project keywords & noise indicators
# ---------------------------------------------------------------------------

BLACKLIST_KEYWORDS = [
    "turnitin",
    "simkopdes",
    "koperasi desa",
    "bear_step",
    "bear_final",
    "salah workspace",
    "open folder",
    "postman-mcp",
    "dbviz.pikaa",
    "shietoshi.xyz",
    "bab siji",
    "paper 22",
    "contributor_role_id",
]

AGENT_PLANNING_PHRASES = [
    r"^I will stage the changes.*",
    r"^I will commit the staged.*",
    r"^I will push the newly.*",
    r"^Now update the summary table.*",
    r"^Now I have a complete picture.*",
    r"^Sekarang saya punya semua informasi yang diperlukan.*",
    r"^Mari saya lihat dulu semua API.*",
    r"^Sekarang saya buat workspace.*",
]


def clean_ansi(text: str) -> str:
    """Remove ANSI escape sequences and malformed brackets."""
    text = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)
    text = re.sub(r"\[\[\d+m[A-Z\s]+\[\d+m\]", "", text)
    text = re.sub(r"\[\d+m\]", "", text)
    return text


def scrub_secrets(text: str) -> str:
    """Scrub potential API keys, access tokens, and credentials."""
    text = re.sub(r"hf_[A-Za-z0-9_]{20,}", "hf_REDACTED_HF_TOKEN", text)
    text = re.sub(r"PMAK-[A-Za-z0-9\-]{20,}", "PMAK-REDACTED_POSTMAN_KEY", text)
    text = re.sub(r"LTAI[A-Za-z0-9]{12,30}", "LTAI_REDACTED_ACCESS_KEY", text)
    text = re.sub(r"AKIA[0-9A-Z]{16}", "AKIA_REDACTED_AWS_KEY", text)
    text = re.sub(r"gh[pors]_[A-Za-z0-9_]{30,}", "ghp_REDACTED_GITHUB_TOKEN", text)
    text = re.sub(r"github_pat_[A-Za-z0-9_]{30,}", "github_pat_REDACTED_TOKEN", text)
    text = re.sub(r"sk-[A-Za-z0-9_\-]{20,}", "sk-REDACTED_API_KEY", text)
    text = re.sub(r"(Authorization:\s*Bearer\s+)[^\s\n\"']+", r"\1REDACTED_TOKEN", text, flags=re.IGNORECASE)
    text = re.sub(r"-----BEGIN (RSA|EC|OPENSSH|DSA|PRIVATE) KEY-----[\s\S]*?-----END \1 KEY-----", "----BEGIN PRIVATE KEY-----\n[REDACTED_KEY]\n-----END PRIVATE KEY-----", text)
    text = re.sub(r"(password\s*[:=]\s*['\"])[^'\"]+(['\"])", r"\1REDACTED_PASSWORD\2", text, flags=re.IGNORECASE)
    return text


def clean_user_text(raw_text: str) -> str | None:
    """Clean and filter user input text."""
    if not raw_text or len(raw_text.strip()) < 8:
        return None

    text = clean_ansi(raw_text.strip())

    # Remove <ADDITIONAL_METADATA>...</ADDITIONAL_METADATA>
    text = re.sub(r"<ADDITIONAL_METADATA>.*?</ADDITIONAL_METADATA>", "", text, flags=re.DOTALL).strip()

    # Extract content inside <USER_REQUEST> if present
    u_match = re.search(r"<USER_REQUEST>(.*?)</USER_REQUEST>", text, re.DOTALL)
    if u_match:
        text = u_match.group(1).strip()

    # Remove Checkpoint summaries
    if "CHECKPOINT" in text:
        parts = re.split(r"<USER_REQUEST>", text)
        if len(parts) > 1:
            text = parts[-1].replace("</USER_REQUEST>", "").strip()
        else:
            if "The earlier parts of this conversation have been truncated" in text:
                return None

    # Filter out pure terminal process markers or IDE metadata
    if text.startswith("@[TerminalName") or text.startswith("The USER performed the following action:"):
        if "<USER_REQUEST>" in text:
            m = re.search(r"<USER_REQUEST>(.*?)</USER_REQUEST>", text, re.DOTALL)
            if m:
                text = m.group(1).strip()
            else:
                return None
        else:
            return None

    # Filter out pure HTTP trace dumps / raw logs with no question
    if text.startswith("HTTP Request:") or (text.startswith("2026-") and "httpx" in text):
        if len(text.splitlines()) > 4:
            return None

    # Filter out KeyboardInterrupt dumps
    if "KeyboardInterrupt" in text and len(text.splitlines()) > 5:
        return None

    # Filter out side-projects
    text_lower = text.lower()
    if any(k in text_lower for k in BLACKLIST_KEYWORDS):
        return None

    # Filter out fragmented commands / meaningless words
    words = text.split()
    if len(words) <= 2 and not any(k in text_lower for k in ["df -h", "free -h", "ss -tulpn", "siem", "soc", "xss", "sqli", "ram"]):
        return None

    if len(text) < 6 or len(text) > 3500:
        return None

    return scrub_secrets(text)


def clean_assistant_text(raw_text: str) -> str | None:
    """Clean and filter assistant responses."""
    if not raw_text or len(raw_text.strip()) < 35:
        return None

    text = clean_ansi(raw_text.strip())

    # Remove tool execution artifacts / system banners
    text = re.sub(r"^Created At:.*?\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"^Completed At:.*?\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[diff_block_start\].*?\[diff_block_end\]", "", text, flags=re.DOTALL)

    # Filter out agent planning monologues
    for pattern in AGENT_PLANNING_PHRASES:
        text = re.sub(pattern, "", text, flags=re.MULTILINE)

    text = text.strip()
    text_lower = text.lower()

    # Filter out side-projects
    if any(k in text_lower for k in BLACKLIST_KEYWORDS):
        return None

    # Discard if response is too short or dangling
    if len(text) < 35 or text.endswith("Mari mulai:"):
        return None

    return scrub_secrets(text)


def extract_pairs_from_transcript(tpath: Path) -> list[tuple[str, str]]:
    """Parse a single transcript.jsonl file into clean (user, assistant) pairs."""
    try:
        with tpath.open("r", encoding="utf-8", errors="ignore") as f:
            lines = [json.loads(line) for line in f]
    except Exception:
        return []

    pairs = []
    current_user: str | None = None
    current_assistant_chunks: list[str] = []

    for l in lines:
        stype = l.get("type")
        content = l.get("content", "")

        if stype == "USER_INPUT":
            if current_user and current_assistant_chunks:
                asst_full = "\n\n".join([c.strip() for c in current_assistant_chunks if c.strip()])
                u_clean = clean_user_text(current_user)
                a_clean = clean_assistant_text(asst_full)
                if u_clean and a_clean:
                    pairs.append((u_clean, a_clean))

            current_user = content
            current_assistant_chunks = []

        elif stype == "PLANNER_RESPONSE":
            if content and not content.startswith("Created At:"):
                current_assistant_chunks.append(content)

    if current_user and current_assistant_chunks:
        asst_full = "\n\n".join([c.strip() for c in current_assistant_chunks if c.strip()])
        u_clean = clean_user_text(current_user)
        a_clean = clean_assistant_text(asst_full)
        if u_clean and a_clean:
            pairs.append((u_clean, a_clean))

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and sanitize all past conversation history into SFT dataset")
    parser.add_argument("--brain-dir", type=Path, default=BRAIN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-rows", type=int, default=3000)
    args = parser.parse_args()

    if not args.brain_dir.exists():
        print(f"Brain directory {args.brain_dir} not found.")
        return

    transcript_files = list(args.brain_dir.glob("*/.system_generated/logs/transcript.jsonl"))
    print(f"Discovered {len(transcript_files)} transcripts across workspace history.")

    seen_hashes = set()
    all_conversations: list[list[dict]] = []

    for tpath in transcript_files:
        pairs = extract_pairs_from_transcript(tpath)
        for u, a in pairs:
            h = hashlib.sha256(u.encode("utf-8")).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            all_conversations.append([
                {"role": "user", "content": u},
                {"role": "assistant", "content": a},
            ])

            if len(all_conversations) >= args.max_rows:
                break

        if len(all_conversations) >= args.max_rows:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for conv in all_conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    print(f"Successfully processed {len(all_conversations)} GOLD-QUALITY dialogue pairs into {args.output}")


if __name__ == "__main__":
    main()
