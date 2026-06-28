"""
CustomJSON task for loading conversations from JSONL files.
Each line in the JSONL file should be a JSON array of messages.
"""

import os
import json
from tasks.common import Task


# Part types allowed inside a list-form assistant message (tool-calling SFT).
_VALID_PART_TYPES = ("text", "python", "python_output", "tool", "tool_output")


def _is_valid_conversation(messages) -> bool:
    """
    Validate a conversation against mesosfer's user/assistant alternation.

    - Must be a list of >= 2 message dicts with 'role' and 'content'.
    - Roles must alternate starting with 'user'.
    - User content must be a string.
    - Assistant content may be a string OR a list of part-dicts (tool calls), each
      with 'type' (in _VALID_PART_TYPES) and 'text'. This mirrors RobustCustomJSON
      so tool-calling data in local SFT files does not crash the run.
    """
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    for i, message in enumerate(messages):
        if not isinstance(message, dict) or "role" not in message or "content" not in message:
            return False
        expected_role = "user" if i % 2 == 0 else "assistant"
        if message["role"] != expected_role:
            return False
        content = message["content"]
        if expected_role == "user":
            if not isinstance(content, str):
                return False
        else:  # assistant
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict) or "type" not in part or "text" not in part:
                        return False
                    if part["type"] not in _VALID_PART_TYPES:
                        return False
            elif not isinstance(content, str):
                return False
    return True


class CustomJSON(Task):
    """
    Load conversations from a JSONL file.
    Each line should be a JSON array of message objects with 'role' and 'content' fields.
    Example line: [{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"}]

    Tolerant loader: malformed lines (bad JSON or non-conforming schema) are skipped with
    a single summary line instead of crashing the whole training run. Assistant messages
    may use list-of-parts content for tool calls.
    """

    def __init__(self, filepath, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        self.conversations = []

        # Load all conversations from the JSONL file
        if not os.path.exists(filepath):
            # Helpful error message due to recent change. Will be removed in the future.
            print("-" * 80)
            print(f"Warning: File {filepath} does not exist")
            print("HINT (Oct 21 2025)")
            print("If you recently did a git pull and suddenly see this, it might be due to the new addition of identity conversations")
            print("See this discussion for more details: https://github.com/karpathy/mesosfer/discussions/139")
            print("Quick fix: simply run the following command to download the file and you're done:")
            print(f"curl -L -o {filepath} https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl")
            print("-" * 80)

        else:
            skipped = 0
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:  # skip empty lines
                        continue
                    try:
                        messages = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue
                    if _is_valid_conversation(messages):
                        self.conversations.append(messages)
                    else:
                        skipped += 1
            if skipped > 0:
                print(f"[CustomJSON] {os.path.basename(filepath)}: loaded {len(self.conversations)} valid, skipped {skipped} (bad JSON / non-conforming schema)")

        self.length = len(self.conversations)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        messages = self.conversations[index]
        conversation = {
            "messages": messages,
        }
        return conversation
