"""
Cybersecurity SFT tasks for mesosfer.

Wraps the JSONL files in data/sft/ as Task objects so they can be mixed into
the SFT training pipeline alongside SmolTalk, MMLU, GSM8K, etc.

Each file in data/sft/ follows the same format as identity_conversations.jsonl:
each line is a JSON array of {"role": ..., "content": ...} message objects.

Conversations that do not match the user/assistant alternation are skipped with a
single warning per file rather than crashing the pipeline (see RobustCustomJSON).

Row counts below are post-dedup. Several files shipped with heavy internal
duplication (cyber_defensive was 5000 rows but only 120 unique, i.e. every example
repeated ~42x). Duplicates were removed so that repetition is expressed through the
`*_epochs` knobs instead — visible in the config rather than hidden in the data.
Epochs are chosen to keep each unique example under ~8 exposures per run.

Available datasets:
  - cyber_defensive_conversations.jsonl      (96 rows, ID + EN variants)
  - mesosfer_validation_conversations.jsonl  (24 ID / 32 EN — HELD-OUT, epochs=0, never trained)
  - cloud_security_sft.jsonl                 (6 rows, ID + EN variants)
  - identity_conversations.jsonl             (747 rows, ID + EN variants)
  - multi_turn_soc_sft.jsonl                 (4 rows — DISABLED, epochs=0: the `tool` and
                                              `assistant` turns are mislabelled user input,
                                              so there are no real assistant answers to learn)
  - mythos_combined_sft.jsonl                (55 rows, ID + EN — multi-step tool use; the
                                              `tool` role was converted to tool/tool_output
                                              parts and tool calls normalised to
                                              {"name":"shell","arguments":{"command":...}})
  - mythos_tool_calling.jsonl                (110 rows, ID + EN variants)
  - tool_oriented_cyber_sft.jsonl            (8 rows)
  - tool_calling_conversations_en.jsonl      (122 rows, calc/tool parts)
  - gemini_teacher_conversations.jsonl       (373 rows)
  - alpaca_indonesian_sft.jsonl              (45626 rows — general Indonesian instruction;
                                              dominates the mix at ~1 epoch, see README)
"""

import json
import os
from pathlib import Path

from tasks.common import Task


_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SFT_DIR = _REPO_ROOT / "data" / "sft"

# Module-level cache: filepath → (conversations, skipped_count)
# Avoids re-reading and re-warning when the same file is loaded multiple times
# (which happens with epoch-based oversampling in build_cybersec_sft_tasks).
_FILE_CACHE: dict = {}


def _sft_path(filename: str, sft_dir: str | None = None) -> str:
    base = Path(sft_dir) if sft_dir else DEFAULT_SFT_DIR
    return str(base / filename)


def _is_valid_conversation(messages) -> bool:
    """
    Check if a conversation matches mesosfer's expected user/assistant alternation.

    Returns False if any message has a role other than user/assistant, or if the
    pattern doesn't alternate starting with user.

    An optional leading system message is allowed (the tokenizer merges it into
    the first user message via render_conversation).

    Assistant content may be a string (simple text) or a list of parts (tool
    calls using ``<|calc_start|>`` / ``<|output_start|>`` / ``<|tool_start|>``
    tokens).  User content must always be a string.
    """
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    # Strip optional leading system message (tokenizer handles the merge)
    if isinstance(messages[0], dict) and messages[0].get("role") == "system":
        messages = messages[1:]
        if len(messages) < 2:
            return False
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            return False
        if "role" not in message or "content" not in message:
            return False
        expected_role = "user" if i % 2 == 0 else "assistant"
        if message["role"] != expected_role:
            return False
        content = message["content"]
        if expected_role == "user":
            if not isinstance(content, str):
                return False
        else:  # assistant
            if not isinstance(content, (str, list)):
                return False
            if isinstance(content, list):
                # Validate each part has the required fields
                for part in content:
                    if not isinstance(part, dict):
                        return False
                    if "type" not in part or "text" not in part:
                        return False
                    if part["type"] not in ("text", "calc", "calc_output", "tool", "tool_output"):
                        return False
    return True


class RobustCustomJSON(Task):
    """
    Like CustomJSON but tolerant of malformed conversations:
    - Skips conversations with non-standard roles (e.g. `tool`, `system`)
    - Skips conversations that don't alternate user/assistant
    - Logs a single summary line per file instead of crashing
    """

    def __init__(self, filepath, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath

        # Reuse cached load if available (avoid re-reading + re-warning per epoch)
        if filepath in _FILE_CACHE:
            self.conversations, _ = _FILE_CACHE[filepath]
            self.length = len(self.conversations)
            return

        self.conversations = []
        skipped = 0

        if not os.path.exists(filepath):
            print(f"[cybersec_sft] WARNING: file not found: {filepath}")
            self.length = 0
            _FILE_CACHE[filepath] = (self.conversations, skipped)
            return

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
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
            print(f"[cybersec_sft] {Path(filepath).name}: loaded {len(self.conversations)} valid, skipped {skipped} (non-standard schema)")

        self.length = len(self.conversations)
        _FILE_CACHE[filepath] = (self.conversations, skipped)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        messages = self.conversations[index]
        return {"messages": messages}


# -----------------------------------------------------------------------------
# Cybersecurity SFT task wrappers

class CyberDefensiveConversations(RobustCustomJSON):
    """5K rows of defensive cybersecurity Q&A (SOC analyst, threat triage, IR)."""
    def __init__(self, language="id", sft_dir=None, **kwargs):
        assert language in ("id", "en"), f"language must be 'id' or 'en', got {language}"
        suffix = "" if language == "id" else "_en"
        super().__init__(filepath=_sft_path(f"cyber_defensive_conversations{suffix}.jsonl", sft_dir), **kwargs)


class CloudSecuritySFT(RobustCustomJSON):
    """Cloud security incident response (AWS, Azure, GCP). Small dataset (6 rows)."""
    def __init__(self, language="id", sft_dir=None, **kwargs):
        assert language in ("id", "en"), f"language must be 'id' or 'en', got {language}"
        suffix = "" if language == "id" else "_en"
        super().__init__(filepath=_sft_path(f"cloud_security_sft{suffix}.jsonl", sft_dir), **kwargs)


class MultiTurnSOC(RobustCustomJSON):
    """Multi-turn SOC analyst dialogues. Very small dataset (4 rows) — oversample heavily."""
    def __init__(self, sft_dir=None, **kwargs):
        super().__init__(filepath=_sft_path("multi_turn_soc_sft.jsonl", sft_dir), **kwargs)


class ToolOrientedCyber(RobustCustomJSON):
    """Tool-oriented cybersecurity conversations (nmap, burp, sqlmap, etc.). 8 rows."""
    def __init__(self, sft_dir=None, **kwargs):
        super().__init__(filepath=_sft_path("tool_oriented_cyber_sft.jsonl", sft_dir), **kwargs)


class ToolCallingConversations(RobustCustomJSON):
    """Tool-calling SFT data that teaches <|calc_start|>/<|output_start|> usage.

    Generated by dev/generate_tool_calling_sft.py.  Conversations include
    multi-part assistant messages with tool invocations (WHOIS, dig, hash
    checks, log parsing) and tool-use refusals for offensive requests.
    """
    def __init__(self, sft_dir=None, **kwargs):
        super().__init__(filepath=_sft_path("tool_calling_conversations_en.jsonl", sft_dir), **kwargs)


class MythosToolCalling(RobustCustomJSON):
    """Mythos scenarios converted to native <|tool_start|> tool-calling format.

    Generated by dev/convert_mythos_to_tool_calling.py from mythos_combined_sft.
    CLI tool calls (nmap, tshark, grep, strings, testssl.sh, Suricata, fail2ban,
    etc.) are emitted as generic shell tool calls
    ``{"name": "shell", "arguments": {"command": ...}}``. Contains 110 rows
    (59 with tool calls + 51 text-only kept as-is).
    """
    def __init__(self, language="id", sft_dir=None, **kwargs):
        assert language in ("id", "en"), f"language must be 'id' or 'en', got {language}"
        suffix = "" if language == "id" else "_en"
        super().__init__(filepath=_sft_path(f"mythos_tool_calling{suffix}.jsonl", sft_dir), **kwargs)


class MythosCombined(RobustCustomJSON):
    """Combined mythos (offensive + defensive narrative scenarios). 110 rows."""
    def __init__(self, language="id", sft_dir=None, **kwargs):
        assert language in ("id", "en"), f"language must be 'id' or 'en', got {language}"
        suffix = "" if language == "id" else "_en"
        super().__init__(filepath=_sft_path(f"mythos_combined_sft{suffix}.jsonl", sft_dir), **kwargs)


class mesosferValidation(RobustCustomJSON):
    """mesosfer validation conversations — domain alignment dataset. 300 rows."""
    def __init__(self, language="id", sft_dir=None, **kwargs):
        assert language in ("id", "en"), f"language must be 'id' or 'en', got {language}"
        suffix = "" if language == "id" else "_en"
        super().__init__(filepath=_sft_path(f"mesosfer_validation_conversations{suffix}.jsonl", sft_dir), **kwargs)


class GeminiTeacher(RobustCustomJSON):
    """Gemini-distilled teacher conversations. 373 rows."""
    def __init__(self, sft_dir=None, **kwargs):
        super().__init__(filepath=_sft_path("gemini_teacher_conversations.jsonl", sft_dir), **kwargs)


class PrimusInstruct(RobustCustomJSON):
    """Trend Micro Primus-Instruct cybersecurity instruction pairs. ~100K rows (gated)."""
    def __init__(self, sft_dir=None, **kwargs):
        super().__init__(filepath=_sft_path("primus_instruct.jsonl", sft_dir), **kwargs)


class PrimusReasoning(RobustCustomJSON):
    """Trend Micro Primus-Reasoning cybersecurity reasoning distillation. ~50K rows (gated)."""
    def __init__(self, sft_dir=None, **kwargs):
        super().__init__(filepath=_sft_path("primus_reasoning.jsonl", sft_dir), **kwargs)


class CyberNativeVulnDPO(RobustCustomJSON):
    """CyberNative vulnerable vs fixed code pairs converted to SFT format. ~4.6K rows."""
    def __init__(self, sft_dir=None, **kwargs):
        super().__init__(filepath=_sft_path("cybernative_vuln_dpo_sft.jsonl", sft_dir), **kwargs)


class OpenHermesSFT(RobustCustomJSON):
    """OpenHermes-2.5 general instruction dataset (capped at 50K rows)."""
    def __init__(self, sft_dir=None, **kwargs):
        super().__init__(filepath=_sft_path("openhermes_sft.jsonl", sft_dir), **kwargs)


class UltraChatSFT(RobustCustomJSON):
    """UltraChat 200K multi-turn conversations (capped at 100K rows)."""
    def __init__(self, sft_dir=None, **kwargs):
        super().__init__(filepath=_sft_path("ultrachat_sft.jsonl", sft_dir), **kwargs)


class TrendyolCyberSFT(RobustCustomJSON):
    """Trendyol Cybersecurity Instruction Tuning — 53K defensive cybersec rows."""
    def __init__(self, sft_dir=None, **kwargs):
        super().__init__(filepath=_sft_path("trendyol_cyber_sft.jsonl", sft_dir), **kwargs)


class TiamzCybersecSFT(RobustCustomJSON):
    """Tiamz cybersecurity instruction dataset — 12K Q&A pairs."""
    def __init__(self, sft_dir=None, **kwargs):
        super().__init__(filepath=_sft_path("tiamz_cybersec_sft.jsonl", sft_dir), **kwargs)


class AlpacaIndonesianSFT(RobustCustomJSON):
    """Alpaca Cleaned Indonesian instruction dataset. ~52K clean rows."""
    def __init__(self, sft_dir=None, **kwargs):
        super().__init__(filepath=_sft_path("alpaca_indonesian_sft.jsonl", sft_dir), **kwargs)


class AlpacaGPT4IndonesianSFT(RobustCustomJSON):
    """Alpaca GPT-4 Indonesian dataset. ~52K high-quality Indonesian instruction rows."""
    def __init__(self, sft_dir=None, **kwargs):
        super().__init__(filepath=_sft_path("alpaca_gpt4_indonesian_sft.jsonl", sft_dir), **kwargs)


class MagpieReasoningSFT(RobustCustomJSON):
    """Magpie-Reasoning-V2 dataset converted to SFT format."""
    def __init__(self, sft_dir=None, **kwargs):
        super().__init__(filepath=_sft_path("magpie_reasoning_sft.jsonl", sft_dir), **kwargs)


class OpenThoughtsSFT(RobustCustomJSON):
    """OpenThoughts-114k dataset converted to SFT format."""
    def __init__(self, sft_dir=None, **kwargs):
        super().__init__(filepath=_sft_path("open_thoughts_sft.jsonl", sft_dir), **kwargs)


class HermesFuncCallingSFT(RobustCustomJSON):
    """Hermes multi-turn function calling. Ungated counterpart to the gated xLAM set."""
    def __init__(self, sft_dir=None, **kwargs):
        super().__init__(filepath=_sft_path("hermes_func_calling_sft.jsonl", sft_dir), **kwargs)


class HermesGlaiveFuncCallingSFT(RobustCustomJSON):
    """Glaive function calling, cleaned into ShareGPT turns by Nous."""
    def __init__(self, sft_dir=None, **kwargs):
        super().__init__(filepath=_sft_path("hermes_glaive_func_calling_sft.jsonl", sft_dir), **kwargs)


class KimiK3TracesSFT(RobustCustomJSON):
    """Kimi K3 agentic coding/debugging trajectories (multi-step tool use). ~4K rows."""
    def __init__(self, sft_dir=None, **kwargs):
        super().__init__(filepath=_sft_path("kimi_k3_traces_sft.jsonl", sft_dir), **kwargs)


class CoTCollectionSFT(RobustCustomJSON):
    """CoT Collection chain-of-thought rationales across many NLP tasks."""
    def __init__(self, sft_dir=None, **kwargs):
        super().__init__(filepath=_sft_path("cot_collection_sft.jsonl", sft_dir), **kwargs)


class QyrouReasoningSFT(RobustCustomJSON):
    """Reasoning corpus distilled from frontier models (explicit thought traces)."""
    def __init__(self, sft_dir=None, **kwargs):
        super().__init__(filepath=_sft_path("qyrou_reasoning_sft.jsonl", sft_dir), **kwargs)


class NistCybersecSFT(RobustCustomJSON):
    """NIST cybersecurity training conversations. ~50K rows."""
    def __init__(self, sft_dir=None, **kwargs):
        super().__init__(filepath=_sft_path("nist_cybersec_sft.jsonl", sft_dir), **kwargs)


class FenrirV2SFT(RobustCustomJSON):
    """Fenrir v2.1 cybersecurity Q&A (OWASP/MITRE/NIST). ~99K rows."""
    def __init__(self, sft_dir=None, **kwargs):
        super().__init__(filepath=_sft_path("fenrir_v2_sft.jsonl", sft_dir), **kwargs)


class CodeFeedbackSFT(RobustCustomJSON):
    """Multi-language coding feedback and instruction pairs. ~50K rows."""
    def __init__(self, sft_dir=None, **kwargs):
        super().__init__(filepath=_sft_path("code_feedback_sft.jsonl", sft_dir), **kwargs)


class NuminaMathCoTSFT(RobustCustomJSON):
    """NuminaMath-CoT competition math with chain-of-thought. ~50K rows."""
    def __init__(self, sft_dir=None, **kwargs):
        super().__init__(filepath=_sft_path("numinamath_cot_sft.jsonl", sft_dir), **kwargs)


class AquilaXSecurityReasoning(RobustCustomJSON):
    """AquilaX cybersecurity vulnerability-analysis reasoning (CoT). ~18K rows."""
    def __init__(self, sft_dir=None, **kwargs):
        super().__init__(filepath=_sft_path("aquilax_security_reasoning_sft.jsonl", sft_dir), **kwargs)


class XLAMFunctionCalling(RobustCustomJSON):
    """xLAM (APIGen) function-calling — generic named tool calls rendered with
    ``<|tool_start|>`` / ``<|tool_end|>``. ~20K rows (gated)."""
    def __init__(self, sft_dir=None, **kwargs):
        super().__init__(filepath=_sft_path("xlam_function_calling_sft.jsonl", sft_dir), **kwargs)



# -----------------------------------------------------------------------------
# Mixture builders

def build_cybersec_sft_tasks(
    cyber_defensive_epochs: int = 8,
    cloud_security_epochs: int = 8,
    multi_turn_soc_epochs: int = 0,
    tool_oriented_epochs: int = 8,
    tool_calling_epochs: int = 8,
    mythos_epochs: int = 8,
    mythos_tool_calling_epochs: int = 4,
    mesosfer_validation_epochs: int = 0,
    gemini_teacher_epochs: int = 2,
    primus_instruct_epochs: int = 1,
    primus_reasoning_epochs: int = 1,
    cybernative_vuln_epochs: int = 3,
    openhermes_epochs: int = 1,
    ultrachat_epochs: int = 1,
    trendyol_cyber_epochs: int = 1,
    tiamz_cybersec_epochs: int = 2,
    alpaca_indonesian_epochs: int = 1,
    alpaca_gpt4_indonesian_epochs: int = 1,
    magpie_reasoning_epochs: int = 1,
    open_thoughts_epochs: int = 1,
    hermes_func_calling_epochs: int = 2,
    hermes_glaive_func_calling_epochs: int = 1,
    kimi_k3_traces_epochs: int = 4,
    cot_collection_epochs: int = 1,
    qyrou_reasoning_epochs: int = 1,
    nist_cybersec_epochs: int = 1,
    fenrir_v2_epochs: int = 1,
    code_feedback_epochs: int = 1,
    numinamath_cot_epochs: int = 1,
    aquilax_security_reasoning_epochs: int = 2,
    xlam_function_calling_epochs: int = 1,
    include_english: bool = True,
    sft_dir: str | None = None,
) -> list:
    """
    Build a list of cybersecurity Task objects for use in TaskMixture.

    Default epoch counts oversample tiny datasets (cloud_security 6 rows × 20 = 120,
    multi_turn_soc 4 rows × 30 = 120) so they aren't drowned out by the larger
    cyber_defensive (5K rows). Tune these in chat_sft.py CLI flags.

    Args:
        cyber_defensive_epochs: epochs of cyber_defensive_conversations (5K rows each)
        cloud_security_epochs: epochs of cloud_security_sft (6 rows each, oversample)
        multi_turn_soc_epochs: epochs of multi_turn_soc_sft (4 rows, oversample)
        tool_oriented_epochs: epochs of tool_oriented_cyber_sft (8 rows, oversample)
        tool_calling_epochs: epochs of tool_calling_conversations_en (tool-use with special tokens)
        mythos_epochs: epochs of mythos_combined_sft (110 rows each, text-only compatible)
        mythos_tool_calling_epochs: epochs of mythos_tool_calling (110 rows, native tool format with CLI tools)
        mesosfer_validation_epochs: epochs of mesosfer validation (300 rows each)
        gemini_teacher_epochs: epochs of gemini teacher (373 rows)
        primus_instruct_epochs: epochs of Primus-Instruct (~100K rows, gated)
        primus_reasoning_epochs: epochs of Primus-Reasoning (~50K rows, gated)
        cybernative_vuln_epochs: epochs of CyberNative vuln DPO (~4.6K rows)
        openhermes_epochs: epochs of OpenHermes-2.5 (50K rows)
        ultrachat_epochs: epochs of UltraChat 200K (100K rows)
        include_english: whether to also include _en variants of bilingual datasets
        sft_dir: override path to data/sft/ (default: repo_root/data/sft)

    Returns:
        list of Task objects (each can be passed multiple times to oversample)
    """
    tasks = []

    languages = ["id", "en"] if include_english else ["id"]

    # Bilingual datasets: include each language separately × N epochs
    for lang in languages:
        for _ in range(cyber_defensive_epochs):
            tasks.append(CyberDefensiveConversations(language=lang, sft_dir=sft_dir))
        for _ in range(cloud_security_epochs):
            tasks.append(CloudSecuritySFT(language=lang, sft_dir=sft_dir))
        for _ in range(mythos_epochs):
            tasks.append(MythosCombined(language=lang, sft_dir=sft_dir))
        for _ in range(mesosfer_validation_epochs):
            tasks.append(mesosferValidation(language=lang, sft_dir=sft_dir))
        for _ in range(mythos_tool_calling_epochs):
            tasks.append(MythosToolCalling(language=lang, sft_dir=sft_dir))

    # Monolingual datasets
    for _ in range(multi_turn_soc_epochs):
        tasks.append(MultiTurnSOC(sft_dir=sft_dir))
    for _ in range(tool_oriented_epochs):
        tasks.append(ToolOrientedCyber(sft_dir=sft_dir))
    for _ in range(tool_calling_epochs):
        tasks.append(ToolCallingConversations(sft_dir=sft_dir))
    for _ in range(gemini_teacher_epochs):
        tasks.append(GeminiTeacher(sft_dir=sft_dir))

    # External HF datasets (only included if files exist — downloaded via download_sft_data.py)
    for _ in range(primus_instruct_epochs):
        tasks.append(PrimusInstruct(sft_dir=sft_dir))
    for _ in range(primus_reasoning_epochs):
        tasks.append(PrimusReasoning(sft_dir=sft_dir))
    for _ in range(cybernative_vuln_epochs):
        tasks.append(CyberNativeVulnDPO(sft_dir=sft_dir))
    for _ in range(openhermes_epochs):
        tasks.append(OpenHermesSFT(sft_dir=sft_dir))
    for _ in range(ultrachat_epochs):
        tasks.append(UltraChatSFT(sft_dir=sft_dir))
    for _ in range(trendyol_cyber_epochs):
        tasks.append(TrendyolCyberSFT(sft_dir=sft_dir))
    for _ in range(tiamz_cybersec_epochs):
        tasks.append(TiamzCybersecSFT(sft_dir=sft_dir))
    for _ in range(alpaca_indonesian_epochs):
        tasks.append(AlpacaIndonesianSFT(sft_dir=sft_dir))
    for _ in range(alpaca_gpt4_indonesian_epochs):
        tasks.append(AlpacaGPT4IndonesianSFT(sft_dir=sft_dir))
    for _ in range(magpie_reasoning_epochs):
        tasks.append(MagpieReasoningSFT(sft_dir=sft_dir))
    for _ in range(open_thoughts_epochs):
        tasks.append(OpenThoughtsSFT(sft_dir=sft_dir))
    for _ in range(hermes_func_calling_epochs):
        tasks.append(HermesFuncCallingSFT(sft_dir=sft_dir))
    for _ in range(hermes_glaive_func_calling_epochs):
        tasks.append(HermesGlaiveFuncCallingSFT(sft_dir=sft_dir))
    for _ in range(kimi_k3_traces_epochs):
        tasks.append(KimiK3TracesSFT(sft_dir=sft_dir))
    for _ in range(cot_collection_epochs):
        tasks.append(CoTCollectionSFT(sft_dir=sft_dir))
    for _ in range(qyrou_reasoning_epochs):
        tasks.append(QyrouReasoningSFT(sft_dir=sft_dir))
    for _ in range(nist_cybersec_epochs):
        tasks.append(NistCybersecSFT(sft_dir=sft_dir))
    for _ in range(fenrir_v2_epochs):
        tasks.append(FenrirV2SFT(sft_dir=sft_dir))
    for _ in range(code_feedback_epochs):
        tasks.append(CodeFeedbackSFT(sft_dir=sft_dir))
    for _ in range(numinamath_cot_epochs):
        tasks.append(NuminaMathCoTSFT(sft_dir=sft_dir))
    for _ in range(aquilax_security_reasoning_epochs):
        tasks.append(AquilaXSecurityReasoning(sft_dir=sft_dir))
    for _ in range(xlam_function_calling_epochs):
        tasks.append(XLAMFunctionCalling(sft_dir=sft_dir))


    # Filter out empty tasks (e.g. if all rows were skipped)
    tasks = [t for t in tasks if len(t) > 0]
    return tasks


def total_cybersec_rows(tasks: list) -> int:
    """Helper to sum the total row count across a list of cybersec SFT tasks."""
    return sum(len(t) for t in tasks)


if __name__ == "__main__":
    print(f"SFT dir: {DEFAULT_SFT_DIR}")
    print(f"Exists: {DEFAULT_SFT_DIR.exists()}")
    if DEFAULT_SFT_DIR.exists():
        tasks = build_cybersec_sft_tasks()
        print(f"\nBuilt {len(tasks)} task instances")
        print(f"Total rows (with epoch oversampling): {total_cybersec_rows(tasks):,}")
        from collections import Counter
        breakdown = Counter(type(t).__name__ for t in tasks)
        print("\nBreakdown by class:")
        for cls, n in sorted(breakdown.items()):
            print(f"  {cls}: {n} instances")
