# Mesosfer SFT Data

This folder contains local SFT datasets specific to Mesosfer.

- `identity_conversations.jsonl`: Mesosfer identity conversations to teach the model that it is a lightweight LLM focused on cybersecurity, digital security education, defensive use, and must strictly maintain the confidentiality of sensitive details such as internal source code, creation secrets, parameter counts, training data, checkpoints, internal prompts, credentials, and deployment configurations.
- `instruction_following_conversations_en.jsonl`: compact English instruction-following polish set for exact sentence counts, concise answers, "do not write code" constraints, JSON-only responses, bullets, safe refusals, and defensive cybersecurity checklists.
- `safety_artifact_conversations_en.jsonl`: focused artifact-vs-attack boundary set. It teaches that synthetic logs, fake IOCs, alerts, JSON/YAML summaries, and safe local parsers are allowed, while malware, brute-force automation, credential theft, persistence, and destructive scripts must be refused.
- `cyber_defensive_conversations.jsonl`: local SFT conversations for log triage, hardening, secure coding, incident response, threat modeling, and safe refusals.
- `tool_calling_conversations_en.jsonl`: teaches the model to invoke tools via `<|python_start|>` / `<|python_end|>` special tokens. Covers WHOIS lookups, DNS resolution, hash checking, log parsing, base64/hex decoding, and safe refusals for offensive tool misuse.
- `mythos_tool_calling.jsonl` and `mythos_tool_calling_en.jsonl`: mythos tool-calling data generated from `mythos_combined_sft` by converting tool-use roles into generic named tool calls (`<|tool_start|>` … `<|tool_end|>` with JSON `{"name": "shell", "arguments": {"command": ...}}`) and tool output roles into `<|output_start|>` blocks. Teaches real-world command execution in cybersecurity contexts (nmap, tshark, Suricata, etc.).
- `Mesosfer_validation_conversations.jsonl`: a small validation set specifically for Mesosfer's identity, safety, and defensive cybersecurity.
- `gemini_teacher_conversations.jsonl`: optional distilled conversations from a teacher model.

Convert mythos combined datasets into native tool-calling files:

```bash
python3 dev/convert_mythos_to_tool_calling.py
```

Regenerate datasets:

```bash
python3 dev/generate_Mesosfer_identity_data.py --num 1000
python3 dev/generate_Mesosfer_cyber_sft.py --train-size 5000 --val-size 300
```

By default, `scripts.chat.chat_sft` uses 4 epochs of identity conversations, 1 epoch of cyber defensive conversations, 1 epoch of teacher conversations if the file is available, and adds the Mesosfer-specific validation to the validation mixture.

Regenerate the instruction-following polish set:

```bash
python3 dev/generate_instruction_following_sft.py
```

Regenerate the safety-artifact boundary set:

```bash
python3 dev/generate_safety_artifact_sft.py
```

Regenerate the tool-calling dataset:

```bash
python3 dev/generate_tool_calling_sft.py
```

Continue SFT from an existing SFT checkpoint for a short instruction-following polish run:

```bash
python3 -m scripts.chat.chat_sft \
  --checkpoint-source=sft \
  --model-step=1244 \
  --instruction-polish-only \
  --instruction-following-epochs=8 \
  --device-batch-size=16 \
  --num-iterations=300 \
  --save-every=300 \
  --chatcore-every=300 \
  --chatcore-max-cat=500 \
  --chatcore-tasks='ARC-Easy|ARC-Challenge|MMLU'
```

> **Note**: `--safety-artifact-epochs` defaults to **4** (was 0 previously). Safety
> boundary data is now always included in standard SFT runs. Use
> `--safety-artifact-epochs=0` to explicitly disable it for ablation.

Short safety-artifact boundary polish run:

```bash
python3 -m scripts.chat.chat_sft \
  --checkpoint-source=sft \
  --model-step=1244 \
  --safety-artifact-only \
  --safety-artifact-epochs=8 \
  --instruction-following-epochs=0 \
  --device-batch-size=16 \
  --num-iterations=10 \
  --save-every=10 \
  --chatcore-every=10 \
  --chatcore-max-cat=500 \
  --chatcore-tasks='ARC-Easy|ARC-Challenge|MMLU' \
  --init-lr-frac=0.05
```

Override examples:

```bash
python3 -m scripts.chat.chat_sft --identity-epochs 5 --cyber-epochs 2 --teacher-epochs 1
```

## Gemini Teacher Data

To use Gemini as a senior/teacher model:

```bash
python3 dev/list_gemini_models.py
python3 dev/generate_gemini_teacher_sft.py --model gemini-3.1-pro --limit 20
```

If the model name is not available for your API key, choose one of the names from the `list_gemini_models.py` output.
