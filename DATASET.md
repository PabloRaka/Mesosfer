# Dataset Architecture (mesosfer)

This document describes the datasets that power the three training/evaluation
stages of **mesosfer**, kept in sync with the source of truth in code:

- **Pretraining** → `scripts/data/prepare_data.py` (`DATASET_SOURCES`, `DOMAIN_SAMPLING_WEIGHTS`)
- **SFT** → `scripts/data/download_sft_data.py` (`SOURCES`) + local `data/sft/*.jsonl` wired in `tasks/cybersec_sft.py`
- **Eval** → `scripts/eval/base_eval.py` + `scripts/chat/chat_eval.py`

---

## 1. Token Budget Summary (Depth 32)


| Mode                                             | Scaling Params | Ratio | Tokens Needed | Tokens Available | Status              |
| :------------------------------------------------- | :--------------- | :------ | :-------------- | :----------------- | :-------------------- |
| Speedrun (`--target-param-data-ratio=10`)        | ~6.8B          | 10×  | ~68B          | ~103B            | ✅ surplus          |
| Recommended (`--target-param-data-ratio=15`)     | ~6.8B          | 15×  | ~103B         | ~103B            | ✅ ~covered (marginal) |
| Compute-optimal (`--target-param-data-ratio=18`) | ~6.8B          | 18×  | ~123B         | ~103B            | ⚠️ under            |

> **Depth 32 config:** `n_embd = 32 × 128 = 4096`, **96K vocab**, ~13.7B total params (legacy VE; ~9.3B with `--ve-layers=4`), ~6.8B scaling params (transformer matrices + lm_head, excl. embeddings).
> Recommended: `--depth=32 --target-param-data-ratio=10` (~68B, comfortable surplus) or `=15` (~103B, marginal). Use `--ve-layers=4 --grad-checkpoint` to cut memory.

### Pretraining mix by category (sum of `max_tokens`)


| Category         | Tokens      | Share |
| :----------------- | :------------ | :------ |
| Cybersecurity    | ~37.1B      | ~37%  |
| Code             | ~30.1B      | ~30%  |
| Reasoning / Math | ~21.0B      | ~21%  |
| General          | ~15.2B      | ~15%  |
| **Total**        | **~103.4B** | 100%  |

> General includes ~3.2B Indonesian (`fineweb2_id`, `wikipedia_id`); the rest is English.

---

## 2. Pretraining Data

Sampling weights drive temperature-based interleaved shuffling: higher weight = a
domain appears more frequently in mixed shards. `max_tokens` is the hard upper
bound on volume per source.

> **Shuffle config:** `cluster_size=32`, `sampling_temperature=1.2` (reduce domain-burst loss spikes).

### Weight guidelines

- **2.0+ (critical):** Exploit/vuln patches, SOC-critical intel, curated seeds, local security data.
- **1.5–1.9 (high):** Structured cybersecurity knowledge, CVE feeds, threat intel, advisories, detection rules.
- **1.2–1.4 (elevated):** Secure code, refined code.
- **0.9–1.1 (normal):** Mathematical / reasoning corpora.
- **0.5–0.7 (reduced):** General web reference content (prevent domination).

### Pretraining domain table

****


| Domain                      | Category      | Weight | Max Tokens | Source / Type                               | Notes                                    |
| :---------------------------- | :-------------- | :------- | :----------- | :-------------------------------------------- | :----------------------------------------- |
| `circl_vuln_patch`          | Cybersecurity | 2.3    | 2.4B       | `CIRCL/vulnerability-cwe-patch`             | Real vuln+patch pairs.                   |
| `local_incident_response`   | Cybersecurity | 2.2    | 1.44B      | `data/synthetic-ir/`                        | Synthetic IR reports.                    |
| `primus_seed`               | Cybersecurity | 2.2    | 2.0B<br /> | `trendmicro-ailab/Primus-Seed`              | Curated cybersec seed corpus.            |
| `primus_nemotron_cc`        | Cybersecurity | 2.1    | 7.6B       | `trend-cybertron/Primus-Nemotron-CC`        | Cybersec text filtered from Nemotron-CC. |
| `local_soc_synthetic`       | Cybersecurity | 2.1    | 1.44B      | `data/synthetic-soc/`                       | Synthetic SOC dialogues.                 |
| `local_reverse_engineering` | Cybersecurity | 2.0    | 1.2B       | `data/reverse-engineering/`                 | RE/exploitation analysis.                |
| `primus_reasoning`          | Cybersecurity | 2.0    | 1.5B       | `trendmicro-ailab/Primus-Reasoning`         | Distilled cybersec chain-of-thought.     |
| `brightdata_cybersec`       | Cybersecurity | 2.0    | 500M       | brightdata_scraper                          | Real-time threat intel via proxy.        |
| `project_zero`              | Cybersecurity | 2.0    | 40M        | rss                                         | Google Project Zero.                     |
| `mandiant`                  | Cybersecurity | 2.0    | 40M        | rss                                         | Mandiant threat intel.                   |
| `dfir_report`               | Cybersecurity | 2.0    | 35M        | rss                                         | DFIR Report intrusion analysis.          |
| `cisa_kev`                  | Cybersecurity | 2.0    | 10M        | url_json                                    | CISA Known Exploited Vulnerabilities.    |
| `primus_fineweb`            | Cybersecurity | 1.9    | 5.0B       | `trendmicro-ailab/Primus-FineWeb`           | Cybersec-filtered FineWeb.               |
| `local_cloud_security`      | Cybersecurity | 1.9    | 1.2B       | `data/cloud_nl/`                            | Cloud audit logs (NL narratives).        |
| `local_security_logs`       | Cybersecurity | 1.9    | 1.2B       | `data/log_nl/`                              | Auth/web/Sysmon logs (NL narratives).    |
| `unit42`                    | Cybersecurity | 1.9    | 40M        | rss                                         | Palo Alto Unit 42.                       |
| `talos`                     | Cybersecurity | 1.9    | 40M        | rss                                         | Cisco Talos.                             |
| `github_advisory`           | Cybersecurity | 1.9    | 100M       | github_repo                                 | GitHub Security Advisory DB.             |
| `all_cve_records`           | Cybersecurity | 1.8    | 6.0B       | nvd_json_feeds                              | NVD CVE JSON feeds 2002–present.        |
| `nvd_cve`                   | Cybersecurity | 1.8    | 4.8B       | circl_ndjson_dump                           | CIRCL CVE List v5 dump.                  |
| `academic_security_papers`  | Cybersecurity | 1.8    | 120M       | pdf_manifest                                | Security conference papers.              |
| `mitre_attack_stix`         | Cybersecurity | 1.8    | 70M        | github_repo                                 | MITRE ATT&CK STIX.                       |
| `exploitdb`                 | Code          | 1.8    | 140M       | github_repo                                 | Exploit Database PoC corpus.             |
| `metasploit`                | Code          | 1.7    | 180M       | github_repo                                 | Metasploit Framework modules.            |
| `cloudflare_security`       | Cybersecurity | 1.6    | 30M        | rss                                         | Cloudflare security research.            |
| `sigmahq_rules`             | Cybersecurity | 1.6    | 70M        | github_repo                                 | SigmaHQ detection rules.                 |
| `elastic_rules`             | Cybersecurity | 1.6    | 70M        | github_repo                                 | Elastic detection rules.                 |
| `threat_report_pdfs`        | Cybersecurity | 1.5    | 90M        | pdf_manifest                                | Vendor threat report PDFs.               |
| `splunk_rules`              | Cybersecurity | 1.5    | 60M        | github_repo                                 | Splunk security detections.              |
| `zeek_scripts`              | Cybersecurity | 1.5    | 40M        | github_repo                                 | Zeek scripts and packages.               |
| `code_powershell`           | Code          | 1.6    | 800M       | `bigcode/the-stack-dedup`                   | PowerShell — Windows admin + offensive.  |
| `code_assembly`             | Code          | 1.6    | 800M       | `bigcode/the-stack-dedup`                   | Assembly — reverse engineering.          |
| `code_sql`                  | Code          | 1.5    | 1.5B       | `bigcode/the-stack-dedup`                   | SQL — injection context.                 |
| `secure_code_python`        | Code          | 1.4    | 4.0B       | `bigcode/the-stack-dedup`                   | Python (dedup).                          |
| `secure_code_c`             | Code          | 1.4    | 3.0B       | `bigcode/the-stack-dedup`                   | C (dedup).                               |
| `secure_code_shell`         | Code          | 1.4    | 2.0B       | `bigcode/the-stack-dedup`                   | Shell (dedup).                           |
| `secure_code_javascript`    | Code          | 1.3    | 2.4B       | `bigcode/the-stack-dedup`                   | JavaScript (dedup).                      |
| `secure_code_typescript`    | Code          | 1.3    | 2.0B       | `bigcode/the-stack-dedup`                   | TypeScript (dedup).                      |
| `secure_code_java`          | Code          | 1.3    | 2.0B       | `bigcode/the-stack-dedup`                   | Java (dedup).                            |
| `secure_code_php`           | Code          | 1.3    | 1.6B       | `bigcode/the-stack-dedup`                   | PHP (dedup).                             |
| `swallow_code_v2`           | Code          | 1.3    | 5.0B       | `tokyotech-llm/swallow-code-v2`             | Refined Python (4-stage LLM pipeline).   |
| `secure_code_cpp`           | Code          | 1.2    | 3.0B       | `bigcode/the-stack-dedup`                   | C++ (dedup).                             |
| `secure_code_rust`          | Code          | 1.2    | 2.4B       | `bigcode/the-stack-dedup`                   | Rust (dedup).                            |
| `secure_code_go`            | Code          | 1.2    | 2.4B       | `bigcode/the-stack-dedup`                   | Go (dedup).                              |
| `code_csharp`               | Code          | 1.2    | 2.0B       | `bigcode/the-stack-dedup`                   | C# — .NET applications and tooling.      |
| `code_jupyter`              | Code          | 1.0    | 2.0B       | `bigcode/the-stack-dedup`                   | Notebooks — machine learning / AI code.  |
| `nemotron_cc_math`          | Reasoning     | 1.0    | 18.0B      | `nvidia/Nemotron-CC-Math-v1` (`4plus`)      | High-quality math/reasoning corpus.      |
| `finemath`                  | Reasoning     | 0.9    | 3.0B       | `HuggingFaceTB/finemath` (`finemath-4plus`) | Math reasoning.                          |
| `fineweb_edu`               | General       | 0.7    | 4.0B       | `HuggingFaceFW/fineweb-edu` (`sample-10BT`) | Educational web content.                 |
| `climbmix`                  | General       | 0.6    | 5.0B       | `karpathy/climbmix-400b-shuffle`            | General pretraining, suppressed.         |
| `wikipedia`                 | General       | 0.5    | 3.0B       | `wikimedia/wikipedia` (`20231101.en`)       | English Wikipedia, suppressed.           |
| `fineweb2_id`               | General       | 0.6    | 3.0B       | `HuggingFaceFW/fineweb-2` (`ind_Latn`)      | Indonesian web text.                     |
| `wikipedia_id`              | General       | 0.5    | 0.2B       | `wikimedia/wikipedia` (`20231101.id`)       | Indonesian Wikipedia.                    |

> **48 sources total.** Instruction/chat/DPO-style datasets are **not** here — they live in the SFT pipeline (Section 3).

> 🇮🇩 **Indonesian in pretraining:** `fineweb2_id` + `wikipedia_id` contribute ~3.2B tokens (~3% of the
> mix). This is deliberately small — it exists so the 96K tokenizer learns real Indonesian sub-words and
> the base model has a language foundation to build on. The Indonesian SFT sets (Section 3) sit on top of
> it; SFT alone cannot teach a language the base model never saw during pretraining. Weights are kept at
> the general-knowledge tier (0.5–0.6) because weight controls how often a domain appears in mixed
> shards, and bursty language switching is a known loss-spike source.

> ⚠️ **Loss-spike prevention:** Raw logs (`.log`, `.xml`, `.json`) are converted to natural-language
> narratives by `convert_logs_to_nl.py` before pretraining:
>
> ```
> data/log/*   ──► convert_logs_to_nl.py ──► data/log_nl/*.jsonl     (local_security_logs)
> data/cloud/* ──► convert_logs_to_nl.py ──► data/cloud_nl/*.jsonl   (local_cloud_security)
> ```

---

## 3. SFT Data

SFT trains on a mixture of (A) bundled local datasets in `data/sft/` and (B) external
HuggingFace datasets downloaded by `scripts/data/download_sft_data.py`. Epoch counts
oversample small high-value datasets.

### 3A. Local bundled datasets (`data/sft/*.jsonl`)

Wired into the training mixture via `tasks/cybersec_sft.py` (default epochs shown) and `scripts/chat/chat_sft.py`.


| Dataset              | File                                           | Language | Default Epochs | Notes                                                |
| :--------------------- | :----------------------------------------------- | :--------- | :--------------- | :----------------------------------------------------- |
| CyberDefensive       | `cyber_defensive_conversations(_en).jsonl`     | ID + EN  | 1              | Defensive cybersec Q&A (SOC, triage, IR).            |
| CloudSecurity        | `cloud_security_sft(_en).jsonl`                | ID + EN  | 20             | Cloud IR (AWS, Azure, GCP).                          |
| MultiTurnSOC         | `multi_turn_soc_sft.jsonl`                     | EN       | 30             | Multi-turn SOC dialogues (oversampled).              |
| ToolOrientedCyber    | `tool_oriented_cyber_sft.jsonl`                | EN       | 20             | Tool-oriented cybersec (nmap, burp…).               |
| ToolCalling          | `tool_calling_conversations_en.jsonl`          | EN       | 15             | Tool-calling (WHOIS, dig, hashes).                   |
| MythosCombined       | `mythos_combined_sft(_en).jsonl`               | ID + EN  | 4              | Offensive/defensive narrative scenarios.             |
| MythosToolCalling    | `mythos_tool_calling(_en).jsonl`               | ID + EN  | 4              | Mythos with native tool calling.                     |
| mesosferValidation   | `mesosfer_validation_conversations(_en).jsonl` | ID + EN  | 2              | Domain alignment validation.                         |
| GeminiTeacher        | `gemini_teacher_conversations.jsonl`           | EN       | 2              | Gemini-distilled teacher conversations.              |
| Identity             | `identity_conversations(_en).jsonl`            | ID + EN  | 2              | Synthetic identity conversations.                    |
| Rules                | `rules.jsonl`                                  | EN       | 4              | Safety/format/behavioral instructions.               |
| SafetyArtifact       | `safety_artifact_conversations_en.jsonl`       | EN       | 4              | Refusals for attack automation vs allowed artifacts. |
| InstructionFollowing | `instruction_following_conversations_en.jsonl` | EN       | 4              | Format constraints (json, word count).               |

### 3B. External datasets (downloaded by `download_sft_data.py` into `data/sft/`)


| Key                          | Source / Origin                                         | Max Rows | Default Epochs | Notes                                                                                                       |
| :----------------------------- | :-------------------------------------------------------- | :--------- | :--------------- | :------------------------------------------------------------------------------------------------------------ |
| `primus_instruct`            | `trendmicro-ailab/Primus-Instruct`                      | 100K     | 1              | Trend Micro cybersec instructions (**gated**).                                                              |
| `primus_reasoning`           | `trendmicro-ailab/Primus-Reasoning`                     | 50K      | 1              | Cybersec chain-of-thought (**gated**).                                                                      |
| `cybernative_vuln_dpo`       | `CyberNative/Code_Vulnerability_Security_DPO`           | 10K      | 3              | Vulnerable vs fixed code (DPO→SFT).                                                                        |
| `openhermes`                 | `teknium/OpenHermes-2.5`                                | 50K      | 1              | General conversational instruction.                                                                         |
| `ultrachat`                  | `HuggingFaceH4/ultrachat_200k`                          | 100K     | 1              | Multi-turn conversations.                                                                                   |
| `trendyol_cyber_sft`         | `Trendyol/...-Cybersecurity-Instruction-Tuning-Dataset` | 53K      | 1              | Defensive cybersec Q&A.                                                                                     |
| `tiamz_cybersec`             | `Tiamz/cybersecurity-instruction-dataset`               | 15K      | 2              | Cybersec instruction Q&A.                                                                                   |
| `alpaca_indonesian`          | `ilhamfadheel/alpaca-cleaned-indonesian`                | 55K      | 1              | Indonesian instruction-following.                                                                           |
| `magpie_reasoning_sft`       | `Magpie-Align/Magpie-Reasoning-V2`                      | 50K      | 1              | DeepSeek-R1-Llama reasoning.                                                                                |
| `open_thoughts_sft`          | `open-thoughts/OpenThoughts-114k`                       | 50K      | 1              | Reasoning / CoT conversations.                                                                              |
| `nist_cybersec`              | `ethanolivertroy/nist-cybersecurity-training`           | 50K      | 1              | NIST cybersec training conversations.                                                                       |
| `fenrir_v2`                  | `AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1`        | 99K      | 1              | Cybersec Q&A (OWASP/MITRE/NIST).                                                                            |
| `code_feedback`              | `m-a-p/Code-Feedback`                                   | 50K      | 1              | Multi-language coding feedback/instruction.                                                                 |
| `numinamath_cot`             | `AI-MO/NuminaMath-CoT`                                  | 50K      | 1              | Competition math with chain-of-thought.                                                                     |
| `aquilax_security_reasoning` | `tuandunghcmut/AquilaX-AI-security-assistant-reasoning` | 18.3K    | 2              | Cybersec vuln-analysis reasoning (CoT). Reasoning template stripped of Llama-3 reserved tokens (**gated**). |
| `xlam_function_calling`      | `Salesforce/xlam-function-calling-60k`                  | 20K      | 1              | Generic named tool-calling (APIGen, cc-by-4.0); rendered as `<\|tool_start\|>` JSON `{name, arguments}` — shell, scanners, SQL, HTTP (**gated**). |
| `hermes_func_calling`        | `NousResearch/hermes-function-calling-v1` (`func_calling`) | 20K   | 2              | Multi-turn tool calling, apache-2.0. Ungated counterpart to xLAM — without `HF_TOKEN` this is the main external tool-calling source. `<tool_call>` XML → native tool/tool_output parts. |
| `hermes_glaive_func_calling` | `NousResearch/hermes-function-calling-v1` (`glaive_func_calling`) | 20K | 1     | Glaive function calling, already cleaned into ShareGPT turns. Preferred over raw `glaiveai/glaive-function-calling-v2`, which needs brittle text parsing. |
| `kimi_k3_traces`             | `greghavens/kimi-k3-coding-and-debugging-traces`        | 4K       | 4              | Agentic coding/debugging trajectories with multi-step tool use and reasoning (cc-by-4.0). OpenAI-style `tool_calls`; `arguments` JSON string decoded to dict so `parse_tool_call` accepts it. |
| `cot_collection`             | `kaist-ai/CoT-Collection` (`en`)                        | 50K      | 1              | Chain-of-thought rationales across many NLP tasks (cc-by-4.0). Repo ships only a loading script, so it is read via `revision=refs/convert/parquet`. |
| `qyrou_reasoning`            | `Qyrou/reasoning-corpus-4K-5M-v1`                       | 50K      | 1              | Reasoning distilled from frontier models, apache-2.0. Has an explicit `thought_trace` column, so no thinking-tag parsing needed. |

> All external datasets above are wired into the default training mixture
> (`tasks/cybersec_sft.py` → `build_cybersec_sft_tasks`, with CLI epoch flags in
> `scripts/chat/chat_sft.py`). A source is only trained on if its `data/sft/*.jsonl`
> file exists (download it first via `download_sft_data.py`); missing files are skipped.

---

## 4. Eval Data

Evaluation runs through CORE + domain probes (`scripts/eval/base_eval.py`,
`scripts/chat/chat_eval.py`). MCQ tasks report centered accuracy vs random baseline.

### 4A. General (CORE)


| Task                     | Source                 | Shots | Notes                           |
| :------------------------- | :----------------------- | :------ | :-------------------------------- |
| ARC-Easy / ARC-Challenge | `tasks/arc.py`         | —    | Science MCQ.                    |
| MMLU                     | `cais/mmlu`            | 5     | Broad knowledge MCQ.            |
| GSM8K                    | `tasks/gsm8k.py`       | —    | Grade-school math (generative). |
| HumanEval                | `tasks/humaneval.py`   | —    | Code generation (pass@1).       |
| SpellingBee              | `tasks/spellingbee.py` | —    | Character/spelling probe.       |

### 4B. Cybersecurity domain


| Task                     | Source                                          | Shots | Notes                                                          |
| :------------------------- | :------------------------------------------------ | :------ | :--------------------------------------------------------------- |
| `mmlu_computer_security` | `cais/mmlu` (computer_security)                 | 5     | Cybersec subset of MMLU.                                       |
| `cybermetric_500`        | `tuandunghcmut/cybermetric_500_v1`              | 3     | Cybersecurity knowledge MCQ.                                   |
| `secbench_mcq_en`        | `secbench-hf/SecBench` (`data/MCQs_2730.jsonl`) | 3     | English single-answer MCQs (~652); includes Logical Reasoning. |

### 4C. Coding domain (CodeMMLU, 3-shot)


| Task                            | Subset               | Source               |
| :-------------------------------- | :--------------------- | :--------------------- |
| `codemmlu_programming_syntax`   | programming_syntax   | `Fsoft-AIC/CodeMMLU` |
| `codemmlu_software_principles`  | software_principles  | `Fsoft-AIC/CodeMMLU` |
| `codemmlu_code_completion`      | code_completion      | `Fsoft-AIC/CodeMMLU` |
| `codemmlu_code_repair`          | code_repair          | `Fsoft-AIC/CodeMMLU` |
| `codemmlu_execution_prediction` | execution_prediction | `Fsoft-AIC/CodeMMLU` |

---

## 5. Security Vocabulary Markers

Keyword tuples used for filtering, quality scoring, and routing in `prepare_data.py`:

* **`CAUSAL_MARKERS`** — cause-effect / vulnerability impact (e.g. *"exploits"*, *"bypasses"*, *"allows attacker to"*).
* **`MECHANISM_TERMS`** — memory corruption / RE concepts (e.g. *"buffer overflow"*, *"rop chain"*, *"use-after-free"*).
* **`MARKETING_TERMS`** — promotional phrases (e.g. *"book a demo"*, *"free trial"*); used as a **negative filter**.
* **`SECURITY_STACK_TERMS`** — infrastructure / defensive tooling (e.g. *"firewall"*, *"siem"*, *"yara"*).
* **`SECURITY_CODE_TERMS`** — low-level / exploitation vocabulary (e.g. *"shellcode"*, *"ghidra"*, *"vtable"*, *"syscall"*).

---

## 6. Data Source Schema

A frozen dataclass (`Source`) defines the config contract for RSS/GitHub/PDF/scraper sources:
`name`, `source_type` (`rss`, `github_repo`, `local_files`, `brightdata_scraper`, `url_json`, `pdf_manifest`,
`nvd_json_feeds`, `circl_ndjson_dump`), `domain` & `primary_subdomain`, `expected_tier`
(`GOLD`/`SILVER`/`BRONZE_CLEAN`), `estimated_tokens` & `max_tokens`, `description`, and `config`.

HuggingFace sources are configured directly in `DATASET_SOURCES` with `hf_name`, optional `hf_subset`,
`text_column`, `split`, `streaming`, and `max_tokens`.

---

## 7. Technical Notes

1. **Keyword filtering:** `MARKETING_TERMS` suppresses vendor "sales" language; technical content with
   `CAUSAL_MARKERS` / `MECHANISM_TERMS` is preferred.
2. **Dynamic token scaling:** `int(base_value * scale)` with `scale = target_tokens / 500_000_000`
   keeps inter-domain ratios balanced when scaling toward ~100B tokens.
3. **Data tiering:** *Gold* (curated cybersec knowledge, synthetic SOC/IR, Primus-Seed/Reasoning),
   *Silver* (GitHub repos, Sigma/ExploitDB, The-Stack code), *Bronze→Gold* (raw logs upgraded after
   NL-narrative conversion).
4. **Interleaving:** Temperature-weighted sampling with `cluster_size=32`, `sampling_temperature=1.2`
   in `interleaved_shuffle_main` reduces abrupt domain transitions and single-domain dominance.
5. **Pretraining vs SFT boundary:** Pretraining uses raw-text corpora only. Instruction/chat/DPO/CoT
   datasets belong to SFT (`download_sft_data.py`), never to pretraining.

## 8. Tool-Calling Token Protocol

Tool-calling is taught during SFT using mesosfer's own special tokens (no
dependency on Llama-3 / OpenAI tool formats):

| Token pair | Meaning | Supervised? |
| :--- | :--- | :--- |
| `<\|calc_start\|>` … `<\|calc_end\|>` | Inline expression evaluator — arithmetic and simple string ops, run by the engine at inference (not a Python REPL) | yes |
| `<\|tool_start\|>` … `<\|tool_end\|>` | Generic named tool call. Payload is JSON `{"name": ..., "arguments": {...}}` — covers shell, network scanners, SQL, HTTP, etc. | yes |
| `<\|output_start\|>` … `<\|output_end\|>` | Tool / REPL result returned from the environment | no (not supervised) |

Sources:
- **Local** (`data/sft/`): `tool_calling_conversations_en.jsonl`, `mythos_tool_calling*.jsonl`, `tool_oriented_cyber_sft.jsonl` (generated by `dev/` scripts).
- **External** (HF), all converted to the generic `<|tool_start|>` format in `download_sft_data.py`:
  `xlam_function_calling` via `convert_function_calling` (**gated**);
  `hermes_func_calling` and `hermes_glaive_func_calling` via `convert_sharegpt_tool_calls` (`<tool_call>` XML tags);
  `kimi_k3_traces` via `convert_openai_tool_calls` (OpenAI `tool_calls`, multi-step agentic).
  The Hermes and Kimi sets are ungated, so tool-calling data is available without `HF_TOKEN`.

> Every converter emits `arguments` as a JSON **object**, because
> `mesosfer.eval.harness.parse_tool_call` rejects a non-dict `arguments` and turns the
> call into an error. Source datasets that use a string or a list (raw Glaive, the old
> `mythos_combined` payloads) are normalised before being written.

> The generic `<|tool_start|>` payload carries the tool name, so a single token
> pair supports every tool family. At inference the model only *emits* tool calls;
> an external harness is responsible for execution — shell/network tools must never
> be auto-executed by the model server.
