# Mesosfer - Cybersecurity LLM Framework

[![License MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](pyproject.toml)
[![PyTorch 2.6+](https://img.shields.io/badge/pytorch-2.6+-red.svg)](https://pytorch.org/)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-green.svg)](RUN_NVIDIA.md)
[![ROCm 7.0](https://img.shields.io/badge/ROCm-7.0-orange.svg)](RUN_AMD.md)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow.svg)](https://huggingface.co/Pabloraka)
[![Sponsor](https://img.shields.io/badge/%E2%9D%A4%20Sponsor-GitHub-ea4aaa.svg)](https://github.com/sponsors/PabloRaka)

A minimal full-stack cybersecurity-focused language model — from pretraining to chat.

Mesosfer is inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) and follows Andrej Karpathy's educational approach to LLM training: clean, readable code that actually works. It is specifically optimized for cybersecurity and secure coding domains.

## Features

- **Pretraining** — Train base models from scratch using scaling laws
- **Tokenizer Training** — Custom BPE tokenizer (96K vocab, cybersec-aware: hex/hash-aware splitting)
- **Supervised Fine-Tuning (SFT)** — Instruction tuning with cybersecurity datasets
- **Reinforcement Learning (RL)** — GRPO-style RL on cybersecurity tasks
- **RLHF Data Collection** — Human feedback via thumbs up/down UI, stored to `data/rlhf/`
- **Evaluation** — CORE benchmark + cybersecurity domain probes
- **Chat Interface** — CLI and WebUI for model interaction
  - Syntax-highlighted code blocks (Python, Rust, JSON, etc.)
  - Markdown rendering (headings, lists, bold/italic, inline code)
  - Welcome screen with centered input on new conversation

## Architecture Highlights

| Feature | Implementation |
|---------|----------------|
| Attention | Group-Query Attention (GQA) with Flash Attention 2/3 |
| Positional Encoding | Rotary embeddings (RoPE) |
| Activation | ReLU² |
| Normalization | RMSNorm (no learnable parameters) |
| Optimizer | MuonAdamW (Muon for matrices, AdamW for embeddings) |
| Training Precision | BF16 (all GPUs) / FP8 (Hopper/H100 only) |
| Value Embeddings | ResFormer-style alternating layers |

## Hardware Support

| Vendor | GPUs | Backend |
|--------|------|---------|
| NVIDIA | H100, H200, B200, A100, L40S, L4, RTX 4090, T4 | CUDA |
| AMD | MI355X, MI350X, MI325X, MI300X, MI250X, MI210 | ROCm 7.0+ |
| Apple | M1/M2/M3 Pro/Max | MPS |
| Intel/x86 | CPU | CPU |

> For detailed setup instructions, see:
> - [RUN_NVIDIA.md](RUN_NVIDIA.md) — NVIDIA GPU setup
> - [RUN_AMD.md](RUN_AMD.md) — AMD GPU setup (ROCm 7.0 pre-built image)

---

## Quick Start

### 1. Install Dependencies

```bash
# Clone the repository
git clone https://github.com/your-repo/mesosfer.git
cd mesosfer

# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.12
source .venv/bin/activate

# NVIDIA CUDA
uv sync --extra gpu

# AMD ROCm 7.0 (pre-built image — PyTorch already installed)
# uv sync --extra rocm --no-build-isolation
# pip install flash-attn --no-build-isolation
```

---

### 2. Prepare Data

Data preparation must be completed **before** training. Run these steps in order:

#### Step 2a — Download and Prepare Pretraining Datasets

This single command automatically downloads the ClimbMix general pretraining shards (~17 GB) and prepares/interleaves the cybersecurity domain datasets (CVE feeds, HuggingFace cybersec datasets, and local files):

```bash
python -m scripts.data.prepare_data

# Check progress
python -m scripts.data.prepare_data --status

# Dry-run to preview sources
python -m scripts.data.prepare_data --dry-run
```

#### Step 2b — Convert raw security logs to natural language

Converts `data/log/` and `data/cloud/` files to NL narratives (prevents loss spikes).
Output: `data/log_nl/` and `data/cloud_nl/`

```bash
python -m scripts.data.convert_logs_to_nl

# Preview first 3 documents without writing
python -m scripts.data.convert_logs_to_nl --dry-run
```

> Step 2b can run in parallel with 2a. It only reads from `data/` in the repo.

---

### 3. Train Tokenizer

Train a 96K BPE tokenizer on the prepared data. Requires Step 2a to be complete.

```bash
python -m scripts.train.tok_train

# Evaluate tokenizer compression ratio vs GPT-2 and GPT-4
python -m scripts.eval.tok_eval
```

---

### 4. Pretrain Base Model

Requires Steps 2a, 2b, and 3 to be complete.

```bash
# Depth 16 (~2.82B params @ 96K vocab with default value embeddings; use --ve-layers=2 since
# VE tables otherwise outweigh the transformer at this depth) — ~10B tokens at ratio=10
python -m scripts.train.base_train \
    --depth=16 \
    --aspect-ratio=128 \
    --target-param-data-ratio=10 \
    --ve-layers=2 \
    --grad-checkpoint \
    --device-batch-size=32 \
    --warmup-steps=500 \
    --window-pattern=L \
    --save-every=1000 \
    --core-metric-every=1000 \
    --run=d16_run

# Or use the full pipeline script (handles setup + tokenizer + pretrain + SFT)
WANDB_RUN=my_run bash runs/speedrun.sh
```

> **AMD ROCm note:** Use `--window-pattern=L` (full attention). Sliding window (`SSL`, `SSSL`)
> is not yet supported in the ROCm FA2 Triton backward pass.

---

### 5. Evaluate Base Model

```bash
python -m scripts.eval.base_eval \
    --model-tag d16 \
    --device-batch-size 32
```

---

### 6. Supervised Fine-Tuning (SFT)

Requires Step 4 to be complete.

#### Step 6a — Download external SFT datasets (recommended)

The local bundled SFT files in `data/sft/*.jsonl` ship with the repo, but the
external HuggingFace instruction/reasoning datasets must be fetched first.
Gated datasets (`Primus-*`, `AquilaX`) require `HF_TOKEN` and accepting the
dataset terms — without a token they are simply skipped.

```bash
python -m scripts.data.download_sft_data

# List available SFT sources
python -m scripts.data.download_sft_data --list

# Download only specific sources
# python -m scripts.data.download_sft_data --sources openhermes ultrachat
```

#### Step 6b — Run SFT

Automatically mixes the local + downloaded cybersecurity SFT datasets from
`data/sft/` (any missing file is skipped with a warning).

```bash
python -m scripts.chat.chat_sft \
    --device-batch-size=32 \
    --run=sft_run

# Disable cybersec SFT (for ablation)
# python -m scripts.chat.chat_sft --disable-cybersec-sft
```

---

### 7. Chat with Your Model

```bash
# CLI chat
python -m scripts.chat.chat_cli -p "Explain CVE-2021-44228 (Log4Shell)"

# Web UI (opens http://localhost:8000)
python -m scripts.chat.chat_web

# Web UI — multi-GPU (4 workers)
python -m scripts.chat.chat_web --num-gpus 4

# Web UI — load specific checkpoint
python -m scripts.chat.chat_web --model-tag d16 --step 14000
```

The Web UI includes:
- **Welcome screen** — centered input shown before the first message
- **Syntax-highlighted code blocks** — Python, Rust, JSON, Bash, and more, with a one-click copy button
- **Markdown rendering** — headings, lists, bold/italic, inline code
- **Thumbs up / down feedback** — per-response rating that saves to `data/rlhf/feedback.jsonl`

---

### 8. Collect RLHF Feedback

Human preference data is collected automatically while chatting via the Web UI.
Each 👍 or 👎 click on an assistant response appends a record to `data/rlhf/feedback.jsonl`.

```jsonc
{
  "timestamp": "2026-05-20T10:00:00+00:00",
  "message_index": 1,
  "rating": "negative",
  "reason": "factually_incorrect",
  "comment": "The CVE number is wrong.",
  "conversation": [
    { "role": "user",      "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```

Use this data to train a reward model or run DPO fine-tuning in a future step.

---

## Web UI API Endpoints

The Web UI server (`scripts/chat/chat_web.py`) exposes the following endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serve the chat UI (`mesosfer/ui.html`) |
| `GET` | `/logo.svg` | Serve the Mesosfer logo |
| `GET` | `/interface/*` | Serve static assets (CSS, JS) |
| `POST` | `/chat/completions` | Streaming chat completion (SSE) |
| `POST` | `/feedback` | Submit thumbs up/down feedback (saved to `data/rlhf/feedback.jsonl`) |
| `GET` | `/health` | Health check + worker pool status |
| `GET` | `/stats` | Worker pool statistics and GPU utilization |

### Feedback endpoint payload

```json
POST /feedback
{
  "message_index": 1,
  "rating": "negative",
  "reason": "factually_incorrect",
  "comment": "Optional free-text",
  "conversation": [
    { "role": "user",      "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```

Valid `reason` values: `inappropriate_response`, `continuous_repetition`, `factually_incorrect`, `too_verbose`, `formatting_issues`, `other`.

---

## Full Pipeline Summary

```
Step 2a: scripts.data.prepare_data         ─┐  (auto-downloads ClimbMix shards)
Step 2b: scripts.data.convert_logs_to_nl   ─┘  can run in parallel
         ↓
Step 3:  scripts.train.tok_train
         scripts.eval.tok_eval
         ↓
Step 4:  scripts.train.base_train
         ↓
Step 5:  scripts.eval.base_eval
         ↓
Step 6a: scripts.data.download_sft_data    (fetch external SFT datasets)
Step 6b: scripts.chat.chat_sft
         ↓
Step 7:  scripts.chat.chat_cli / chat_web
         ↓
Step 8:  Collect RLHF feedback via Web UI  →  data/rlhf/feedback.jsonl
         (future: reward model training / DPO)
```

---

## Project Structure

```
mesosfer/
├── mesosfer/               # Library (importable)
│   ├── model/              # GPT model, attention, optimization
│   ├── data/               # Dataset download, dataloader, tokenizer
│   ├── eval/               # CORE eval, BPB, engine
│   ├── utils/              # Common utilities, checkpointing, reporting
│   ├── interface/          # Web UI static assets
│   │   ├── style.css       # All UI styles (chat, code blocks, feedback, empty state)
│   │   ├── chat.js         # Chat logic, streaming, slash commands, markdown rendering
│   │   ├── feedback.js     # Thumbs up/down, feedback modal, POST /feedback
│   │   └── markdown.js     # Markdown parser + syntax-highlighted code block renderer
│   └── ui.html             # HTML shell (loads interface/ assets)
├── scripts/
│   ├── train/              # base_train.py, tok_train.py
│   ├── chat/               # chat_sft.py, chat_rl.py, chat_cli.py, chat_web.py
│   ├── eval/               # base_eval.py, tok_eval.py
│   └── data/               # prepare_data.py, convert_logs_to_nl.py, download_sft_data.py
├── tasks/                  # Eval tasks (MMLU, GSM8K, cybersec_sft, cybersec_rl)
├── data/
│   ├── log/                # Raw security logs (input to convert_logs_to_nl)
│   ├── cloud/              # Raw cloud audit logs (input to convert_logs_to_nl)
│   ├── log_nl/             # NL narratives from logs (output, used in training)
│   ├── cloud_nl/           # NL narratives from cloud logs (output, used in training)
│   ├── sft/                # Cybersecurity SFT conversations
│   ├── rlhf/               # Human feedback collected via Web UI
│   │   └── feedback.jsonl  # Appended at runtime (gitignored)
│   ├── synthetic-ir/       # Synthetic incident response data
│   ├── synthetic-soc/      # Synthetic SOC analyst data
│   └── reverse-engineering/ # RE/exploitation analysis data
├── runs/                   # Training run scripts (speedrun, miniseries, etc.)
└── tests/                  # Unit tests
```

---

## Training Scripts

| Script | Description | Hardware |
|--------|-------------|----------|
| `runs/speedrun.sh` | Full pipeline: pretrain + SFT + eval | 8x H100/A100 or 1x MI300X |
| `runs/scaling_laws.sh` | Research: optimal model configs | 8x GPU |
| `runs/miniseries.sh` | Train multiple depths (12-26) | 8x GPU |
| `runs/runcpu.sh` | Demo on CPU/MacBook | CPU/MPS |

---

## Key Training Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--depth` | 32 | Transformer depth (32 ≈ 13.7B params at 96K vocab / legacy VE; ~9.3B with `--ve-layers=4`) |
| `--aspect-ratio` | 128 | Model dimension = depth × aspect-ratio |
| `--head-dim` | 128 | Attention head dimension |
| `--max-seq-len` | 2048 | Maximum context length |
| `--device-batch-size` | 32 | Batch size per GPU |
| `--target-param-data-ratio` | 15 | Data-to-parameters ratio (~100B tokens at depth 32; 10 = faster/undertrained) |
| `--warmup-steps` | 200 | LR warmup steps (200 for depth 20+) |
| `--window-pattern` | L | Attention window: L=full, S=sliding (L required for ROCm) |
| `--ve-layers` | -1 | Value-embedding placement: -1 = legacy (~half layers); K≥0 = first K layers + last only (cuts vocab-sized VE tables & memory) |
| `--grad-checkpoint` | off | Activation checkpointing: recompute each block in backward to save activation memory (~30% extra compute; needed for deep/wide models) |
| `--save-every` | 1000 | Checkpoint every N steps |
| `--core-metric-every` | 5000 | CORE eval every N steps |

---

## Configuration

### Environment Variables

```bash
# Backend selection (auto-detected if not set)
export mesosfer_TORCH_BACKEND=cuda    # cuda, rocm, cpu

# Compute dtype override
export mesosfer_DTYPE=bfloat16        # bfloat16, float16, float32

# Cache directory (datasets, checkpoints, tokenizer)
export mesosfer_BASE_DIR="$HOME/.cache/mesosfer"

# Wandb logging
export WANDB_RUN=my_training_run
```

### Dataset Configuration

See [DATASET.md](DATASET.md) for dataset sources, sampling weights, and token budgets.

---

## Requirements

- Python 3.12+
- PyTorch 2.6.0+ (ROCm 7.0) or 2.9.1+ (CUDA 12.8)
- NVIDIA GPU (CUDA 12.8+) or AMD GPU (ROCm 7.0+)
- 16GB+ GPU VRAM (80GB+ recommended for depth 32)

---

## Development

```bash
# Install dev dependencies
uv sync --dev

# Run tests
pytest tests/

# Format code
ruff format .

# Lint
ruff check .
```

---

## Model Performance

| Model | Parameters | Tokens | Validation BPB | CORE Score |
|-------|-----------|--------|----------------|------------|
| GPT-2 (reference) | ~124M | ~5B | ~0.9700 | ~0.2500 |
| mesosfer d24 (Base, ratio=10) | ~1.38B | ~7.3B | 0.7337 | 0.2541 |
| mesosfer d24 (SFT, step 1250) | ~1.38B | ~7.3B + ~1.2B | 0.7312 | **0.3486 (ChatCORE)** |
| mesosfer d32 | ~13.7B | ~100B target (ratio=15) | TBA | TBA |

> CORE score: average of MMLU (5-shot), GSM8K (COT), ARC-C, HumanEval (pass@1), Only base model

---

## Scaling to Larger Models

The architecture is designed to scale without fundamental changes. Model size is controlled by two arguments:

```
model_dim = depth × aspect_ratio  (rounded up to nearest multiple of head_dim)
num_heads = model_dim / head_dim
```

### Depth × Aspect Ratio → Parameter Count

| Depth | Aspect Ratio | Model Dim | ~Total Params | Dataset needed (ratio=10) | Covered by ~100B dataset? |
| -------| --------------| -----------| ---------------| ---------------------------| ---------------------------|
| 16    | 64           | 1024      | ~0.9B         | ~2.7B                     | ✅                         |
| 16    | 128          | 2048      | ~2.82B        | ~10B                      | ✅                         |
| 18    | 64           | 1152      | ~1.2B         | ~3.6B                     | ✅                         |
| 20    | 64           | 1280      | ~1.5B         | ~4.8B                     | ✅                         |
| 22    | 64           | 1408      | ~1.8B         | ~6.2B                     | ✅                         |
| 24    | 64           | 1536      | ~2.2B         | ~7.8B                     | ✅                         |
| 28    | 64           | 1792      | ~3.1B         | ~12.0B                    | ✅                         |
| 32    | 128          | 4096      | ~13.7B        | ~68B                      | ✅                         |
| 36    | 128          | 4608      | ~18.2B        | ~96B                      | ✅                         |
| 40    | 128          | 5120      | ~23.7B        | ~131B                     | ❌ need more data          |
| 44    | 128          | 5632      | ~30.0B        | ~173B                     | ❌ need more data          |
| 48    | 128          | 6144      | ~37.5B        | ~223B                     | ❌ need more data          |

> "Dataset needed" = scaling params × ratio=10. Scaling params = transformer matrices + lm_head (excludes embeddings).
> Current dataset (~100B tokens) covers depth 16–36 at ratio=10. For depth 40+, additional data sources are required.
> For Chinchilla-optimal training (ratio=20), double the token requirements above.
> Total params are at 96K vocab with default (legacy) value embeddings. `--ve-layers=K` cuts the vocab-sized
> value-embedding tables substantially (e.g. d32 drops from ~13.7B to ~9.3B at `--ve-layers=4`), saving memory.
> At depth 16, default `--ve-layers=-1` (8 VE tables) outweighs the ~805M-param transformer matrices — pass `--ve-layers=2` (3 tables) for a well-balanced ~1B-class model.

To train a larger model, simply pass the desired depth and aspect ratio:

```bash
# ~3B model (depth 28, aspect-ratio 64)
python -m scripts.train.base_train \
    --depth=28 \
    --aspect-ratio=64 \
    --head-dim=128 \
    --target-param-data-ratio=10 \
    --run=d28_run

# ~7B model (depth 40, aspect-ratio 64)
python -m scripts.train.base_train \
    --depth=40 \
    --aspect-ratio=64 \
    --head-dim=128 \
    --target-param-data-ratio=10 \
    --run=d40_run
```

Features like GQA, Flash Attention 2/3, RoPE, RMSNorm, and BF16/FP8 are already implemented and designed for large-scale training. The main practical constraints for 7B+ models are **data volume** (current dataset ~100B tokens covers up to ~6.5B scaling params at ratio≈15) and **hardware** (7B in BF16 requires ~14GB weights + optimizer state, recommend 2–4× A100/H100 80GB or 1× MI300X).

---

## Sponsors

If you find Mesosfer useful and want to support its ongoing development, please consider sponsoring the project!

[![Sponsor on GitHub](https://img.shields.io/badge/Sponsor-PabloRaka-ea4aaa?style=for-the-badge&logo=github-sponsors&logoColor=white)](https://github.com/sponsors/PabloRaka)

---

## Acknowledgments

- [Andrej Karpathy](https://karpathy.ai/) — nanoGPT inspiration
- [PyTorch](https://pytorch.org/) — Deep learning framework
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) — Efficient attention
- [Muon optimizer](https://kellerjordan.github.io/posts/muon/) — Matrix parameter optimization

## License

[MIT License](LICENSE)
