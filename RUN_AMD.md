# Running Mesosfer on AMD GPUs

This guide covers the setup and execution of Mesosfer on AMD GPUs using ROCm.

## Prerequisites

### Hardware

| GPU | Architecture | Compute Unit | BF16 FLOPS | Memory | Notes |
|-----|-------------|--------------|------------|--------|-------|
| MI355X | CDNA 4 | ~304 | 2500 TFLOPS | 256GB HBM3 | Latest, compute-focused |
| MI350X | CDNA 3 | ~304 | 1.6 PetaFLOPS | 256GB HBM3 | Compute powerhouse |
| MI325X | CDNA 3 | ~304 | 1.3 PetaFLOPS | 256GB HBM3 | Good value |
| MI300X | CDNA 3 | ~228 | 1.3 PetaFLOPS | 192GB HBM3 | Popular for inference |
| MI300A | CDNA 3 | ~228 | 980 TFLOPS | 192GB HBM3 | APU variant |
| MI250X | CDNA 2 | ~208 | 383 TFLOPS | 128GB HBM2 | Older compute GPU |
| MI250 | CDNA 2 | ~208 | 362 TFLOPS | 128GB HBM2 | MI250X variant |
| MI210 | CDNA 2 | ~104 | 181 TFLOPS | 64GB HBM2 | Entry compute GPU |

> **Note:** AMD ROCm GPUs all support BF16 natively, including MI210 (older generation).

### Software

- **OS:** Linux only (Ubuntu 20.04+, RHEL/CentOS 8+, SLES 15+)
- **Python:** 3.12+
- **ROCm:** 7.0+ (recommended — use the `PyTorch 2.6.0 - ROCm 7.0` pre-built image)
- **PyTorch:** 2.6.0+ with ROCm support (pre-installed in the image)

> **Important:** ROCm is Linux-only. Windows support via WSL2 is experimental.

### Check ROCm Installation

```bash
# Check ROCm version
rocm-smi --version

# List available GPUs
rocm-smi

# Check rocminfo
rocminfo | grep -A 10 "Agent"
```

---

## Installation

### 1. Use the Pre-Built ROCm Image (Recommended)

The easiest way to get started is with the official PyTorch ROCm image:

```
PyTorch 2.6.0 - ROCm 7.0.0
```

This image has PyTorch, ROCm drivers, and all GPU libraries pre-installed. No manual ROCm installation needed.

> If you need to install ROCm manually on bare metal, follow the [ROCm Installation Guide](https://rocm.docs.amd.com/projects/install/en/latest/).

### 2. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Create Virtual Environment

```bash
cd /path/to/mesosfer
uv venv --python 3.12
source .venv/bin/activate
```

### 4. Install with ROCm Support

```bash
# For ROCm 7.0 pre-built images (PyTorch 2.6.0 already installed in image):
uv sync --extra rocm --no-build-isolation

# Install Flash Attention 2 for ROCm (enables faster training vs SDPA fallback)
uv pip install flash-attn --no-build-isolation
```

> **Why `--no-build-isolation`?** The `PyTorch 2.6.0 - ROCm 7.0` image has torch
> pre-installed at the system level. This flag tells uv to reuse it instead of
> downloading from the WHL index (which only carries ROCm 6.4 builds).
> The same flag is needed for `flash-attn` because it compiles against the
> pre-installed ROCm torch headers.

### 5. Verify Installation

```bash
# Check ROCm PyTorch
python -c "import torch; print(f'ROCm: {torch.cuda.is_available()}, Version: {torch.version.hip}')"
# Expected: ROCm: True, Version: 7.0.0...

# Verify GPU detected
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
# Expected: GPU: AMD Instinct MI300X (or similar)
```

---

## Environment Variables

### Required

```bash
# Set backend explicitly (required for AMD)
export mesosfer_TORCH_BACKEND=rocm

# Optimize memory allocator
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# Limit OpenMP threads
export OMP_NUM_THREADS=1

# ROCm specific
export HIP_VISIBLE_DEVICES=0,1,2,3  # Select GPUs (0-indexed)
```

### Optional

```bash
# Override compute dtype (bfloat16 recommended for AMD)
export mesosfer_DTYPE=bfloat16

# Cache directory (default: ~/.cache/mesosfer)
export mesosfer_BASE_DIR="$HOME/.cache/mesosfer"

# Weights & Biases logging
export WANDB_RUN=my_training_run
# To enable wandb: wandb login

# ROCm tuning
export ROCM_BLIS_LC=1  # Enable ROCm BLIS optimizations
```

### Multi-GPU Setup

```bash
# For 8-GPU MI300X node:
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# For specific GPU selection:
export HIP_VISIBLE_DEVICES=0,2,4,6  # Use GPUs 0, 2, 4, 6 only
```

---

## GPU-Specific Notes

### MI355X / MI350X / MI325X (CDNA 3/4)

Latest AMD compute GPUs with highest performance:
- Full BF16 support
- Flash Attention 2 via ROCm 7.0
- ROCm 7.0+ recommended

```bash
export mesosfer_TORCH_BACKEND=rocm
export ROCM_BLIS_LC=1

# 8-GPU configuration with depth 24 best-practice flags
torchrun --standalone --nproc_per_node=8 -m scripts.train.base_train -- \
    --depth=24 \
    --target-param-data-ratio=10 \
    --device-batch-size=16 \
    --warmup-steps=200 \
    --window-pattern=SSL \
    --save-every=1000 \
    --core-metric-every=5000 \
    --run=$WANDB_RUN
```

### MI300X / MI300A

Popular for large-scale inference and training:
- 8 GCDs (MI300X) or 4 GCDs (MI300A)
- 192GB HBM3 per GPU
- ROCm 7.0+ recommended

> **MI300X GCD note:** MI300X exposes 8 GCDs but PyTorch sees them as **1 logical device**
> with 192GB unified VRAM. `torch.cuda.device_count()` returns 1, not 8.
> Use `--nproc_per_node=1` for single-node training on MI300X.

```bash
# Single MI300X (1 logical device = 8 GCDs unified)
export HIP_VISIBLE_DEVICES=0
torchrun --standalone --nproc_per_node=1 -m scripts.train.base_train -- \
    --depth=24 \
    --target-param-data-ratio=10 \
    --device-batch-size=16 \
    --warmup-steps=200 \
    --window-pattern=SSL \
    --save-every=500 \
    --core-metric-every=5000 \
    --run=$WANDB_RUN
```

### MI250X / MI250

Older but capable compute GPUs:
- 2 GCDs per GPU
- 128GB HBM2 per GPU
- ROCm 5.7+ recommended

```bash
# Each GCD is a separate "device" in PyTorch
# MI250X shows as 2 devices
export HIP_VISIBLE_DEVICES=0,1,2,3  # First 2 GPUs (4 GCDs)
```

### MI210

Entry-level compute GPU:
- Single GCD
- 64GB HBM2
- Good for single-GPU training or development

```bash
# Single MI210
export HIP_VISIBLE_DEVICES=0

python -m scripts.train.base_train -- \
    --depth=12 \
    --device-batch-size=8 \
    --run=dummy
```

---

## Training Scripts

### Data Pipeline Overview

mesosfer auto-merges parquet shards from two sources for pretraining:

```
~/.cache/mesosfer/
├── base_data_climbmix/           ← downloaded by `mesosfer.data.dataset -n 170`
│   ├── shard_00000.parquet
│   ├── ...
│   └── shard_06542.parquet       ← always used as validation shard
└── base_data_cybersecurity/      ← created by `scripts.data.prepare_data` (auto-merged)
    ├── shard_00000.parquet
    └── ...
```

The dataloader (`mesosfer/data/dataloader.py`) reads from both directories. Auxiliary
shards from `base_data_cybersecurity/` are placed BEFORE primary ClimbMix shards in
the iteration order, so the validation shard always comes from ClimbMix.

**Recommended pipeline order:**

```bash
# 1. Download ClimbMix general pretraining data (~17GB)
python -m mesosfer.data.dataset -n 170

# 2. Convert raw security logs to natural language (already done in repo)
python -m scripts.data.convert_logs_to_nl

# 3. Prepare cybersecurity dataset (downloads + interleaves cybersec sources)
python -m scripts.data.prepare_data
# Output: ~/.cache/mesosfer/base_data_cybersecurity/shard_*.parquet

# 4. Train tokenizer (uses ClimbMix only — fast, ~10 min)
python -m scripts.train.tok_train

# 5. Pretrain depth 24 (auto-merges both directories at training time)
torchrun --standalone --nproc_per_node=1 -m scripts.train.base_train -- \
    --depth=24 --target-param-data-ratio=10 --device-batch-size=16 \
    --warmup-steps=200 --window-pattern=SSL --run=$WANDB_RUN

# 6. SFT (cybersec mixture auto-included via tasks/cybersec_sft.py)
torchrun --standalone --nproc_per_node=1 -m scripts.chat.chat_sft -- \
    --device-batch-size=16 --run=$WANDB_RUN
```

> **Steps 1, 2, and 3 can run in parallel.** Tokenizer training (step 4) only
> needs step 1 to be complete. Pretraining (step 5) needs steps 1, 3, and 4.

### Quick Start: GPT-2 Class Training

**Note:** Training times on AMD GPUs vary by GPU model. MI300X 8x similar to H100.

```bash
# Basic run (no wandb logging)
bash runs/speedrun.sh

# With wandb logging
WANDB_RUN=my_run bash runs/speedrun.sh

# In a screen session
screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh
```

> **Note:** Some scripts default to `--extra gpu` (NVIDIA). For AMD, ensure your `uv sync` used `--extra rocm`.

### Scaling Laws Research

```bash
export NPROC_PER_NODE=8
export WANDB_RUN=scaling_rocm
export mesosfer_BASE_DIR="$HOME/.cache/mesosfer"

bash runs/scaling_laws.sh
```

### Miniseries (Multiple Depth Sweep)

```bash
# Default series
bash runs/miniseries.sh

# Custom name
bash runs/miniseries.sh mi300x_run

# Skip setup (if already done)
SKIP_SETUP=1 bash runs/miniseries.sh
```

### CPU Demo

```bash
bash runs/runcpu.sh
```

---

## Command Line Arguments

Common arguments for `scripts/train/base_train.py`:

| Argument | Default | Description |
|----------|---------|-------------|
| `--depth` | 20 | Transformer depth |
| `--aspect-ratio` | 64 | model_dim = depth * aspect_ratio |
| `--head-dim` | 128 | Attention head dimension |
| `--max-seq-len` | 2048 | Context length |
| `--device-batch-size` | 32 | Per-GPU batch size |
| `--total-batch-size` | -1 | Auto-compute if -1 |
| `--target-param-data-ratio` | 12 | Chinchilla-optimal ~20 |
| `--target-flops` | -1 | Fixed FLOP budget |
| `--num-iterations` | -1 | Auto-compute if -1 |
| `--run` | dummy | wandb run name |

### Batch Size Guidelines by GPU

```bash
# MI355X/MI350X (256GB memory) — depth 24 can use --device-batch-size=32
--device-batch-size=32

# MI325X (256GB memory) — depth 24
--device-batch-size=32

# MI300X (192GB memory) — depth 24 can use 16, even 24 with FP8
--device-batch-size=16

# MI250X (128GB memory) — depth 24
--device-batch-size=8

# MI210 (64GB memory) — depth 16 max recommended
--device-batch-size=4
```

> **Note:** MI300X has 2.4× the VRAM of H100 (80GB). Don't be too conservative
> — `--device-batch-size=16` is the sweet spot for d24 with 2048 seq_len, and
> 24-32 is achievable when training without optimizer state replication.

---

## Post-Training: Chat Interface

```bash
# CLI Chat
python -m scripts.chat.chat_cli -p "Why is the sky blue?"

# Web UI
python -m scripts.chat.chat_web
```

---

## Troubleshooting

### "ROCm not available" Error

1. **Verify ROCm installation:**
   ```bash
   rocm-smi
   ```

2. **Check PyTorch ROCm support:**
   ```bash
   python -c "import torch; print(f'ROCm: {torch.cuda.is_available()}, Version: {torch.version.hip}')"
   ```

3. **Check HIP_VISIBLE_DEVICES:**
   ```bash
   echo $HIP_VISIBLE_DEVICES
   rocm-smi
   ```

4. **Reinstall with rocm extra:**
   ```bash
   uv sync --extra rocm
   ```

### Out of Memory (OOM)

1. **Reduce batch size:**
   ```bash
   --device-batch-size=4
   ```

2. **Reduce sequence length:**
   ```bash
   --max-seq-len=1024
   ```

3. **Reduce model size:**
   ```bash
   --depth=8
   --head-dim=64
   ```

4. **Check memory:**
   ```bash
   rocm-smi --showmeminfo vram
   ```

### Flash Attention Issues

mesosfer uses Flash Attention 2 on ROCm via `flash-attn` package:

```bash
# Check which attention backend is active at startup
# Should print: "✓ Using Flash Attention 2 for training attention (ROCm)."

# If SDPA fallback is used instead, install flash-attn for ROCm:
pip install flash-attn --no-build-isolation

# Verify ROCm 7.0 is detected:
python -c "import torch; print(torch.version.hip)"
```

### Multi-GPU Training Fails

1. **Verify NCCL (ROCm variant):**
   ```bash
   python -c "import torch; print(torch.distributed.is_nccl_available())"
   ```

2. **Check all GPUs visible:**
   ```bash
   rocm-smi
   python -c "import torch; print(torch.cuda.device_count())"
   ```

3. **Test with fewer GPUs:**
   ```bash
   export HIP_VISIBLE_DEVICES=0,1  # Only 2 GPUs
   torchrun --standalone --nproc_per_node=2 -m scripts.train.base_train -- ...
   ```

### Performance Issues

1. **Enable BLIS optimizations:**
   ```bash
   export ROCM_BLIS_LC=1
   ```

2. **Check MFU in logs:**
   ```bash
   # Model FLOP Utilization should be 45-55% for well-tuned runs
   ```

3. **Profile with ROCm tools:**
   ```bash
   # Enable ROCM profiling
   export ROCM_PROFILER_ENABLE=1
   ```

### Common ROCm Errors

| Error | Solution |
|-------|----------|
| `HIP error: hipErrorNoBinaryForGpu` | Update ROCm, check GPU compatibility |
| `ROCm version mismatch` | Ensure PyTorch ROCm version matches installed ROCm |
| `Out of memory` | Reduce batch size, sequence length, or model depth |
| `NCCL timeout` | Increase NCCL timeout, check network connectivity |

---

## GPU Comparison Table

| GPU | CDNA | Memory | BF16 FLOPS | GCDs | Best For |
|-----|------|--------|------------|-------|----------|
| MI355X | 4 | 256GB HBM3 | 2.5 PFLOPS | 8 | Research/compute |
| MI350X | 3 | 256GB HBM3 | 1.6 PFLOPS | 8 | Research/compute |
| MI325X | 3 | 256GB HBM3 | 1.3 PFLOPS | 8 | Balanced compute |
| MI300X | 3 | 192GB HBM3 | 1.3 PFLOPS | 8 | LLM inference/training |
| MI300A | 3 | 192GB HBM3 | 980 TFLOPS | 4 | APU workloads |
| MI250X | 2 | 128GB HBM2 | 383 TFLOPS | 2 | Older compute |
| MI210 | 2 | 64GB HBM2 | 181 TFLOPS | 1 | Entry/dev workloads |

---

## Quick Verification

```bash
# 1. Verify ROCm detection
rocm-smi

# 2. Verify PyTorch ROCm 7.0
python -c "import torch; print(f'ROCm: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}, Version: {torch.version.hip}')"
# Expected: ROCm: True, Device: AMD Instinct MI300X, Version: 7.0.0...

# 3. Verify compute dtype (should be bf16)
python -c "from mesosfer.utils.common import COMPUTE_DTYPE; print(f'Compute dtype: {COMPUTE_DTYPE}')"

# 4. Verify attention backend (should be FA2 on ROCm)
python -c "from mesosfer.model.flash_attention import ATTENTION_BACKEND; print(f'Attention: {ATTENTION_BACKEND}')"
# Expected: Attention: fa2

# 5. Quick test run (depth 4, 10 steps, no GPU memory pressure)
export mesosfer_TORCH_BACKEND=rocm
python -m scripts.train.base_train -- --depth=4 --num-iterations=10 --run=dummy --device-batch-size=1 --core-metric-every=-1
```

---

## Additional Resources

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [ROCm Installation Guide](https://rocm.docs.amd.com/projects/install/en/latest/)
- [PyTorch ROCm Support](https://pytorch.org/)