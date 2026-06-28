# Running Mesosfer on NVIDIA GPUs

This guide covers the setup and execution of Mesosfer on NVIDIA GPUs using CUDA.

## Prerequisites

### Hardware

| GPU | Compute Capability | Architecture | BF16 FLOPS | Memory | Notes |
|-----|-------------------|--------------|------------|--------|-------|
| H100 (SXM/NVL) | SM 90 | Hopper | 989 TFLOPS | 80GB HBM3 | FP8 supported, recommended |
| H100 (PCIe) | SM 90 | Hopper | 756 TFLOPS | 80GB HBM3 | FP8 supported |
| H200 | SM 90 | Hopper | 989 TFLOPS | 80GB HBM3 | FP8 supported |
| GB200 | SM 100 | Blackwell | 2500 TFLOPS | 192GB HBM3e | Latest generation |
| B200 | SM 100 | Blackwell | 2250 TFLOPS | 192GB HBM3e | Latest generation |
| A100 (SXM) | SM 80 | Ampere | 312 TFLOPS | 40/80GB HBM2e | Minimum for bf16 training |
| A800 | SM 80 | Ampere | 312 TFLOPS | 40/80GB HBM2e | Data center variant |
| L40S | SM 89 | Ada | 362 TFLOPS | 48GB GDDR6 | Data center inference |
| L4 | SM 89 | Ada | 121 TFLOPS | 24GB GDDR6 | Data center inference, efficient |
| RTX 4090 | SM 89 | Ada | 165 TFLOPS | 24GB GDDR6X | Consumer best |
| RTX 3090 | SM 86 | Ampere | 71 TFLOPS | 24GB GDDR6X | Consumer older gen |
| T4 | SM 75 | Turing | ~65 TFLOPS* | 16GB GDDR6 | Inference GPU, older gen |

> *T4 uses Tensor Core INT8/FP16, not native BF16. Will fall back to fp32/fp16 compute.

> **Note:** GPUs older than Ampere (SM < 80) will fall back to fp32 compute. T4 (SM 75) uses fp16 instead of bf16.

### Software

- **OS:** Linux (Ubuntu 20.04+), Windows with WSL2, or macOS (CPU only)
- **Python:** 3.12+
- **NVIDIA Driver:** 525.60.13+ (for CUDA 12.x)
- **CUDA:** 12.8+ (bundled with PyTorch wheels)

### Check GPU Availability

```bash
nvidia-smi
```

Expected output shows GPU model, memory, and driver version.

---

## Installation

### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create Virtual Environment

```bash
cd /path/to/mesosfer
uv venv
source .venv/bin/activate
```

### 3. Install with CUDA Support

```bash
uv sync --extra gpu
```

This installs PyTorch with CUDA 12.8 support from PyPI.

---

## Environment Variables

### Required

```bash
# Set backend explicitly (optional, auto-detected)
export mesosfer_TORCH_BACKEND=cuda

# Optimize memory allocator
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# Limit OpenMP threads
export OMP_NUM_THREADS=1
```

### Optional

```bash
# Override compute dtype: bfloat16, float16, float32
# Note: Use float16 for T4 (no bf16 support)
export mesosfer_DTYPE=bfloat16

# Cache directory (default: ~/.cache/mesosfer)
export mesosfer_BASE_DIR="$HOME/.cache/mesosfer"

# Weights & Biases logging
export WANDB_RUN=my_training_run
# To enable wandb: wandb login
```

---

## GPU-Specific Notes

### H100 / H200 / Blackwell (B200/GB200)

- FP8 training supported with `--fp8` flag
- Flash Attention 3.0 available
- TF32 enabled automatically
- Recommended for production training

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.train.base_train -- \
    --depth=24 \
    --fp8 \
    --device-batch-size=16 \
    --run=$WANDB_RUN
```

### A100 / A800

- BF16 training supported natively
- Flash Attention 2.0 available
- TF32 enabled automatically
- Good balance of cost and performance

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.train.base_train -- \
    --depth=20 \
    --device-batch-size=16 \
    --run=$WANDB_RUN
```

### L40S / L4 / RTX 4090

- BF16 training supported
- Flash Attention 2.0 or SDPA fallback
- TF32 enabled automatically
- Excellent for single-GPU or small cluster training

```bash
# Single GPU training
python -m scripts.train.base_train -- \
    --depth=12 \
    --device-batch-size=8 \
    --run=dummy

# 4-GPU L4/L40S node
torchrun --standalone --nproc_per_node=4 -m scripts.train.base_train -- \
    --depth=16 \
    --device-batch-size=16 \
    --run=$WANDB_RUN
```

### T4

- **No native BF16 support** - uses fp16 instead
- Flash Attention SDPA fallback only
- No TF32 acceleration
- Inference-optimized, training will be slower
- Set `mesosfer_DTYPE=float16` explicitly

```bash
# T4 optimized settings
export mesosfer_DTYPE=float16
export PYTORCH_ALLOC_CONF="expandable_segments:True,max_split_size_mb=512"

# Single T4 training (small model recommended)
python -m scripts.train.base_train -- \
    --depth=6 \
    --head-dim=64 \
    --max-seq-len=1024 \
    --device-batch-size=4 \
    --num-iterations=1000 \
    --run=dummy

# Multi-T4 inference cluster
torchrun --standalone --nproc_per_node=4 -m scripts.train.base_train -- \
    --depth=8 \
    --device-batch-size=8 \
    --max-seq-len=1024 \
    --run=$WANDB_RUN
```

---

## Training Scripts

### Quick Start: GPT-2 Class Training (~3 hours on 8x H100)

```bash
# Basic run (no wandb logging)
bash runs/speedrun.sh

# With wandb logging
WANDB_RUN=my_run bash runs/speedrun.sh

# In a screen session (recommended for long runs)
screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh
```

### Scaling Laws Research

Analyzes optimal model configurations across FLOP budgets.

```bash
# Customize via environment variables
export NPROC_PER_NODE=8
export WANDB_RUN=scaling_jan26
export mesosfer_BASE_DIR="$HOME/.cache/mesosfer"

bash runs/scaling_laws.sh
```

**Key configurations:**
- FLOP budgets: 1e18, 2.15e18, 4.64e18, 1e19
- Depths: 10, 12, 14, 16, 18, 20
- Results saved to `~/.cache/mesosfer/scaling_laws_results_<label>/results.csv`

### Miniseries (Multiple Depth Sweep)

Trains models with depths 12-26 in a single run.

```bash
# Default series name is today's date
bash runs/miniseries.sh

# Custom series name
bash runs/miniseries.sh feb15

# With wandb
WANDB_RUN=feb15_miniseries bash runs/miniseries.sh feb15

# Skip setup (if already done)
SKIP_SETUP=1 bash runs/miniseries.sh feb15
```

**Customize parallelism:**
```bash
NPROC_PER_NODE=4 bash runs/miniseries.sh
```

### CPU/MPS Demo (MacBook or CPU-only)

```bash
bash runs/runcpu.sh
```

> **Warning:** CPU training is for demo/educational purposes only. Will not produce useful models.

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
| `--fp8` | false | Enable FP8 training (H100+ only) |
| `--target-param-data-ratio` | 12 | Chinchilla-optimal ~20 |
| `--target-flops` | -1 | Fixed FLOP budget (scaling laws) |
| `--num-iterations` | -1 | Auto-compute if -1 |
| `--run` | dummy | wandb run name |

### Batch Size Guidelines by GPU

```bash
# H100/A100/A800 (80GB+ memory)
--device-batch-size=16

# L40S (48GB memory)
--device-batch-size=12

# L4/RTX 4090 (24GB memory)
--device-batch-size=8

# T4 (16GB memory)
--device-batch-size=4

# Small models (depth < 12)
--device-batch-size=32  # Adjust down based on GPU memory
```

---

## Post-Training: Chat Interface

After training completes, you can interact with the model:

### CLI Chat

```bash
python -m scripts.chat.chat_cli -p "Why is the sky blue?"
```

### Interactive CLI

```bash
python -m scripts.chat.chat_cli
# Type your questions, Ctrl+D to exit
```

### Web UI

```bash
python -m scripts.chat.chat_web
# Opens http://localhost:8000
```

---

## Troubleshooting

### CUDA Out of Memory (OOM)

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
   --depth=6
   --head-dim=64
   ```

### "CUDA not available" Error

1. Verify PyTorch CUDA installation:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. Check CUDA driver:
   ```bash
   nvidia-smi
   ```

3. Reinstall with correct extra:
   ```bash
   uv sync --extra gpu
   ```

### Flash Attention Not Available

```bash
# Check FA status at startup - should show:
# "Using Flash Attention 3.0" (H100+)
# "Using Flash Attention 2.0" (A100+)
# "Using SDPA" (fallback for L4/T4/RTX)

# For H100: Install Flash Attention manually (optional):
pip install flash-attn --no-build-isolation
```

### T4 Issues

1. **Force float16:**
   ```bash
   export mesosfer_DTYPE=float16
   ```

2. **Check memory fragmentation:**
   ```bash
   export PYTORCH_ALLOC_CONF="expandable_segments:True,max_split_size_mb=512"
   ```

3. **Smaller batches required:**
   ```bash
   --device-batch-size=2
   --max-seq-len=512
   ```

### Multi-GPU Training Fails

1. **Verify NCCL:**
   ```bash
   python -c "import torch; print(torch.distributed.is_nccl_available())"
   ```

2. **Check all GPUs visible:**
   ```bash
   python -c "import torch; print(torch.cuda.device_count())"
   ```

3. **Run with correct parallelism:**
   ```bash
   # For 4 GPUs:
   torchrun --standalone --nproc_per_node=4 -m scripts.train.base_train -- ...
   ```

### Performance Issues

1. **Enable TF32 (automatic on Ampere+):**
   ```bash
   # Verify TF32 is enabled:
   python -c "import torch; print(torch.get_float32_matmul_precision())"
   ```

2. **Enable FP8 (H100+ only):**
   ```bash
   torchrun ... --fp8
   ```

3. **Profile with MFU:**
   - Check log output for "Model FLOP Utilization (MFU)"
   - Target: 45-55% for well-tuned runs

---

## GPU Comparison Table

| GPU | Architecture | SM | BF16 Support | FP8 Support | Memory | Best For |
|-----|-------------|-----|-------------|--------------|--------|----------|
| GB200 | Blackwell | 100 | Yes | Yes | 192GB | Research |
| B200 | Blackwell | 100 | Yes | Yes | 192GB | Research |
| H100 | Hopper | 90 | Yes | Yes | 80GB | Production training |
| H200 | Hopper | 90 | Yes | Yes | 80GB | Production training |
| A100 | Ampere | 80 | Yes | No | 40/80GB | Balanced training |
| A800 | Ampere | 80 | Yes | No | 40/80GB | China market |
| L40S | Ada | 89 | Yes | No | 48GB | Data center inference |
| L4 | Ada | 89 | Yes | No | 24GB | Efficient inference |
| RTX 4090 | Ada | 89 | Yes | No | 24GB | Consumer training |
| RTX 3090 | Ampere | 86 | Yes | No | 24GB | Consumer training |
| T4 | Turing | 75 | No (fp16) | No | 16GB | Inference/small models |

---

## Quick Verification

```bash
# 1. Verify GPU detection
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

# 2. Verify compute dtype
python -c "import torch; print(f'Compute dtype: {torch.float32 if not torch.cuda.is_available() else torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16}')"

# 3. Verify Flash Attention
python -c "from mesosfer.model.flash_attention import flash_attn_func; print('Flash Attention available')"

# 4. Quick test run
python -m scripts.train.base_train --depth=4 --num-iterations=10 --run=dummy --device-batch-size=1
```