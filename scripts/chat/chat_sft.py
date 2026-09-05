"""
Supervised fine-tuning (SFT) the model.
Run as:

python -m scripts.chat.chat_sft

Or torchrun for training:

torchrun --standalone --nproc_per_node=8 -m scripts.chat.chat_sft -- --device-batch-size=16
"""

import gc
import argparse
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import time
import wandb
import torch
from tqdm import tqdm
from mesosfer.utils.common import compute_init, compute_cleanup, print0, DummyWandb, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
from mesosfer.data.tokenizer import get_token_bytes
from mesosfer.utils.checkpoint_manager import save_checkpoint, load_model, load_optimizer_state
from mesosfer.eval.loss_eval import evaluate_bpb
import torch.distributed as dist
from mesosfer.model.flash_attention import ATTENTION_BACKEND, HAS_FA2, HAS_FA3, _is_rocm
from mesosfer.eval.engine import Engine
from scripts.chat.chat_eval import run_chat_eval

from tasks.common import TaskMixture
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
from tasks.customjson import CustomJSON
from tasks.spellingbee import SimpleSpelling, SpellingBee
from tasks.cybersec_sft import build_cybersec_sft_tasks, total_cybersec_rows

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Supervised fine-tuning (SFT) the model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Model loading
parser.add_argument("-i", "--checkpoint-source", "--source", type=str, default="base", choices=["base", "sft"], help="checkpoint source to fine-tune from: base|sft")
parser.add_argument("-g", "--model-tag", "--depth", type=str, default=None, help="model tag or depth to load from")
parser.add_argument("-s", "--model-step", "--step", type=int, default=None, help="model step to load from")
parser.add_argument("--load-optimizer", type=int, default=1, help="warm-start optimizer from pretrained checkpoint (0=no, 1=yes)")
# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1, help="number of optimization steps (-1 = full epoch)")
# Batch sizes (default: inherit from pretrained checkpoint)
parser.add_argument("--max-seq-len", type=int, default=None, help="max context length (default: inherit from pretrain)")
parser.add_argument("--device-batch-size", type=int, default=None, help="per-device batch size (default: inherit from pretrain)")
parser.add_argument("--total-batch-size", type=int, default=None, help="total batch size in tokens (default: inherit from pretrain)")
# Optimization (default: inherit from pretrained checkpoint)
parser.add_argument("--learning-rate", "--lr", type=float, default=None, help="override learning rate (AdamW/Muon)")
parser.add_argument("--embedding-lr", type=float, default=None, help="learning rate for embedding parameters (Adam) (default: inherit from pretrain)")
parser.add_argument("--unembedding-lr", type=float, default=None, help="learning rate for unembedding parameters (Adam) (default: inherit from pretrain)")
parser.add_argument("--matrix-lr", type=float, default=None, help="learning rate for matrix parameters (Muon) (default: inherit from pretrain)")
parser.add_argument("--init-lr-frac", type=float, default=0.8, help="initial LR as fraction of base LR")
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
# Evaluation
parser.add_argument("--eval-every", type=int, default=200, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=40*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--chatcore-every", type=int, default=200, help="evaluate ChatCORE metric every N steps (-1 = disable)")
parser.add_argument("--chatcore-max-cat", type=int, default=-1, help="max problems per categorical task for ChatCORE")
parser.add_argument("--chatcore-max-sample", type=int, default=24, help="max problems per generative task for ChatCORE")
parser.add_argument("--chatcore-tasks", type=str, default=None,
                    help="pipe-separated ChatCORE tasks to run (default: all). "
                         "Use ARC-Easy|ARC-Challenge|MMLU for ROCm-safe categorical-only eval.")
# Data mixture
parser.add_argument("--mmlu-epochs", type=int, default=3, help="number of epochs of MMLU in training mixture (teaches Multiple Choice)")
parser.add_argument("--gsm8k-epochs", type=int, default=4, help="number of epochs of GSM8K in training mixture (teaches Math and Tool Use)")
# Cybersecurity SFT data mixture
parser.add_argument("--cyber-defensive-epochs", type=int, default=1, help="epochs of cyber_defensive_conversations (96 rows x language, deduped from 5K)")
parser.add_argument("--cloud-security-epochs", type=int, default=2, help="epochs of cloud_security_sft (6 rows x language, oversampled)")
parser.add_argument("--multi-turn-soc-epochs", type=int, default=0, help="epochs of multi_turn_soc_sft (DISABLED: turns are mislabelled, no real assistant answers)")
parser.add_argument("--tool-oriented-epochs", type=int, default=1, help="epochs of tool_oriented_cyber_sft (8 rows, oversampled)")
parser.add_argument("--mythos-epochs", type=int, default=0, help="epochs of mythos_combined_sft (0=disable paranoid forensic narrative)")
parser.add_argument("--mythos-tool-calling-epochs", type=int, default=0, help="epochs of mythos_tool_calling (0=disable fake compromise scenarios)")
parser.add_argument("--mesosfer-validation-epochs", type=int, default=0, help="epochs of mesosfer_validation_conversations (0 = held-out eval set, do not train)")
parser.add_argument("--gemini-teacher-epochs", type=int, default=4, help="epochs of gemini_teacher_conversations (373 rows)")
parser.add_argument("--chat-history-epochs", type=int, default=4, help="epochs of chat_history_distilled_sft (pair-programming & troubleshooting traces)")
parser.add_argument("--real-agentic-epochs", type=int, default=2, help="epochs of real_agentic_tool_calling_sft (real-world tool traces in Mesosfer format)")
parser.add_argument("--primus-instruct-epochs", type=int, default=1, help="epochs of Primus-Instruct (~100K rows, gated, 0=skip)")
parser.add_argument("--primus-reasoning-epochs", type=int, default=1, help="epochs of Primus-Reasoning (~50K rows, gated, 0=skip)")
parser.add_argument("--cybernative-vuln-epochs", type=int, default=2, help="epochs of CyberNative vuln DPO (~4.6K rows)")
parser.add_argument("--openhermes-epochs", type=int, default=1, help="epochs of OpenHermes-2.5 (50K rows, 0=skip)")
parser.add_argument("--ultrachat-epochs", type=int, default=1, help="epochs of UltraChat 200K (100K rows, 0=skip)")
parser.add_argument("--trendyol-cyber-epochs", type=int, default=1, help="epochs of Trendyol Cybersecurity Instruction (53K rows, 0=skip)")
parser.add_argument("--tiamz-cybersec-epochs", type=int, default=2, help="epochs of Tiamz cybersecurity Q&A (12K rows)")
parser.add_argument("--alpaca-indonesian-epochs", type=int, default=2, help="epochs of Alpaca Cleaned Indonesian instruction dataset (~52K rows, 0=skip)")
parser.add_argument("--alpaca-gpt4-indonesian-epochs", type=int, default=2, help="epochs of Alpaca GPT-4 Indonesian dataset (~52K rows, 0=skip)")
parser.add_argument("--magpie-reasoning-epochs", type=int, default=1, help="epochs of magpie_reasoning_sft (~50K rows, 0=skip)")
parser.add_argument("--open-thoughts-epochs", type=int, default=1, help="epochs of open_thoughts_sft (~50K rows, 0=skip)")
parser.add_argument("--hermes-func-calling-epochs", type=int, default=3, help="epochs of Hermes function calling (20K rows, ungated tool-calling)")
parser.add_argument("--hermes-glaive-func-calling-epochs", type=int, default=1, help="epochs of Glaive function calling via Hermes (20K rows)")
parser.add_argument("--kimi-k3-traces-epochs", type=int, default=4, help="epochs of Kimi K3 agentic coding/debugging traces (~4K rows)")
parser.add_argument("--cot-collection-epochs", type=int, default=1, help="epochs of CoT Collection chain-of-thought rationales (50K rows)")
parser.add_argument("--qyrou-reasoning-epochs", type=int, default=1, help="epochs of Qyrou reasoning corpus with thought traces (50K rows)")
parser.add_argument("--nist-cybersec-epochs", type=int, default=1, help="epochs of nist_cybersec_sft (~50K rows, 0=skip)")
parser.add_argument("--fenrir-v2-epochs", type=int, default=1, help="epochs of fenrir_v2_sft (~99K rows, 0=skip)")
parser.add_argument("--code-feedback-epochs", type=int, default=1, help="epochs of code_feedback_sft (~50K rows, 0=skip)")
parser.add_argument("--numinamath-cot-epochs", type=int, default=1, help="epochs of numinamath_cot_sft (~50K rows, 0=skip)")
parser.add_argument("--aquilax-security-reasoning-epochs", type=int, default=2, help="epochs of aquilax_security_reasoning_sft cybersec CoT (~18K rows, gated, 0=skip)")
parser.add_argument("--xlam-function-calling-epochs", type=int, default=2, help="epochs of xlam_function_calling_sft generic tool-calling (~20K rows, gated, 0=skip)")
parser.add_argument("--include-english-sft", type=int, default=1, help="1 = include _en variants of bilingual cybersec datasets, 0 = ID only")
parser.add_argument("--disable-cybersec-sft", action="store_true", help="disable all cybersecurity SFT datasets (for ablation)")
parser.add_argument("--rules-epochs", type=int, default=4, help="epochs of rules.jsonl (behavioral/safety/format rules)")
parser.add_argument("--tool-calling-epochs", type=int, default=24, help="epochs of tool_calling_conversations_en (calc/tool/subagent parts)")
parser.add_argument("--instruction-following-epochs", type=int, default=4, help="epochs of instruction_following_conversations_en.jsonl (format/count/conciseness polish)")
parser.add_argument("--instruction-polish-only", action="store_true", help="train only on local identity/rules/instruction-following polish data")
parser.add_argument("--safety-artifact-epochs", type=int, default=4, help="epochs of safety_artifact_conversations_en.jsonl (artifact-vs-attack boundary)")
parser.add_argument("--safety-artifact-only", action="store_true", help="train only on local identity/rules/safety-artifact boundary data")
parser.add_argument("--local-sft-only", action="store_true", help="train only on local SFT datasets (skip downloading SmolTalk, MMLU, GSM8K, and Spelling tasks)")
parser.add_argument("--targeted-domain-sft", action="store_true", help="train specifically on cybersecurity, tool-use, code feedback, and CoT reasoning without generic chatter")
parser.add_argument("--save-every", type=int, default=200, help="save intermediate checkpoint every N steps (-1 = only at end)")
args = parser.parse_args()

if args.targeted_domain_sft:
    args.local_sft_only = True
    args.ultrachat_epochs = 0
    args.openhermes_epochs = 0
    args.alpaca_indonesian_epochs = 0
    args.cyber_defensive_epochs = 8
    args.cloud_security_epochs = 8
    args.mythos_epochs = 8
    args.mythos_tool_calling_epochs = 8
    args.tool_calling_epochs = 12
    args.tool_oriented_epochs = 8
    args.gemini_teacher_epochs = 6
    args.rules_epochs = 8
    args.instruction_following_epochs = 8
    args.safety_artifact_epochs = 6

user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="mesosfer-sft", name=args.run, config=user_config)

# Flash Attention status
if ATTENTION_BACKEND == "fa3":
    print0("✓ Using Flash Attention 3 for SFT attention.")
elif ATTENTION_BACKEND == "fa2":
    platform = "ROCm" if _is_rocm() else "CUDA"
    print0(f"✓ Using Flash Attention 2 for SFT attention ({platform}).")
else:
    if _is_rocm():
        print0("WARNING: AMD ROCm detected but Flash Attention 2 is not available/usable.")
    elif HAS_FA3 and COMPUTE_DTYPE != torch.bfloat16:
        print0(f"WARNING: Flash Attention 3 only supports bf16 here, but COMPUTE_DTYPE={COMPUTE_DTYPE}.")
    elif HAS_FA2 and COMPUTE_DTYPE not in {torch.float16, torch.bfloat16}:
        print0(f"WARNING: Flash Attention 2 requires fp16/bf16, but COMPUTE_DTYPE={COMPUTE_DTYPE}.")
    else:
        print0("WARNING: Flash Attention 3/2 not available.")
    print0("WARNING: Using PyTorch SDPA fallback. SFT training may be less efficient.")

# Load the model and tokenizer
model, tokenizer, meta = load_model(args.checkpoint_source, device, phase="train", model_tag=args.model_tag, step=args.model_step)
loaded_checkpoint_step = int(meta.get("step", 0) or 0) if args.checkpoint_source == "sft" else 0
if loaded_checkpoint_step > 0:
    print0(f"Continuing SFT from {args.checkpoint_source} checkpoint step {loaded_checkpoint_step}")

# Inherit training hyperparameters from pretrained checkpoint (None = inherit, explicit value = override)
pretrain_user_config = meta.get("user_config", {})
for name, fallback, source in [
    ("max_seq_len",       2048,  meta),
    ("device_batch_size", 32,    meta),
    ("total_batch_size",  524288, meta),
    ("embedding_lr",      0.3,   pretrain_user_config),
    ("unembedding_lr",    0.004, pretrain_user_config),
    ("matrix_lr",         0.02,  pretrain_user_config),
]:
    arg_val = getattr(args, name)
    pretrain_val = source.get(name)
    if arg_val is None:
        resolved = pretrain_val if pretrain_val is not None else fallback
        setattr(args, name, resolved)
        print0(f"Inherited {name}={resolved} from pretrained checkpoint")
    elif pretrain_val is not None and arg_val != pretrain_val:
        print0(f"NOTE: --{name.replace('_', '-')}={arg_val} overrides pretrained value of {pretrain_val}")
    else:
        print0(f"Using {name}={arg_val}")

orig_model = model
model = torch.compile(model, dynamic=False)
depth = model.config.n_layer
num_flops_per_token = model.estimate_flops()
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert args.total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {args.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
token_bytes = get_token_bytes(device=device)

# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
# Note that pretraining ramps weight_decay to zero by end of pretraining, so SFT continues with zero
optimizer = model.setup_optimizer(unembedding_lr=args.unembedding_lr, embedding_lr=args.embedding_lr, matrix_lr=args.matrix_lr, weight_decay=0.0)

# Optionally warm-start optimizer from pretrained checkpoint (momentum buffers etc.)
# Note: load_state_dict overwrites param_group metadata (LRs, betas, etc.) with the
# pretrained values. Since pretraining warmdown brings LRs to ~0, we must save and
# restore our fresh SFT LRs after loading.
base_dir = get_base_dir()
if args.load_optimizer:
    optimizer_data = load_optimizer_state(args.checkpoint_source, device, rank=ddp_rank, model_tag=args.model_tag, step=args.model_step)
    if optimizer_data is not None:
        base_lrs = [group["lr"] for group in optimizer.param_groups]
        optimizer.load_state_dict(optimizer_data)
        del optimizer_data
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr
        print0("Loaded optimizer state from pretrained checkpoint (momentum buffers only, LRs reset)")
    else:
        print0("WARNING: optimizer checkpoint not found, starting with fresh optimizer (slightly worse)")

# GradScaler for fp16 training (bf16/fp32 don't need it)
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

# Override the initial learning rate as a fraction of the base learning rate or explicit --learning-rate
if args.learning_rate is not None:
    for group in optimizer.param_groups:
        group["lr"] = args.learning_rate
        group["initial_lr"] = args.learning_rate
    print0(f"Overriding optimizer learning rate with --learning-rate={args.learning_rate}")
else:
    for group in optimizer.param_groups:
        group["lr"] = group["lr"] * args.init_lr_frac
        group["initial_lr"] = group["lr"]

# SFT data mixture and DataLoader
sft_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "sft"))
sft_sample_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "sft_sample"))
if os.path.exists(sft_sample_dir) and not os.path.exists(sft_dir):
    sft_dir = sft_sample_dir
identity_conversations_filepath = os.path.join(sft_dir, "identity_conversations.jsonl")
identity_conversations_en_filepath = os.path.join(sft_dir, "identity_conversations_en.jsonl")
rules_filepath = os.path.join(sft_dir, "rules.jsonl")
instruction_following_filepath = os.path.join(sft_dir, "instruction_following_conversations_en.jsonl")
safety_artifact_filepath = os.path.join(sft_dir, "safety_artifact_conversations_en.jsonl")
if args.instruction_polish_only or args.safety_artifact_only:
    train_tasks = [
        CustomJSON(filepath=identity_conversations_filepath),
    ]
    mode_name = "Safety artifact" if args.safety_artifact_only else "Instruction polish"
    print0(f"{mode_name} mode: skipping broad SmolTalk/MMLU/GSM8K/spelling training tasks")
elif args.local_sft_only:
    train_tasks = [
        CustomJSON(filepath=identity_conversations_filepath),
        CustomJSON(filepath=identity_conversations_filepath),
    ]
    print0("Local SFT mode: training only on local SFT data mixture (skipping SmolTalk/MMLU/GSM8K/Spelling)")
else:
    train_tasks = [
        SmolTalk(split="train"), # 460K rows of general conversations
        CustomJSON(filepath=identity_conversations_filepath), # 1000 rows of synthetic identity conversations
        CustomJSON(filepath=identity_conversations_filepath), # 2 epochs of these
        *[MMLU(subset="all", split="auxiliary_train") for _ in range(args.mmlu_epochs)], # 100K rows per epoch
        *[GSM8K(subset="main", split="train") for _ in range(args.gsm8k_epochs)], # 8K rows per epoch
        SimpleSpelling(size=200000, split="train"), # 200K rows of Simple Spelling (e.g. spell the word 'apple')
        SpellingBee(size=80000, split="train"), # 80K rows of Spelling Bee (e.g. how many 'r' are in 'strawberry'?)
    ]

if args.include_english_sft and os.path.exists(identity_conversations_en_filepath):
    train_tasks.extend([
        CustomJSON(filepath=identity_conversations_en_filepath),
        CustomJSON(filepath=identity_conversations_en_filepath),
    ])
    print0(f"Added identity_conversations_en.jsonl: 2 epoch(s) from {identity_conversations_en_filepath}")
elif args.include_english_sft:
    print0(f"WARNING: identity_conversations_en.jsonl not found at {identity_conversations_en_filepath}, skipping")

# Add rules.jsonl if it exists and rules_epochs > 0
if args.rules_epochs > 0 and os.path.exists(rules_filepath):
    rules_tasks = [CustomJSON(filepath=rules_filepath) for _ in range(args.rules_epochs)]
    train_tasks.extend(rules_tasks)
    print0(f"Added rules.jsonl: {args.rules_epochs} epoch(s) from {rules_filepath}")
elif args.rules_epochs > 0:
    print0(f"WARNING: rules.jsonl not found at {rules_filepath}, skipping")

# Add safety artifact boundary data (synthetic artifacts allowed, attack automation refused)
if args.safety_artifact_epochs > 0 and os.path.exists(safety_artifact_filepath):
    safety_artifact_tasks = [CustomJSON(filepath=safety_artifact_filepath) for _ in range(args.safety_artifact_epochs)]
    train_tasks.extend(safety_artifact_tasks)
    print0(f"Added safety_artifact_conversations_en.jsonl: {args.safety_artifact_epochs} epoch(s) from {safety_artifact_filepath}")
elif args.safety_artifact_epochs > 0:
    print0(f"WARNING: safety_artifact_conversations_en.jsonl not found at {safety_artifact_filepath}, skipping")

# Add instruction-following polish data (exact counts, no-code constraints, JSON-only, concise answers)
if not args.safety_artifact_only and args.instruction_following_epochs > 0 and os.path.exists(instruction_following_filepath):
    instruction_tasks = [CustomJSON(filepath=instruction_following_filepath) for _ in range(args.instruction_following_epochs)]
    train_tasks.extend(instruction_tasks)
    print0(f"Added instruction_following_conversations_en.jsonl: {args.instruction_following_epochs} epoch(s) from {instruction_following_filepath}")
elif not args.safety_artifact_only and args.instruction_following_epochs > 0:
    print0(f"WARNING: instruction_following_conversations_en.jsonl not found at {instruction_following_filepath}, skipping")

# Add cybersecurity SFT mixture (preserves cybersec capability from pretraining)
if args.instruction_polish_only or args.safety_artifact_only:
    mode_name = "Safety artifact" if args.safety_artifact_only else "Instruction polish"
    print0(f"{mode_name} mode: skipping broad cybersec SFT mixture")
elif not args.disable_cybersec_sft:
    cybersec_tasks = build_cybersec_sft_tasks(
        cyber_defensive_epochs=args.cyber_defensive_epochs,
        cloud_security_epochs=args.cloud_security_epochs,
        multi_turn_soc_epochs=args.multi_turn_soc_epochs,
        tool_oriented_epochs=args.tool_oriented_epochs,
        tool_calling_epochs=args.tool_calling_epochs,
        mythos_epochs=args.mythos_epochs,
        mythos_tool_calling_epochs=args.mythos_tool_calling_epochs,
        mesosfer_validation_epochs=args.mesosfer_validation_epochs,
        gemini_teacher_epochs=args.gemini_teacher_epochs,
        chat_history_epochs=args.chat_history_epochs,
        real_agentic_epochs=args.real_agentic_epochs,
        primus_instruct_epochs=args.primus_instruct_epochs,
        primus_reasoning_epochs=args.primus_reasoning_epochs,
        cybernative_vuln_epochs=args.cybernative_vuln_epochs,
        openhermes_epochs=args.openhermes_epochs,
        ultrachat_epochs=args.ultrachat_epochs,
        trendyol_cyber_epochs=args.trendyol_cyber_epochs,
        tiamz_cybersec_epochs=args.tiamz_cybersec_epochs,
        alpaca_indonesian_epochs=args.alpaca_indonesian_epochs,
        alpaca_gpt4_indonesian_epochs=args.alpaca_gpt4_indonesian_epochs,
        magpie_reasoning_epochs=args.magpie_reasoning_epochs,
        open_thoughts_epochs=args.open_thoughts_epochs,
        nist_cybersec_epochs=args.nist_cybersec_epochs,
        fenrir_v2_epochs=args.fenrir_v2_epochs,
        code_feedback_epochs=args.code_feedback_epochs,
        numinamath_cot_epochs=args.numinamath_cot_epochs,
        aquilax_security_reasoning_epochs=args.aquilax_security_reasoning_epochs,
        xlam_function_calling_epochs=args.xlam_function_calling_epochs,
        include_english=bool(args.include_english_sft),
        sft_dir=sft_dir,
    )
    train_tasks.extend(cybersec_tasks)
    print0(f"Added cybersec SFT: {len(cybersec_tasks)} task instances, {total_cybersec_rows(cybersec_tasks):,} total rows")
else:
    print0("Cybersec SFT disabled via --disable-cybersec-sft")

train_dataset = TaskMixture(train_tasks)
if args.local_sft_only:
    print0(f"Training mixture (local SFT): {len(train_dataset):,} rows")
    val_tasks = []
    val_id = os.path.join(sft_dir, "mesosfer_validation_conversations.jsonl")
    val_en = os.path.join(sft_dir, "mesosfer_validation_conversations_en.jsonl")
    if os.path.exists(val_id):
        val_tasks.append(CustomJSON(filepath=val_id))
    if os.path.exists(val_en):
        val_tasks.append(CustomJSON(filepath=val_en))
    if not val_tasks:
        val_tasks.append(CustomJSON(filepath=identity_conversations_filepath))
    val_dataset = TaskMixture(val_tasks)
    print0(f"Validation mixture (local): {len(val_dataset):,} rows")
else:
    print0(f"Training mixture: {len(train_dataset):,} rows (MMLU x{args.mmlu_epochs}, GSM8K x{args.gsm8k_epochs})")
    val_dataset = TaskMixture([
        SmolTalk(split="test"), # 24K rows in test set
        MMLU(subset="all", split="test", stop=5200), # 14K rows in test set, use only 5.2K to match the train ratios
        GSM8K(subset="main", split="test", stop=420), # 1.32K rows in test set, use only 420 to match the train ratios
    ]) # total: 24K + 5.2K + 0.42K ~= 29.6K rows
# DataLoader is defined here, it emits inputs, targets : 2D tensors of shape (device_batch_size, max_seq_len)
# A big problem is that we don't know the final num_iterations in advance. So we create
# these two global variables and update them from within the data generator.
last_step = False # we will toggle this to True when we reach the end of the training dataset
approx_progress = 0.0 # will go from 0 to 1 over the course of the epoch
current_epoch = 1 # track epoch for logging
def sft_data_generator_bos_bestfit(split, buffer_size=100):
    """
    BOS-aligned dataloader for SFT with bestfit-pad packing.

    Each row in the batch starts with BOS (beginning of a conversation).
    Conversations are packed using best-fit algorithm. When no conversation fits,
    the row is padded (instead of cropping) to ensure no tokens are ever discarded.
    Padding positions have targets masked with -1 (ignore_index for cross-entropy).
    """
    global last_step, approx_progress, current_epoch
    assert split in {"train", "val"}, "split must be 'train' or 'val'"
    dataset = train_dataset if split == "train" else val_dataset
    dataset_size = len(dataset)
    assert dataset_size > 0
    row_capacity = args.max_seq_len + 1  # +1 for target at last position
    bos_token = tokenizer.get_bos_token_id()

    # Conversation buffer: list of (token_ids, loss_mask) tuples
    conv_buffer = []
    cursor = ddp_rank  # Each rank processes different conversations (for fetching)
    consumed = ddp_rank  # Track actual consumption separately from buffering
    epoch = 1
    it = 0  # iteration counter

    def refill_buffer():
        nonlocal cursor, epoch
        while len(conv_buffer) < buffer_size:
            conversation = dataset[cursor]
            ids, mask = tokenizer.render_conversation(conversation, max_tokens=row_capacity)
            conv_buffer.append((ids, mask))
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor = cursor % dataset_size
                epoch += 1
                # Note: last_step is now triggered based on consumption, not fetching

    while True:
        rows = []
        mask_rows = []
        row_lengths = []  # Track actual content length (excluding padding) for each row
        for _ in range(args.device_batch_size):
            row = []
            mask_row = []
            padded = False
            while len(row) < row_capacity:
                # Ensure buffer has conversations
                while len(conv_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - len(row)

                # Find largest conversation that fits entirely
                best_idx = -1
                best_len = 0
                for i, (conv, _) in enumerate(conv_buffer):
                    conv_len = len(conv)
                    if conv_len <= remaining and conv_len > best_len:
                        best_idx = i
                        best_len = conv_len

                if best_idx >= 0:
                    # Found a conversation that fits - use it entirely
                    conv, conv_mask = conv_buffer.pop(best_idx)
                    row.extend(conv)
                    mask_row.extend(conv_mask)
                    consumed += ddp_world_size  # Track actual consumption
                else:
                    # No conversation fits - pad the remainder instead of cropping
                    # This ensures we never discard any tokens
                    content_len = len(row)
                    row.extend([bos_token] * remaining)  # Pad with BOS tokens
                    mask_row.extend([0] * remaining)
                    padded = True
                    break  # Row is now full (with padding)

            # Track content length: full row if no padding, otherwise the length before padding
            if padded:
                row_lengths.append(content_len)
            else:
                row_lengths.append(row_capacity)
            rows.append(row[:row_capacity])
            mask_rows.append(mask_row[:row_capacity])

        # Dataloader-local iteration counter. This counts micro-batches, not
        # optimizer steps, so --num-iterations is handled in the outer training
        # loop where `step` counts optimizer updates.
        it += 1

        # Update progress tracking (based on consumed, not cursor, to account for buffering)
        if split == "train":
            current_epoch = epoch
            if args.num_iterations <= 0:
                approx_progress = consumed / dataset_size
            # Trigger last_step when we've consumed the dataset only for epoch-based
            # runs. If --num-iterations is set, keep cycling through the dataset so
            # small polish datasets can train for the requested number of steps.
            if args.num_iterations <= 0 and consumed >= dataset_size:
                last_step = True

        # Build tensors
        use_cuda = device_type == "cuda"
        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        inputs = batch_tensor[:, :-1].to(device=device, dtype=torch.int32, non_blocking=use_cuda).contiguous()
        targets = batch_tensor[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda).contiguous()

        # Apply the loss mask from render_conversation (mask=1 for assistant completions,
        # mask=0 for user prompts, BOS, special tokens, tool outputs). mask[1:] aligns
        # with targets (shifted by 1). Unmasked positions get -1 (ignore_index).
        mask_tensor = torch.tensor(mask_rows, dtype=torch.int8)
        mask_targets = mask_tensor[:, 1:].to(device=device)
        targets[mask_targets == 0] = -1

        # Mask out padding positions in targets (set to -1 = ignore_index)
        # For each row, positions >= (content_length - 1) in targets should be masked
        for i, content_len in enumerate(row_lengths):
            if content_len < row_capacity:
                targets[i, content_len-1:] = -1

        yield inputs, targets

train_loader = sft_data_generator_bos_bestfit("train")
build_val_loader = lambda: sft_data_generator_bos_bestfit("val")
progress = 0 # will go from 0 to 1 over the course of the epoch

# Learning rate schedule (linear warmup, constant, linear warmdown)
# Same shape as base_train but uses progress (0→1) instead of absolute step counts,
# because SFT doesn't always know num_iterations in advance (dataset-driven stopping).
def get_lr_multiplier(progress):
    if progress < args.warmup_ratio:
        return (progress + 1e-8) / args.warmup_ratio
    elif progress <= 1.0 - args.warmdown_ratio:
        return 1.0
    else:
        decay = (progress - (1.0 - args.warmdown_ratio)) / args.warmdown_ratio
        return (1 - decay) * 1.0 + decay * args.final_lr_frac

# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# -----------------------------------------------------------------------------
# Training loop
x, y = next(train_loader) # prefetch the very first batch of data
min_val_bpb = float("inf")
val_bpb = None
val_loss = None
smooth_train_loss = 0 # EMA of training loss
ema_beta = 0.9 # EMA decay factor
total_training_time = 0 # total wall-clock time of training
step = 0

# Estimate total steps from dataset size and batch config for tqdm
# SFT doesn't know exact num_iterations upfront, so we estimate from dataset size
if args.num_iterations > 0:
    _estimated_total = args.num_iterations
else:
    _estimated_total = max(1, len(train_dataset) // (args.device_batch_size * ddp_world_size))
pbar = tqdm(
    total=_estimated_total,
    desc="sft",
    disable=not master_process,
    dynamic_ncols=True,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
)

while True:
    if args.num_iterations > 0 and step >= args.num_iterations:
        last_step = True

    flops_so_far = num_flops_per_token * args.total_batch_size * step

    # Synchronize last_step across all ranks to avoid hangs in the distributed setting
    if ddp:
        last_step_tensor = torch.tensor(last_step, dtype=torch.int32, device=device)
        dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
        last_step = bool(last_step_tensor.item())

    # once in a while: evaluate the val bpb (all ranks participate)
    if last_step or (args.eval_every > 0 and step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        val_bpb, val_loss = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        if master_process:
            tqdm.write(f"Step {step:05d} | Validation bpb: {val_bpb:.4f} | Val loss: {val_loss:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # once in a while: estimate the ChatCORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    chatcore_results = {}
    if args.chatcore_every > 0 and (last_step or (step > 0 and step % args.chatcore_every == 0)):
        model.eval()
        engine = Engine(orig_model, tokenizer)
        default_chatcore_tasks = ['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval', 'SpellingBee']
        all_tasks = default_chatcore_tasks if args.chatcore_tasks is None else [
            task.strip() for task in args.chatcore_tasks.split('|') if task.strip()
        ]
        categorical_tasks = {'ARC-Easy', 'ARC-Challenge', 'MMLU'}
        baseline_accuracies = {
            'ARC-Easy': 0.25, 'ARC-Challenge': 0.25, 'MMLU': 0.25,
            'GSM8K': 0.0, 'HumanEval': 0.0, 'SpellingBee': 0.0,
        }
        unknown_tasks = set(all_tasks) - set(default_chatcore_tasks)
        if unknown_tasks:
            raise ValueError(f"Unknown ChatCORE task(s): {sorted(unknown_tasks)}")
        task_results = {}
        for task_name in all_tasks:
            limit = args.chatcore_max_cat if task_name in categorical_tasks else args.chatcore_max_sample
            max_problems = None if limit < 0 else limit  # -1 means no limit
            acc = run_chat_eval(task_name, orig_model, tokenizer, engine,
                                batch_size=args.device_batch_size, max_problems=max_problems)
            task_results[task_name] = acc
            print0(f"  {task_name}: {100*acc:.2f}%")
        # Compute ChatCORE metrics (mean centered accuracy, ranges from 0=random to 1=perfect)
        def centered_mean(tasks):
            return sum((task_results[t] - baseline_accuracies[t]) / (1.0 - baseline_accuracies[t]) for t in tasks) / len(tasks)
        chatcore = centered_mean(all_tasks)
        evaluated_categorical_tasks = [task for task in all_tasks if task in categorical_tasks]
        chatcore_cat = centered_mean(evaluated_categorical_tasks) if evaluated_categorical_tasks else 0.0
        if master_process:
            tqdm.write(f"Step {step:05d} | ChatCORE: {chatcore:.4f} | ChatCORE_cat: {chatcore_cat:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "chatcore_metric": chatcore,
            "chatcore_cat": chatcore_cat,
            **{f"chatcore/{task_name}": acc for task_name, acc in task_results.items()},
        })
        model.train()

    # save checkpoint at the end of the run (all ranks participate so each saves its optimizer shard)
    if last_step or (args.save_every > 0 and step > 0 and step % args.save_every == 0):
        output_dirname = args.model_tag if args.model_tag else f"d{depth}" # e.g. d12
        checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", output_dirname)
        checkpoint_step = loaded_checkpoint_step + step
        save_checkpoint(
            checkpoint_dir,
            checkpoint_step,
            orig_model.state_dict(),
            optimizer.state_dict(),
            {
                "step": checkpoint_step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": {
                    "sequence_len": args.max_seq_len,
                    "vocab_size": tokenizer.get_vocab_size(),
                    "n_layer": depth,
                    "n_head": model.config.n_head,
                    "n_kv_head": model.config.n_kv_head,
                    "n_embd": model.config.n_embd,
                    "window_pattern": model.config.window_pattern,
                    # Architecture-affecting fields must be persisted or the SFT checkpoint
                    # will rebuild with the wrong value-embedding layout (load_state_dict
                    # key mismatch / silent corruption). ve_layers is critical; grad_checkpoint
                    # is harmless to shapes but saved for completeness.
                    "ve_layers": model.config.ve_layers,
                    "grad_checkpoint": model.config.grad_checkpoint,
                },
                "user_config": user_config, # inputs to the training script
            },
            rank=ddp_rank,
        )

    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    nan_detected = False
    for micro_step in range(grad_accum_steps):
        loss = model(x, y)
        train_loss = loss.detach() # for logging
        # Skip step if loss is NaN/Inf to prevent gradient corruption
        if not torch.isfinite(train_loss):
            nan_detected = True
            model.zero_grad(set_to_none=True)
            x, y = next(train_loader)
            if args.num_iterations <= 0:
                progress = max(progress, approx_progress)
            break
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        x, y = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
        if args.num_iterations <= 0:
            progress = max(progress, approx_progress) # only increase progress monotonically

    if nan_detected:
        print0(f"WARNING: NaN/Inf loss at step {step}, skipping step")
        step += 1
        continue

    # step the optimizer
    if args.num_iterations > 0:
        progress = min(1.0, step / args.num_iterations)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
    if scaler is not None:
        scaler.unscale_(optimizer)
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # State
    step += 1
    if args.num_iterations > 0:
        progress = min(1.0, step / args.num_iterations)

    # logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * progress
    tok_per_sec = int(args.total_batch_size / dt)
    flops_per_sec = num_flops_per_token * args.total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps

    # Update tqdm progress bar
    if master_process:
        postfix = {
            "loss": f"{debiased_smooth_loss:.4f}",
            "lr": f"{lrm:.2f}",
            "tok/s": f"{tok_per_sec:,}",
            "mfu%": f"{mfu:.1f}",
            "epoch": current_epoch,
        }
        if min_val_bpb < float("inf"):
            postfix["val_bpb"] = f"{val_bpb:.4f}" if 'val_bpb' in dir() and val_bpb is not None else "—"
            postfix["best"] = f"{min_val_bpb:.4f}"
        # Update total estimate dynamically based on approx_progress
        if approx_progress > 0 and args.num_iterations < 0:
            pbar.total = max(pbar.n + 1, int(step / approx_progress))
        pbar.set_postfix(postfix, refresh=False)
        pbar.update(1)

    if step % 10 == 0:
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": current_epoch,
        })

    # The garbage collector spends ~500ms scanning for cycles quite frequently.
    # We manually manage it to avoid these pauses during training.
    if step == 1:
        gc.collect() # manually collect a lot of garbage from setup
        gc.freeze() # freeze all currently surviving objects and exclude them from GC
        gc.disable() # disable GC entirely except:
    elif step % 5000 == 0: # every 5000 steps...
        gc.collect() # manually collect, just to be safe for very long runs

# print a few more stats
pbar.close()
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# Log to report
from mesosfer.utils.report import get_report
get_report().log(section="SFT", data=[
    user_config, # CLI args
    { # stats about the training setup
        "Number of iterations": step,
        "DDP world size": ddp_world_size,
    },
    { # stats about training outcomes
        "Minimum validation bpb": min_val_bpb,
    }
])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
