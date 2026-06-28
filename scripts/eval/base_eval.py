"""
Unified evaluation script for base models.

Supports three evaluation modes (comma-separated):
  --eval core    : CORE metric (accuracy on ICL tasks)
  --eval bpb     : Bits per byte on train/val splits
  --eval sample  : Generate samples from the model

Default is all three: --eval core,bpb,sample

Examples:

    # Evaluate a HuggingFace model (e.g. GPT-2 124M) using 8 GPUs
    torchrun --nproc_per_node=8 -m scripts.eval.base_eval --hf-path openai-community/gpt2

    # Evaluate a mesosfer model (e.g. d24) using 8 GPUs
    torchrun --nproc_per_node=8 -m scripts.eval.base_eval --model-tag d24 --device-batch-size=16

    # Quick/approximate evaluation using a single GPU
    python -m scripts.eval.base_eval --model-tag d24 --device-batch-size=16 --max-per-task=100 --split-tokens=524288
"""
import os
import csv
import time
import json
import yaml
import shutil
import random
import zipfile
import tempfile
import argparse
import torch

from mesosfer.utils.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type, download_file_with_lock
from mesosfer.data.tokenizer import HuggingFaceTokenizer, get_token_bytes
from mesosfer.utils.checkpoint_manager import load_model
from mesosfer.eval.core_eval import evaluate_task
from mesosfer.data.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from mesosfer.eval.loss_eval import evaluate_bpb
from mesosfer.eval.engine import Engine

# -----------------------------------------------------------------------------
# HuggingFace loading utilities

class ModelWrapper:
    """Lightweight wrapper to give HuggingFace models a mesosfer-compatible interface."""
    def __init__(self, model, max_seq_len=None):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids, targets=None, loss_reduction='mean'):
        logits = self.model(input_ids).logits
        if targets is None:
            return logits
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction
        )
        return loss

    def get_device(self):
        return next(self.model.parameters()).device


def load_hf_model(hf_path: str, device):
    """Load a HuggingFace model and tokenizer."""
    print0(f"Loading HuggingFace model from: {hf_path}")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(hf_path)
    model.to(device)
    model.eval()
    max_seq_len = 1024 if "gpt2" in hf_path else None
    model = ModelWrapper(model, max_seq_len=max_seq_len)
    tokenizer = HuggingFaceTokenizer.from_pretrained(hf_path)
    return model, tokenizer


def get_hf_token_bytes(tokenizer, device="cpu"):
    """Compute token_bytes tensor for a HuggingFace tokenizer."""
    vocab_size = tokenizer.tokenizer.get_vocab_size()
    token_bytes = torch.zeros(vocab_size, dtype=torch.int64, device=device)
    for token_id in range(vocab_size):
        token_str = tokenizer.tokenizer.decode([token_id])
        token_bytes[token_id] = len(token_str.encode('utf-8'))
    return token_bytes

# -----------------------------------------------------------------------------
# CORE evaluation

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

HF_MMLU_DATASET = "cais/mmlu"
HF_CYBERMETRIC_DATASET = "tuandunghcmut/cybermetric_500_v1"
HF_SECBENCH_DATASET = "secbench-hf/SecBench"
HF_CODEMMLU_DATASET = "Fsoft-AIC/CodeMMLU"


def _mmlu_answer_to_index(answer):
    """Normalize MMLU answer values from HF datasets into a 0-based index."""
    if isinstance(answer, int):
        return answer
    if isinstance(answer, str):
        answer = answer.strip()
        if answer.isdigit():
            return int(answer)
        if len(answer) == 1 and answer.upper() in "ABCD":
            return ord(answer.upper()) - ord("A")
    raise ValueError(f"Unsupported MMLU answer value: {answer!r}")


def _letter_answer_to_index(answer):
    """Normalize letter answers such as A/B/C/D into a 0-based index."""
    if isinstance(answer, int):
        return answer
    if isinstance(answer, str):
        answer = answer.strip()
        if answer.isdigit():
            return int(answer)
        if len(answer) == 1 and answer.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            return ord(answer.upper()) - ord("A")
    raise ValueError(f"Unsupported answer value: {answer!r}")


def _write_core_jsonl(items, output_path):
    """Atomically write CORE multiple-choice JSONL items."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for item in items:
            choices = list(item["choices"])
            gold = int(item["gold"])
            if len(choices) < 2:
                raise ValueError(f"Expected at least 2 choices, got {len(choices)}")
            if gold < 0 or gold >= len(choices):
                raise ValueError(f"Gold index out of range: {gold}")
            f.write(json.dumps({
                "query": item["query"],
                "choices": choices,
                "gold": gold,
            }, ensure_ascii=False) + "\n")
    os.replace(tmp_path, output_path)


def _convert_hf_mmlu_split_to_core_jsonl(dataset, output_path):
    """Write a HF MMLU split in CORE multiple-choice JSONL format."""
    items = ({
        "query": row["question"],
        "choices": row["choices"],
        "gold": _mmlu_answer_to_index(row["answer"]),
    } for row in dataset)
    _write_core_jsonl(items, output_path)


def _convert_hf_cybermetric_split_to_core_jsonl(dataset, output_path):
    """Write a CyberMetric split in CORE multiple-choice JSONL format."""
    items = ({
        "query": row["question"],
        "choices": [row["option_a"], row["option_b"], row["option_c"], row["option_d"]],
        "gold": _letter_answer_to_index(row["correct_answer"]),
    } for row in dataset)
    _write_core_jsonl(items, output_path)


def _convert_hf_secbench_split_to_core_jsonl(dataset, output_path):
    """Write the English MCQ subset of SecBench in CORE multiple-choice JSONL format.

    SecBench is bilingual (Chinese + English) and some MCQs are multi-answer.
    We keep only English, single-answer (A-D) questions with exactly 4 choices so
    the result is a clean 4-way multiple-choice task comparable to the others.
    """
    def _iter():
        for row in dataset:
            if str(row.get("language", "")).strip().lower() != "english":
                continue
            label = str(row.get("label", "")).strip()
            if len(label) != 1 or label.upper() not in "ABCD":
                continue
            choices = row.get("answers")
            if not isinstance(choices, list) or len(choices) != 4:
                continue
            yield {
                "query": row["question"],
                "choices": choices,
                "gold": _letter_answer_to_index(label),
            }
    _write_core_jsonl(_iter(), output_path)


def _convert_hf_codemmlu_split_to_core_jsonl(dataset, output_path):
    """Write a CodeMMLU split in CORE multiple-choice JSONL format."""
    items = ({
        "query": row["question"],
        "choices": row["choices"],
        "gold": _letter_answer_to_index(row["answer"]),
    } for row in dataset)
    _write_core_jsonl(items, output_path)


def prepare_hf_mmlu_core_jsonl(subject, output_path, split="test"):
    """
    Materialize a HuggingFace MMLU subject into the local CORE eval format.

    Returns (ok, message). Missing/unavailable HF subjects are non-fatal because
    cyber eval is an optional extension on top of the default CORE bundle.
    """
    if os.path.exists(output_path):
        return True, "cached"

    try:
        from datasets import load_dataset
        dataset = load_dataset(HF_MMLU_DATASET, subject, split=split)
    except Exception as exc:
        return False, f"HF subset unavailable ({subject}): {exc}"

    try:
        _convert_hf_mmlu_split_to_core_jsonl(dataset, output_path)
    except Exception as exc:
        return False, f"failed to convert HF subset ({subject}): {exc}"

    return True, f"created from {HF_MMLU_DATASET}/{subject}:{split}"


def prepare_hf_core_jsonl(dataset_name, output_path, subset=None, split="test", converter=None, data_files=None):
    """Materialize a HF dataset/subset into local CORE eval format."""
    if os.path.exists(output_path):
        return True, "cached"

    try:
        from datasets import load_dataset
        load_kwargs = {"split": split}
        if data_files is not None:
            load_kwargs["data_files"] = data_files
        if subset is None:
            dataset = load_dataset(dataset_name, **load_kwargs)
        else:
            dataset = load_dataset(dataset_name, subset, **load_kwargs)
    except Exception as exc:
        target = dataset_name if subset is None else f"{dataset_name}/{subset}"
        return False, f"HF dataset unavailable ({target}): {exc}"

    try:
        converter(dataset, output_path)
    except Exception as exc:
        target = dataset_name if subset is None else f"{dataset_name}/{subset}"
        return False, f"failed to convert HF dataset ({target}): {exc}"

    target = dataset_name if subset is None else f"{dataset_name}/{subset}"
    return True, f"created from {target}:{split}"


def _random_baseline_for_data(data):
    """Estimate random-choice baseline percentage for variable-choice MCQ data."""
    return 100.0 * sum(1.0 / len(item["choices"]) for item in data) / len(data)


def evaluate_core_jsonl_task(model, tokenizer, device, cfg, data_path, max_per_task=-1):
    """Evaluate one local CORE-format JSONL task and return accuracy/centered."""
    print0(f"Evaluating: {cfg['description']}... ", end='')
    start_time = time.time()

    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]

    shuffle_rng = random.Random(1337)
    shuffle_rng.shuffle(data)
    if max_per_task > 0:
        data = data[:max_per_task]

    task_meta = {
        'task_type': cfg['task_type'],
        'dataset_uri': cfg['dataset_uri'],
        'num_fewshot': cfg['num_fewshot'],
        'continuation_delimiter': ' ',
    }
    accuracy = evaluate_task(model, tokenizer, data, device, task_meta)
    baseline = cfg.get('random_baseline')
    if baseline is None:
        baseline = _random_baseline_for_data(data)
    centered = (accuracy - 0.01 * baseline) / (1.0 - 0.01 * baseline)
    elapsed = time.time() - start_time
    print0(f"accuracy: {accuracy:.4f} | centered: {centered:.4f} | time: {elapsed:.2f}s")
    return accuracy, centered


def print_final_eval_summary(core_results=None, cybersec_results=None, coding_results=None, bpb_results=None):
    """Print a compact terminal summary after all eval sections finish."""
    print0("\n" + "="*80)
    print0("Final Evaluation Summary")
    print0("="*80)

    overall_centered = []
    if core_results:
        print0(f"CORE metric: {core_results['core_metric']:.4f}")
        for label, acc in core_results["results"].items():
            centered = core_results["centered_results"][label]
            overall_centered.append(centered)
            print0(f"  {label:<35} accuracy: {acc:.4f} | centered: {centered:.4f}")

    if cybersec_results:
        cybersec_metric = sum(v['centered'] for v in cybersec_results.values()) / len(cybersec_results)
        print0(f"\nCybersec domain metric: {cybersec_metric:.4f}")
        for label, result in cybersec_results.items():
            overall_centered.append(result['centered'])
            print0(f"  {label:<35} accuracy: {result['accuracy']:.4f} | centered: {result['centered']:.4f}")

    if coding_results:
        coding_metric = sum(v['centered'] for v in coding_results.values()) / len(coding_results)
        print0(f"\nCoding domain metric: {coding_metric:.4f}")
        for label, result in coding_results.items():
            overall_centered.append(result['centered'])
            print0(f"  {label:<35} accuracy: {result['accuracy']:.4f} | centered: {result['centered']:.4f}")

    if overall_centered:
        overall_metric = sum(overall_centered) / len(overall_centered)
        print0("")
        print0(f"CORE overall metric: {overall_metric:.4f}")

    if bpb_results:
        print0("\nBPB:")
        for split_name, bpb in bpb_results.items():
            print0(f"  {split_name:<35} bpb: {bpb:.6f}")


def place_eval_bundle(file_path):
    """Unzip eval_bundle.zip and place it in the base directory."""
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print0(f"Placed eval_bundle directory at {eval_bundle_dir}")


def evaluate_core(model, tokenizer, device, max_per_task=-1, skip_tasks=None):
    """
    Evaluate a base model on the CORE benchmark.
    Returns dict with results, centered_results, and core_metric.

    Args:
        skip_tasks: set of task labels to skip (e.g. {'jeopardy'} for tasks
                    that are too general for a cybersecurity-focused model).
    """
    if skip_tasks is None:
        skip_tasks = set()
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    # Download the eval bundle if needed
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)

    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    tasks = config['icl_tasks']

    # Load random baseline values
    random_baselines = {}
    with open(eval_meta_data, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row['Eval Task']
            random_baseline = row['Random baseline']
            random_baselines[task_name] = float(random_baseline)

    # Evaluate each task
    results = {}
    centered_results = {}
    for task in tasks:
        start_time = time.time()
        label = task['label']

        # Skip tasks that are too general for a cybersecurity-focused model
        if label in skip_tasks:
            print0(f"Skipping: {label} (in --skip-tasks list)")
            continue

        task_meta = {
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' ')
        }
        print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ", end='')

        data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]

        # Shuffle for consistent subsampling when using max_per_task
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)
        results[label] = accuracy
        random_baseline = random_baselines[label]
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result
        elapsed = time.time() - start_time
        print0(f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {elapsed:.2f}s")

    core_metric = sum(centered_results.values()) / len(centered_results)
    out = {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric
    }
    return out

# -----------------------------------------------------------------------------
# Main

def main():
    parser = argparse.ArgumentParser(description="Base model evaluation")
    parser.add_argument('--eval', type=str, default='core,bpb,sample', help='Comma-separated evaluations to run: core,bpb,sample (default: all)')
    parser.add_argument('--hf-path', type=str, default=None, help='HuggingFace model path (e.g. openai-community/gpt2-xl)')
    parser.add_argument('--model-tag', type=str, default=None, help='mesosfer model tag to identify the checkpoint directory')
    parser.add_argument('--step', type=int, default=None, help='Model step to load (default = last)')
    parser.add_argument('--max-per-task', type=int, default=-1, help='Max examples per CORE task (-1 = all)')
    parser.add_argument('--device-batch-size', type=int, default=32, help='Per-device batch size for BPB evaluation')
    parser.add_argument('--split-tokens', type=int, default=40*524288, help='Number of tokens to evaluate per split for BPB')
    parser.add_argument('--device-type', type=str, default='', help='cuda|cpu|mps (empty = autodetect)')
    # Mesosfer-specific: skip overly general tasks that are less relevant for a cybersec-focused model
    parser.add_argument('--skip-tasks', type=str, default='jeopardy',
                        help='Comma-separated CORE task labels to skip (default: jeopardy). '
                             'Use "none" to run all tasks.')
    # Mesosfer-specific: evaluate cybersecurity MMLU subsets on top of CORE
    parser.add_argument('--cybersec-eval', type=int, default=1,
                        help='1 = run cybersecurity MMLU subsets after CORE (default), 0 = skip')
    parser.add_argument('--coding-eval', type=int, default=1,
                        help='1 = run CodeMMLU coding subsets after CORE (default), 0 = skip')
    args = parser.parse_args()

    # Parse evaluation modes
    eval_modes = set(mode.strip() for mode in args.eval.split(','))
    valid_modes = {'core', 'bpb', 'sample'}
    invalid = eval_modes - valid_modes
    if invalid:
        parser.error(f"Invalid eval modes: {invalid}. Valid: {valid_modes}")

    # Distributed / precision setup
    device_type = autodetect_device_type() if args.device_type == '' else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    # Load model and tokenizer
    is_hf_model = args.hf_path is not None
    if is_hf_model:
        model, tokenizer = load_hf_model(args.hf_path, device)
        sequence_len = model.max_seq_len or 1024
        token_bytes = get_hf_token_bytes(tokenizer, device=device)
        model_name = args.hf_path
        model_slug = args.hf_path.replace("/", "-")
    else:
        model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
        sequence_len = meta["model_config"]["sequence_len"]
        token_bytes = get_token_bytes(device=device)
        model_name = f"base_model (step {meta['step']})"
        model_slug = f"base_model_{meta['step']:06d}"

    print0(f"Evaluating model: {model_name}")
    print0(f"Eval modes: {', '.join(sorted(eval_modes))}")

    # Results to log
    core_results = None
    bpb_results = {}
    samples = []
    unconditioned_samples = []

    # --- Sampling ---
    if 'sample' in eval_modes and not is_hf_model:
        print0("\n" + "="*80)
        print0("Model Samples")
        print0("="*80)
        if ddp_rank == 0:
            # Mix of general knowledge and cybersecurity prompts to probe domain coverage
            prompts = [
                # General knowledge (sanity checks)
                "The capital of France is",
                "If 5*x + 3 = 13, then x is",
                "The opposite of hot is",
                # Cybersecurity domain probes
                "A buffer overflow vulnerability allows",
                "MITRE ATT&CK technique T1059 refers to",
                "To detect SQL injection attacks, a SOC analyst should",
                "CVE-2021-44228 (Log4Shell) is exploited by",
                "When responding to a ransomware incident, the first step is",
                "A YARA rule for detecting malicious PowerShell would include",
                "The principle of least privilege means",
            ]
            engine = Engine(model, tokenizer)
            print0("\nConditioned samples:")
            for prompt in prompts:
                tokens = tokenizer(prompt, prepend="<|bos|>")
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=32, temperature=0)
                sample_str = tokenizer.decode(sample[0])
                print0("-" * 80)
                print0(sample_str)
                samples.append(sample_str)

            print0("\nUnconditioned samples:")
            tokens = tokenizer("", prepend="<|bos|>")
            uncond, _ = engine.generate_batch(tokens, num_samples=8, max_tokens=128, temperature=1.0)
            for sample in uncond:
                sample_str = tokenizer.decode(sample)
                print0("-" * 80)
                print0(sample_str)
                unconditioned_samples.append(sample_str)
    elif 'sample' in eval_modes and is_hf_model:
        print0("\nSkipping sampling for HuggingFace models (not supported)")

    # --- BPB evaluation ---
    if 'bpb' in eval_modes:
        print0("\n" + "="*80)
        print0("BPB Evaluation")
        print0("="*80)
        tokens_per_step = args.device_batch_size * sequence_len * ddp_world_size
        if args.split_tokens % tokens_per_step != 0:
            # Adjust to nearest multiple
            args.split_tokens = (args.split_tokens // tokens_per_step) * tokens_per_step
            print0(f"Adjusted split_tokens to {args.split_tokens} (must be divisible by {tokens_per_step})")
        steps = args.split_tokens // tokens_per_step

        for split_name in ["train", "val"]:
            loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, sequence_len, split_name, device=device)
            bpb, _ = evaluate_bpb(model, loader, steps, token_bytes)
            bpb_results[split_name] = bpb
            print0(f"{split_name} bpb: {bpb:.6f}")

    # --- CORE evaluation ---
    if 'core' in eval_modes:
        print0("\n" + "="*80)
        print0("CORE Evaluation")
        print0("="*80)
        # Parse skip list
        skip_tasks_set = set()
        if args.skip_tasks.lower() != 'none':
            skip_tasks_set = {t.strip() for t in args.skip_tasks.split(',') if t.strip()}
        if skip_tasks_set:
            print0(f"Skipping tasks: {', '.join(sorted(skip_tasks_set))}")
        core_results = evaluate_core(model, tokenizer, device,
                                     max_per_task=args.max_per_task,
                                     skip_tasks=skip_tasks_set)

        # Write CSV output
        if ddp_rank == 0:
            base_dir = get_base_dir()
            output_csv_path = os.path.join(base_dir, "base_eval", f"{model_slug}.csv")
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
                f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
                for label in core_results["results"]:
                    acc = core_results["results"][label]
                    centered = core_results["centered_results"][label]
                    f.write(f"{label:<35}, {acc:<10.6f}, {centered:<10.6f}\n")
                f.write(f"{'CORE':<35}, {'':<10}, {core_results['core_metric']:<10.6f}\n")
            print0(f"\nResults written to: {output_csv_path}")
            print0(f"CORE metric: {core_results['core_metric']:.4f}")

    # --- Cybersecurity domain evaluation (Mesosfer-specific) ---
    cybersec_results = {}
    if 'core' in eval_modes and args.cybersec_eval and not is_hf_model:
        print0("\n" + "="*80)
        print0("Cybersecurity Domain Evaluation (Mesosfer)")
        print0("="*80)

        base_dir = get_base_dir()
        eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
        data_base_path = os.path.join(eval_bundle_dir, "eval_data")

        # Cybersecurity-relevant MMLU subsets available in the eval bundle
        # These measure domain knowledge directly relevant to Mesosfer's focus
        cybersec_tasks_cfg = [
            {
                "label": "mmlu_computer_security",
                "dataset_uri": "mmlu_computer_security.jsonl",
                "hf_subject": "computer_security",
                "task_type": "multiple_choice",
                "num_fewshot": 5,
                "random_baseline": 25.0,
                "description": "MMLU Computer Security (5-shot)",
            },
            {
                "label": "cybermetric_500",
                "dataset_uri": "cybermetric_500.jsonl",
                "hf_dataset": HF_CYBERMETRIC_DATASET,
                "hf_subset": None,
                "hf_split": "train",
                "hf_converter": _convert_hf_cybermetric_split_to_core_jsonl,
                "task_type": "multiple_choice",
                "num_fewshot": 3,
                "random_baseline": 25.0,
                "description": "CyberMetric 500 (3-shot)",
            },
            {
                "label": "secbench_mcq_en",
                "dataset_uri": "secbench_mcq_en.jsonl",
                "hf_dataset": HF_SECBENCH_DATASET,
                "hf_subset": None,
                "hf_split": "train",
                "hf_data_files": "data/MCQs_2730.jsonl",
                "hf_converter": _convert_hf_secbench_split_to_core_jsonl,
                "task_type": "multiple_choice",
                "num_fewshot": 3,
                "random_baseline": 25.0,
                "description": "SecBench MCQ English (3-shot)",
            },
        ]

        for cfg in cybersec_tasks_cfg:
            data_path = os.path.join(data_base_path, cfg["dataset_uri"])
            if not os.path.exists(data_path):
                if "hf_subject" in cfg:
                    ok, message = prepare_hf_mmlu_core_jsonl(cfg["hf_subject"], data_path)
                else:
                    ok, message = prepare_hf_core_jsonl(
                        cfg["hf_dataset"],
                        data_path,
                        subset=cfg.get("hf_subset"),
                        split=cfg.get("hf_split", "test"),
                        converter=cfg["hf_converter"],
                        data_files=cfg.get("hf_data_files"),
                    )
                print0(f"  {cfg['label']}: {message}")
                if not ok:
                    continue

            accuracy, centered = evaluate_core_jsonl_task(
                model, tokenizer, device, cfg, data_path, max_per_task=args.max_per_task)
            cybersec_results[cfg['label']] = {'accuracy': accuracy, 'centered': centered}

        if cybersec_results:
            cybersec_metric = sum(v['centered'] for v in cybersec_results.values()) / len(cybersec_results)
            print0(f"\nCybersec domain metric: {cybersec_metric:.4f}")

            # Append to CSV
            if ddp_rank == 0 and core_results is not None:
                output_csv_path = os.path.join(get_base_dir(), "base_eval", f"{model_slug}.csv")
                with open(output_csv_path, 'a', encoding='utf-8', newline='') as f:
                    f.write(f"\n# Cybersecurity Domain Tasks\n")
                    for label, v in cybersec_results.items():
                        f.write(f"{label:<35}, {v['accuracy']:<10.6f}, {v['centered']:<10.6f}\n")
                    f.write(f"{'CybersecDomain':<35}, {'':<10}, {cybersec_metric:<10.6f}\n")

    # --- Coding domain evaluation (Mesosfer-specific) ---
    coding_results = {}
    if 'core' in eval_modes and args.coding_eval and not is_hf_model:
        print0("\n" + "="*80)
        print0("Coding Domain Evaluation (CodeMMLU)")
        print0("="*80)

        base_dir = get_base_dir()
        eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
        data_base_path = os.path.join(eval_bundle_dir, "eval_data")

        coding_tasks_cfg = [
            ("codemmlu_programming_syntax", "programming_syntax", 3),
            ("codemmlu_software_principles", "software_principles", 3),
            ("codemmlu_code_completion", "code_completion", 3),
            ("codemmlu_code_repair", "code_repair", 3),
            ("codemmlu_execution_prediction", "execution_prediction", 3),
        ]

        for label, subset, fewshot in coding_tasks_cfg:
            cfg = {
                "label": label,
                "dataset_uri": f"{label}.jsonl",
                "task_type": "multiple_choice",
                "num_fewshot": fewshot,
                "random_baseline": None,
                "description": f"CodeMMLU {subset} ({fewshot}-shot)",
            }
            data_path = os.path.join(data_base_path, cfg["dataset_uri"])
            if not os.path.exists(data_path):
                ok, message = prepare_hf_core_jsonl(
                    HF_CODEMMLU_DATASET,
                    data_path,
                    subset=subset,
                    converter=_convert_hf_codemmlu_split_to_core_jsonl,
                )
                print0(f"  {label}: {message}")
                if not ok:
                    continue

            accuracy, centered = evaluate_core_jsonl_task(
                model, tokenizer, device, cfg, data_path, max_per_task=args.max_per_task)
            coding_results[label] = {'accuracy': accuracy, 'centered': centered}

        if coding_results:
            coding_metric = sum(v['centered'] for v in coding_results.values()) / len(coding_results)
            print0(f"\nCoding domain metric: {coding_metric:.4f}")

            if ddp_rank == 0 and core_results is not None:
                output_csv_path = os.path.join(get_base_dir(), "base_eval", f"{model_slug}.csv")
                with open(output_csv_path, 'a', encoding='utf-8', newline='') as f:
                    f.write(f"\n# Coding Domain Tasks\n")
                    for label, v in coding_results.items():
                        f.write(f"{label:<35}, {v['accuracy']:<10.6f}, {v['centered']:<10.6f}\n")
                    f.write(f"{'CodingDomain':<35}, {'':<10}, {coding_metric:<10.6f}\n")

    # --- Log to report ---
    from mesosfer.utils.report import get_report
    report_data = [{"model": model_name}]

    if core_results:
        report_data[0]["CORE metric"] = core_results["core_metric"]
        report_data.append(core_results["centered_results"])

    if cybersec_results:
        cybersec_metric = sum(v['centered'] for v in cybersec_results.values()) / len(cybersec_results)
        report_data[0]["CybersecDomain metric"] = cybersec_metric
        report_data.append({k: v['accuracy'] for k, v in cybersec_results.items()})

    if coding_results:
        coding_metric = sum(v['centered'] for v in coding_results.values()) / len(coding_results)
        report_data[0]["CodingDomain metric"] = coding_metric
        report_data.append({k: v['accuracy'] for k, v in coding_results.items()})

    overall_centered = []
    if core_results:
        overall_centered.extend(core_results["centered_results"].values())
    if cybersec_results:
        overall_centered.extend(v["centered"] for v in cybersec_results.values())
    if coding_results:
        overall_centered.extend(v["centered"] for v in coding_results.values())
    if overall_centered:
        report_data[0]["CORE overall metric"] = sum(overall_centered) / len(overall_centered)

    if bpb_results:
        report_data[0]["train bpb"] = bpb_results.get("train")
        report_data[0]["val bpb"] = bpb_results.get("val")

    if samples:
        report_data.append({f"sample {i}": s for i, s in enumerate(samples)})
    if unconditioned_samples:
        report_data.append({f"unconditioned {i}": s for i, s in enumerate(unconditioned_samples)})

    print_final_eval_summary(
        core_results=core_results,
        cybersec_results=cybersec_results,
        coding_results=coding_results,
        bpb_results=bpb_results,
    )

    get_report().log(section="Base model evaluation", data=report_data)

    compute_cleanup()


if __name__ == "__main__":
    main()
