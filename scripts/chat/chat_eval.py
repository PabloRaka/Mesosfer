"""
Evaluate the Chat model.
All the generic code lives here, and all the evaluation-specific
code lives in mesosfer directory and is imported from here.

Example runs:
python -m scripts.chat.chat_eval -a ARC-Easy
torchrun --nproc_per_node=8 -m scripts.chat.chat_eval -- -a ARC-Easy
"""

import argparse
from functools import partial
import torch
import torch.distributed as dist

from mesosfer.utils.common import compute_init, compute_cleanup, get_dist_info, print0, autodetect_device_type, get_base_dir
from mesosfer.utils.checkpoint_manager import load_model
from mesosfer.eval.engine import Engine
from mesosfer.eval.core_eval import evaluate_task
from scripts.eval.base_eval import (
    HF_CODEMMLU_DATASET,
    HF_CYBERMETRIC_DATASET,
    HF_SECBENCH_DATASET,
    _convert_hf_codemmlu_split_to_core_jsonl,
    _convert_hf_cybermetric_split_to_core_jsonl,
    _convert_hf_secbench_split_to_core_jsonl,
    _random_baseline_for_data,
    prepare_hf_core_jsonl,
    prepare_hf_mmlu_core_jsonl,
)

from tasks.humaneval import HumanEval
from tasks.mmlu import MMLU
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.spellingbee import SpellingBee

# -----------------------------------------------------------------------------
# Generative evaluation loop (we go one problem at a time, sample, evaluate)

def run_generative_eval(task_object, tokenizer, model, engine, num_samples, max_new_tokens, temperature, top_k, max_problems=None):

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    # Run the evaluation
    num_passed, total = 0, 0
    for i in range(ddp_rank, num_problems, ddp_world_size):
        conversation = task_object[i]

        # Tokenize the prompt
        encoded_prompt = tokenizer.render_for_completion(conversation)
        # Get the completions
        results, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=num_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        # Decode the completions as text
        prefix_length = len(encoded_prompt)
        completions = [tokenizer.decode(result_tokens[prefix_length:]) for result_tokens in results]
        # Evaluate success criteria
        outcomes = [task_object.evaluate(conversation, completion) for completion in completions]
        passed = any(outcomes)

        # Keep stats
        total += 1
        num_passed += int(passed)

        # Logging (overwrite the same line in the console)
        print(f"\r\033[KRank {ddp_rank} | {num_passed}/{total} ({100*num_passed/total:.2f}%)", end='', flush=True)

    # Finish the in-place progress line with a newline before final summary
    print()

    # Aggregate results across all ranks
    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    print0("=" * 50)
    print0(f"Final: {num_passed}/{total} ({100*num_passed/total:.2f}%)")

    # Return the accuracy
    return num_passed/total

# -----------------------------------------------------------------------------
# Categorical evaluation loop
# A lot easier because we don't have to sample. Therefore, we can actually go
# batches at a time and just check the logits for correct answer choices.

def run_categorical_eval(task_object, tokenizer, model, batch_size, max_problems=None):

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()
    bos = tokenizer.get_bos_token_id() # use BOS as pad token is ok, these positions are ignored

    # We'll process batches of independent problems at a time because there is no sampling needed
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    ceil_div = lambda x, y: -(-x // y)
    num_batches = ceil_div(num_problems, batch_size)

    # Run the evaluation
    letter_to_id_cache = {} # many letters will repeat often, let's save the tokenizer some work
    num_passed, total = 0, 0
    for i in range(ddp_rank, num_batches, ddp_world_size):
        i0, i1 = i * batch_size, min((i + 1) * batch_size, num_problems)

        # Prepare the batch of problems. They might all be of different length, so we pad/collate them.
        conversations = [task_object[ii] for ii in range(i0, i1)]
        prompt_ids = [tokenizer.render_for_completion(conversation) for conversation in conversations] # TODO: remake the way this works
        max_length = max(len(ids) for ids in prompt_ids)
        answer_time_positions = [len(ids) - 1 for ids in prompt_ids] # where the last token is (and the predicted answer)
        padded_prompt_ids = [ids + [bos] * (max_length - len(ids)) for ids in prompt_ids]
        prompt_ids = torch.tensor(padded_prompt_ids, dtype=torch.long, device=device)

        # Get the logits for the whole batch of conversations in parallel (efficiency win here)
        with torch.no_grad():
            logits = model(prompt_ids) # (B, T, V)

        # Focus on the available answer on just the letters corresponding to choices
        # Note that this helps the evaluation a lot because it specifically narrows the focus to only the available letters
        # The much harder alternative would be to just generate from the Assistant and check if it responded with the correct
        # letter (e.g. A, B, C, D), but evaluations typically make the task easier in this way.
        for idx, conversation in enumerate(conversations):
            # get the token ids of all the available letters of this problem
            letters = conversation['letters']
            letter_ids = []
            for letter in letters:
                if not letter in letter_to_id_cache:
                    encoded_letter = tokenizer.encode(letter)
                    assert len(encoded_letter) == 1, "Each letter must be a single token"
                    letter_to_id_cache[letter] = encoded_letter[0]
                letter_ids.append(letter_to_id_cache[letter])
            # focus logits just down to the answer position and the available letters of the answer
            answer_pos = answer_time_positions[idx]
            focus_logits = logits[idx, answer_pos, letter_ids]
            # get the argmax letter (the predicted answer)
            argmax_letter_id = focus_logits.argmax(dim=-1).item()
            predicted_letter = letters[argmax_letter_id]
            # evaluate the outcome
            outcome = task_object.evaluate(conversation, predicted_letter)
            num_passed += int(outcome)
            total += 1

    # Aggregate results across all ranks
    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    average = num_passed/total
    print0(f"Final: {num_passed}/{total} ({100*average:.2f}%)")
    return average

# -----------------------------------------------------------------------------

def run_chat_eval(task_name, model, tokenizer, engine,
                   batch_size=1, num_samples=1, max_new_tokens=512, temperature=0.0, top_k=50,
                   max_problems=None):
    # Create the evaluation object
    task_module = {
        'HumanEval': HumanEval,
        'MMLU': partial(MMLU, subset="all", split="test"),
        'ARC-Easy': partial(ARC, subset="ARC-Easy", split="test"),
        'ARC-Challenge': partial(ARC, subset="ARC-Challenge", split="test"),
        'GSM8K': partial(GSM8K, subset="main", split="test"),
        'SpellingBee': partial(SpellingBee, size=256, split="test"),
    }[task_name]
    task_object = task_module()
    # Run the evaluation
    if task_object.eval_type == 'generative':
        acc = run_generative_eval(task_object, tokenizer, model, engine, num_samples, max_new_tokens, temperature, top_k, max_problems=max_problems)
    elif task_object.eval_type == 'categorical':
        acc = run_categorical_eval(task_object, tokenizer, model, batch_size, max_problems=max_problems)
    else:
        raise ValueError(f"Unsupported task evaluation type: {task_object.eval_type}")
    return acc

# -----------------------------------------------------------------------------

def _evaluate_domain_jsonl_task(model, tokenizer, device, cfg, data_path, max_problems=None):
    print0(f"Evaluating: {cfg['description']}... ", end='')
    import json
    import random
    import time
    start_time = time.time()

    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]

    shuffle_rng = random.Random(1337)
    shuffle_rng.shuffle(data)
    if max_problems is not None:
        data = data[:max_problems]

    task_meta = {
        'task_type': 'multiple_choice',
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


def run_chat_domain_eval(model, tokenizer, device, max_problems=None, domains="cyber,coding"):
    base_dir = get_base_dir()
    data_base_path = f"{base_dir}/eval_bundle/eval_data"
    requested_domains = {domain.strip() for domain in domains.split(',') if domain.strip()}

    results = {}
    centered_results = {}

    if "cyber" in requested_domains:
        print0("\n" + "=" * 80)
        print0("Chat Cybersecurity Domain Evaluation")
        print0("=" * 80)
        cyber_tasks = [
            {
                "label": "mmlu_computer_security",
                "dataset_uri": "mmlu_computer_security.jsonl",
                "prepare": lambda path: prepare_hf_mmlu_core_jsonl("computer_security", path),
                "num_fewshot": 5,
                "random_baseline": 25.0,
                "description": "MMLU Computer Security (5-shot)",
            },
            {
                "label": "cybermetric_500",
                "dataset_uri": "cybermetric_500.jsonl",
                "prepare": lambda path: prepare_hf_core_jsonl(
                    HF_CYBERMETRIC_DATASET,
                    path,
                    split="train",
                    converter=_convert_hf_cybermetric_split_to_core_jsonl,
                ),
                "num_fewshot": 3,
                "random_baseline": 25.0,
                "description": "CyberMetric 500 (3-shot)",
            },
            {
                "label": "secbench_mcq_en",
                "dataset_uri": "secbench_mcq_en.jsonl",
                "prepare": lambda path: prepare_hf_core_jsonl(
                    HF_SECBENCH_DATASET,
                    path,
                    split="train",
                    data_files="data/MCQs_2730.jsonl",
                    converter=_convert_hf_secbench_split_to_core_jsonl,
                ),
                "num_fewshot": 3,
                "random_baseline": 25.0,
                "description": "SecBench MCQ English (3-shot)",
            },
        ]
        for cfg in cyber_tasks:
            data_path = f"{data_base_path}/{cfg['dataset_uri']}"
            ok, message = cfg["prepare"](data_path)
            print0(f"  {cfg['label']}: {message}")
            if not ok:
                continue
            accuracy, centered = _evaluate_domain_jsonl_task(model, tokenizer, device, cfg, data_path, max_problems)
            results[cfg["label"]] = accuracy
            centered_results[cfg["label"]] = centered

    if "coding" in requested_domains:
        print0("\n" + "=" * 80)
        print0("Chat Coding Domain Evaluation")
        print0("=" * 80)
        coding_tasks = [
            ("codemmlu_programming_syntax", "programming_syntax", 3),
            ("codemmlu_software_principles", "software_principles", 3),
            ("codemmlu_code_completion", "code_completion", 3),
            ("codemmlu_code_repair", "code_repair", 3),
            ("codemmlu_execution_prediction", "execution_prediction", 3),
        ]
        for label, subset, fewshot in coding_tasks:
            cfg = {
                "label": label,
                "dataset_uri": f"{label}.jsonl",
                "num_fewshot": fewshot,
                "random_baseline": None,
                "description": f"CodeMMLU {subset} ({fewshot}-shot)",
            }
            data_path = f"{data_base_path}/{cfg['dataset_uri']}"
            ok, message = prepare_hf_core_jsonl(
                HF_CODEMMLU_DATASET,
                data_path,
                subset=subset,
                converter=_convert_hf_codemmlu_split_to_core_jsonl,
            )
            print0(f"  {label}: {message}")
            if not ok:
                continue
            accuracy, centered = _evaluate_domain_jsonl_task(model, tokenizer, device, cfg, data_path, max_problems)
            results[label] = accuracy
            centered_results[label] = centered

    metrics = {}
    for domain_name, prefix in [("ChatCyberDomain metric", ("mmlu_", "cybermetric_", "secbench_")),
                                ("ChatCodingDomain metric", ("codemmlu_",))]:
        domain_values = [centered for label, centered in centered_results.items() if label.startswith(prefix)]
        if domain_values:
            metrics[domain_name] = sum(domain_values) / len(domain_values)

    if metrics:
        print0("\n" + "=" * 80)
        print0("Chat Domain Evaluation Summary")
        print0("=" * 80)
        for name, value in metrics.items():
            print0(f"{name}: {value:.4f}")
        for label, accuracy in results.items():
            print0(f"  {label:<35} accuracy: {accuracy:.4f} | centered: {centered_results[label]:.4f}")

    return results, centered_results, metrics


def print_final_chat_eval_summary(results, baseline_accuracies, chatcore_metric_dict,
                                  domain_results=None, domain_centered_results=None, domain_metrics=None):
    print0("\n" + "=" * 80)
    print0("Final Chat Evaluation Summary")
    print0("=" * 80)

    overall_centered = []
    if results:
        if chatcore_metric_dict:
            print0(f"ChatCORE metric: {chatcore_metric_dict['ChatCORE metric']:.4f}")
        for task_name, acc in results.items():
            baseline_acc = baseline_accuracies.get(task_name, 0.0)
            centered = (acc - baseline_acc) / (1.0 - baseline_acc)
            overall_centered.append(centered)
            print0(f"  {task_name:<35} accuracy: {acc:.4f} | centered: {centered:.4f}")

    if domain_metrics:
        print0("")
        for name, value in domain_metrics.items():
            print0(f"{name}: {value:.4f}")

    if domain_results:
        for label, accuracy in domain_results.items():
            centered = domain_centered_results[label]
            overall_centered.append(centered)
            print0(f"  {label:<35} accuracy: {accuracy:.4f} | centered: {centered:.4f}")

    if overall_centered:
        overall_metric = sum(overall_centered) / len(overall_centered)
        print0("")
        print0(f"ChatCORE overall metric: {overall_metric:.4f}")

# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--source', type=str, required=True, help="Source of the model: sft|rl")
    parser.add_argument('-a', '--task-name', type=str, default=None, help="Task name. Default = all tasks. Use | to split multiple tasks. Use 'none' for domain eval only.")
    parser.add_argument('-t', '--temperature', type=float, default=0.0)
    parser.add_argument('-m', '--max-new-tokens', type=int, default=512)
    parser.add_argument('-n', '--num-samples', type=int, default=1)
    parser.add_argument('-k', '--top-k', type=int, default=50)
    parser.add_argument('-b', '--batch-size', type=int, default=8, help='Batch size for categorical evaluation')
    parser.add_argument('-g', '--model-tag', '--depth', type=str, default=None, help='Model tag or depth to load (e.g. d12)')
    parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
    parser.add_argument('-x', '--max-problems', type=int, default=None, help='Max problems to evaluate')
    parser.add_argument('--domain-eval', type=int, default=0, help='1 = also run cyber/coding domain evals')
    parser.add_argument('--domain-eval-domains', type=str, default='cyber,coding', help='Comma-separated domain eval groups: cyber,coding')
    parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)

    # Get the tasks to evaluate on
    all_tasks = ['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval', 'SpellingBee']
    baseline_accuracies = {
        'ARC-Easy': 0.25, # multiple choice 1 of 4 => 25%
        'ARC-Challenge': 0.25, # multiple choice 1 of 4 => 25%
        'MMLU': 0.25, # multiple choice 1 of 4 => 25%
        'GSM8K': 0.0, # open-ended => 0%
        'HumanEval': 0.0, # open-ended => 0%
        'SpellingBee': 0.0, # open-ended => 0%
    }
    if args.task_name is None:
        task_names = all_tasks
    elif args.task_name.lower() == 'none':
        task_names = []
    else:
        task_names = args.task_name.split('|')

    # Run all the task evaluations sequentially
    results = {}
    for task_name in task_names:
        acc = run_chat_eval(
            task_name,
            model, tokenizer, engine,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            max_problems=args.max_problems,
        )
        results[task_name] = acc
        print0(f"{task_name} accuracy: {100 * acc:.2f}%")

    domain_results = {}
    domain_centered_results = {}
    domain_metrics = {}
    if args.domain_eval:
        domain_results, domain_centered_results, domain_metrics = run_chat_domain_eval(
            model, tokenizer, device,
            max_problems=args.max_problems,
            domains=args.domain_eval_domains,
        )

    # Log to report
    from mesosfer.utils.report import get_report
    all_tasks_were_evaluated = all(task_name in results for task_name in all_tasks)
    # calculate the ChatCORE metric if we can (similar to CORE, it's the mean centered accuracy)
    # this way, ChatCORE ranges from 0 (at random baseline) to 1 (peak performance)
    chatcore_metric_dict = {}
    if all_tasks_were_evaluated:
        centered_mean = 0
        for task_name, acc in results.items():
            baseline_acc = baseline_accuracies.get(task_name, 0.0)
            centered_acc = (acc - baseline_acc) / (1.0 - baseline_acc)
            centered_mean += centered_acc
        chatcore_metric = centered_mean / len(results)
        chatcore_metric_dict = {"ChatCORE metric": chatcore_metric}

    overall_centered = []
    for task_name, acc in results.items():
        baseline_acc = baseline_accuracies.get(task_name, 0.0)
        overall_centered.append((acc - baseline_acc) / (1.0 - baseline_acc))
    overall_centered.extend(domain_centered_results.values())
    overall_metric_dict = {}
    if overall_centered:
        overall_metric_dict = {"ChatCORE overall metric": sum(overall_centered) / len(overall_centered)}

    print_final_chat_eval_summary(
        results,
        baseline_accuracies,
        chatcore_metric_dict,
        domain_results=domain_results,
        domain_centered_results=domain_centered_results,
        domain_metrics=domain_metrics,
    )

    get_report().log(section="Chat evaluation " + args.source, data=[
        vars(args), # CLI args
        results,
        chatcore_metric_dict,
        domain_results,
        domain_centered_results,
        domain_metrics,
        overall_metric_dict,
    ])

    compute_cleanup()
