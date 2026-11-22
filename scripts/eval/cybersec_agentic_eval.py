"""
Run the offline agentic cybersecurity evaluation.

The model is given a lab/CTF task, may call tools, and the tool results are
replayed from a scripted dataset instead of executing anything. This measures
tool selection, multi-step chaining, and whether the final answer reaches the
expected finding.

Example:
    python -m scripts.eval.cybersec_agentic_eval -i sft
    python -m scripts.eval.cybersec_agentic_eval -i sft -x 4 --output eval_agentic.json
"""

import argparse
import json
from pathlib import Path

from mesosfer.utils.common import (
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    print0,
)
from mesosfer.utils.checkpoint_manager import load_model
from mesosfer.eval.engine import Engine
from mesosfer.eval.harness import run_agentic
from mesosfer.eval.cybersec_agentic import format_report, load_eval_cases, make_mock_harness, score_case, summarize

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "data" / "eval" / "cybersec_agentic_lab_eval.jsonl"


def run_case(engine, tokenizer, case, max_tool_calls, max_new_tokens, temperature, top_k, seed):
    harness, _state = make_mock_harness(case)
    conversation = {"messages": [{"role": "user", "content": case["prompt"]}]}
    result = run_agentic(
        engine,
        tokenizer,
        conversation,
        harness,
        max_tool_calls=max_tool_calls,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
    )
    return score_case(case, result)


def main() -> int:
    parser = argparse.ArgumentParser(description="Agentic cybersecurity eval with mock tools")
    parser.add_argument("-i", "--source", type=str, required=True, choices=["sft", "rl"], help="model source")
    parser.add_argument("-g", "--model-tag", "--depth", type=str, default=None, help="model tag/depth to load")
    parser.add_argument("-s", "--step", type=int, default=None, help="step to load")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="eval JSONL path")
    parser.add_argument("-x", "--max-cases", type=int, default=None, help="evaluate at most N cases")
    parser.add_argument("--max-tool-calls", type=int, default=6, help="tool-call budget per case")
    parser.add_argument("-m", "--max-new-tokens", type=int, default=512)
    parser.add_argument("-t", "--temperature", type=float, default=0.0)
    parser.add_argument("-k", "--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps"],
                        help="device type; empty => autodetect")
    parser.add_argument("--output", type=Path, default=None, help="optional JSON results path")
    parser.add_argument("--show-answers", action="store_true", help="include final answers in the JSON output")
    args = parser.parse_args()

    cases = load_eval_cases(args.dataset)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    if not cases:
        print0("No eval cases to run.")
        return 1

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    model, tokenizer, meta = load_model(args.source, device, phase="eval",
                                        model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)

    # Shard across ranks like the other eval loops.
    local_rows = []
    for index in range(ddp_rank, len(cases), ddp_world_size):
        case = cases[index]
        row = run_case(
            engine, tokenizer, case,
            max_tool_calls=args.max_tool_calls,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed + index,
        )
        local_rows.append(row)
        print0(f"[rank {ddp_rank}] {row['id']}: {'PASS' if row['passed'] else 'FAIL'}")

    # Gather rows from every rank so the report is global.
    import torch.distributed as dist

    gathered = [None] * ddp_world_size
    if ddp:
        dist.all_gather_object(gathered, local_rows)
    else:
        gathered[0] = local_rows
    rows = [row for chunk in gathered if chunk for row in chunk]
    rows.sort(key=lambda r: r["id"])

    metrics = summarize(rows)
    step_val = args.step if args.step is not None else meta.get("step", "latest")
    model_info = f"Model: {args.source.upper()} | Depth: {args.model_tag or 'd12'} | Step: {step_val}"
    print0("\n" + format_report(rows, metrics, model_info=model_info))

    if args.output is not None:
        payload = {"model_info": model_info, "dataset": str(args.dataset), "metrics": metrics, "rows": rows}
        if not args.show_answers:
            for row in payload["rows"]:
                row.pop("final_answer", None)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print0(f"Saved results to {args.output.resolve()}")

    try:
        from mesosfer.utils.report import get_report
        get_report().log(section="Cybersec agentic eval " + args.source, data=[
            model_info, metrics, rows,
        ])
    except Exception as exc:  # reporting must never fail the eval
        print0(f"Notice: could not log report ({exc})")

    compute_cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
