"""
Focused tokenizer BPB sweep orchestrator.

Trains a small grid of candidate tokenizers (vocab size x split-pattern preset),
prints a cheap compression pre-screen, then trains a small model per candidate on an
IDENTICAL token budget and reports validation BPB — so the best tokenizer is chosen by
the real downstream metric rather than a proxy.

Design spec: docs/superpowers/specs/2026-07-11-tokenizer-bpb-sweep-design.md

Fairness protocol: every candidate uses the SAME model config and the SAME token budget
(depth 8, pinned --total-batch-size and --num-iterations), so compute is equal across
candidates. BPB is per-byte (via each tokenizer's own token_bytes.pt), so it is comparable
across different vocab sizes.

Phases (run individually or `all`):
  tokenizers  Phase 1: train the candidate tokenizers (CPU).
  prescreen   Phase 2: rank candidates by compression (cheap; guidance only by default).
  bpb         Phase 3: train a small model per candidate, capture validation BPB (needs GPU).
  report      Phase 4: aggregate BPB results into a markdown table.
  all         tokenizers -> prescreen -> bpb -> report.

Everything is idempotent: existing tokenizers / completed BPB runs are skipped unless --force.
Intended to run on the A100 box after `prepare_data.py` has produced parquet text.
"""

import os
import re
import sys
import json
import argparse
import itertools
import subprocess

from mesosfer.utils.common import get_base_dir

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOK_TRAIN = os.path.join(REPO_ROOT, "scripts", "train", "tok_train.py")
BASE_TRAIN = os.path.join(REPO_ROOT, "scripts", "train", "base_train.py")
TOK_EVAL = os.path.join(REPO_ROOT, "scripts", "eval", "tok_eval.py")

DEFAULT_VOCAB_SIZES = [65536, 98304, 131072]
DEFAULT_SPLIT_PRESETS = ["num12", "num13"]

VAL_BPB_RE = re.compile(r"Validation bpb:\s*([0-9]*\.?[0-9]+)")


# -----------------------------------------------------------------------------
# Path helpers

def candidate_name(vocab, preset):
    return f"v{vocab}_{preset}"


def sweep_root():
    return os.path.join(get_base_dir(), "tok_sweep")


def candidate_tokenizer_dir(name):
    # tok_train saves tokenizer.pkl + token_bytes.pt into this directory
    return os.path.join(sweep_root(), name, "tokenizer")


def candidate_log_path(name):
    return os.path.join(sweep_root(), name, "bpb_train.log")


def model_tag(name):
    return f"toksweep_{name}"


def results_json_path():
    return os.path.join(sweep_root(), "results.json")


# -----------------------------------------------------------------------------
# Subprocess helper

def run(cmd, log_path=None):
    """Run a subprocess from the repo root, streaming output. If log_path is given, tee
    combined stdout+stderr to that file and return the captured text; else stream live."""
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    if log_path is None:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        return None
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    captured = []
    with subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as proc, \
         open(log_path, "w", encoding="utf-8") as logf:
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
            captured.append(line)
        ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)
    return "".join(captured)


# -----------------------------------------------------------------------------
# Phase 1: train candidate tokenizers

def phase_tokenizers(candidates, max_chars, doc_cap, force):
    print("\n=== Phase 1: train candidate tokenizers ===")
    for vocab, preset in candidates:
        name = candidate_name(vocab, preset)
        out_dir = candidate_tokenizer_dir(name)
        done_marker = os.path.join(out_dir, "token_bytes.pt")
        if os.path.exists(done_marker) and not force:
            print(f"[tok] skip {name} (already trained)")
            continue
        os.makedirs(out_dir, exist_ok=True)
        run([sys.executable, TOK_TRAIN,
             "--vocab-size", str(vocab),
             "--split-pattern-preset", preset,
             "--out-dir", out_dir,
             "--max-chars", str(max_chars),
             "--doc-cap", str(doc_cap)])


# -----------------------------------------------------------------------------
# Phase 2: compression pre-screen (guidance)

def phase_prescreen(candidates, prescreen_keep):
    print("\n=== Phase 2: compression pre-screen ===")
    names = [candidate_name(v, p) for v, p in candidates]
    dirs = [candidate_tokenizer_dir(n) for n in names]
    # Reuse tok_eval's candidate-ranking path for a human-readable table.
    run([sys.executable, TOK_EVAL, "--tokenizer-dirs", *dirs])

    if prescreen_keep is None or prescreen_keep >= len(candidates):
        print(f"[prescreen] keeping all {len(candidates)} candidates for BPB "
              f"(compression is only guidance; set --prescreen-keep to prune).")
        return candidates

    # Optional pruning: keep the top-K by byte-weighted compression on val text.
    from mesosfer.data.tokenizer import RustBPETokenizer
    from mesosfer.data.dataset import parquets_iter_batched
    try:
        val_text = "\n".join(next(parquets_iter_batched(split="val")))
    except Exception as e:
        print(f"[prescreen] could not load val text ({e}); keeping all candidates.")
        return candidates
    scored = []
    for (v, p), d in zip(candidates, dirs):
        tok = RustBPETokenizer.from_directory(d)
        enc = tok.encode(val_text)
        ratio = len(val_text.encode("utf-8")) / len(enc)
        scored.append(((v, p), ratio))
    scored.sort(key=lambda x: x[1], reverse=True)
    kept = [c for c, _ in scored[:prescreen_keep]]
    print(f"[prescreen] pruned to top {prescreen_keep} by val compression: "
          f"{[candidate_name(v, p) for v, p in kept]}")
    return kept


# -----------------------------------------------------------------------------
# Phase 3: BPB runs

def phase_bpb(candidates, target_tokens, depth, device_batch_size, max_seq_len,
              warmup_steps, eval_every, force):
    print("\n=== Phase 3: BPB runs (train small model per candidate) ===")
    # Pin tokens-per-optimizer-step so the token budget is exact and identical across
    # candidates regardless of vocab size (single GPU, no grad accumulation).
    total_batch_size = device_batch_size * max_seq_len
    num_iterations = max(1, -(-target_tokens // total_batch_size))  # ceil division
    print(f"[bpb] depth={depth} tokens/step={total_batch_size:,} "
          f"num_iterations={num_iterations:,} (~{num_iterations * total_batch_size:,} tokens)")

    results = load_results()
    for vocab, preset in candidates:
        name = candidate_name(vocab, preset)
        if name in results and results[name].get("val_bpb") is not None and not force:
            print(f"[bpb] skip {name} (val_bpb={results[name]['val_bpb']})")
            continue
        tok_dir = candidate_tokenizer_dir(name)
        if not os.path.exists(os.path.join(tok_dir, "token_bytes.pt")):
            print(f"[bpb] WARNING: tokenizer for {name} missing; run phase 'tokenizers' first. Skipping.")
            continue
        log_path = candidate_log_path(name)
        cmd = [sys.executable, BASE_TRAIN,
               "--depth", str(depth),
               "--tokenizer-dir", tok_dir,
               "--model-tag", model_tag(name),
               "--num-iterations", str(num_iterations),
               "--total-batch-size", str(total_batch_size),
               "--device-batch-size", str(device_batch_size),
               "--max-seq-len", str(max_seq_len),
               "--warmup-steps", str(warmup_steps),
               "--eval-every", str(eval_every),
               "--core-metric-every", "-1",  # skip CORE metric — we only need BPB
               "--sample-every", "-1",       # skip sampling
               "--run", "dummy"]             # disable wandb
        try:
            output = run(cmd, log_path=log_path)
        except subprocess.CalledProcessError as e:
            print(f"[bpb] {name} FAILED (exit {e.returncode}); see {log_path}")
            results[name] = {"vocab": vocab, "preset": preset, "val_bpb": None, "status": "failed"}
            save_results(results)
            continue
        val_bpb = parse_min_val_bpb(output)
        results[name] = {"vocab": vocab, "preset": preset, "val_bpb": val_bpb, "status": "ok"}
        save_results(results)
        print(f"[bpb] {name}: val_bpb={val_bpb}")
    return results


def parse_min_val_bpb(text):
    """Return the minimum 'Validation bpb: X' value seen in the run output, or None."""
    vals = [float(m) for m in VAL_BPB_RE.findall(text or "")]
    return min(vals) if vals else None


# -----------------------------------------------------------------------------
# Results persistence + Phase 4 report

def load_results():
    path = results_json_path()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(results):
    os.makedirs(sweep_root(), exist_ok=True)
    with open(results_json_path(), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def phase_report(candidates):
    print("\n=== Phase 4: BPB results ===")
    results = load_results()
    rows = []
    for vocab, preset in candidates:
        name = candidate_name(vocab, preset)
        entry = results.get(name, {})
        rows.append((name, vocab, preset, entry.get("val_bpb"), entry.get("status", "missing")))

    # Rank by val_bpb ascending (lower is better); Nones last.
    rows.sort(key=lambda r: (r[3] is None, r[3] if r[3] is not None else 0.0))

    print(f"\n{'Candidate':<22}{'Vocab':<9}{'Preset':<8}{'Val BPB':<12}{'Status':<10}")
    print("-" * 61)
    for name, vocab, preset, bpb, status in rows:
        bpb_s = f"{bpb:.6f}" if bpb is not None else "-"
        print(f"{name:<22}{vocab:<9}{preset:<8}{bpb_s:<12}{status:<10}")
    winner = next((r for r in rows if r[3] is not None), None)
    if winner:
        print(f"\nWinner (lowest val BPB): {winner[0]}  (val_bpb={winner[3]:.6f})")

    # Log to the run report as markdown.
    try:
        from mesosfer.utils.report import get_report
        lines = ["### Tokenizer BPB sweep results", "",
                 "| Candidate | Vocab | Split preset | Val BPB | Status |",
                 "|-----------|-------|--------------|---------|--------|"]
        for name, vocab, preset, bpb, status in rows:
            bpb_s = f"{bpb:.6f}" if bpb is not None else "-"
            lines.append(f"| {name} | {vocab} | {preset} | {bpb_s} | {status} |")
        if winner:
            lines += ["", f"**Winner:** {winner[0]} (val BPB {winner[3]:.6f})"]
        get_report().log(section="Tokenizer BPB sweep", data=["\n".join(lines)])
    except Exception as e:
        print(f"[report] could not log to report ({e}); results are in {results_json_path()}")
    return rows


# -----------------------------------------------------------------------------
# Main

def main():
    parser = argparse.ArgumentParser(description="Focused tokenizer BPB sweep")
    parser.add_argument("--phase", choices=["all", "tokenizers", "prescreen", "bpb", "report"],
                        default="all", help="which phase to run (default: all)")
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=DEFAULT_VOCAB_SIZES,
                        help=f"vocab sizes to sweep (default: {DEFAULT_VOCAB_SIZES})")
    parser.add_argument("--split-presets", nargs="+", default=DEFAULT_SPLIT_PRESETS,
                        help=f"split-pattern presets to sweep (default: {DEFAULT_SPLIT_PRESETS})")
    # Tokenizer training
    parser.add_argument("--max-chars", type=int, default=2_000_000_000,
                        help="chars to train each tokenizer on (default: 2B, same as tok_train)")
    parser.add_argument("--doc-cap", type=int, default=10_000, help="max chars per doc (default: 10,000)")
    # BPB screening model
    parser.add_argument("--target-tokens", type=int, default=1_000_000_000,
                        help="identical token budget per candidate BPB run (default: 1B)")
    parser.add_argument("--depth", type=int, default=8, help="screening model depth (default: 8)")
    parser.add_argument("--device-batch-size", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--warmup-steps", type=int, default=40, help="LR warmup steps (40 ok for tiny models)")
    parser.add_argument("--eval-every", type=int, default=500, help="evaluate val BPB every N steps")
    parser.add_argument("--prescreen-keep", type=int, default=None,
                        help="if set, keep only top-K candidates by val compression before BPB "
                             "(default: keep all — compression is guidance only)")
    parser.add_argument("--force", action="store_true", help="recompute even if artifacts exist")
    args = parser.parse_args()

    candidates = list(itertools.product(args.vocab_sizes, args.split_presets))
    print(f"Sweep root: {sweep_root()}")
    print(f"Candidates ({len(candidates)}): {[candidate_name(v, p) for v, p in candidates]}")

    if args.phase in ("all", "tokenizers"):
        phase_tokenizers(candidates, args.max_chars, args.doc_cap, args.force)
    if args.phase in ("all", "prescreen"):
        candidates = phase_prescreen(candidates, args.prescreen_keep)
    if args.phase in ("all", "bpb"):
        phase_bpb(candidates, args.target_tokens, args.depth, args.device_batch_size,
                  args.max_seq_len, args.warmup_steps, args.eval_every, args.force)
    if args.phase in ("all", "report"):
        phase_report(candidates)


if __name__ == "__main__":
    main()
