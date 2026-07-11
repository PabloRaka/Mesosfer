"""
Install a swept tokenizer as the default (<base_dir>/tokenizer).

After `tok_sweep.py` has produced val-BPB results, promote the winning candidate (or any
named candidate / arbitrary directory) to the default tokenizer location so that ordinary
`base_train.py` / `chat_sft.py` runs pick it up automatically.

The winning tokenizer needs NO retraining: tokenizer quality depends only on corpus + vocab
+ split-pattern (all finalized in the sweep's Phase 1 on the full 2B chars). The `depth 8`
screening model only measured BPB — it is not part of the tokenizer.

Safety: the current default tokenizer is BACKED UP to <base_dir>/tokenizer.backup-<timestamp>
before it is overwritten. Use --dry-run to preview, --force to skip the "already default" guard.

Usage:
  python scripts/eval/tok_install.py                 # install lowest-BPB winner from results.json
  python scripts/eval/tok_install.py --name v65536_num12
  python scripts/eval/tok_install.py --from-dir /path/to/tokenizer
  python scripts/eval/tok_install.py --dry-run
"""

import os
import sys
import time
import shutil
import argparse

# Reuse the sweep's path/result helpers (sibling module in scripts/eval/).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tok_sweep

from mesosfer.utils.common import get_base_dir

REQUIRED_FILES = ["tokenizer.pkl", "token_bytes.pt"]


def default_tokenizer_dir():
    return os.path.join(get_base_dir(), "tokenizer")


def pick_winner():
    """Return (name, val_bpb) of the lowest-BPB successful candidate, or (None, None)."""
    results = tok_sweep.load_results()
    ok = [(name, e) for name, e in results.items()
          if e.get("status") == "ok" and e.get("val_bpb") is not None]
    if not ok:
        return None, None
    name, entry = min(ok, key=lambda kv: kv[1]["val_bpb"])
    return name, entry["val_bpb"]


def resolve_source(args):
    """Resolve the source tokenizer directory from --from-dir / --name / results winner."""
    if args.from_dir:
        return args.from_dir, None
    if args.name:
        return tok_sweep.candidate_tokenizer_dir(args.name), None
    name, val_bpb = pick_winner()
    if name is None:
        sys.exit("No successful BPB results found in results.json. Run `tok_sweep.py --phase bpb` "
                 "first, or pass --name / --from-dir explicitly.")
    return tok_sweep.candidate_tokenizer_dir(name), (name, val_bpb)


def validate_source(src):
    if not os.path.isdir(src):
        sys.exit(f"Source tokenizer dir does not exist: {src}")
    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(src, f))]
    if missing:
        sys.exit(f"Source {src} is missing required files: {missing}")


def backup_existing(dest, dry_run):
    """Back up an existing non-empty default tokenizer to a timestamped sibling dir."""
    if not (os.path.isdir(dest) and os.listdir(dest)):
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup_dir = f"{dest.rstrip(os.sep)}.backup-{stamp}"
    print(f"[backup] existing default tokenizer -> {backup_dir}")
    if not dry_run:
        shutil.copytree(dest, backup_dir)
    return backup_dir


def install(src, dest, dry_run):
    print(f"[install] {src} -> {dest}")
    if dry_run:
        for f in sorted(os.listdir(src)):
            print(f"    would copy: {f}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    shutil.copytree(src, dest)


def main():
    parser = argparse.ArgumentParser(description="Install a swept tokenizer as the default")
    src_group = parser.add_mutually_exclusive_group()
    src_group.add_argument("--name", help="candidate name to install (e.g. v65536_num12)")
    src_group.add_argument("--from-dir", help="install an arbitrary tokenizer directory")
    parser.add_argument("--dry-run", action="store_true", help="preview actions without changing anything")
    parser.add_argument("--force", action="store_true",
                        help="install even if the source already IS the current default")
    args = parser.parse_args()

    src, winner = resolve_source(args)
    dest = default_tokenizer_dir()
    validate_source(src)

    if winner:
        print(f"Selected winner from results.json: {winner[0]} (val BPB {winner[1]:.6f})")
    print(f"Source     : {src}")
    print(f"Destination: {dest}")

    if os.path.abspath(src) == os.path.abspath(dest) and not args.force:
        print("Source is already the default tokenizer; nothing to do. (use --force to re-copy)")
        return

    backup_existing(dest, args.dry_run)
    install(src, dest, args.dry_run)

    if args.dry_run:
        print("\n[dry-run] no changes made.")
    else:
        print("\nDone. The new default tokenizer is installed.")
        print("Verify:   python scripts/eval/tok_eval.py        # 'Ours' now reflects the new tokenizer")
        print("Retrain:  python scripts/train/base_train.py --depth <target>   # full run on the new default")


if __name__ == "__main__":
    main()
