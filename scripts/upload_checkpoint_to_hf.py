#!/usr/bin/env python3
"""
Upload mesosfer artifacts (model checkpoints, tokenizer, dataset) to
HuggingFace Hub.

Run without arguments for interactive mode (top-level selection menu).
Or use flags directly for non-interactive / scripting.

Usage:
    # Interactive mode (recommended) — top-level menu picks artifact
    python scripts/upload_checkpoint_to_hf.py

    # ---- Model checkpoints (default --artifact model) -----------------------
    # Upload the latest base checkpoint
    python scripts/upload_checkpoint_to_hf.py --latest

    # Upload the best SFT / RL checkpoint
    python scripts/upload_checkpoint_to_hf.py --best --source sft
    python scripts/upload_checkpoint_to_hf.py --best --source rl

    # Upload a specific step / model + meta only (skip optimizer)
    python scripts/upload_checkpoint_to_hf.py --step 8000
    python scripts/upload_checkpoint_to_hf.py --best --model-only

    # ---- Tokenizer ---------------------------------------------------------
    python scripts/upload_checkpoint_to_hf.py --artifact tokenizer

    # ---- Dataset (parquet shards) ------------------------------------------
    # Upload local cybersecurity shards (default dir)
    python scripts/upload_checkpoint_to_hf.py --artifact dataset

    # Upload from a custom dir under ~/.cache/mesosfer/
    python scripts/upload_checkpoint_to_hf.py --artifact dataset \\
        --dataset-dirs base_data_cybersecurity

Checkpoint sources:
    base  → ~/.cache/mesosfer/base_checkpoints/<depth>/
    sft   → ~/.cache/mesosfer/chatsft_checkpoints/<depth>/
    rl    → ~/.cache/mesosfer/chatrl_checkpoints/<depth>/

HF repo layout produced by this script:
    <repo>/
    ├── base/<depth>/{model,meta,optim}_XXXXXX.{pt,json}
    ├── sft/<depth>/...
    ├── rl/<depth>/...
    ├── tokenizer/{tokenizer.pkl, token_bytes.pt}
    └── dataset/<dataset-name>/shard_XXXXX.parquet
"""

import os
import sys
import json
import argparse
from pathlib import Path

# ── UI helpers (shared with download script) ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from _ui import (  # noqa: E402
    box, menu, confirm, section, badge, spinner,
    info, success, warn, err, dim,
    BOLD, CYAN, BRIGHT_BLACK, BRIGHT_GREEN, RESET,
)


# ── .env loader (no external deps) ───────────────────────────────────────────

def _load_dotenv() -> None:
    """Load key=value pairs from .env (repo root) into os.environ if not set."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

_load_dotenv()


def _resolve_username(cli_value: str | None) -> str:
    """
    Resolve the HF username with this priority:
      1. Explicit username from CLI (if user passed full repo, extract username)
      2. HF_USERNAME env var / .env file
      3. Extract from HF_REPO in env (e.g. johndoe/mesosfer-checkpoints -> johndoe)
      4. Auto-detect from active HF login whoami
      5. Prompt the user interactively
    """
    if cli_value:
        if "/" in cli_value:
            return cli_value.split("/")[0]
        return cli_value

    env_username = os.environ.get("HF_USERNAME", "").strip()
    if env_username and "your_hf_username" not in env_username:
        return env_username

    env_repo = os.environ.get("HF_REPO", "").strip()
    if env_repo and "your_hf_username" not in env_repo and "/" in env_repo:
        return env_repo.split("/")[0]

    # Auto-detect via HfApi
    try:
        from huggingface_hub import HfApi
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        api = HfApi(token=token)
        user = api.whoami()
        if "name" in user:
            return user["name"]
    except Exception:
        pass

    # Interactive fallback
    warn("HF_USERNAME is not set in .env and auto-detect failed.")
    try:
        username = input("  Enter your Hugging Face username (e.g. mesosfer): ").strip()
    except (KeyboardInterrupt, EOFError):
        username = ""
    if not username:
        err("ERROR: No Hugging Face username specified. Aborting.")
        sys.exit(1)
    return username


# Files that make up the tokenizer artifact (mirrors tok_train.py output)
TOKENIZER_FILES = ("tokenizer.pkl", "token_bytes.pt")


# ── Helpers ──────────────────────────────────────────────────────────────────

def find_best_checkpoint(ckpt_dir: Path) -> tuple[int, float]:
    """Search for the checkpoint with the lowest val_bpb across all meta_*.json files."""
    meta_files = sorted(ckpt_dir.glob("meta_*.json"))
    if not meta_files:
        raise FileNotFoundError(f"Could not find any meta_*.json files. {ckpt_dir}")

    best_step = None
    best_bpb = float("inf")

    for meta_path in meta_files:
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            val_bpb = meta.get("val_bpb")
            if val_bpb is None:
                continue
            step = int(meta_path.stem.split("_")[1])  # meta_008000.json -> 8000
            if val_bpb < best_bpb:
                best_bpb = val_bpb
                best_step = step
        except Exception as e:
            print(f"  WARN: Failed to read {meta_path.name}: {e}")

    if best_step is None:
        raise ValueError("Could not find any checkpoints with a valid val_bpb value.")

    return best_step, best_bpb


def find_latest_checkpoint(ckpt_dir: Path) -> tuple[int, float | None]:
    """Find the checkpoint with the highest step (latest)."""
    meta_files = sorted(ckpt_dir.glob("meta_*.json"))
    if not meta_files:
        raise FileNotFoundError(f"No meta_*.json files found in the current directory. {ckpt_dir}")

    latest_meta = meta_files[-1]
    step = int(latest_meta.stem.split("_")[1])

    try:
        with open(latest_meta, encoding="utf-8") as f:
            meta = json.load(f)
        val_bpb = meta.get("val_bpb")
    except Exception:
        val_bpb = None

    return step, val_bpb


def load_all_checkpoints(ckpt_dir: Path) -> list[dict]:
    """
    Read all meta_*.json files and return a list of dicts:
    [{"step": int, "val_bpb": float|None, "label": str}, ...]
    sorted from lowest to highest step.
    """
    meta_files = sorted(ckpt_dir.glob("meta_*.json"))
    rows = []
    best_bpb = float("inf")

    for meta_path in meta_files:
        try:
            step = int(meta_path.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        val_bpb = None
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            val_bpb = meta.get("val_bpb")
        except Exception:
            pass
        if isinstance(val_bpb, float) and val_bpb < best_bpb:
            best_bpb = val_bpb
        rows.append({"step": step, "val_bpb": val_bpb})

    # Mark best
    for row in rows:
        bpb = row["val_bpb"]
        bpb_str = f"{bpb:.6f}" if isinstance(bpb, float) else "N/A    "
        is_best = isinstance(bpb, float) and bpb == best_bpb
        tag = " ← BEST" if is_best else ""
        row["label"] = f"step {row['step']:>7,}   val_bpb={bpb_str}{tag}"

    return rows


def list_checkpoints(ckpt_dir: Path):
    """Show all checkpoints along with their val_bpb."""
    rows = load_all_checkpoints(ckpt_dir)
    if not rows:
        print("No checkpoints found.")
        return

    print(f"\n{'Step':<10} {'val_bpb':<12} {'Status'}")
    print("-" * 35)
    for row in rows:
        bpb = row["val_bpb"]
        bpb_str = f"{bpb:.6f}" if isinstance(bpb, float) else "N/A"
        tag = "← BEST" if "← BEST" in row["label"] else ""
        print(f"{row['step']:<10} {bpb_str:<12} {tag}")
    print()


# ── Checkbox multi-select (prompt_toolkit) ───────────────────────────────────

def checkbox_select(rows: list[dict]) -> list[int]:
    """
    Display checkpoints as an interactive checkbox list.
    Navigation: ↑/↓ or j/k to move, SPACE to toggle, ENTER to confirm, q/ESC to cancel.
    Return a list of selected steps..
    """
    try:
        from prompt_toolkit import Application
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import Layout
        from prompt_toolkit.layout.containers import Window
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.formatted_text import HTML
        from prompt_toolkit.styles import Style
    except ImportError:
        print("ERROR: prompt_toolkit is not installed.")
        print("Run: pip install prompt_toolkit")
        sys.exit(1)

    if not rows:
        print("No checkpoints found.")
        return []

    n = len(rows)
    cursor = [0]          # current cursor position
    selected = set()      # selected indices
    cancelled = [False]

    style = Style.from_dict({
        "cursor":   "bold ansicyan",
        "checked":  "bold ansigreen",
        "unchecked": "",
        "tag":      "ansiyellow",
        "hint":     "ansibrightblack",
    })

    def get_text():
        lines = []
        lines.append(HTML(
            "<hint>  ↑/↓ navigate   SPACE select/deselect   ENTER confirm   q cancel\n\n</hint>"
        ))
        for i, row in enumerate(rows):
            is_cursor = (i == cursor[0])
            is_sel = (i in selected)

            checkbox = "[x]" if is_sel else "[ ]"
            label = row["label"]

            if is_cursor and is_sel:
                lines.append(HTML(f"<cursor>  ▶ <checked>{checkbox} {label}</checked></cursor>\n"))
            elif is_cursor:
                lines.append(HTML(f"<cursor>  ▶ {checkbox} {label}</cursor>\n"))
            elif is_sel:
                lines.append(HTML(f"    <checked>{checkbox} {label}</checked>\n"))
            else:
                lines.append(HTML(f"    {checkbox} {label}\n"))

        sel_count = len(selected)
        lines.append(HTML(f"\n<hint>  {sel_count} checkpoint(s) selected</hint>\n"))
        return lines

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("k")
    def move_up(event):
        cursor[0] = (cursor[0] - 1) % n
        event.app.invalidate()

    @kb.add("down")
    @kb.add("j")
    def move_down(event):
        cursor[0] = (cursor[0] + 1) % n
        event.app.invalidate()

    @kb.add("space")
    def toggle(event):
        idx = cursor[0]
        if idx in selected:
            selected.discard(idx)
        else:
            selected.add(idx)
        event.app.invalidate()

    @kb.add("enter")
    def confirm(event):
        event.app.exit()

    @kb.add("q")
    @kb.add("escape")
    @kb.add("c-c")
    def cancel(event):
        cancelled[0] = True
        event.app.exit()

    layout = Layout(Window(content=FormattedTextControl(get_text, focusable=True)))
    app = Application(layout=layout, key_bindings=kb, style=style, full_screen=False)
    app.run()

    if cancelled[0]:
        return []

    return sorted(rows[i]["step"] for i in selected)


# ── Checkpoint sub-menu ───────────────────────────────────────────────────────

def interactive_menu(ckpt_dir: Path) -> tuple[str, list[int]]:
    """
    Checkpoint upload mode selection.
    Returns (mode, steps):
      mode: 'latest' | 'best' | 'choose' | 'list' | 'quit'
    """
    meta_files = sorted(ckpt_dir.glob("meta_*.json"))
    steps_info = ""
    if meta_files:
        steps = []
        for m in meta_files:
            try:
                steps.append(int(m.stem.split("_")[1]))
            except Exception:
                pass
        if steps:
            steps_info = f"{len(steps)} checkpoints  ({min(steps):,} – {max(steps):,})"

    idx = menu(
        "Upload Checkpoint",
        ["Latest", "Best val_bpb", "Choose (multi-select)", "List all", "Cancel"],
        [
            "highest step number",
            "lowest val_bpb score",
            "manually pick one or more steps",
            "show table, no upload",
            "",
        ],
        subtitle=f"{ckpt_dir}  {steps_info}",
    )
    modes = ["latest", "best", "choose", "list", "quit"]
    return (modes[idx] if idx != -1 else "quit"), []


# ── Upload satu step ──────────────────────────────────────────────────────────

def upload_step(api, step: int, ckpt_dir: Path, repo: str, depth: str, source: str, model_only: bool):
    """Upload all files for one step. Returns (uploaded, total).

    Files are stored in the repo under:
      <source>/<depth>/model_XXXXXX.pt   (e.g. sft/d24/model_001000.pt)
    This keeps base, sft, and rl checkpoints cleanly separated.
    """
    step_str = f"{step:06d}"
    files = [f"model_{step_str}.pt", f"meta_{step_str}.json"]
    if not model_only:
        files.append(f"optim_{step_str}_rank0.pt")

    # Repo path prefix: e.g. "sft/d24/" or "base/d24/"
    repo_prefix = f"{source}/{depth}"

    uploaded = 0
    for filename in files:
        filepath = ckpt_dir / filename
        if not filepath.exists():
            print(f"  SKIP: {filename} not found")
            continue
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  Uploading {filename} ({size_mb:.1f} MB) → {repo_prefix}/{filename}")
        api.upload_file(
            path_or_fileobj=str(filepath),
            path_in_repo=f"{repo_prefix}/{filename}",
            repo_id=repo,
            repo_type="model",
        )
        uploaded += 1
        print(f"  ✓ {filename} uploaded")

    return uploaded, len(files)


# ── Tokenizer upload ──────────────────────────────────────────────────────────

def upload_tokenizer(api, base_dir: str, repo: str) -> None:
    """Upload tokenizer.pkl + token_bytes.pt to <repo>/ (Dedicated Tokenizer Repo)."""
    tokenizer_dir = Path(base_dir) / "tokenizer"
    if not tokenizer_dir.exists():
        err(f"ERROR: Tokenizer directory not found: {tokenizer_dir}")
        err("  Run scripts/train/tok_train.py first to generate the tokenizer.")
        return

    info(f"\nTokenizer dir: {tokenizer_dir}")
    info(f"Repo:          {repo}/\n")

    uploaded = 0
    for filename in TOKENIZER_FILES:
        filepath = tokenizer_dir / filename
        if not filepath.exists():
            warn(f"  SKIP: {filename} not found in {tokenizer_dir}")
            continue
        size_mb = filepath.stat().st_size / (1024 * 1024)
        info(f"  Uploading {filename} ({size_mb:.2f} MB) → {filename}")
        api.upload_file(
            path_or_fileobj=str(filepath),
            path_in_repo=f"{filename}",
            repo_id=repo,
            repo_type="model",
        )
        uploaded += 1
        success(f"  ✓ {filename} uploaded")

    if uploaded == 0:
        err(f"\nNo tokenizer files found in {tokenizer_dir}.")
    else:
        success(f"\nDone! {uploaded}/{len(TOKENIZER_FILES)} tokenizer file(s) uploaded to {repo}/")


# ── Dataset upload ────────────────────────────────────────────────────────────

# Dataset directories produced locally (relative to mesosfer base_dir).
# prepare_data.py  → base_data_cybersecurity/  (default --output-dir)
# dataset.py       → base_data_climbmix/       (ClimbMix parquet shards)
DATASET_DIRS = [
    "base_data_cybersecurity",   # output of scripts/data/prepare_data.py
    "base_data_climbmix",        # output of mesosfer/data/dataset.py
]


def _list_local_parquets(data_dir: Path) -> list[Path]:
    """Return sorted list of .parquet files in data_dir (no .tmp files)."""
    if not data_dir.exists():
        return []
    return sorted(p for p in data_dir.glob("*.parquet") if not p.name.endswith(".tmp"))


def upload_dataset(api, base_dir: str, repo: str, dataset_names: list[str] | None = None) -> None:
    """
    Upload parquet shards from one or more local dataset directories to HF Hub.

    Repo layout:  dataset/<dir-name>/shard_XXXXX.parquet

    By default uploads from both:
      - base_data_cybersecurity/  (prepare_data.py output)
      - base_data_climbmix/       (dataset.py / ClimbMix shards)

    Pass dataset_names to restrict to specific dirs.
    Files already present in the repo are skipped (idempotent).
    """
    dirs_to_upload = dataset_names if dataset_names else DATASET_DIRS

    # Pre-flight: collect existing files in repo to enable idempotent skip
    with spinner("Fetching existing files in repo…"):
        try:
            existing_in_repo: set[str] = set(api.list_repo_files(repo_id=repo, repo_type="model"))
        except Exception as e:
            warn(f"  Could not list repo files ({e}). Will attempt all uploads.")
            existing_in_repo = set()

    grand_uploaded = grand_skipped = grand_missing = 0

    for dir_name in dirs_to_upload:
        data_dir = Path(base_dir) / dir_name
        parquets = _list_local_parquets(data_dir)

        if not parquets:
            warn(f"\n  [{dir_name}] No parquet files found — skipping.")
            warn(f"    Expected path: {data_dir}")
            if dir_name == "base_data_cybersecurity":
                warn("    Run: python -m scripts.data.prepare_data")
            elif dir_name == "base_data_climbmix":
                warn("    Run: python -m mesosfer.data.dataset -n 170")
            grand_missing += 1
            continue

        repo_prefix = f"{dir_name}"
        info(f"\n  [{dir_name}]  {len(parquets)} shard(s) → {repo}/{dir_name}/")

        pending = [
            p for p in parquets
            if f"{repo_prefix}/{p.name}" not in existing_in_repo
            and f"dataset/{repo_prefix}/{p.name}" not in existing_in_repo
        ]
        skipped = len(parquets) - len(pending)
        if skipped:
            info(f"    {skipped} shard(s) already in repo — skipping")

        # One upload_file call is one commit, and the Hub caps commits at 128/hour, so a
        # per-file loop dies partway through any real dataset. upload_folder preuploads
        # the LFS blobs and then makes a single commit for the whole batch, so shard
        # count stops mattering. A failed run re-uses the already-uploaded blobs.
        uploaded = 0
        if pending:
            total_mb = sum(p.stat().st_size for p in pending) / (1024 * 1024)
            info(f"    Uploading {len(pending)} shard(s), {total_mb:.0f} MB, in one commit…")
            api.upload_folder(
                folder_path=str(data_dir),
                path_in_repo=repo_prefix,
                repo_id=repo,
                repo_type="model",
                allow_patterns=[p.name for p in pending],
                commit_message=f"Add {len(pending)} {dir_name} shards",
            )
            uploaded = len(pending)

        success(f"  [{dir_name}] Done: {uploaded} uploaded, {skipped} skipped")
        grand_uploaded += uploaded
        grand_skipped += skipped

    print()
    if grand_missing == len(dirs_to_upload):
        err("ERROR: No dataset directories found locally. Nothing was uploaded.")
    else:
        success(
            f"Dataset upload complete: {grand_uploaded} uploaded, "
            f"{grand_skipped} skipped (already in repo)"
        )
        success(f"Repo: {repo}/")


# ── Interactive model option prompts ─────────────────────────────────────────

def _prompt_model_options(args: argparse.Namespace) -> None:
    """Ask source and depth interactively when user picks 'model' from top menu."""
    section("Checkpoint source")
    idx = menu(
        "Select source",
        ["base  — pretraining", "sft   — chat SFT", "rl    — RL/GRPO"],
        [
            "~/.cache/mesosfer/base_checkpoints/",
            "~/.cache/mesosfer/chatsft_checkpoints/",
            "~/.cache/mesosfer/chatrl_checkpoints/",
        ],
    )
    if idx == -1:
        return
    args.source = ["base", "sft", "rl"][idx]

    try:
        depth_input = input(f"  Depth tag [{args.depth}]: ").strip()
    except (KeyboardInterrupt, EOFError):
        depth_input = ""
    if depth_input:
        args.depth = depth_input

    args.model_only = confirm("Skip optimizer state? (model + meta only)", default=False)


# ── Top-level interactive menu ────────────────────────────────────────────────

def top_level_menu(base_dir: str) -> str:
    """Show the top-level artifact selection menu. Returns artifact key or 'exit'."""
    idx = menu(
        "Upload Mesosfer Artifacts to HuggingFace",
        [
            "Model checkpoint",
            "Tokenizer",
            "Dataset (parquet shards)",
            "Exit",
        ],
        [
            "model weights + optimizer + meta.json",
            "tokenizer.pkl + token_bytes.pt",
            "base_data_cybersecurity + base_data_climbmix",
            "",
        ],
        subtitle=f"Cache dir: {base_dir}",
    )
    return ["model", "tokenizer", "dataset", "exit"][idx] if idx != -1 else "exit"


# ── HF login helper ───────────────────────────────────────────────────────────

def _hf_login(repo: str, private: bool = True) -> object | None:
    """Import HfApi, verify login, ensure repo exists. Returns api or None."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        err("ERROR: huggingface_hub is not installed. Run: pip install huggingface_hub")
        return None

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)
    try:
        user = api.whoami()
        success(f"Logged in as: {user['name']}")
    except Exception as e:
        err(f"ERROR: Could not log in to HuggingFace — {e}")
        err("Run: hf auth login")
        return None

    repo_type = "dataset" if repo.endswith("/dataset") else "model"
    api.create_repo(repo_id=repo, repo_type=repo_type, private=private, exist_ok=True)
    info(f"Repo: {repo}\n")
    return api


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Upload mesosfer artifacts (model / tokenizer / dataset) to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Artifact selector (optional — omit for interactive top-level menu)
    parser.add_argument(
        "--artifact", choices=["model", "tokenizer", "dataset"], default=None,
        help="Which artifact to upload. Omit for interactive menu.",
    )
    # Common
    parser.add_argument("--repo", type=str, default=None,
                        help="HuggingFace repo ID (default: HF_REPO env var, e.g. johndoe/mesosfer-checkpoints)")
    parser.add_argument("--base-dir", type=str, default=None,
                        help="Override mesosfer cache dir (default: ~/.cache/mesosfer)")
    parser.add_argument("--public", action="store_true",
                        help="Create the repo public if it does not exist yet (default: private). "
                             "Never changes an existing repo's visibility.")

    parser.add_argument("--public", action="store_true", default=False,
                        help="Make the HuggingFace repository public")
    parser.add_argument("--private", action="store_true", default=False,
                        help="Make the HuggingFace repository private (default)")

    # Model checkpoint flags (backward-compatible)
    g = parser.add_argument_group("model checkpoint")
    sel = g.add_mutually_exclusive_group()
    sel.add_argument("--step", type=int, help="Upload a specific step (e.g. 8000)")
    sel.add_argument("--best", action="store_true",
                     help="Upload the checkpoint with the best val_bpb")
    sel.add_argument("--latest", action="store_true",
                     help="Upload the latest checkpoint (highest step)")
    sel.add_argument("--list", action="store_true",
                     help="List local checkpoints and exit")
    g.add_argument("--depth", type=str, default="d32",
                   help="Model depth tag (default: d32)")
    g.add_argument("--source", type=str, default="base", choices=["base", "sft", "rl"],
                   help="Checkpoint source: base | sft | rl (default: base)")
    g.add_argument("--model-only", action="store_true",
                   help="Upload model + meta only (skip optimizer state)")

    # Dataset flags
    g = parser.add_argument_group("dataset")
    g.add_argument(
        "--dataset-dirs", nargs="*", default=None,
        metavar="DIR_NAME",
        help=(
            "Dataset dir name(s) under mesosfer cache to upload "
            "(default: base_data_cybersecurity base_data_climbmix). "
            "Example: --dataset-dirs base_data_cybersecurity"
        ),
    )

    args = parser.parse_args()

    base_dir = args.base_dir or os.path.expanduser("~/.cache/mesosfer")

    # Resolve HF username
    username = _resolve_username(args.repo)

    # ── Determine artifact via top-level menu if not given by flag ────────────
    model_flags_given = bool(args.step or args.best or args.latest or args.list)

    if args.artifact is None and not model_flags_given:
        # No flags at all → show top-level menu
        artifact = top_level_menu(base_dir)
        if artifact == "exit":
            info("\nGoodbye.")
            return
        args.artifact = artifact
        # For model: ask source + depth interactively before touching the filesystem
        if artifact == "model":
            _prompt_model_options(args)
    elif args.artifact is None:
        # Legacy: model flags given without --artifact → default to model
        args.artifact = "model"

    # Hardcode repo path depending on artifact type
    if args.artifact == "model":
        args.repo = f"{username}/model"
    elif args.artifact == "tokenizer":
        args.repo = f"{username}/tokenizer"
    elif args.artifact == "dataset":
        args.repo = f"{username}/dataset"

    # ── HF login (shared for all artifacts) ──────────────────────────────────
    is_private = False if args.public else True
    api = _hf_login(args.repo, private=is_private)
    if api is None:
        return

    # ── Dispatch ─────────────────────────────────────────────────────────────
    if args.artifact == "tokenizer":
        upload_tokenizer(api, base_dir, args.repo)
        return

    if args.artifact == "dataset":
        upload_dataset(api, base_dir, args.repo, args.dataset_dirs)
        return

    # ── artifact == "model" ───────────────────────────────────────────────────
    source_dir_map = {
        "base": "base_checkpoints",
        "sft":  "chatsft_checkpoints",
        "rl":   "chatrl_checkpoints",
    }
    ckpt_subdir = source_dir_map[args.source]
    ckpt_dir = Path(base_dir) / ckpt_subdir / args.depth

    if not ckpt_dir.exists():
        err(f"ERROR: Checkpoint directory not found: {ckpt_dir}")
        err(f"  source={args.source!r} maps to: {ckpt_subdir}/{args.depth}/")
        if args.source == "sft":
            err("  Make sure SFT training has completed and saved a checkpoint.")
        elif args.source == "rl":
            err("  Make sure RL training has completed and saved a checkpoint.")
        return

    # ── Interactive checkpoint sub-menu (no model flags given) ───────────────
    chosen_steps: list[int] = []
    if not model_flags_given:
        info(f"\n  Source: {args.source} ({ckpt_dir})")
        mode, _ = interactive_menu(ckpt_dir)
        if mode == "quit":
            return
        elif mode == "list":
            info(f"\nCheckpoint dir: {ckpt_dir}")
            list_checkpoints(ckpt_dir)
            return
        elif mode == "latest":
            args.latest = True
        elif mode == "best":
            args.best = True
        elif mode == "choose":
            rows = load_all_checkpoints(ckpt_dir)
            if not rows:
                err("No checkpoints found.")
                return
            info("\n  Use SPACE to select, ENTER to confirm:\n")
            chosen_steps = checkbox_select(rows)
            if not chosen_steps:
                warn("\nNo checkpoints selected. Cancelled.")
                return
            info(f"\n  Selected: {len(chosen_steps)} checkpoint(s) — {chosen_steps}")

    # Mode --list (CLI flag)
    if args.list:
        info(f"Checkpoint dir: {ckpt_dir}")
        list_checkpoints(ckpt_dir)
        return

    # Determine steps to upload
    steps_to_upload: list[int] = []
    if chosen_steps:
        steps_to_upload = chosen_steps
        for s in steps_to_upload:
            meta_path = ckpt_dir / f"meta_{s:06d}.json"
            val_bpb: float | str = "N/A"
            if meta_path.exists():
                try:
                    with open(meta_path, encoding="utf-8") as f:
                        val_bpb = json.load(f).get("val_bpb", "N/A")
                except Exception:
                    pass
            bpb_str = f"{val_bpb:.6f}" if isinstance(val_bpb, float) else str(val_bpb)
            info(f"  • step {s:,}  (val_bpb={bpb_str})")
    elif args.best:
        step, best_bpb = find_best_checkpoint(ckpt_dir)
        info(f"\nBest checkpoint: step {step} (val_bpb={best_bpb:.6f})")
        steps_to_upload = [step]
    elif args.latest:
        step, val_bpb = find_latest_checkpoint(ckpt_dir)
        bpb_info = f"val_bpb={val_bpb:.6f}" if isinstance(val_bpb, float) else "val_bpb=N/A"
        info(f"\nLatest checkpoint: step {step} ({bpb_info})")
        steps_to_upload = [step]
    elif args.step:
        step = args.step
        meta_path = ckpt_dir / f"meta_{step:06d}.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                val_bpb = json.load(f).get("val_bpb", "N/A")
            info(f"Uploading step {step} (val_bpb={val_bpb})")
        else:
            info(f"Uploading step {step} (meta not found)")
        steps_to_upload = [step]
    else:
        parser.print_help()
        return

    # Upload
    grand_uploaded = grand_total = 0
    for i, step in enumerate(steps_to_upload, 1):
        if len(steps_to_upload) > 1:
            info(f"── [{i}/{len(steps_to_upload)}] step {step:,} ──")
        uploaded, total = upload_step(api, step, ckpt_dir, args.repo, args.depth, args.source, args.model_only)
        grand_uploaded += uploaded
        grand_total += total
        if len(steps_to_upload) > 1:
            print()

    success(f"Done! {grand_uploaded}/{grand_total} file(s) uploaded to {args.repo}/{args.source}/{args.depth}/")


if __name__ == "__main__":
    main()
