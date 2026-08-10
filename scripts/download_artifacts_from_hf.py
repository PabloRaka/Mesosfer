#!/usr/bin/env python3
"""
Download mesosfer artifacts (model checkpoints, tokenizer, dataset shards) from
HuggingFace Hub into the local mesosfer cache directory.

Path layout is IDENTICAL to upload_checkpoint_to_hf.py — no path changes needed:

    HF Repo                                    Local (~/.cache/mesosfer/)
    ─────────────────────────────────────────  ──────────────────────────────────────────
    base/d24/model_XXXXXX.pt              →    base_checkpoints/d24/model_XXXXXX.pt
    sft/d24/model_XXXXXX.pt               →    chatsft_checkpoints/d24/model_XXXXXX.pt
    rl/d24/model_XXXXXX.pt                →    chatrl_checkpoints/d24/model_XXXXXX.pt
    tokenizer/tokenizer.pkl               →    tokenizer/tokenizer.pkl
    tokenizer/token_bytes.pt              →    tokenizer/token_bytes.pt
    dataset/base_data_cybersecurity/*.parquet → base_data_cybersecurity/*.parquet
    dataset/base_data_climbmix/*.parquet  →    base_data_climbmix/*.parquet

Usage examples
--------------
    # Interactive
    python scripts/download_artifacts_from_hf.py

    # Direct: latest base checkpoint (model + optimizer + meta)
    python scripts/download_artifacts_from_hf.py --artifact model --latest

    # Direct: best SFT checkpoint, model + meta only (skip optimizer)
    python scripts/download_artifacts_from_hf.py --artifact model --best \\
        --source sft --model-only

    # Direct: tokenizer only
    python scripts/download_artifacts_from_hf.py --artifact tokenizer

    # Direct: all dataset shards (both dirs)
    python scripts/download_artifacts_from_hf.py --artifact dataset

    # Direct: only cybersecurity shards
    python scripts/download_artifacts_from_hf.py --artifact dataset \\
        --dataset-dirs base_data_cybersecurity
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

# ── UI helpers (shared with upload script) ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from _ui import (  # noqa: E402
    box, menu, confirm, section, badge, spinner,
    info, success, warn, err, dim,
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


# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_REPO = None  # resolved at runtime via _resolve_username()
DEFAULT_DEPTH = "d32"

SOURCE_DIR_MAP = {
    "base": "base_checkpoints",
    "sft":  "chatsft_checkpoints",
    "rl":   "chatrl_checkpoints",
}

TOKENIZER_FILES = ("tokenizer.pkl", "token_bytes.pt")

# Dataset dirs — MUST match DATASET_DIRS in upload_checkpoint_to_hf.py exactly.
# Upload stores:   dataset/<dir_name>/shard_XXXXX.parquet  (in HF repo)
# Download reads:  dataset/<dir_name>/shard_XXXXX.parquet  → ~/.cache/mesosfer/<dir_name>/
DATASET_DIRS = [
    "base_data_cybersecurity",   # output of scripts/data/prepare_data.py
    "base_data_climbmix",        # output of mesosfer/data/dataset.py (ClimbMix)
]

# Tokenizer + dataset constants are imported lazily so that --artifact model
# does not require pyarrow / requests / mesosfer to be importable.


# ── Cache resolution (mirrors mesosfer.utils.common.get_base_dir) ────────────

def get_base_dir(override: str | None = None) -> Path:
    if override:
        p = Path(os.path.expanduser(override))
    elif os.environ.get("mesosfer_BASE_DIR"):
        p = Path(os.path.expanduser(os.environ["mesosfer_BASE_DIR"]))
    else:
        p = Path.home() / ".cache" / "mesosfer"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── HF API helpers ───────────────────────────────────────────────────────────

def _require_hf_hub():
    try:
        from huggingface_hub import HfApi, hf_hub_download  # noqa: F401
    except ImportError:
        err("ERROR: huggingface_hub is not installed. Run: pip install huggingface_hub")
        sys.exit(1)
    from huggingface_hub import HfApi, hf_hub_download
    return HfApi, hf_hub_download


def _place_downloaded(cached: str, local_path) -> None:
    """Move a freshly downloaded file out of the HF cache into the mesosfer layout.

    hf_hub_download returns a path inside the snapshot dir, which is a symlink whose
    target is *relative* (../../blobs/<sha>). os.replace on that path moves the link
    itself, so the target only still resolves if the destination happens to sit at the
    same directory depth — which is why the dataset survived and the tokenizer did not.
    Move the blob it points at instead, and drop the now-dangling snapshot link.
    """
    import shutil

    real = os.path.realpath(cached)
    try:
        os.replace(real, local_path)
    except OSError:
        shutil.copy2(real, local_path)  # cross-device
    if os.path.islink(cached):
        os.unlink(cached)


def _whoami(api) -> str | None:
    try:
        return api.whoami().get("name")
    except Exception:
        return None  # tokenizer/dataset usually live in public repos


# ── 1. MODEL CHECKPOINT DOWNLOAD ─────────────────────────────────────────────

def list_remote_checkpoints(api, repo: str, source: str, depth: str) -> tuple[list[dict], str]:
    """
    Return (rows, prefix)
    discovered under <source>/<depth>/ or <depth>/ in the HF repo, sorted by step.
    """
    try:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        files = api.list_repo_files(repo_id=repo, repo_type="model", token=token)
    except Exception as e:
        err(f"ERROR: cannot list files in {repo}: {e}")
        return [], ""

    # Check if files exist with the source prefix (e.g., base/d24/)
    source_prefix = f"{source}/{depth}/"
    has_source_prefix = any(f.startswith(source_prefix) for f in files)
    
    prefix = source_prefix if has_source_prefix else f"{depth}/"

    by_step: dict[int, dict] = {}
    for path in files:
        if not path.startswith(prefix):
            continue
        name = path[len(prefix):]
        # Expected stems: model_XXXXXX.pt | meta_XXXXXX.json | optim_XXXXXX_rank0.pt
        try:
            stem = name.split(".", 1)[0]
            parts = stem.split("_")
            step = int(parts[1])
        except (IndexError, ValueError):
            continue
        by_step.setdefault(step, {"step": step, "val_bpb": None, "files": set()})
        by_step[step]["files"].add(name)

    rows = sorted(by_step.values(), key=lambda r: r["step"])
    if not rows:
        return rows, prefix

    # Fetch val_bpb for each meta file (cheap: small JSON) — best effort
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    from huggingface_hub import hf_hub_download
    for row in rows:
        meta_name = f"meta_{row['step']:06d}.json"
        if meta_name not in row["files"]:
            continue
        try:
            local = hf_hub_download(
                repo_id=repo, repo_type="model",
                filename=f"{prefix}{meta_name}",
                token=token,
            )
            with open(local, encoding="utf-8") as f:
                row["val_bpb"] = json.load(f).get("val_bpb")
        except Exception:
            pass

    # Build display label with BEST tag
    best_bpb = float("inf")
    for r in rows:
        if isinstance(r["val_bpb"], float) and r["val_bpb"] < best_bpb:
            best_bpb = r["val_bpb"]
    for r in rows:
        bpb = r["val_bpb"]
        bpb_str = f"{bpb:.6f}" if isinstance(bpb, float) else "N/A     "
        tag = " ← BEST" if isinstance(bpb, float) and bpb == best_bpb else ""
        r["label"] = f"step {r['step']:>7,}   val_bpb={bpb_str}{tag}"
    return rows, prefix


def _download_one_step(api, hf_hub_download, repo: str, prefix: str,
                       step: int, out_dir: Path, model_only: bool, available: set[str]) -> tuple[int, int]:
    """Download a single step's files into out_dir. Returns (downloaded, total)."""
    step_str = f"{step:06d}"
    targets = [f"model_{step_str}.pt", f"meta_{step_str}.json"]
    if not model_only:
        targets.append(f"optim_{step_str}_rank0.pt")

    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for name in targets:
        if name not in available:
            warn(f"  SKIP: {name} not present in repo")
            continue
        local_path = out_dir / name
        if local_path.exists():
            info(f"  ✓ {name} already present (skipping)")
            downloaded += 1
            continue
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        cached = hf_hub_download(
            repo_id=repo, repo_type="model",
            filename=f"{prefix}{name}",
            token=token,
        )
        _place_downloaded(cached, local_path)
        success(f"  ✓ {name} → {local_path}")
        downloaded += 1
    return downloaded, len(targets)


def download_model(args, base_dir: Path) -> None:
    HfApi, hf_hub_download = _require_hf_hub()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)

    user = _whoami(api)
    if user:
        info(f"Logged in as: {user}")
    else:
        warn("Not logged in to HuggingFace. Public repos work; private ones will fail.")
        warn("  → Run: hf auth login")

    out_dir = base_dir / SOURCE_DIR_MAP[args.source] / args.depth
    info(f"\nRepo:        {args.repo}")
    
    rows, prefix = list_remote_checkpoints(api, args.repo, args.source, args.depth)
    if not rows:
        err(f"ERROR: no checkpoints found at {args.repo} with prefix {prefix}")
        return

    info(f"Remote path: {prefix}")
    info(f"Local dir:   {out_dir}")

    # Decide which steps to fetch
    steps: list[int] = []

    if args.list:
        print(f"\n{'Step':<10} {'val_bpb':<14} {'Status'}")
        print("-" * 38)
        for r in rows:
            bpb = r["val_bpb"]
            bpb_str = f"{bpb:.6f}" if isinstance(bpb, float) else "N/A"
            tag = "← BEST" if "← BEST" in r["label"] else ""
            print(f"{r['step']:<10} {bpb_str:<14} {tag}")
        print()
        return

    if args.latest:
        steps = [rows[-1]["step"]]
    elif args.best:
        valid = [r for r in rows if isinstance(r["val_bpb"], float)]
        if not valid:
            err("ERROR: no checkpoints with a val_bpb value found.")
            return
        best = min(valid, key=lambda r: r["val_bpb"])
        steps = [best["step"]]
    elif args.step is not None:
        steps = [args.step]
    else:
        # Sub-menu (interactive)
        steps = _checkpoint_submenu(rows)
        if not steps:
            warn("Cancelled.")
            return

    info(f"\nSelected {len(steps)} step(s): {steps}")
    if args.model_only:
        warn("Skipping optimizer state (--model-only)")

    grand_dl, grand_total = 0, 0
    for i, step in enumerate(steps, 1):
        if len(steps) > 1:
            print(f"\n── [{i}/{len(steps)}] step {step:,} ──")
        avail = next((r["files"] for r in rows if r["step"] == step), set())
        if not avail:
            err(f"  step {step} not found in repo, skipping")
            continue
        d, t = _download_one_step(
            api, hf_hub_download, args.repo, prefix,
            step, out_dir, args.model_only, avail,
        )
        grand_dl += d
        grand_total += t

    success(f"\nDone! {grand_dl}/{grand_total} file(s) downloaded to {out_dir}")


def _checkpoint_submenu(rows: list[dict]) -> list[int]:
    """Interactive sub-menu: latest / best / pick / cancel."""
    idx = menu(
        "Select checkpoint",
        ["Latest", "Best val_bpb", "Choose (multi-select)", "Cancel"],
        [
            "highest step number",
            "lowest val_bpb score",
            "manually pick one or more steps",
            "",
        ],
    )
    if idx == -1 or idx == 3:
        return []
    if idx == 0:
        return [rows[-1]["step"]]
    if idx == 1:
        valid = [r for r in rows if isinstance(r["val_bpb"], float)]
        if not valid:
            err("  No checkpoints with a val_bpb value found.")
            return []
        return [min(valid, key=lambda r: r["val_bpb"])["step"]]
    # idx == 2: choose
    return _checkbox_select(rows)


def _checkbox_select(rows: list[dict]) -> list[int]:
    """Multi-select checkpoints. Falls back to numeric input if prompt_toolkit
    is missing or stdout is not a TTY (graceful degradation)."""
    if not sys.stdout.isatty():
        return _numeric_multiselect(rows)
    try:
        from prompt_toolkit import Application
        from prompt_toolkit.formatted_text import HTML
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import Layout
        from prompt_toolkit.layout.containers import Window
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.styles import Style
    except ImportError:
        warn("  prompt_toolkit not installed — falling back to numeric input.")
        return _numeric_multiselect(rows)

    n = len(rows)
    cursor = [0]
    selected: set[int] = set()
    cancelled = [False]

    style = Style.from_dict({
        "cursor":   "bold ansicyan",
        "checked":  "bold ansigreen",
        "hint":     "ansibrightblack",
    })

    def get_text():
        lines = [HTML("<hint>  ↑/↓ navigate   SPACE toggle   ENTER confirm   q cancel\n\n</hint>")]
        for i, r in enumerate(rows):
            cb = "[x]" if i in selected else "[ ]"
            lbl = r["label"]
            if i == cursor[0] and i in selected:
                lines.append(HTML(f"<cursor>  ▶ <checked>{cb} {lbl}</checked></cursor>\n"))
            elif i == cursor[0]:
                lines.append(HTML(f"<cursor>  ▶ {cb} {lbl}</cursor>\n"))
            elif i in selected:
                lines.append(HTML(f"    <checked>{cb} {lbl}</checked>\n"))
            else:
                lines.append(HTML(f"    {cb} {lbl}\n"))
        lines.append(HTML(f"\n<hint>  {len(selected)} checkpoint(s) selected</hint>\n"))
        return lines

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("k")
    def _(e):
        cursor[0] = (cursor[0] - 1) % n
        e.app.invalidate()

    @kb.add("down")
    @kb.add("j")
    def _(e):
        cursor[0] = (cursor[0] + 1) % n
        e.app.invalidate()

    @kb.add("space")
    def _(e):
        idx = cursor[0]
        selected.discard(idx) if idx in selected else selected.add(idx)
        e.app.invalidate()

    @kb.add("enter")
    def _(e): e.app.exit()

    @kb.add("q")
    @kb.add("escape")
    @kb.add("c-c")
    def _(e):
        cancelled[0] = True
        e.app.exit()

    Application(
        layout=Layout(Window(content=FormattedTextControl(get_text, focusable=True))),
        key_bindings=kb, style=style, full_screen=False,
    ).run()

    if cancelled[0]:
        return []
    return sorted(rows[i]["step"] for i in selected)


def _numeric_multiselect(rows: list[dict]) -> list[int]:
    print()
    for i, r in enumerate(rows, 1):
        print(f"    {i:>3}. {r['label']}")
    print()
    try:
        raw = input("  Enter indices (comma-separated, e.g. 1,3,5) or 'all': ").strip()
    except (KeyboardInterrupt, EOFError):
        return []
    if not raw:
        return []
    if raw.lower() == "all":
        return [r["step"] for r in rows]
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part.isdigit():
            continue
        idx = int(part) - 1
        if 0 <= idx < len(rows):
            out.append(rows[idx]["step"])
    return sorted(set(out))


# ── 2. TOKENIZER DOWNLOAD ────────────────────────────────────────────────────

def download_tokenizer(args, base_dir: Path) -> None:
    HfApi, hf_hub_download = _require_hf_hub()
    out_dir = base_dir / "tokenizer"
    out_dir.mkdir(parents=True, exist_ok=True)

    repo = args.repo
    info(f"\nRepo:      {repo}/")
    info(f"Local dir: {out_dir}\n")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    try:
        api = HfApi(token=token)
        files = set(api.list_repo_files(repo, repo_type="model", token=token))
    except Exception:
        files = set()

    downloaded = 0
    for name in TOKENIZER_FILES:
        local_path = out_dir / name
        if local_path.exists() and not args.force:
            info(f"  ✓ {name} already present (use --force to overwrite)")
            downloaded += 1
            continue
        info(f"  Downloading {name} …")
        
        if name in files:
            repo_path = name
        elif f"tokenizer/{name}" in files:
            repo_path = f"tokenizer/{name}"
        else:
            repo_path = name

        try:
            cached = hf_hub_download(
                repo_id=repo,
                repo_type="model",
                filename=repo_path,
                token=token,
            )
        except Exception as e:
            err(f"  ✗ {name}: {e}")
            continue
        _place_downloaded(cached, local_path)
        success(f"  ✓ {name} → {local_path}")
        downloaded += 1

    if downloaded == 0:
        err("\nNo tokenizer files were downloaded.")
    else:
        success(f"\nDone! {downloaded}/{len(TOKENIZER_FILES)} tokenizer files in {out_dir}")


# ── 3. DATASET (PARQUET) DOWNLOAD ────────────────────────────────────────────

def download_dataset(args, base_dir: Path) -> None:
    """
    Download parquet shards from HF repo back to local cache.

    Mirrors upload_checkpoint_to_hf.py exactly:
      HF repo:    dataset/<dir_name>/shard_XXXXX.parquet
      Local path: ~/.cache/mesosfer/<dir_name>/shard_XXXXX.parquet

    Supports both dataset dirs:
      - base_data_cybersecurity/  (prepare_data.py output)
      - base_data_climbmix/       (dataset.py / ClimbMix shards)

    Files already present locally are skipped (idempotent).
    """
    HfApi, hf_hub_download = _require_hf_hub()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)

    repo = getattr(args, "repo", DEFAULT_REPO)
    dirs_to_download = getattr(args, "dataset_dirs", None) or DATASET_DIRS

    # List all files in repo once (cheap, avoids per-file HEAD requests)
    info(f"\nRepo: {repo}")
    with spinner("Fetching file list from repo…"):
        try:
            all_repo_files: list[str] = list(api.list_repo_files(repo_id=repo, repo_type="model", token=token))
        except Exception as e:
            err(f"ERROR: cannot list files in {repo}: {e}")
            err("  Make sure you are logged in or HF_TOKEN is correct in .env")
            return

    grand_dl = grand_skip = grand_missing = 0

    for dir_name in dirs_to_download:
        repo_prefix = f"{dir_name}/"
        # Filter remote shards (handle both <dir_name>/ and dataset/<dir_name>/)
        remote_shards = sorted(
            f for f in all_repo_files
            if (f.startswith(repo_prefix) or f.startswith(f"dataset/{repo_prefix}")) and f.endswith(".parquet")
        )

        if not remote_shards:
            warn(f"\n  [{dir_name}] No parquet shards found in repo under {repo_prefix}")
            warn(f"    Upload first: python scripts/upload_checkpoint_to_hf.py --artifact dataset")
            grand_missing += 1
            continue

        out_dir = base_dir / dir_name
        out_dir.mkdir(parents=True, exist_ok=True)

        info(f"\n  [{dir_name}]  {len(remote_shards)} shard(s) → {out_dir}")

        downloaded = skipped = 0
        for i, repo_path in enumerate(remote_shards, 1):
            if repo_path.startswith("dataset/"):
                filename = repo_path[len(f"dataset/{repo_prefix}"):]
            else:
                filename = repo_path[len(repo_prefix):]
            local_path = out_dir / filename

            if local_path.exists() and not getattr(args, "force", False):
                info(f"    [{i}/{len(remote_shards)}] SKIP {filename} (already exists locally)")
                skipped += 1
                continue

            info(f"    [{i}/{len(remote_shards)}] Downloading {filename}…")
            try:
                cached = hf_hub_download(
                    repo_id=repo,
                    repo_type="model",
                    filename=repo_path,
                    token=token,
                )
            except Exception as e:
                err(f"    ✗ {filename}: {e}")
                continue

            _place_downloaded(cached, local_path)
            success(f"    ✓ {filename} → {local_path}")
            downloaded += 1

        success(f"  [{dir_name}] Done: {downloaded} downloaded, {skipped} skipped")
        grand_dl += downloaded
        grand_skip += skipped

    print()
    if grand_missing == len(dirs_to_download):
        err("ERROR: No dataset shards found in repo. Nothing was downloaded.")
    else:
        success(f"Dataset download complete: {grand_dl} downloaded, {grand_skip} skipped (already local)")


# ── Top-level interactive menu ───────────────────────────────────────────────

def main_menu(base_dir: Path) -> str:
    """Show top-level download menu. Returns artifact key or 'exit'."""
    idx = menu(
        "Download Mesosfer Artifacts from HuggingFace",
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


def prompt_model_options(args: argparse.Namespace) -> None:
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


def prompt_dataset_options(args: argparse.Namespace) -> None:
    print()
    info(f"  Default dirs: {', '.join(DATASET_DIRS)}")
    raw = input("  Download specific dirs? (comma-separated, or Enter for all): ").strip()
    if raw:
        dirs = [d.strip() for d in raw.split(",") if d.strip()]
        if dirs:
            args.dataset_dirs = dirs


# ── Argparse ─────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download mesosfer artifacts from HuggingFace Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--artifact",
        choices=["model", "tokenizer", "dataset"],
        help="Which artifact to download. Omit for interactive menu.",
    )
    # Common
    p.add_argument("--repo", type=str, default=None,
                   help="HF repo ID (default: HF_REPO env var, e.g. johndoe/mesosfer-checkpoints)")
    p.add_argument("--base-dir", type=str, default=None,
                   help="Override the mesosfer cache directory (default: ~/.cache/mesosfer)")
    p.add_argument("--force", action="store_true",
                   help="Re-download files even if they already exist locally.")

    # Model checkpoint
    g = p.add_argument_group("model checkpoint")
    g.add_argument("--source", type=str, default="base", choices=["base", "sft", "rl"],
                   help="Checkpoint source. Default: base")
    g.add_argument("--depth", type=str, default=DEFAULT_DEPTH,
                   help=f"Model depth tag (default: {DEFAULT_DEPTH})")
    sel = g.add_mutually_exclusive_group()
    sel.add_argument("--latest", action="store_true",
                     help="Download the latest checkpoint (highest step)")
    sel.add_argument("--best", action="store_true",
                     help="Download the checkpoint with the best val_bpb")
    sel.add_argument("--step", type=int,
                     help="Download a specific step (e.g. 8000)")
    sel.add_argument("--list", action="store_true",
                     help="List remote checkpoints and exit")
    g.add_argument("--model-only", action="store_true",
                   help="Skip optimizer state (faster, can't resume training)")

    # Dataset — mirrors upload_checkpoint_to_hf.py --dataset-dirs
    g = p.add_argument_group("dataset")
    g.add_argument(
        "--dataset-dirs", nargs="*", default=None,
        metavar="DIR_NAME",
        help=(
            "Dataset dir name(s) to download from repo "
            f"(default: {' '.join(DATASET_DIRS)}). "
            "Example: --dataset-dirs base_data_cybersecurity"
        ),
    )
    return p


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    base_dir = get_base_dir(args.base_dir)

    # Resolve HF username
    username = _resolve_username(args.repo)

    # Pre-execution validation (RULES §5)
    if not os.access(base_dir, os.W_OK):
        err(f"ERROR: cache dir is not writable: {base_dir}")
        return 2

    if args.artifact is None:
        # Interactive entry point
        choice = main_menu(base_dir)
        if choice == "exit":
            dim("\nGoodbye.")
            return 0
        args.artifact = choice
        if choice == "model":
            prompt_model_options(args)

    # Hardcode repo path depending on artifact type
    if args.artifact == "model":
        args.repo = f"{username}/model"
    elif args.artifact == "tokenizer":
        args.repo = f"{username}/tokenizer"
    elif args.artifact == "dataset":
        args.repo = f"{username}/dataset"

    try:
        if args.artifact == "model":
            download_model(args, base_dir)
        elif args.artifact == "tokenizer":
            download_tokenizer(args, base_dir)
        elif args.artifact == "dataset":
            download_dataset(args, base_dir)
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        warn("\nInterrupted by user.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
