"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

The dataloader supports merging multiple parquet directories so cybersecurity
data prepared by `scripts.data.prepare_data` (output to `base_data_cybersecurity/`)
is automatically picked up alongside the ClimbMix general pretraining data
(`base_data_climbmix/`).

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import os
import argparse
import random
import re
import time
import pyarrow.parquet as pq


from mesosfer.utils.common import get_base_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# The URL on the internet where the data is hosted and downloaded from on demand
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542 # the last datashard is shard_06542.parquet
index_to_filename = lambda index: f"shard_{index:05d}.parquet" # format of the filenames
base_dir = get_base_dir()

# Primary dataset: ClimbMix general pretraining data
DATA_DIR = os.path.join(base_dir, "base_data_climbmix")

# Auxiliary dataset directories that are auto-merged if they exist.
# These are produced by other pipelines (e.g. scripts/data/prepare_data.py).
# Order matters: directories listed first are placed earlier in the parquet
# sequence. Validation shards are identified by name (a `val_` prefix, written
# by prepare_data.py's val_shard_path()) via split_parquet_paths(), not by
# position — any directory's val_*.parquet is picked up. Corpora written before
# that convention (no val_* file anywhere) fall back to the last file in this
# merged list, which is typically ClimbMix's shard_06542.parquet.
AUXILIARY_DATA_DIRS = [
    os.path.join(base_dir, "base_data_cybersecurity"),          # output of scripts.data.prepare_data
    os.path.join(base_dir, "base_data_cybersecurity_sample"),   # sample output of --sample-10k
]

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def list_parquet_files(data_dir=None, warn_on_legacy=False, include_auxiliary=True):
    """
    Return all parquet file paths used for training/validation.

    By default this merges the primary ClimbMix directory with any auxiliary
    directories listed in AUXILIARY_DATA_DIRS that exist on disk. Auxiliary
    shards are placed BEFORE primary shards. split_parquet_paths() picks the
    validation file(s) out of this list by name (`val_*`), so directory order
    here no longer determines which shard is validation.

    Args:
        data_dir: if provided, only read from this single directory (no merge)
        warn_on_legacy: print legacy-fallback warning if primary dir is missing
        include_auxiliary: if False, skip the auxiliary dirs even when present

    Returns:
        sorted list of parquet file paths
    """
    # If caller specified an explicit data_dir, honor it exactly (no merging)
    if data_dir is not None:
        return _list_parquet_in_dir(data_dir)

    # Default: merge primary + auxiliary directories
    primary_dir = DATA_DIR
    primary_missing = not os.path.exists(primary_dir)
    if primary_missing:
        # Legacy fallback: ClimbMix dir doesn't exist, try old FinewebEdu dir
        primary_dir = os.path.join(base_dir, "base_data")

    primary_files = _list_parquet_in_dir(primary_dir)

    # Collect auxiliary files
    aux_files = []
    if include_auxiliary:
        for aux_dir in AUXILIARY_DATA_DIRS:
            if os.path.exists(aux_dir):
                found = _list_parquet_in_dir(aux_dir)
                if found:
                    aux_files.extend(found)
                    if warn_on_legacy:  # also acts as the rank-0-train flag
                        print(f"  ✓ Merged auxiliary parquet dir: {aux_dir} ({len(found)} shards)")

    # Only nag about the missing ClimbMix download when there is genuinely nothing to
    # train on. A prepare_data corpus in an auxiliary dir is a complete dataset in its
    # own right — telling that user to download ClimbMix and retrain the tokenizer would
    # dilute their mixture and move the validation shard off their own distribution.
    if primary_missing and warn_on_legacy and not aux_files:
        _print_legacy_warning(DATA_DIR)

    # Auxiliary shards come first, then primary.
    return aux_files + primary_files


def _list_parquet_in_dir(data_dir):
    """List sorted parquet files in a single directory (no merging logic)."""
    if not os.path.exists(data_dir):
        return []
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    return [os.path.join(data_dir, f) for f in parquet_files]


def _print_legacy_warning(missing_dir):
    print()
    print("=" * 80)
    print("  WARNING: DATASET UPGRADE REQUIRED")
    print("=" * 80)
    print()
    print(f"  Could not find: {missing_dir}")
    print()
    print("  mesosfer recently switched from FinewebEdu-100B to ClimbMix-400B.")
    print("  Everyone who does `git pull` as of March 4, 2026 is expected to see this message.")
    print("  To upgrade to the new ClimbMix-400B dataset, run these two commands:")
    print()
    print("    python -m mesosfer.data.dataset -n 170     # download ~170 shards, enough for GPT-2")
    print("    python -m scripts.train.tok_train          # re-train tokenizer on new ClimbMix data")
    print()
    print("  For now, falling back to your old FinewebEdu-100B dataset...")
    print("=" * 80)
    print()

# Seed for the train-shard permutation. Every reader must use the same one, or DDP
# ranks disagree about which shard index means which file.
SHARD_SHUFFLE_SEED = 1337


def split_parquet_paths(parquet_paths, split):
    """Split into train/val.

    Validation files are identified by name: any path whose basename starts with
    `val_` (written by prepare_data.py's val_shard_path()) is validation, and those
    are excluded from train regardless of where they sort in the merged
    aux+primary list. This is what lets a prepare_data corpus's own multi-domain
    validation shard actually get used as validation, instead of being shuffled
    into train while the last ClimbMix shard silently stands in for it.

    If no `val_*` file is present (corpora prepared before this convention existed,
    where the validation shard is indistinguishable from a training shard by name),
    fall back to the legacy behavior exactly: val is the last file, train is
    everything else.

    prepare_data interleaves sources *within* one run, but each `--sources` batch
    appends its own contiguous block of shards. Filename order is therefore download
    order, so reading it as-is trains on one domain at a time (and makes tok_train's
    --max-chars cap see only the earliest batches). Fixed seed so every DDP rank and
    every restart agree on the same permutation.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    val_paths = [p for p in parquet_paths if os.path.basename(p).startswith("val_")]
    if val_paths:
        train_paths = [p for p in parquet_paths if p not in val_paths]
    else:
        val_paths = parquet_paths[-1:]
        train_paths = parquet_paths[:-1]
    if split == "val":
        return val_paths
    train_paths = list(train_paths)  # a copy, safe to shuffle in place
    random.Random(SHARD_SHUFFLE_SEED).shuffle(train_paths)
    return train_paths


# the-stack stores notebooks as raw .ipynb JSON, execution outputs included, and image
# outputs are base64 blobs. code_jupyter is 15.5% of this corpus by characters and most
# of those characters are those blobs. Substring guard first: it is a cheap C-level scan
# that the ~90% of documents which are not notebooks fail immediately.
_NOTEBOOK_MARKER = '"output_type"'
_NOTEBOOK_BLOB_RE = re.compile(r'"(image/\w+|application/pdf)": "[A-Za-z0-9+/=\\n]{200,}"')


def strip_notebook_blobs(text):
    """Replace base64 output payloads in raw notebook JSON with a short placeholder.

    Keeps the notebook's code and markdown, which is the machine-learning and AI content
    code_jupyter was added for, and drops the rendered images that teach nothing.
    """
    if _NOTEBOOK_MARKER not in text:
        return text
    return _NOTEBOOK_BLOB_RE.sub(r'"\1": "<stripped>"', text)


def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". See split_parquet_paths() for how val is selected.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    parquet_paths = split_parquet_paths(list_parquet_files(), split)
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = [strip_notebook_blobs(t) for t in rg.column('text').to_pylist()]
            yield texts

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n  ℹ️ INFO: The dataset downloader has been unified into prepare_data.py.")
    print("  To download ClimbMix pretraining shards, please run:")
    print("    python scripts/data/prepare_data.py --download-climbmix 170\n")

