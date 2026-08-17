"""hf_hub_download hands back a snapshot symlink, not the file itself.

Its target is relative (../../blobs/<sha>), so moving the link to a directory at a
different depth leaves a dangling path — the tokenizer landed one level shallower
than the dataset and broke, while the shards happened to survive.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from download_artifacts_from_hf import _place_downloaded  # noqa: E402


def _fake_hf_cache(tmp_path, payload=b"tokenizer bytes"):
    """Mirror the HF layout: blobs/<sha> with snapshots/<rev>/<name> -> ../../blobs/<sha>."""
    blobs = tmp_path / "blobs"
    snapshot = tmp_path / "snapshots" / "abc123"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    blob = blobs / "deadbeef"
    blob.write_bytes(payload)
    link = snapshot / "tokenizer.pkl"
    os.symlink(os.path.join("..", "..", "blobs", "deadbeef"), link)
    return link, blob


def test_moving_to_a_shallower_dir_still_yields_readable_content(tmp_path):
    link, _ = _fake_hf_cache(tmp_path)
    dest_dir = tmp_path / "out"
    dest_dir.mkdir()
    dest = dest_dir / "tokenizer.pkl"

    _place_downloaded(str(link), dest)

    assert dest.exists(), "destination must exist"
    assert not dest.is_symlink(), "a relative symlink would dangle after the move"
    assert dest.read_bytes() == b"tokenizer bytes"


def test_dangling_snapshot_link_is_removed(tmp_path):
    link, _ = _fake_hf_cache(tmp_path)
    dest = tmp_path / "out.pkl"

    _place_downloaded(str(link), dest)

    assert not os.path.lexists(link), "stale link makes the HF cache look valid when it is not"


def test_plain_file_input_is_handled(tmp_path):
    """HF_HUB_DISABLE_SYMLINKS and Windows return a real file, not a link."""
    src = tmp_path / "real.pt"
    src.write_bytes(b"weights")
    dest = tmp_path / "out.pt"

    _place_downloaded(str(src), dest)

    assert dest.read_bytes() == b"weights"
    assert not src.exists()
