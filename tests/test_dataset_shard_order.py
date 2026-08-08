"""Train shards must not be read in download order.

prepare_data appends one contiguous block of shards per `--sources` batch, so
sorted-by-filename order groups every domain together. Training that order means
one domain at a time; tok_train's max_chars cap would only ever see the first batches.
"""
import pyarrow as pa
import pyarrow.parquet as pq

from mesosfer.data import dataset


def _write_shards(tmp_path, n):
    for i in range(n):
        table = pa.table({"text": [f"shard {i}"]})
        pq.write_table(table, tmp_path / f"shard_{i:05d}.parquet")


def test_train_shards_are_shuffled_not_download_order(tmp_path, monkeypatch):
    _write_shards(tmp_path, 20)
    paths = sorted(str(p) for p in tmp_path.glob("*.parquet"))
    monkeypatch.setattr(dataset, "list_parquet_files", lambda: paths)

    seen = [batch[0] for batch in dataset.parquets_iter_batched(split="train")]

    assert len(seen) == 19, "last shard is held out for val"
    assert seen != [f"shard {i}" for i in range(19)], "still reading in download order"
    assert sorted(seen) == sorted(f"shard {i}" for i in range(19)), "no shard lost or duplicated"


def test_shard_order_is_stable_across_calls(tmp_path, monkeypatch):
    """DDP ranks each call this independently — they must agree."""
    _write_shards(tmp_path, 20)
    paths = sorted(str(p) for p in tmp_path.glob("*.parquet"))
    monkeypatch.setattr(dataset, "list_parquet_files", lambda: paths)

    first = [batch[0] for batch in dataset.parquets_iter_batched(split="train")]
    second = [batch[0] for batch in dataset.parquets_iter_batched(split="train")]

    assert first == second


def test_val_split_is_unaffected(tmp_path, monkeypatch):
    _write_shards(tmp_path, 20)
    paths = sorted(str(p) for p in tmp_path.glob("*.parquet"))
    monkeypatch.setattr(dataset, "list_parquet_files", lambda: paths)

    seen = [batch[0] for batch in dataset.parquets_iter_batched(split="val")]

    assert seen == ["shard 19"]
