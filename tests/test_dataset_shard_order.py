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


def test_val_prefixed_file_is_selected_regardless_of_position():
    """prepare_data's val_shard.parquet must win over the last-file fallback, and it
    can land anywhere in the aux+primary merged list (list_parquet_files puts aux dirs
    first), not just at the end.
    """
    paths = ["aux/shard_00000.parquet", "aux/val_shard.parquet", "primary/shard_00001.parquet"]

    assert dataset.split_parquet_paths(paths, "val") == ["aux/val_shard.parquet"]
    train = dataset.split_parquet_paths(paths, "train")
    assert "aux/val_shard.parquet" not in train
    assert sorted(train) == ["aux/shard_00000.parquet", "primary/shard_00001.parquet"]


def test_no_val_prefixed_file_falls_back_to_legacy_last_file():
    """Corpora prepared before val_shard.parquet existed have no way to name their
    validation shard, so this must reproduce the old last-file behavior exactly:
    val is the last file, train is everything else (shuffled with the fixed seed).
    """
    paths = [f"shard_{i:05d}.parquet" for i in range(5)]
    import random
    expected_train = paths[:-1]
    random.Random(dataset.SHARD_SHUFFLE_SEED).shuffle(expected_train)

    assert dataset.split_parquet_paths(paths, "val") == paths[-1:]
    assert dataset.split_parquet_paths(paths, "train") == expected_train


def test_pretrain_dataloader_uses_the_same_order():
    """base_train reads through dataloader._document_batches, not parquets_iter_batched.

    They must agree: if only one of them shuffles, tok_train and base_train see
    different corpora, and the tokenizer is fitted to data the model never reads
    in that order.
    """
    from mesosfer.data import dataloader

    paths = [f"shard_{i:05d}.parquet" for i in range(20)]

    assert dataloader.split_parquet_paths is dataset.split_parquet_paths
    assert dataset.split_parquet_paths(paths, "train") == dataset.split_parquet_paths(paths, "train")
    assert dataset.split_parquet_paths(paths, "train") != paths[:-1]
    assert dataset.split_parquet_paths(paths, "val") == paths[-1:]
