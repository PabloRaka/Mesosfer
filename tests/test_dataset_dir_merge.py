"""The "DATASET UPGRADE REQUIRED" nag must not fire when a corpus is actually present.

A prepare_data corpus lives in an auxiliary dir. Warning there tells the user to
download ClimbMix and retrain the tokenizer, which would dilute their mixture and
move the validation shard off their own distribution.
"""
import pyarrow as pa
import pyarrow.parquet as pq

from mesosfer.data import dataset


def _shards(path, n):
    path.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        pq.write_table(pa.table({"text": [f"doc {i}"]}), path / f"shard_{i:05d}.parquet")
    return path


def test_no_upgrade_nag_when_auxiliary_corpus_exists(tmp_path, monkeypatch, capsys):
    aux = _shards(tmp_path / "base_data_cybersecurity", 3)
    monkeypatch.setattr(dataset, "DATA_DIR", str(tmp_path / "base_data_climbmix"))
    monkeypatch.setattr(dataset, "base_dir", str(tmp_path))
    monkeypatch.setattr(dataset, "AUXILIARY_DATA_DIRS", [str(aux)])

    files = dataset.list_parquet_files(warn_on_legacy=True)

    assert len(files) == 3
    assert "UPGRADE REQUIRED" not in capsys.readouterr().out


def test_upgrade_nag_still_fires_with_nothing_to_train_on(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dataset, "DATA_DIR", str(tmp_path / "base_data_climbmix"))
    monkeypatch.setattr(dataset, "base_dir", str(tmp_path))
    monkeypatch.setattr(dataset, "AUXILIARY_DATA_DIRS", [str(tmp_path / "nope")])

    files = dataset.list_parquet_files(warn_on_legacy=True)

    assert files == []
    assert "UPGRADE REQUIRED" in capsys.readouterr().out
