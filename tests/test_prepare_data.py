import sys
from dataclasses import FrozenInstanceError

import pytest

from scripts.data import prepare_data


def test_stream_local_jsonl_text_rows(tmp_path):
    local_file = tmp_path / "ir.jsonl"
    local_file.write_text(
        '{"text":"Incident response report with ransomware triage, containment, and recovery details."}\n'
        '{"text":"short"}\n',
        encoding="utf-8",
    )

    source_config = {
        "source_type": "local_files",
        "local_paths": [str(local_file)],
        "category": "cybersecurity",
    }

    docs = list(prepare_data.stream_local_file_texts("local_incident_response", source_config, float("inf")))

    assert docs == [
        "Source: local_incident_response\n"
        f"File: {local_file.name}\n\n"
        "Incident response report with ransomware triage, containment, and recovery details."
    ]


def test_stream_local_jsonl_conversation_list(tmp_path):
    local_file = tmp_path / "soc.jsonl"
    local_file.write_text(
        '[{"role":"user","content":"How do I triage suspicious Windows login failures?"},'
        '{"role":"assistant","content":"Review Event ID 4625, source IPs, successful follow-up logins, and preserve evidence."}]\n',
        encoding="utf-8",
    )

    source_config = {
        "source_type": "local_files",
        "local_paths": [str(local_file)],
        "category": "cybersecurity",
    }

    docs = list(prepare_data.stream_local_file_texts("local_soc_synthetic", source_config, float("inf")))

    assert len(docs) == 1
    assert "user: How do I triage suspicious Windows login failures?" in docs[0]
    assert "assistant: Review Event ID 4625" in docs[0]


def test_stream_local_raw_files_include_file_label(tmp_path):
    local_file = tmp_path / "apache.log"
    local_file.write_text(
        "203.0.113.77 - - [16/May/2026:08:00:00] "
        '"GET /login.php?id=1%27%20UNION%20SELECT HTTP/1.1" 403 421\n',
        encoding="utf-8",
    )

    source_config = {
        "source_type": "local_files",
        "local_paths": [str(local_file)],
        "category": "cybersecurity",
    }

    docs = list(prepare_data.stream_local_file_texts("local_security_logs", source_config, float("inf")))

    assert docs == [
        "Source: local_security_logs\n"
        f"File: {local_file.name}\n\n"
        "203.0.113.77 - - [16/May/2026:08:00:00] "
        '"GET /login.php?id=1%27%20UNION%20SELECT HTTP/1.1" 403 421'
    ]


def test_default_sources_include_local_but_exclude_sft():
    local_sources = [
        name
        for name, config in prepare_data.DATASET_SOURCES.items()
        if config.get("source_type") == "local_files"
    ]

    assert {
        "local_incident_response",
        "local_soc_synthetic",
        "local_reverse_engineering",
        "local_cloud_security",
        "local_security_logs",
    }.issubset(local_sources)
    assert all("data/sft" not in path for name in local_sources for path in prepare_data.DATASET_SOURCES[name]["local_paths"])


def test_climbmix_keeps_general_text_without_security_keywords():
    text = "This is a high quality general explanation about literature, history, and scientific reasoning."

    assert prepare_data.is_high_quality_security_text(text, "climbmix") is True


def test_sampling_probabilities_prefer_cyber_and_local_over_general():
    # Weights rank sources INSIDE a bucket; cross-bucket size is TARGET_DOMAIN_SHARE.
    cyber = ["local_incident_response", "circl_vuln_patch"]
    general = ["climbmix", "wikipedia"]
    cyber_probs = dict(zip(cyber, prepare_data._build_sampling_probs(cyber)))
    general_probs = dict(zip(general, prepare_data._build_sampling_probs(general)))

    assert cyber_probs["circl_vuln_patch"] > cyber_probs["local_incident_response"]
    assert general_probs["climbmix"] > general_probs["wikipedia"]


def test_indonesian_sources_are_their_own_share_bucket():
    assert prepare_data._source_bucket("wikipedia_id") == "indonesian"
    assert prepare_data._source_bucket("climbmix") == "general"
    assert sum(prepare_data.TARGET_DOMAIN_SHARE.values()) == pytest.approx(1.0)


def test_bucket_picker_drives_the_mix_to_the_target_shares():
    """Greedy deficit picking must land on TARGET_DOMAIN_SHARE despite uneven doc sizes."""
    names = ["climbmix", "wikipedia_id", "secure_code_python", "nvd_cve", "finemath"]
    samplers = prepare_data._build_bucket_samplers(names)
    # Wildly different document lengths per bucket — the whole point of char-based deficits.
    doc_chars = {"climbmix": 4000, "wikipedia_id": 8000, "secure_code_python": 500,
                 "nvd_cve": 300, "finemath": 2000}
    stats = {n: {"docs": 0, "chars": 0} for n in names}

    for _ in range(4000):
        bucket = prepare_data._pick_bucket(samplers, stats)
        picked = samplers[bucket][0][0]  # single source per bucket here
        stats[picked]["docs"] += 1
        stats[picked]["chars"] += doc_chars[picked]

    total = sum(s["chars"] for s in stats.values())
    for name in names:
        target = prepare_data.TARGET_DOMAIN_SHARE[prepare_data._source_bucket(name)]
        assert stats[name]["chars"] / total == pytest.approx(target, abs=0.02)


def test_dry_run_exits_before_interleaved_streaming(monkeypatch, tmp_path, capsys):
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("dry-run should not enter interleaved streaming")

    monkeypatch.setattr(prepare_data, "get_base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(prepare_data, "interleaved_shuffle_main", fail_if_called)
    monkeypatch.setattr(sys, "argv", ["prepare_data.py", "--dry-run", "--max-tokens", "1000000"])

    prepare_data.main()

    output = capsys.readouterr().out
    assert called is False
    assert "[DRY RUN]" in output
    assert "Planned mix" in output
    assert "local files" in output


def test_dataset2_adds_advanced_security_vocabulary():
    assert "allows attacker to" in prepare_data.CAUSAL_MARKERS
    assert "use-after-free" in prepare_data.MECHANISM_TERMS
    assert "book a demo" in prepare_data.MARKETING_TERMS
    assert "shellcode" in prepare_data.SECURITY_CODE_TERMS


def test_source_schema_is_frozen_and_scales_token_caps():
    source = prepare_data.Source(
        name="example",
        source_type="rss",
        domain="Cyber",
        primary_subdomain="Threat Intel",
        expected_tier=prepare_data.TIER_GOLD,
        estimated_tokens=10,
        max_tokens=20,
        description="Example source",
        config={"url": "https://example.com/feed.xml"},
    )

    try:
        source.name = "changed"
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("Source should be frozen")

    definitions = {src.name: src for src in prepare_data._source_definitions(target_tokens=1_000_000_000)}
    assert definitions["project_zero"].max_tokens == 80_000_000
    assert definitions["project_zero"].expected_tier == prepare_data.TIER_GOLD


def test_dataset2_sources_are_available_without_breaking_dry_run():
    assert "project_zero" in prepare_data.DATASET_SOURCES
    assert prepare_data.DATASET_SOURCES["project_zero"]["source_type"] == "rss"
    assert prepare_data.DOMAIN_SAMPLING_WEIGHTS["project_zero"] > prepare_data.DOMAIN_SAMPLING_WEIGHTS["wikipedia"]


def test_dataset2_vocabulary_only_applies_to_dataset2_sources():
    dataset2_marker_only = (
        "The advisory explains that this condition allows attacker to alter the control flow "
        "after malformed input reaches a parser boundary."
    )
    old_source_marker_only = (
        "The repository note says this condition allows attacker to alter application behavior "
        "after malformed input reaches a parser boundary."
    )

    assert prepare_data.is_high_quality_security_text(dataset2_marker_only, "project_zero") is True
    assert prepare_data.is_high_quality_security_text(old_source_marker_only, "secure_code_python") is False


def test_dataset2_marketing_filter_only_applies_to_dataset2_sources():
    dataset2_marketing = (
        "CVE-2026-0001 overview with vendor pricing language. Book a demo to learn more "
        "about the platform and customer story."
    )
    old_source_marketing = (
        "CVE-2026-0001 proof of concept notes include a book a demo string copied from "
        "a README banner in the repository."
    )

    assert prepare_data.is_high_quality_security_text(dataset2_marketing, "project_zero") is False
    assert prepare_data.is_high_quality_security_text(old_source_marketing, "secure_code_python") is True


def test_competition_math_is_not_a_pretraining_source():
    # Instruction-tuning data must never be in the pretraining mixture. The SFT entry
    # that used to hold it was dropped: hendrycks/competition_math no longer exists on
    # the Hub, and numinamath_cot covers the same material with chain-of-thought.
    assert "competition_math" not in prepare_data.DATASET_SOURCES
    assert "competition_math" not in prepare_data.DOMAIN_SAMPLING_WEIGHTS

    from scripts.data import download_sft_data

    assert "competition_math_sft" not in download_sft_data.SOURCES


def test_cot_collection_asks_for_a_config_that_exists():
    """The parquet-conversion branch exposes one config named "default".

    Requesting the upstream name ("en") raises BuilderConfig not found, which the
    download loop swallows — the source reports 0 rows instead of failing loudly.
    """
    from scripts.data import download_sft_data

    cfg = download_sft_data.SOURCES["cot_collection"]

    assert cfg["revision"] == "refs/convert/parquet"
    assert cfg["hf_subset"] == "default"


def test_sft_only_sources_excluded_from_pretraining():
    # The instruction/chat/DPO datasets that were moved to SFT must not leak
    # back into the pretraining mixture.
    moved = (
        "trendyol_cyber",
        "nist_cybersec",
        "fenrir_v2",
        "cybernative_vuln_dpo",
        "openhermes",
        "code_feedback",
        "numinamath_cot",
        "competition_math",
    )
    for name in moved:
        assert name not in prepare_data.DATASET_SOURCES
        assert name not in prepare_data.DOMAIN_SAMPLING_WEIGHTS



def test_writer_reader_column_contract(tmp_path):
    """prepare_data.write_shard must produce a 'text' column read the exact same
    way the pretraining reader (dataset.parquets_iter_batched) and tokenizer
    training (tok_train) consume it. This locks the writer<->reader contract."""
    import pyarrow.parquet as pq

    shard = tmp_path / "shard_00000.parquet"
    docs = ["first doc about CVE-2021-44228", "second doc about SOC triage"]
    prepare_data.write_shard(docs, str(shard))

    # Mimic dataset.parquets_iter_batched's exact access pattern
    pf = pq.ParquetFile(str(shard))
    read_back = []
    for rg_idx in range(pf.num_row_groups):
        rg = pf.read_row_group(rg_idx)
        read_back.extend(rg.column("text").to_pylist())

    assert read_back == docs


def test_writer_output_dir_matches_reader_auxiliary_dir():
    """The directory prepare_data writes cybersecurity shards into must be one of
    the auxiliary directories the pretraining reader auto-merges, otherwise the
    prepared data would silently never be trained on."""
    import os
    from mesosfer.data import dataset

    aux_basenames = {os.path.basename(p) for p in dataset.AUXILIARY_DATA_DIRS}
    # default --output-dir in prepare_data is "base_data_cybersecurity"
    assert "base_data_cybersecurity" in aux_basenames


# ---------------------------------------------------------------------------
# Incremental --sources batches (interleaved mode)
#
# interleaved_shuffle_main used to start at shard_00000 on every invocation, so a
# second run with different --sources overwrote the first run's shards and replaced
# its completed_sources. Only the tail of the older run survived, silently. These
# tests pin the resume behaviour that makes incremental batching safe.
# ---------------------------------------------------------------------------

class _Args:
    max_tokens = None
    shard_size = 500
    val_ratio = 0.008
    fresh = False
    cluster_size = 8
    sampling_temperature = 1.2
    checkpoint_dir = None
    checkpoint_every_gb = 9999.0


def _patch_streaming(monkeypatch):
    """Replace network streaming + checkpointing with local stand-ins."""
    monkeypatch.setattr(
        prepare_data, "stream_dataset_texts",
        lambda name, config, *a, **k: (f"{name} doc {i} " + "x" * 400 for i in range(2000)),
    )
    monkeypatch.setattr(prepare_data, "create_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(prepare_data, "verify_shard_balance", lambda *a, **k: None)


def _shard_count(directory):
    return len(list(directory.glob("shard_*.parquet")))


def _two_batches():
    names = list(prepare_data.DATASET_SOURCES)
    return names[:2], names[2:4]


def test_second_batch_appends_instead_of_overwriting(monkeypatch, tmp_path, capsys):
    _patch_streaming(monkeypatch)
    first, second = _two_batches()
    out = str(tmp_path)

    prepare_data.interleaved_shuffle_main(_Args(), list(first), out)
    after_first = _shard_count(tmp_path)
    progress_first = prepare_data.load_progress(out)

    prepare_data.interleaved_shuffle_main(_Args(), list(second), out)
    after_second = _shard_count(tmp_path)
    progress_second = prepare_data.load_progress(out)
    capsys.readouterr()

    assert after_second > after_first, "second batch overwrote the first batch's shards"
    assert progress_second["next_shard_idx"] > progress_first["next_shard_idx"]
    # completed_sources is a union across runs, not the last run's list
    assert set(first) <= set(progress_second["completed_sources"])
    assert set(second) <= set(progress_second["completed_sources"])
    # per-source stats from the first batch survive
    for name in first:
        assert progress_second["stats"].get(name, {}).get("docs")


def test_already_completed_sources_are_skipped(monkeypatch, tmp_path, capsys):
    _patch_streaming(monkeypatch)
    first, _ = _two_batches()
    out = str(tmp_path)

    prepare_data.interleaved_shuffle_main(_Args(), list(first), out)
    before = _shard_count(tmp_path)
    prepare_data.interleaved_shuffle_main(_Args(), list(first), out)
    capsys.readouterr()

    assert _shard_count(tmp_path) == before, "re-running finished sources duplicated data"


def test_max_tokens_budget_is_cumulative_across_batches(monkeypatch, tmp_path, capsys):
    _patch_streaming(monkeypatch)
    first, second = _two_batches()
    out = str(tmp_path)

    class Capped(_Args):
        max_tokens = 100_000

    prepare_data.interleaved_shuffle_main(Capped(), list(first), out)
    prepare_data.interleaved_shuffle_main(Capped(), list(second), out)
    capsys.readouterr()

    stats = prepare_data.load_progress(out)["stats"]
    total = sum(s["chars"] for s in stats.values()) / prepare_data.CHARS_PER_TOKEN
    assert total <= Capped.max_tokens * 1.05, (
        f"budget is per-run instead of cumulative: {total:,.0f} tokens"
    )


def test_checkpoint_every_gb_can_disable_checkpoints(monkeypatch, tmp_path, capsys):
    """A large --checkpoint-every-gb must suppress the final checkpoint too.

    The end-of-run checkpoint used to be unconditional, so the flag could not turn
    checkpointing off. Each archive is a full tar.gz of the output dir, so running N
    incremental --sources batches left N snapshots behind: seven batches over a 15 GB
    dataset produced 46 GB of archives and filled the disk.
    """
    _patch_streaming(monkeypatch)
    first, _ = _two_batches()
    archives = tmp_path / "ckpt"
    archives.mkdir()

    calls = []
    monkeypatch.setattr(prepare_data, "create_checkpoint",
                        lambda *a, **k: calls.append(a))

    class NoCheckpoint(_Args):
        checkpoint_dir = str(archives)
        checkpoint_every_gb = 9999.0

    prepare_data.interleaved_shuffle_main(NoCheckpoint(), list(first), str(tmp_path))
    capsys.readouterr()

    assert calls == [], f"checkpoint written despite a 9999 GB threshold: {calls}"
