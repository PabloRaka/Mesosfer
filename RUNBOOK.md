# Runbook — data prep on CPU, training on GPU

Three stages (pretrain → CPT → SFT), each split across two machines: a cheap CPU box
that prepares data, and a GPU VM that trains. HuggingFace is the transport between them.

Depth 16 throughout. Adjust `--depth`/`--d16` together if you change it.

---

## 0. Setup (both machines, once)

```bash
git clone https://github.com/PabloRaka/Mesosfer.git && cd Mesosfer
```

CPU box (data prep only — no GPU deps needed):

```bash
uv sync --extra cpu
```

GPU VM:

```bash
uv sync --extra gpu     # NVIDIA; use --extra rocm --no-build-isolation for AMD
```

Both machines need `.env` (copy from `.env.example`):

```bash
HF_TOKEN=hf_xxxxxxxxxxxx
HF_USERNAME=pabloraka
```

`HF_TOKEN` needs **write** scope on the CPU box and is required on the GPU VM too —
several pretraining sources (`Primus-*`, `AquilaX`) are gated and are silently skipped
without it. Alternatively `hf auth login`.

The upload script creates three repos under your account: `<user>/dataset`,
`<user>/tokenizer`, `<user>/model`. **New repos are private unless you pass `--public`.**
`--public` never changes the visibility of a repo that already exists.

---

## 1. PRETRAIN

### 1a. CPU box — prepare the corpus

```bash
python -m scripts.data.prepare_data --dry-run --d16
```

Read the "Planned mix" table before committing hours of download: it shows what each
bucket wants against what the sources can really supply. Then run it for real:

```bash
python -m scripts.data.prepare_data --d16
```

Writes ~15B tokens to `~/.cache/mesosfer/base_data_cybersecurity/` at the pretrain mix
(general 50% / code 25% / cybersecurity 15% / reasoning 10%). Resumable — re-run the
same command after an interruption. Check progress any time:

```bash
python -m scripts.data.prepare_data --status
```

**8 GB RAM box:** `--sources a,b,c` prepares a subset of sources and resumes cleanly, so
a small machine can do the corpus in batches instead of holding ~52 concurrent
HuggingFace streaming iterators open at once (each one holds an open response and a
parquet row-group buffer — that concurrency is the memory floor, and no amount of
shard-size tuning reduces it). Raise `--checkpoint-every-gb` (default 10) if the periodic
tar.gz snapshots of the output dir are also a concern — note `0` is rejected by argparse
validation (`--checkpoint-every-gb must be > 0`), it does not disable snapshots.

### 1b. CPU box — train the tokenizer

The tokenizer reads the corpus you just built, so it must be trained here, and the same
tokenizer must be used by every later stage. Do this once.

```bash
python -m scripts.train.tok_train
python -m scripts.eval.tok_eval
```

### 1c. CPU box — upload corpus + tokenizer

```bash
python -m scripts.upload_checkpoint_to_hf --artifact dataset --public --dataset-dirs base_data_cybersecurity
```

```bash
python -m scripts.upload_checkpoint_to_hf --artifact tokenizer --public
```

Only `base_data_cybersecurity` is uploaded on purpose. `base_data_climbmix` is a local
2-shard scratch dir the prep step needs; the GPU VM does not want it (see DATASET.md).

### 1d. GPU VM — download

```bash
python -m scripts.download_artifacts_from_hf --artifact dataset --dataset-dirs base_data_cybersecurity
```

```bash
python -m scripts.download_artifacts_from_hf --artifact tokenizer
```

### 1e. GPU VM — pretrain

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.train.base_train -- --depth=16 --aspect-ratio=128 --ve-layers=2 --data-dir=$HOME/.cache/mesosfer/base_data_cybersecurity --target-param-data-ratio=15 --device-batch-size=32 --window-pattern=L --fp8 --model-tag=d16 --run=d16_pretrain
```

Single GPU: drop `torchrun --standalone --nproc_per_node=8 -m` for `python -m`, lower
`--device-batch-size`, and drop `--fp8` unless you are on H100+.

`--ve-layers=2` matters at this depth: with the default the value-embedding tables are
twice the size of the transformer itself. `--data-dir` is explicit so the run cannot
silently pick up whatever else is in the cache dir.

Evaluate, then publish the checkpoint:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.eval.base_eval -- --model-tag=d16 --device-batch-size=32
```

```bash
python -m scripts.upload_checkpoint_to_hf --artifact model --source base --depth d16 --best --public
```

---

## 2. CPT (continued pretraining)

### 2a. CPU box — prepare the CPT corpus

```bash
python -m scripts.data.prepare_data --dry-run --stage cpt --max-tokens 3000000000
```

```bash
python -m scripts.data.prepare_data --stage cpt --max-tokens 3000000000
```

Writes to `~/.cache/mesosfer/base_data_cpt/` at the CPT mix — cybersecurity 50% split
across 9 subdomains, code 30% across 6 language groups, general 10%, reasoning 10%.

Do **not** use `--d16` here: that flag sizes a full pretraining budget (~15B tokens).
CPT is a top-up pass, so set `--max-tokens` yourself. 3B is a reasonable starting point
(~20% of the pretraining budget); raise it if you have the compute.

When it finishes, read the per-subdomain realized-vs-target table it prints. A subdomain
far below target means the corpus simply does not contain enough of it — that is a
sourcing problem, not a bug, and no flag will fix it.

Do not retrain the tokenizer. CPT must use the pretraining tokenizer.

### 2b. CPU box — upload

```bash
python -m scripts.upload_checkpoint_to_hf --artifact dataset --public --dataset-dirs base_data_cpt
```

### 2c. GPU VM — download

```bash
python -m scripts.download_artifacts_from_hf --artifact dataset --dataset-dirs base_data_cpt
```

If the GPU VM is a fresh machine, also pull the tokenizer and the pretrained checkpoint:

```bash
python -m scripts.download_artifacts_from_hf --artifact tokenizer
```

```bash
python -m scripts.download_artifacts_from_hf --artifact model --source base --depth d16 --best
```

### 2d. GPU VM — run CPT

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.train.base_train -- --init-from=d16 --data-dir=$HOME/.cache/mesosfer/base_data_cpt --model-tag=d16_cpt --num-iterations=4000 --device-batch-size=32 --window-pattern=L --fp8 --run=d16_cpt
```

`--init-from` loads only the weights and starts a fresh run: new optimizer, new LR
schedule, step 0, its own checkpoint dir. It refuses to overwrite the source checkpoint.
Do not pass `--depth`, `--aspect-ratio`, `--ve-layers` or `--max-seq-len` — the config
comes from the checkpoint, and passing a conflicting value is a hard error by design.

Set `--num-iterations` explicitly. Left to the automatic budget, CPT would size itself
like a full pretraining run.

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.eval.base_eval -- --model-tag=d16_cpt --device-batch-size=32
```

```bash
python -m scripts.upload_checkpoint_to_hf --artifact model --source base --depth d16_cpt --best --public
```

---

## 3. SFT

**The HF round-trip does not apply here.** The SFT sets are JSONL, and the dataset
uploader only handles parquet shards — it would upload nothing. The SFT data reaches the
GPU VM two other ways, both already working:

- the bundled conversations are committed to the repo under `data/sft/*.jsonl`, so
  `git clone` already brought them;
- the external sets are fetched straight from HuggingFace by the downloader below.

So run the fetch on the GPU VM itself. There is no upload step.

### 3a. GPU VM — fetch the external SFT sets

```bash
python -m scripts.data.download_sft_data --list
```

```bash
python -m scripts.data.download_sft_data
```

Writes into `data/sft/` alongside the bundled files. Gated sources are skipped silently
without a valid `HF_TOKEN` — check the summary it prints rather than assuming.

### 3b. GPU VM — run SFT

From the CPT checkpoint:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat.chat_sft -- --checkpoint-source=base --model-tag=d16_cpt --run=d16_sft
```

To fine-tune the pretrained model instead, use `--model-tag=d16`.

```bash
python -m scripts.upload_checkpoint_to_hf --artifact model --source sft --depth d16_cpt --best --public
```

### 3c. Try it

```bash
python -m scripts.chat.chat_cli -p "Explain how a SQL injection is exploited and mitigated."
```

```bash
python -m scripts.chat.chat_web
```

---

## Notes

**Always pass `--depth` to the HF scripts.** Both default to `d32`, which does not exist
in this project's runs — the upload fails with "checkpoint directory not found" and the
download reports "no checkpoints found". Neither is silent, but both waste a cycle.

**Uploads are idempotent.** Shards already in the repo are skipped, so an interrupted
upload is safe to re-run.

**Validation.** `prepare_data` writes `val_shard.parquet` — a held-out split spanning
every domain in the mix — and the loader selects it by name. Validation loss from a
corpus prepared before that existed is not comparable: it measured ClimbMix only.
