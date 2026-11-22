use std::collections::{HashMap, VecDeque};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Cursor, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use aho_corasick::{AhoCorasick, AhoCorasickBuilder};
use anyhow::{anyhow, Context, Result};
use arrow_array::{Array, LargeStringArray, RecordBatch, StringArray, StringViewArray};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use clap::Parser;
use crossbeam_channel::{bounded, Receiver, Sender};
use flate2::read::MultiGzDecoder;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use reqwest::blocking::Client;
use serde::Deserialize;

const CHARS_PER_TOKEN: f64 = 4.0;
const MIN_DOC_CHARS: usize = 50;
const MAX_DOC_CHARS: usize = 500_000;

#[derive(Parser, Debug)]
#[command(name = "fast_prepare", about = "Multi-threaded Rust pretraining dataset preparer for Mesosfer")]
struct Args {
    /// Target total tokens (default: 40.2B for d24 ratio 15)
    #[arg(long, default_value_t = 40_202_409_240)]
    max_tokens: u64,

    /// Output directory (default: ~/.cache/mesosfer/base_data_cybersecurity)
    #[arg(long)]
    output_dir: Option<PathBuf>,

    /// Path to Mesosfer repo root for local data/* jsonl files (default: current dir)
    #[arg(long)]
    repo_dir: Option<PathBuf>,

    /// Number of parallel download + parse worker threads
    #[arg(long, default_value_t = 16)]
    workers: usize,

    /// Number of parallel Parquet shard writer threads
    #[arg(long, default_value_t = 4)]
    writer_threads: usize,

    /// Max docs per train shard
    #[arg(long, default_value_t = 100_000)]
    shard_docs: usize,

    /// Max raw text bytes per train shard (MB)
    #[arg(long, default_value_t = 256)]
    shard_mb: usize,

    /// Validation fraction (default 0.8%)
    #[arg(long, default_value_t = 0.008)]
    val_ratio: f64,

    /// Clear existing shards in output_dir before starting
    #[arg(long)]
    fresh: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Bucket {
    General,
    Code,
    Cybersecurity,
    Reasoning,
}

impl Bucket {
    const ALL: [Bucket; 4] = [
        Bucket::General,
        Bucket::Code,
        Bucket::Cybersecurity,
        Bucket::Reasoning,
    ];
    fn name(self) -> &'static str {
        match self {
            Bucket::General => "general",
            Bucket::Code => "code",
            Bucket::Cybersecurity => "cybersecurity",
            Bucket::Reasoning => "reasoning",
        }
    }
    fn idx(self) -> usize {
        match self {
            Bucket::General => 0,
            Bucket::Code => 1,
            Bucket::Cybersecurity => 2,
            Bucket::Reasoning => 3,
        }
    }
    fn target_share(self) -> f64 {
        match self {
            Bucket::General => 0.50,
            Bucket::Code => 0.25,
            Bucket::Cybersecurity => 0.15,
            Bucket::Reasoning => 0.10,
        }
    }
}

#[derive(Clone, Debug)]
struct SourceSpec {
    name: &'static str,
    repo: &'static str,
    prefix: &'static str,
    text_col: &'static str,
    bucket: Bucket,
    weight: f64,
    max_tokens: u64,
}

fn default_sources() -> Vec<SourceSpec> {
    vec![
        // General (50%)
        SourceSpec { name: "climbmix", repo: "karpathy/climbmix-400b-shuffle", prefix: "", text_col: "text", bucket: Bucket::General, weight: 1.0, max_tokens: 60_000_000_000 },
        SourceSpec { name: "fineweb2_id", repo: "HuggingFaceFW/fineweb-2", prefix: "data/ind_Latn/train/", text_col: "text", bucket: Bucket::General, weight: 0.30, max_tokens: 3_000_000_000 },
        SourceSpec { name: "wikipedia_id", repo: "wikimedia/wikipedia", prefix: "20231101.id/", text_col: "text", bucket: Bucket::General, weight: 0.15, max_tokens: 200_000_000 },
        // Code (25%)
        SourceSpec { name: "swallow_code_v2", repo: "tokyotech-llm/swallow-code-v2", prefix: "stage4-llm-rewrite/", text_col: "text", bucket: Bucket::Code, weight: 1.3, max_tokens: 5_000_000_000 },
        SourceSpec { name: "secure_code_python", repo: "bigcode/the-stack-dedup", prefix: "data/python/", text_col: "content", bucket: Bucket::Code, weight: 1.4, max_tokens: 4_000_000_000 },
        SourceSpec { name: "secure_code_c", repo: "bigcode/the-stack-dedup", prefix: "data/c/", text_col: "content", bucket: Bucket::Code, weight: 1.4, max_tokens: 3_000_000_000 },
        SourceSpec { name: "secure_code_cpp", repo: "bigcode/the-stack-dedup", prefix: "data/cpp/", text_col: "content", bucket: Bucket::Code, weight: 1.2, max_tokens: 3_000_000_000 },
        SourceSpec { name: "secure_code_rust", repo: "bigcode/the-stack-dedup", prefix: "data/rust/", text_col: "content", bucket: Bucket::Code, weight: 1.2, max_tokens: 2_400_000_000 },
        SourceSpec { name: "secure_code_go", repo: "bigcode/the-stack-dedup", prefix: "data/go/", text_col: "content", bucket: Bucket::Code, weight: 1.2, max_tokens: 2_400_000_000 },
        SourceSpec { name: "secure_code_shell", repo: "bigcode/the-stack-dedup", prefix: "data/shell/", text_col: "content", bucket: Bucket::Code, weight: 1.4, max_tokens: 2_000_000_000 },
        SourceSpec { name: "secure_code_javascript", repo: "bigcode/the-stack-dedup", prefix: "data/javascript/", text_col: "content", bucket: Bucket::Code, weight: 1.3, max_tokens: 2_400_000_000 },
        SourceSpec { name: "secure_code_typescript", repo: "bigcode/the-stack-dedup", prefix: "data/typescript/", text_col: "content", bucket: Bucket::Code, weight: 1.3, max_tokens: 2_000_000_000 },
        SourceSpec { name: "secure_code_java", repo: "bigcode/the-stack-dedup", prefix: "data/java/", text_col: "content", bucket: Bucket::Code, weight: 1.3, max_tokens: 2_000_000_000 },
        SourceSpec { name: "secure_code_php", repo: "bigcode/the-stack-dedup", prefix: "data/php/", text_col: "content", bucket: Bucket::Code, weight: 1.3, max_tokens: 1_600_000_000 },
        SourceSpec { name: "code_sql", repo: "bigcode/the-stack-dedup", prefix: "data/sql/", text_col: "content", bucket: Bucket::Code, weight: 1.5, max_tokens: 1_500_000_000 },
        SourceSpec { name: "code_powershell", repo: "bigcode/the-stack-dedup", prefix: "data/powershell/", text_col: "content", bucket: Bucket::Code, weight: 1.6, max_tokens: 800_000_000 },
        SourceSpec { name: "code_assembly", repo: "bigcode/the-stack-dedup", prefix: "data/assembly/", text_col: "content", bucket: Bucket::Code, weight: 1.6, max_tokens: 800_000_000 },
        SourceSpec { name: "code_csharp", repo: "bigcode/the-stack-dedup", prefix: "data/c-sharp/", text_col: "content", bucket: Bucket::Code, weight: 1.2, max_tokens: 2_000_000_000 },
        SourceSpec { name: "code_jupyter", repo: "bigcode/the-stack-dedup", prefix: "data/jupyter-notebook/", text_col: "content", bucket: Bucket::Code, weight: 1.0, max_tokens: 2_000_000_000 },
        // Cybersecurity (15%)
        SourceSpec { name: "primus_nemotron_cc", repo: "trend-cybertron/Primus-Nemotron-CC", prefix: "data/", text_col: "text", bucket: Bucket::Cybersecurity, weight: 2.1, max_tokens: 7_600_000_000 },
        SourceSpec { name: "primus_fineweb", repo: "trendmicro-ailab/Primus-FineWeb", prefix: "data/", text_col: "text", bucket: Bucket::Cybersecurity, weight: 1.9, max_tokens: 5_000_000_000 },
        SourceSpec { name: "primus_seed", repo: "trendmicro-ailab/Primus-Seed", prefix: "data/", text_col: "text", bucket: Bucket::Cybersecurity, weight: 2.2, max_tokens: 2_000_000_000 },
        // Reasoning / Math (10%)
        SourceSpec { name: "nemotron_cc_math", repo: "nvidia/Nemotron-CC-Math-v1", prefix: "4plus/", text_col: "text", bucket: Bucket::Reasoning, weight: 1.0, max_tokens: 18_000_000_000 },
        SourceSpec { name: "finemath", repo: "HuggingFaceTB/finemath", prefix: "finemath-4plus/", text_col: "text", bucket: Bucket::Reasoning, weight: 0.9, max_tokens: 3_000_000_000 },
    ]
}

fn load_hf_token() -> Option<String> {
    if let Ok(tok) = std::env::var("HF_TOKEN") {
        let t = tok.trim().to_string();
        if !t.is_empty() {
            return Some(t);
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        let p = PathBuf::from(home).join(".cache/huggingface/token");
        if let Ok(tok) = fs::read_to_string(p) {
            let t = tok.trim().to_string();
            if !t.is_empty() {
                return Some(t);
            }
        }
    }
    None
}

#[derive(Deserialize)]
struct HfTreeEntry {
    #[serde(rename = "type")]
    entry_type: String,
    path: String,
    size: Option<u64>,
}

fn list_hf_files(client: &Client, token: Option<&str>, repo: &str, prefix: &str) -> Result<Vec<(String, u64)>> {
    let clean_prefix = prefix.trim_matches('/');
    let mut url = if clean_prefix.is_empty() {
        format!("https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true")
    } else {
        format!("https://huggingface.co/api/datasets/{repo}/tree/main/{clean_prefix}?recursive=true")
    };
    let mut out = Vec::new();
    loop {
        let mut req = client.get(&url);
        if let Some(t) = token {
            req = req.bearer_auth(t);
        }
        let resp = req.send().with_context(|| format!("HF tree request failed: {url}"))?;
        if !resp.status().is_success() {
            return Err(anyhow!("HF tree HTTP {} for {repo}/{clean_prefix}", resp.status()));
        }
        let next_link = resp
            .headers()
            .get("link")
            .and_then(|v| v.to_str().ok())
            .and_then(parse_next_link);
        let entries: Vec<HfTreeEntry> = serde_json::from_reader(resp)?;
        for e in entries {
            if e.entry_type == "file"
                && (e.path.ends_with(".parquet")
                    || e.path.ends_with(".jsonl")
                    || e.path.ends_with(".jsonl.gz")
                    || e.path.ends_with(".json.gz"))
            {
                out.push((e.path, e.size.unwrap_or(50_000_000)));
            }
        }
        match next_link {
            Some(next) => url = next,
            None => break,
        }
    }
    Ok(out)
}

fn parse_next_link(header: &str) -> Option<String> {
    for part in header.split(',') {
        if part.contains("rel=\"next\"") {
            let start = part.find('<')? + 1;
            let end = part.find('>')?;
            if end > start {
                return Some(part[start..end].to_string());
            }
        }
    }
    None
}

struct FileTask {
    source_idx: usize,
    source_name: &'static str,
    repo: &'static str,
    path: String,
    est_chars: u64,
    text_col: &'static str,
    bucket: Bucket,
}

struct SourceState {
    spec: SourceSpec,
    files: VecDeque<(String, u64)>,
    chars_cap: u64,
    chars_est_dispatched: u64,
    chars_actual: u64,
    docs_actual: u64,
}

struct Scheduler {
    sources: Vec<SourceState>,
    bucket_target_chars: HashMap<Bucket, u64>,
    bucket_dispatched_chars: HashMap<Bucket, u64>,
    bucket_actual_chars: HashMap<Bucket, u64>,
}

impl Scheduler {
    fn next_task(&mut self, bucket_written: &[AtomicU64; 4], source_written: &[AtomicU64]) -> Option<FileTask> {
        // Pick bucket with lowest completion ratio (dispatched / target) that is still under target
        let mut best_bucket: Option<(Bucket, f64)> = None;
        for b in Bucket::ALL {
            let target = *self.bucket_target_chars.get(&b).unwrap_or(&1);
            if bucket_written[b.idx()].load(Ordering::Relaxed) >= target {
                continue;
            }
            let disp = *self.bucket_dispatched_chars.get(&b).unwrap_or(&0);
            if disp >= target + target / 20 {
                continue;
            }
            let ratio = (disp as f64) / (target.max(1) as f64);
            let has_work = self.sources.iter().enumerate().any(|(i, s)| {
                s.spec.bucket == b
                    && !s.files.is_empty()
                    && s.chars_est_dispatched < s.chars_cap
                    && source_written[i].load(Ordering::Relaxed) < s.chars_cap
            });
            if !has_work {
                continue;
            }
            if best_bucket.is_none() || ratio < best_bucket.unwrap().1 {
                best_bucket = Some((b, ratio));
            }
        }
        let bucket = best_bucket?.0;

        // Inside the bucket, pick source with lowest (dispatched / weight) still under its cap
        let mut best_src: Option<(usize, f64)> = None;
        for (idx, s) in self.sources.iter().enumerate() {
            if s.spec.bucket != bucket
                || s.files.is_empty()
                || s.chars_est_dispatched >= s.chars_cap
                || source_written[idx].load(Ordering::Relaxed) >= s.chars_cap
            {
                continue;
            }
            let score = (s.chars_est_dispatched as f64) / s.spec.weight.max(0.01);
            if best_src.is_none() || score < best_src.unwrap().1 {
                best_src = Some((idx, score));
            }
        }
        let idx = best_src?.0;
        let s = &mut self.sources[idx];
        let (path, fsize) = s.files.pop_front()?;
        // Estimate uncompressed chars from remote compressed file size (~3x expansion for parquet/gz)
        let est = (fsize.saturating_mul(3)).clamp(20_000_000, 4_000_000_000);
        s.chars_est_dispatched += est;
        *self.bucket_dispatched_chars.entry(bucket).or_insert(0) += est;
        Some(FileTask {
            source_idx: idx,
            source_name: s.spec.name,
            repo: s.spec.repo,
            path,
            est_chars: est,
            text_col: s.spec.text_col,
            bucket,
        })
    }

    fn report_actual(&mut self, source_idx: usize, bucket: Bucket, est_chars: u64, docs: u64, chars: u64) {
        let s = &mut self.sources[source_idx];
        s.docs_actual += docs;
        s.chars_actual += chars;
        s.chars_est_dispatched = s.chars_est_dispatched.saturating_sub(est_chars) + chars;
        let disp = self.bucket_dispatched_chars.entry(bucket).or_insert(0);
        *disp = disp.saturating_sub(est_chars) + chars;
        *self.bucket_actual_chars.entry(bucket).or_insert(0) += chars;
    }
}

fn build_junk_filter() -> AhoCorasick {
    let patterns = [
        "node_modules",
        "package-lock.json",
        "yarn.lock",
        ".min.js",
        "generated automatically",
        "auto-generated",
        "webpack",
        "dist/",
        "build/",
    ];
    AhoCorasickBuilder::new()
        .ascii_case_insensitive(true)
        .build(patterns)
        .expect("valid aho-corasick patterns")
}

#[inline]
fn has_excessive_repetition(text: &str) -> bool {
    // 1. Multi-line repetition check: if >= 24 substantial lines have < 25% unique lines
    let mut substantive_lines = 0usize;
    let mut prev_line = "";
    let mut consecutive_dup_lines = 0usize;
    for line in text.lines() {
        let t = line.trim();
        if t.len() >= 20 {
            substantive_lines += 1;
            if t == prev_line {
                consecutive_dup_lines += 1;
                if consecutive_dup_lines >= 4 {
                    return true;
                }
            } else {
                prev_line = t;
                consecutive_dup_lines = 0;
            }
        }
    }
    if substantive_lines >= 24 {
        let mut uniq = std::collections::HashSet::with_capacity(substantive_lines);
        for line in text.lines() {
            let t = line.trim();
            if t.len() >= 20 {
                uniq.insert(t);
            }
        }
        if uniq.len() * 4 < substantive_lines {
            return true;
        }
    }
    // 2. Consecutive 32-byte non-whitespace block repeat check (>= 5 consecutive repeats)
    let bytes = text.as_bytes();
    if bytes.len() >= 160 {
        let step = 32;
        let mut i = 0;
        while i + step * 5 <= bytes.len() {
            let chunk = &bytes[i..i + step];
            if chunk.iter().any(|&b| !b.is_ascii_whitespace())
                && &bytes[i + step..i + step * 2] == chunk
                && &bytes[i + step * 2..i + step * 3] == chunk
                && &bytes[i + step * 3..i + step * 4] == chunk
                && &bytes[i + step * 4..i + step * 5] == chunk
            {
                return true;
            }
            i += step;
        }
    }
    false
}

#[inline]
fn keep_doc(text: &str, junk: &AhoCorasick) -> bool {
    let len = text.len();
    if len < MIN_DOC_CHARS || len > MAX_DOC_CHARS {
        return false;
    }
    if junk.is_match(text) {
        return false;
    }
    !has_excessive_repetition(text)
}

fn download_bytes(client: &Client, token: Option<&str>, url: &str) -> Result<Bytes> {
    let mut last_err = None;
    for attempt in 0..4 {
        let mut req = client.get(url);
        if let Some(t) = token {
            req = req.bearer_auth(t);
        }
        match req.send() {
            Ok(resp) if resp.status().is_success() => match resp.bytes() {
                Ok(b) => return Ok(b),
                Err(e) => last_err = Some(anyhow!("{e}")),
            },
            Ok(resp) => {
                let st = resp.status();
                last_err = Some(anyhow!("HTTP {st} for {url}"));
                if st.as_u16() == 404 || st.as_u16() == 401 || st.as_u16() == 403 {
                    break;
                }
            }
            Err(e) => last_err = Some(anyhow!("{e}")),
        }
        thread::sleep(Duration::from_millis(500 * (1 << attempt)));
    }
    Err(last_err.unwrap_or_else(|| anyhow!("Download failed: {url}")))
}

struct DocBatch {
    source_idx: usize,
    bucket: Bucket,
    docs: Vec<String>,
}

fn send_batch(
    tx: &Sender<DocBatch>,
    source_idx: usize,
    bucket: Bucket,
    docs: &mut Vec<String>,
    stop: &AtomicBool,
) -> bool {
    if docs.is_empty() {
        return !stop.load(Ordering::Relaxed);
    }
    let batch = DocBatch {
        source_idx,
        bucket,
        docs: std::mem::take(docs),
    };
    send_batch_owned(tx, batch, stop)
}

fn send_batch_owned(tx: &Sender<DocBatch>, mut batch: DocBatch, stop: &AtomicBool) -> bool {
    while !stop.load(Ordering::Relaxed) {
        match tx.send_timeout(batch, Duration::from_millis(200)) {
            Ok(()) => return true,
            Err(crossbeam_channel::SendTimeoutError::Timeout(b)) => batch = b,
            Err(crossbeam_channel::SendTimeoutError::Disconnected(_)) => return false,
        }
    }
    false
}

fn parse_parquet_bytes(
    data: Bytes,
    source_idx: usize,
    text_col: &str,
    bucket: Bucket,
    source_cap: u64,
    bucket_cap: u64,
    source_written: &AtomicU64,
    bucket_written: &AtomicU64,
    junk: &AhoCorasick,
    tx: &Sender<DocBatch>,
    stop: &AtomicBool,
) -> Result<(u64, u64)> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(data)?;
    let schema = builder.schema().clone();
    let col_idx = schema
        .index_of(text_col)
        .or_else(|_| schema.index_of("text"))
        .or_else(|_| schema.index_of("content"))
        .with_context(|| format!("Column '{text_col}' not found in parquet schema"))?;
    let mask = parquet::arrow::ProjectionMask::roots(builder.parquet_schema(), [col_idx]);
    let reader = builder.with_projection(mask).with_batch_size(4096).build()?;

    let mut total_docs = 0u64;
    let mut total_chars = 0u64;
    let mut buf = Vec::with_capacity(256);

    for batch_res in reader {
        if stop.load(Ordering::Relaxed)
            || source_written.load(Ordering::Relaxed) >= source_cap
            || bucket_written.load(Ordering::Relaxed) >= bucket_cap
        {
            break;
        }
        let batch = batch_res?;
        let col = batch.column(0);
        let mut hit_cap = false;
        extract_strings_from_col(col.as_ref(), |s| {
            if hit_cap {
                return;
            }
            let trimmed = s.trim();
            if keep_doc(trimmed, junk) {
                let c = trimmed.len() as u64;
                if source_written.load(Ordering::Relaxed) + total_chars >= source_cap
                    || bucket_written.load(Ordering::Relaxed) + total_chars >= bucket_cap
                {
                    hit_cap = true;
                    return;
                }
                buf.push(trimmed.to_string());
                total_docs += 1;
                total_chars += c;
            }
        });
        if buf.len() >= 256 && !send_batch(tx, source_idx, bucket, &mut buf, stop) {
            break;
        }
        if hit_cap {
            break;
        }
    }
    let _ = send_batch(tx, source_idx, bucket, &mut buf, stop);
    Ok((total_docs, total_chars))
}

fn extract_strings_from_col<F: FnMut(&str)>(col: &dyn Array, mut f: F) {
    if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
        for i in 0..arr.len() {
            if !arr.is_null(i) {
                f(arr.value(i));
            }
        }
    } else if let Some(arr) = col.as_any().downcast_ref::<LargeStringArray>() {
        for i in 0..arr.len() {
            if !arr.is_null(i) {
                f(arr.value(i));
            }
        }
    } else if let Some(arr) = col.as_any().downcast_ref::<StringViewArray>() {
        for i in 0..arr.len() {
            if !arr.is_null(i) {
                f(arr.value(i));
            }
        }
    }
}

fn parse_jsonl_reader<R: Read>(
    reader: R,
    source_idx: usize,
    text_col: &str,
    bucket: Bucket,
    source_cap: u64,
    bucket_cap: u64,
    source_written: &AtomicU64,
    bucket_written: &AtomicU64,
    junk: &AhoCorasick,
    tx: &Sender<DocBatch>,
    stop: &AtomicBool,
) -> Result<(u64, u64)> {
    let mut br = BufReader::with_capacity(256 * 1024, reader);
    let mut line = String::new();
    let mut total_docs = 0u64;
    let mut total_chars = 0u64;
    let mut buf = Vec::with_capacity(256);

    loop {
        if stop.load(Ordering::Relaxed)
            || source_written.load(Ordering::Relaxed) >= source_cap
            || bucket_written.load(Ordering::Relaxed) >= bucket_cap
        {
            break;
        }
        line.clear();
        let n = br.read_line(&mut line)?;
        if n == 0 {
            break;
        }
        let s = line.trim();
        if s.is_empty() {
            continue;
        }
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(s) {
            let txt = val
                .get(text_col)
                .and_then(|v| v.as_str())
                .or_else(|| val.get("text").and_then(|v| v.as_str()))
                .or_else(|| val.get("content").and_then(|v| v.as_str()));
            if let Some(t) = txt {
                let trimmed = t.trim();
                if keep_doc(trimmed, junk) {
                    let c = trimmed.len() as u64;
                    if source_written.load(Ordering::Relaxed) + total_chars >= source_cap
                        || bucket_written.load(Ordering::Relaxed) + total_chars >= bucket_cap
                    {
                        break;
                    }
                    buf.push(trimmed.to_string());
                    total_docs += 1;
                    total_chars += c;
                    if buf.len() >= 256 && !send_batch(tx, source_idx, bucket, &mut buf, stop) {
                        break;
                    }
                }
            }
        }
    }
    let _ = send_batch(tx, source_idx, bucket, &mut buf, stop);
    Ok((total_docs, total_chars))
}

fn write_parquet_shard(docs: &[String], path: &Path) -> Result<()> {
    let tmp_path = path.with_extension("parquet.tmp");
    let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
    let array = StringArray::from_iter_values(docs.iter().map(|s| s.as_str()));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)])?;
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let file = File::create(&tmp_path)?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    fs::rename(&tmp_path, path)?;
    Ok(())
}

fn merge_val_parts(output_dir: &Path, val_parts: &[PathBuf]) -> Result<()> {
    if val_parts.is_empty() {
        return Ok(());
    }
    let final_path = output_dir.join("val_shard.parquet");
    let tmp_path = output_dir.join("val_shard.parquet.tmp");
    let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let file = File::create(&tmp_path)?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    for p in val_parts {
        if !p.exists() {
            continue;
        }
        let f = File::open(p)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(f)?.build()?;
        for batch in reader {
            writer.write(&batch?)?;
        }
        let _ = fs::remove_file(p);
    }
    writer.close()?;
    fs::rename(&tmp_path, &final_path)?;
    Ok(())
}

fn load_local_cyber_docs(repo_dir: &Path, junk: &AhoCorasick) -> Vec<String> {
    let dirs = [
        "data/synthetic-ir",
        "data/synthetic-soc",
        "data/reverse-engineering",
        "data/cloud_nl",
        "data/log_nl",
    ];
    let mut docs = Vec::new();
    for rel in dirs {
        let dir = repo_dir.join(rel);
        let Ok(entries) = fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            let Ok(f) = File::open(&p) else { continue };
            for line in BufReader::new(f).lines().flatten() {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
                    if let Some(t) = v.get("text").and_then(|x| x.as_str()) {
                        let trimmed = t.trim();
                        if keep_doc(trimmed, junk) {
                            docs.push(trimmed.to_string());
                        }
                    }
                }
            }
        }
    }
    docs
}

fn ensure_climbmix_min_shards(client: &Client, token: Option<&str>, base_dir: &Path) {
    let dir = base_dir.join("base_data_climbmix");
    let _ = fs::create_dir_all(&dir);
    let existing = fs::read_dir(&dir)
        .map(|rd| {
            rd.flatten()
                .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("parquet"))
                .count()
        })
        .unwrap_or(0);
    if existing >= 2 {
        return;
    }
    println!("[CLIMBMIX] Downloading 2 minimal shards to {}...", dir.display());
    for idx in [0usize, 6542usize] {
        let fname = format!("shard_{idx:05}.parquet");
        let dst = dir.join(&fname);
        if dst.exists() {
            continue;
        }
        let url = format!("https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main/{fname}");
        if let Ok(b) = download_bytes(client, token, &url) {
            let _ = fs::write(&dst, &b);
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let t0 = Instant::now();

    let home = PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".to_string()));
    let base_dir = std::env::var("MESOSFER_BASE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| home.join(".cache/mesosfer"));
    let output_dir = args
        .output_dir
        .clone()
        .unwrap_or_else(|| base_dir.join("base_data_cybersecurity"));
    let repo_dir = args
        .repo_dir
        .clone()
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    if args.fresh && output_dir.exists() {
        for e in fs::read_dir(&output_dir)?.flatten() {
            let p = e.path();
            if p.is_file() {
                let _ = fs::remove_file(p);
            }
        }
    }
    fs::create_dir_all(&output_dir)?;

    let token = load_hf_token();
    println!(
        "=== fast_prepare (Rust) ===\n  Target: {} tokens (~{:.1} GB raw text)\n  Output: {}\n  Workers: {} download/parse + {} Parquet writers | HF token: {}",
        args.max_tokens,
        (args.max_tokens as f64 * CHARS_PER_TOKEN) / 1e9,
        output_dir.display(),
        args.workers,
        args.writer_threads,
        if token.is_some() { "present" } else { "none (public only)" }
    );

    let client = Client::builder()
        .pool_max_idle_per_host(args.workers * 2)
        .timeout(Duration::from_secs(300))
        .connect_timeout(Duration::from_secs(20))
        .build()?;

    ensure_climbmix_min_shards(&client, token.as_deref(), &base_dir);

    let max_chars = (args.max_tokens as f64 * CHARS_PER_TOKEN) as u64;
    let shard_max_chars = args.shard_mb * 1_000_000;
    let specs = default_sources();

    // Parallel file-list discovery across all sources
    println!("[DISCOVERY] Listing remote files across {} HuggingFace sources...", specs.len());
    let mut handles = Vec::new();
    for spec in specs {
        let c = client.clone();
        let t = token.clone();
        handles.push(thread::spawn(move || {
            let files = list_hf_files(&c, t.as_deref(), spec.repo, spec.prefix);
            (spec, files)
        }));
    }

    let mut rng = StdRng::seed_from_u64(42);
    let mut states = Vec::new();
    for h in handles {
        let (spec, res) = h.join().expect("discovery thread panicked");
        match res {
            Ok(mut files) if !files.is_empty() => {
                // Sort then deterministic shuffle so every run has diverse shard coverage
                files.sort_by(|a, b| a.0.cmp(&b.0));
                files.shuffle(&mut rng);
                println!("  + {:24} {:5} files ({})", spec.name, files.len(), spec.bucket.name());
                states.push(SourceState {
                    chars_cap: (spec.max_tokens as f64 * CHARS_PER_TOKEN) as u64,
                    spec,
                    files: VecDeque::from(files),
                    chars_est_dispatched: 0,
                    chars_actual: 0,
                    docs_actual: 0,
                });
            }
            Ok(_) => eprintln!("  ! {:24} 0 files found (skipping)", spec.name),
            Err(e) => eprintln!("  ! {:24} unavailable ({e})", spec.name),
        }
    }

    let num_sources = states.len();
    let source_caps: Arc<Vec<u64>> = Arc::new(states.iter().map(|s| s.chars_cap).collect());
    let source_written: Arc<Vec<AtomicU64>> = Arc::new((0..num_sources).map(|_| AtomicU64::new(0)).collect());
    let bucket_caps: [u64; 4] = [
        (max_chars as f64 * Bucket::General.target_share()) as u64,
        (max_chars as f64 * Bucket::Code.target_share()) as u64,
        (max_chars as f64 * Bucket::Cybersecurity.target_share()) as u64,
        (max_chars as f64 * Bucket::Reasoning.target_share()) as u64,
    ];
    let bucket_written: Arc<[AtomicU64; 4]> = Arc::new([
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
    ]);

    let mut bucket_target_chars = HashMap::new();
    for b in Bucket::ALL {
        bucket_target_chars.insert(b, bucket_caps[b.idx()]);
    }

    let scheduler = Arc::new(Mutex::new(Scheduler {
        sources: states,
        bucket_target_chars,
        bucket_dispatched_chars: HashMap::new(),
        bucket_actual_chars: HashMap::new(),
    }));

    let junk = Arc::new(build_junk_filter());
    let stop = Arc::new(AtomicBool::new(false));
    let (doc_tx, doc_rx): (Sender<DocBatch>, Receiver<DocBatch>) = bounded(args.workers * 4);

    // Seed local cybersecurity JSONL docs immediately
    let local_docs = load_local_cyber_docs(&repo_dir, &junk);
    if !local_docs.is_empty() {
        let lchars: u64 = local_docs.iter().map(|s| s.len() as u64).sum();
        println!("  + local_cyber_files        {:5} docs (~{:.2}M tok)", local_docs.len(), lchars as f64 / 4e6);
        let _ = doc_tx.send(DocBatch {
            source_idx: usize::MAX,
            bucket: Bucket::Cybersecurity,
            docs: local_docs,
        });
    }

    // Spawn Parquet writer pool
    let (write_tx, write_rx): (Sender<(Vec<String>, PathBuf)>, Receiver<(Vec<String>, PathBuf)>) =
        bounded(args.writer_threads * 2);
    let mut writer_handles = Vec::new();
    for _ in 0..args.writer_threads {
        let rx = write_rx.clone();
        writer_handles.push(thread::spawn(move || -> Result<()> {
            for (docs, path) in rx {
                write_parquet_shard(&docs, &path)?;
            }
            Ok(())
        }));
    }
    drop(write_rx);

    // Spawn parallel download + parse workers
    let mut worker_handles = Vec::new();
    for _ in 0..args.workers {
        let sched = scheduler.clone();
        let c = client.clone();
        let t = token.clone();
        let j = junk.clone();
        let tx = doc_tx.clone();
        let st = stop.clone();
        let s_caps = source_caps.clone();
        let s_wr = source_written.clone();
        let b_wr = bucket_written.clone();
        worker_handles.push(thread::spawn(move || {
            while !st.load(Ordering::Relaxed) {
                let task = {
                    let mut guard = sched.lock().unwrap();
                    guard.next_task(&b_wr, &s_wr)
                };
                let Some(task) = task else {
                    // Wait briefly if other buckets are still completing in-flight files
                    if b_wr.iter().enumerate().all(|(i, a)| a.load(Ordering::Relaxed) >= bucket_caps[i]) {
                        break;
                    }
                    thread::sleep(Duration::from_millis(200));
                    let has_any = {
                        let guard = sched.lock().unwrap();
                        guard.sources.iter().enumerate().any(|(i, s)| {
                            !s.files.is_empty()
                                && s_wr[i].load(Ordering::Relaxed) < s.chars_cap
                                && b_wr[s.spec.bucket.idx()].load(Ordering::Relaxed) < bucket_caps[s.spec.bucket.idx()]
                        })
                    };
                    if !has_any {
                        break;
                    }
                    continue;
                };
                let url = format!(
                    "https://huggingface.co/datasets/{}/resolve/main/{}",
                    task.repo, task.path
                );
                let s_cap = s_caps[task.source_idx];
                let b_cap = bucket_caps[task.bucket.idx()];
                match download_bytes(&c, t.as_deref(), &url) {
                    Ok(bytes) => {
                        let res = if task.path.ends_with(".parquet") {
                            parse_parquet_bytes(
                                bytes, task.source_idx, task.text_col, task.bucket,
                                s_cap, b_cap, &s_wr[task.source_idx], &b_wr[task.bucket.idx()],
                                &j, &tx, &st,
                            )
                        } else if task.path.ends_with(".gz") {
                            let gz = MultiGzDecoder::new(Cursor::new(bytes));
                            parse_jsonl_reader(
                                gz, task.source_idx, task.text_col, task.bucket,
                                s_cap, b_cap, &s_wr[task.source_idx], &b_wr[task.bucket.idx()],
                                &j, &tx, &st,
                            )
                        } else {
                            parse_jsonl_reader(
                                Cursor::new(bytes), task.source_idx, task.text_col, task.bucket,
                                s_cap, b_cap, &s_wr[task.source_idx], &b_wr[task.bucket.idx()],
                                &j, &tx, &st,
                            )
                        };
                        match res {
                            Ok((docs, chars)) => {
                                let mut guard = sched.lock().unwrap();
                                guard.report_actual(task.source_idx, task.bucket, task.est_chars, docs, chars);
                            }
                            Err(e) => {
                                eprintln!("  [WARN] Parse failed {} ({}): {e}", task.source_name, task.path);
                                let mut guard = sched.lock().unwrap();
                                guard.report_actual(task.source_idx, task.bucket, task.est_chars, 0, 0);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("  [WARN] Download failed {} ({}): {e}", task.source_name, task.path);
                        let mut guard = sched.lock().unwrap();
                        guard.report_actual(task.source_idx, task.bucket, task.est_chars, 0, 0);
                    }
                }
            }
        }));
    }
    drop(doc_tx);

    // Main interleaver + shard flusher loop
    let mut cur_docs: Vec<String> = Vec::with_capacity(args.shard_docs);
    let mut cur_chars: usize = 0;
    let mut shard_idx: usize = 0;
    let mut val_parts: Vec<PathBuf> = Vec::new();
    let mut total_docs: u64 = 0;
    let mut total_chars: u64 = 0;
    let mut last_log = Instant::now();

    let flush_shard = |docs: &mut Vec<String>,
                       chars: &mut usize,
                       s_idx: &mut usize,
                       v_parts: &mut Vec<PathBuf>,
                       w_tx: &Sender<(Vec<String>, PathBuf)>|
     -> Result<()> {
        if docs.is_empty() {
            return Ok(());
        }
        let mut s_rng = StdRng::seed_from_u64(42 + *s_idx as u64);
        docs.shuffle(&mut s_rng);
        let val_n = ((docs.len() as f64) * args.val_ratio).round() as usize;
        let val_n = val_n.min(docs.len().saturating_sub(1));
        let val_docs: Vec<String> = docs.split_off(docs.len() - val_n);
        let train_docs = std::mem::take(docs);
        *chars = 0;

        let train_path = output_dir.join(format!("shard_{:05}.parquet", *s_idx));
        *s_idx += 1;
        w_tx.send((train_docs, train_path))
            .map_err(|e| anyhow!("Writer channel closed: {e}"))?;

        if !val_docs.is_empty() {
            let vp = output_dir.join(format!("val_part_{:05}.valpart", v_parts.len()));
            v_parts.push(vp.clone());
            w_tx.send((val_docs, vp))
                .map_err(|e| anyhow!("Writer channel closed: {e}"))?;
        }
        Ok(())
    };

    for batch in doc_rx {
        let b_idx = batch.bucket.idx();
        let b_cap = bucket_caps[b_idx];
        let s_idx = batch.source_idx;
        let s_cap = if s_idx < num_sources { source_caps[s_idx] } else { u64::MAX };

        for doc in batch.docs {
            let dlen = doc.len() as u64;
            if bucket_written[b_idx].load(Ordering::Relaxed) >= b_cap {
                break;
            }
            if s_idx < num_sources && source_written[s_idx].load(Ordering::Relaxed) >= s_cap {
                break;
            }
            if total_chars + dlen > max_chars {
                stop.store(true, Ordering::Relaxed);
                break;
            }
            cur_docs.push(doc);
            cur_chars += dlen as usize;
            total_docs += 1;
            total_chars += dlen;
            bucket_written[b_idx].fetch_add(dlen, Ordering::Relaxed);
            if s_idx < num_sources {
                source_written[s_idx].fetch_add(dlen, Ordering::Relaxed);
            }

            if cur_docs.len() >= args.shard_docs || cur_chars >= shard_max_chars {
                flush_shard(&mut cur_docs, &mut cur_chars, &mut shard_idx, &mut val_parts, &write_tx)?;
            }
        }
        if stop.load(Ordering::Relaxed) {
            break;
        }
        if last_log.elapsed() >= Duration::from_secs(10) {
            let elapsed = t0.elapsed().as_secs_f64().max(0.1);
            let est_tok = total_chars as f64 / CHARS_PER_TOKEN;
            let pct = 100.0 * (total_chars as f64) / (max_chars as f64);
            let tok_s = est_tok / elapsed;
            let eta_min = ((args.max_tokens as f64 - est_tok).max(0.0) / tok_s.max(1.0)) / 60.0;
            println!(
                "[PROGRESS] shards={:4} | docs={:10} | ~{:.2}B / {:.2}B tok ({:5.1}%) | {:.2}M tok/s | gen={:.1}B code={:.1}B cyber={:.1}B math={:.1}B | ETA ~{:.1}m",
                shard_idx,
                total_docs,
                est_tok / 1e9,
                args.max_tokens as f64 / 1e9,
                pct,
                tok_s / 1e6,
                bucket_written[0].load(Ordering::Relaxed) as f64 / 4e9,
                bucket_written[1].load(Ordering::Relaxed) as f64 / 4e9,
                bucket_written[2].load(Ordering::Relaxed) as f64 / 4e9,
                bucket_written[3].load(Ordering::Relaxed) as f64 / 4e9,
                eta_min
            );
            last_log = Instant::now();
        }
    }

    stop.store(true, Ordering::Relaxed);
    flush_shard(&mut cur_docs, &mut cur_chars, &mut shard_idx, &mut val_parts, &write_tx)?;
    drop(write_tx);

    for h in writer_handles {
        h.join().expect("writer thread panicked")?;
    }
    for h in worker_handles {
        let _ = h.join();
    }

    println!("[VAL] Merging {} validation parts into val_shard.parquet...", val_parts.len());
    merge_val_parts(&output_dir, &val_parts)?;

    // Write progress.json compatible with prepare_data.py --status
    let guard = scheduler.lock().unwrap();
    let mut stats_json = serde_json::Map::new();
    let mut completed = Vec::new();
    for (i, s) in guard.sources.iter().enumerate() {
        let actual_c = source_written[i].load(Ordering::Relaxed);
        if actual_c > 0 {
            completed.push(serde_json::Value::String(s.spec.name.to_string()));
            stats_json.insert(
                s.spec.name.to_string(),
                serde_json::json!({ "docs": s.docs_actual, "chars": actual_c }),
            );
        }
    }
    let progress = serde_json::json!({
        "completed_sources": completed,
        "requested_sources": completed,
        "next_shard_idx": shard_idx,
        "next_val_part_idx": 0,
        "stats": stats_json,
        "stage": "pretrain",
        "tool": "fast_prepare_rust",
    });
    fs::write(output_dir.join("progress.json"), serde_json::to_string_pretty(&progress)?)?;

    let mins = t0.elapsed().as_secs_f64() / 60.0;
    let est_tok = total_chars as f64 / CHARS_PER_TOKEN;
    println!("\n=== DONE in {:.2} min ===", mins);
    println!("  Shards: {} | Docs: {} | Est. Tokens: {:.2}B", shard_idx, total_docs, est_tok / 1e9);
    for b in Bucket::ALL {
        let c = bucket_written[b.idx()].load(Ordering::Relaxed) as f64;
        let bt = c / CHARS_PER_TOKEN;
        let share = if total_chars > 0 { 100.0 * c / (total_chars as f64) } else { 0.0 };
        println!("  - {:14}: ~{:6.2}B tok ({:4.1}%, target {:2.0}%)", b.name(), bt / 1e9, share, b.target_share() * 100.0);
    }
    Ok(())
}
