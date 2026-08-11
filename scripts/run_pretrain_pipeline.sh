#!/usr/bin/env bash
# =============================================================================
# run_pretrain_pipeline.sh
#
# Strategy: Download tiap batch source -> proses jadi Parquet shard ->
#           upload ke HF -> hapus raw HF cache -> lanjut batch berikutnya.
#
# Ini menghemat disk CPU VM: raw cache HF dihapus tiap selesai 1 batch.
# GPU VM nanti tinggal download shards yang sudah jadi dari HF.
#
# Usage:
#   bash scripts/run_pretrain_pipeline.sh
#   bash scripts/run_pretrain_pipeline.sh --dry-run   # cek apa yang akan dijalankan
#   bash scripts/run_pretrain_pipeline.sh --resume 3  # mulai dari batch ke-3
# =============================================================================

set -euo pipefail

# --- Konfigurasi -------------------------------------------------------------

DEPTH_FLAG="--d16"                          # ubah ke --d12, --d20, dll sesuai target
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
OUTPUT_DIR="base_data_cybersecurity"        # harus konsisten dengan download script

DRY_RUN=false
RESUME_FROM=1

# --- Parse args ---------------------------------------------------------------
for arg in "$@"; do
    case $arg in
        --dry-run)   DRY_RUN=true ;;
        --resume=*)  RESUME_FROM="${arg#*=}" ;;
        --resume)    shift; RESUME_FROM="${1:-1}" ;;
    esac
done

# --- Source Batches -----------------------------------------------------------
# Dikelompokkan dari yang terkecil ke terbesar supaya disk tidak jebol duluan.
# source "local_*" hanya berhasil jika ada file di data/ folder -- skip otomatis
# jika foldernya kosong (prepare_data akan warn tapi tidak crash).

declare -a BATCH_NAMES=(
    "batch1_general_wiki"
    "batch2_general_fineweb"
    "batch3_cyber_local_advisory"
    "batch4_cyber_hf_heavy"
    "batch5_code_secure_systems"
    "batch6_code_web_exploit"
    "batch7_instruction_math"
)

declare -a BATCH_SOURCES=(
    # Batch 1: Wikipedia EN + ID (ringan, cepat)
    "wikipedia,wikipedia_id"

    # Batch 2: FineWeb-Edu + FineWeb2-ID + ClimbMix (besar, streaming)
    "fineweb_edu,fineweb2_id,climbmix"

    # Batch 3: Cybersecurity local + advisory + detection rules (sedang)
    "local_incident_response,local_soc_synthetic,local_reverse_engineering,local_cloud_security,local_security_logs,all_cve_records,nvd_cve,github_advisory,cisa_kev,mitre_attack_stix,sigmahq_rules,elastic_rules,splunk_rules,zeek_scripts"

    # Batch 4: Primus heavy HF datasets + CIRCL (sangat besar, butuh disk ekstra)
    "primus_nemotron_cc,primus_fineweb,primus_seed,circl_vuln_patch,brightdata_cybersec"

    # Batch 5: Secure code - bahasa systems (sedang-besar)
    "secure_code_python,secure_code_c,secure_code_cpp,secure_code_rust,secure_code_go,secure_code_shell,swallow_code_v2"

    # Batch 6: Secure code - bahasa web + exploit frameworks (sedang)
    "secure_code_javascript,secure_code_typescript,secure_code_java,secure_code_php,code_sql,code_powershell,code_assembly,code_csharp,code_jupyter,metasploit,exploitdb"

    # Batch 7: Instruction & Math (sedang)
    "finemath,nemotron_cc_math"
)

# --- Helpers ------------------------------------------------------------------

log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
warn() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $*" >&2; }

run() {
    if $DRY_RUN; then
        echo "[DRY-RUN] $*"
    else
        "$@"
    fi
}

disk_free_gb() {
    df -BG "${HF_CACHE}" 2>/dev/null | awk 'NR==2{gsub("G","",$4); print $4}' || echo "?"
}

# --- Pre-flight checks --------------------------------------------------------

log "=== Mesosfer Pretrain Data Pipeline ==="
log "Depth flag : $DEPTH_FLAG"
log "Output dir : $OUTPUT_DIR"
log "HF cache   : $HF_CACHE"
log "Dry run    : $DRY_RUN"
log "Resume from: batch $RESUME_FROM"
log ""

if [[ "${#BATCH_NAMES[@]}" -ne "${#BATCH_SOURCES[@]}" ]]; then
    echo "ERROR: BATCH_NAMES dan BATCH_SOURCES harus punya jumlah elemen yang sama." >&2
    exit 1
fi

TOTAL_BATCHES="${#BATCH_NAMES[@]}"

# --- Main loop ----------------------------------------------------------------

for i in "${!BATCH_NAMES[@]}"; do
    BATCH_NUM=$((i + 1))
    BATCH_NAME="${BATCH_NAMES[$i]}"
    SOURCES="${BATCH_SOURCES[$i]}"

    if [[ $BATCH_NUM -lt $RESUME_FROM ]]; then
        log "Skipping $BATCH_NAME (batch $BATCH_NUM/$TOTAL_BATCHES) -- resuming from $RESUME_FROM"
        continue
    fi

    echo ""
    log "================================================"
    log "BATCH $BATCH_NUM/$TOTAL_BATCHES: $BATCH_NAME"
    log "Sources  : $SOURCES"
    log "Disk free: $(disk_free_gb) GB"
    log "================================================"

    # -- Step 1: Prepare data (download + shard ke Parquet) ------------------
    log "Step 1: prepare_data untuk $BATCH_NAME ..."
    run python -m scripts.data.prepare_data \
        $DEPTH_FLAG \
        --sources "$SOURCES"

    log "OK: prepare_data selesai untuk $BATCH_NAME"

    # -- Step 2: Upload shards ke HuggingFace --------------------------------
    log "Step 2: Upload $OUTPUT_DIR ke HuggingFace ..."
    run python -m scripts.upload_checkpoint_to_hf \
        --artifact dataset \
        --public \
        --dataset-dirs "$OUTPUT_DIR"

    log "OK: Upload selesai untuk $BATCH_NAME"

    # -- Step 3: Hapus raw HF dataset cache (bebaskan disk) ------------------
    log "Step 3: Bersihkan HF dataset cache ..."
    HF_DATASETS_CACHE="${HF_CACHE}/datasets"
    if [[ -d "$HF_DATASETS_CACHE" ]]; then
        log "   Menghapus: $HF_DATASETS_CACHE"
        run rm -rf "$HF_DATASETS_CACHE"
        log "OK: HF dataset cache dihapus"
    else
        log "   Tidak ada cache HF datasets untuk dihapus"
    fi

    # -- Step 4 (Opsional): Hapus juga local Parquet shards ------------------
    # Shards sudah aman di HF. Uncomment kalau disk masih penuh:
    #
    # MESOSFER_CACHE="${mesosfer_BASE_DIR:-$HOME/.cache/mesosfer}"
    # LOCAL_SHARDS="$MESOSFER_CACHE/$OUTPUT_DIR"
    # if [[ -d "$LOCAL_SHARDS" ]]; then
    #     log "Step 4: Hapus local shards $LOCAL_SHARDS ..."
    #     run rm -rf "$LOCAL_SHARDS"
    #     log "OK: Local shards dihapus"
    # fi

    log "Disk free after cleanup: $(disk_free_gb) GB"
    log "SELESAI: Batch $BATCH_NUM/$TOTAL_BATCHES ($BATCH_NAME)"
done

echo ""
log "================================================"
log "SEMUA BATCH SELESAI!"
log ""
log "GPU VM sekarang bisa download dari HF:"
log "  python -m scripts.download_artifacts_from_hf \\"
log "    --artifact dataset \\"
log "    --dataset-dirs $OUTPUT_DIR"
log ""
log "Lanjut training:"
log "  torchrun --standalone --nproc_per_node=<N_GPU> \\"
log "    -m scripts.train.base_train -- \\"
log "    --depth=16 --aspect-ratio=128 \\"
log "    --data-dir=\$HOME/.cache/mesosfer/$OUTPUT_DIR"
log "================================================"
