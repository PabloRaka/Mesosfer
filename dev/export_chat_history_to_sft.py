#!/usr/bin/env python3
"""
Export and distill pair-programming conversation history into a high-quality SFT dataset for Mesosfer.

Covers real-world AI engineering topics:
- SFT Data Mixture & Token Budgeting
- Balancing Chat Fluency with Autonomous Tool Calling (subagent, terminal, subnet, python)
- Overcoming Paranoia Bias / Scenario Overfitting (Mythos analysis)
- Sequence Length Scaling & GPU Throughput (MI300X ROCm / NVIDIA CUDA)
- Hugging Face Hub Checkpoint Management
- Benchmark Evaluation (ChatCORE, CyberMetric, SecBench, CodeMMLU)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRANSCRIPT = Path(r"C:\Users\Lenovo\.gemini\antigravity-ide\brain\c44310f7-1ea4-45c7-a533-edb08d51e809\.system_generated\logs\transcript.jsonl")
DEFAULT_OUTPUT = REPO_ROOT / "data" / "sft" / "chat_history_distilled_sft.jsonl"


# ---------------------------------------------------------------------------
# High Quality Distilled Conversation Pairs from Pair-Programming Session
# ---------------------------------------------------------------------------

CURATED_SESSION_CONVERSATIONS = [
    {
        "user": "Bagaimana cara menyeimbangkan SFT training agar model lancar ngobrol bahasa Indonesia sekaligus jago function/tool calling?",
        "assistant": (
            "Untuk melatih model agar seimbang antara kemampuan percakapan alami (*conversational fluency*) dan ketepatan pemanggilan alat (*tool calling*), terapkan strategi *Multi-Task SFT Data Mixture* berikut:\n\n"
            "1. **Jaga Fondasi Percakapan (Anchor Data)**:\n"
            "   - Sertakan dataset instruksi umum berkualitas tinggi seperti **Alpaca Cleaned Indonesian** (~50K rows) dan **Alpaca GPT-4 Indonesian** (~50K rows) dengan bobot 1–2 epoch.\n"
            "   - Tambahkan dataset multi-turn chat seperti **UltraChat** atau **OpenHermes** agar model memahami konteks dialog panjang.\n\n"
            "2. **Supervisi Tool Calling dengan Special Tokens**:\n"
            "   - Gunakan format token khusus `<|tool_start|>` dan `<|tool_end|>` untuk memisahkan argumen JSON dari teks percakapan biasa.\n"
            "   - Pastikan token tool output `<|output_start|>` dan `<|output_end|>` diberi mask loss 0 (tidak disupervisi), karena output alat berasal dari runtime environment saat inferensi.\n\n"
            "3. **Cegah Overfitting Kasus Ekstrem (Anti-Paranoia)**:\n"
            "   - Hindari oversampling dataset investigasi malware berlebih yang dapat menyebabkan model mengira setiap pertanyaan admin adalah serangan siber.\n"
            "   - Berikan contoh perintah sistem standar Linux (`df -h`, `free -h`, `uptime`, `ss -tulpn`) dengan output status normal.\n\n"
            "4. **Pendelegasian Subagent Spesifik**:\n"
            "   - Latih model mengenali kapan harus mendelegasikan tugas ke subagent spesifik (`code_auditor`, `soc_analyst`, `threat_intel`, `sysadmin`) dengan skema JSON yang valid."
        ),
    },
    {
        "user": "Apakah menaikkan max sequence length (misalnya dari 2048 ke 6144) membuat training LLM menjadi lebih lama?",
        "assistant": (
            "Ya, menaikkan `max_seq_len` (panjang konteks) secara umum akan meningkatkan beban komputasi dan memori dengan faktor-faktor berikut:\n\n"
            "1. **Kompleksitas Self-Attention**:\n"
            "   - Operasi *Self-Attention* standar berskala kuadratik $O(N^2)$ terhadap panjang urutan token $N$.\n"
            "   - Menikkan context dari 2.048 ke 6.144 ($3\\times$) membutuhkan komputasi matriks attention hingga $\\approx 9\\times$ lebih banyak pada layer attention murni (meskipun Flash Attention 2/3 memitigasi overhead memori I/O secara signifikan).\n\n"
            "2. **Konsumsi Memori VRAM / KV Cache**:\n"
            "   - Aktivasi tensor dan KV cache per sequence menjadi 3x lebih besar. Untuk menghindari Out-Of-Memory (OOM), biasanya `device_batch_size` perlu diturunkan dan diimbangi dengan *gradient accumulation steps*.\n\n"
            "3. **Token Packing & Throughput**:\n"
            "   - Jika dataset Anda terdiri dari dokumen panjang (misal log firewall ribuan baris atau kode sumber lengkap), sequence length 6.144 sangat penting agar konteks tidak terpotong.\n"
            "   - Namun jika mayoritas dataset Anda pendek (< 1.000 token), menaikkan `max_seq_len` tanpa *sample packing* efisien akan membuang komputasi pada token padding.\n\n"
            "**Rekomendasi**: Gunakan `max_seq_len=6144` jika melatih model untuk analisis log, code auditing, dan multi-turn tool traces dengan Flash Attention aktif."
        ),
    },
    {
        "user": "Mengapa model AI bisa mengalami bias paranoid atau selalu mendeteksi malware pada perintah Linux normal?",
        "assistant": (
            "Bias paranoid pada LLM terjadi akibat **ketidakseimbangan distribusi skenario dalam dataset SFT (*Scenario Overfitting*)**:\n\n"
            "### Penyebab Utama:\n"
            "1. **Dominasi Kasus Serangan/Malware**: Jika dataset pelatihan didominasi oleh skenario forensik (di mana setiap kali ada pengecekan sistem selalu ditemukan backdoor, trojan di `/tmp`, atau port 4444 terbuka), model mempelajari prior probabilitas yang salah: *P(Malware | Cek Sistem) ≈ 1.0*.\n"
            "2. **Kurangnya Contoh Baseline Positif (Normalitas)**: Model tidak cukup sering melihat skenario di mana sistem berjalan sehat (`df -h` menghasilkan 25% used, `free -h` normal, `ss -tulpn` hanya port standar).\n\n"
            "### Solusi Perbaikan:\n"
            "1. **Hapus atau Nolkan Epoch Dataset Skenario Paranoid**: Matikan dataset sintetis bertema kompromi ekstrem yang di-oversample berulang.\n"
            "2. **Injeksi Data Administrasi Sistem Sehat**: Tambahkan data SFT perintah harian Linux dengan kesimpulan bahwa sistem beroperasi optimal.\n"
            "3. **Refusal Presisi**: Latih model menolak tindakan ofensif berbahaya (*port scan tanpa izin*) secara tegas tanpa menganggap permintaan defensif/edukasi sebagai ancaman."
        ),
    },
    {
        "user": "Bagaimana cara mengatasi error FileNotFoundError: tokenizer.pkl saat menjalankan script di remote server?",
        "assistant": (
            "Error `FileNotFoundError: /root/.cache/mesosfer/tokenizer/tokenizer.pkl` terjadi karena file tokenizer belum diunduh ke direktori cache lokal server.\n\n"
            "Untuk mengatasinya, unduh artefak tokenizer dari Hugging Face Hub dengan perintah:\n"
            "```bash\n"
            "python scripts/download_artifacts_from_hf.py --artifact tokenizer\n"
            "```\n\n"
            "Perintah ini akan otomatis mengunduh `tokenizer.pkl` dan `token_bytes.pt` ke direktori `~/.cache/mesosfer/tokenizer/` sehingga seluruh pipeline training dan evaluasi dapat berjalan normal."
        ),
    },
    {
        "user": "Bagaimana cara mengunggah checkpoint model SFT ke Hugging Face Hub secara efisien tanpa membuang bandwidth untuk optimizer?",
        "assistant": (
            "Untuk mengunggah model ke Hugging Face Hub dengan cepat, lewati state optimizer (*AdamW/Muon momentum buffers*) yang berukuran besar menggunakan flag `--model-only`:\n\n"
            "```bash\n"
            "python scripts/upload_checkpoint_to_hf.py \\\n"
            "    --source sft \\\n"
            "    --depth d12 \\\n"
            "    --latest \\\n"
            "    --model-only\n"
            "```\n\n"
            "Flag `--model-only` hanya mengunggah file bobot model (`model_XXXXXX.pt`) dan metadata konfigurasi (`meta_XXXXXX.json`), menghemat 60–75% ukuran file dan mempercepat proses upload."
        ),
    },
    {
        "user": "Jelaskan metrik ChatCORE dan bagaimana cara menghitung performa terpusat (centered accuracy) pada evaluasi LLM.",
        "assistant": (
            "**ChatCORE** adalah metrik evaluasi agregat yang mengukur performa model chat di berbagai kapabilitas (*ARC-Easy, ARC-Challenge, MMLU, GSM8K, HumanEval, SpellingBee*).\n\n"
            "### Formula Centered Accuracy:\n"
            "Untuk tugas pilihan ganda (*Multiple Choice* dengan baseline tebakan acak misal $25\\%$ atau $0.25$):\n"
            "$$\\text{Centered Accuracy} = \\frac{\\text{Accuracy} - \\text{Baseline}}{1.0 - \\text{Baseline}}$$\n\n"
            "**Kelebihan Metrik Ini:**\n"
            "- Nilai `0.00` merepresentasikan performa setara tebakan acak (*random baseline*).\n"
            "- Nilai `1.00` merepresentasikan akurasi sempurna ($100\\%$).\n"
            "- Memungkinkan perbandingan yang adil antara benchmark pilihan ganda (4 opsi) dan benchmark generatif terbuka (*open-ended* seperti HumanEval/GSM8K dengan baseline 0%)."
        ),
    },
]


def extract_from_transcript(transcript_path: Path) -> list[dict]:
    """Parse transcript.jsonl for clean real-world Q&A pairs."""
    if not transcript_path.exists():
        return []

    with transcript_path.open("r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    pairs = []
    current_user = None
    current_assistant_chunks = []

    for l in lines:
        stype = l.get("type")
        content = l.get("content", "")
        if stype == "USER_INPUT":
            if current_user and current_assistant_chunks:
                asst_text = "\n".join([c.strip() for c in current_assistant_chunks if c.strip()])
                if len(asst_text) > 40 and not asst_text.startswith("Created At:"):
                    # Validate user text is meaningful
                    if not current_user.startswith("@[TerminalName") and len(current_user) > 10:
                        pairs.append({
                            "role_user": current_user,
                            "role_assistant": asst_text
                        })

            u_match = re.search(r"<USER_REQUEST>(.*?)</USER_REQUEST>", content, re.DOTALL)
            u_text = u_match.group(1).strip() if u_match else content.strip()

            if "CHECKPOINT" in u_text:
                parts = re.split(r"<USER_REQUEST>", u_text)
                if len(parts) > 1:
                    u_text = parts[-1].replace("</USER_REQUEST>", "").strip()

            current_user = u_text
            current_assistant_chunks = []
        elif stype == "PLANNER_RESPONSE":
            if content and not content.startswith("Created At:"):
                current_assistant_chunks.append(content)

    if current_user and current_assistant_chunks:
        asst_text = "\n".join([c.strip() for c in current_assistant_chunks if c.strip()])
        if len(asst_text) > 40 and not current_user.startswith("@[TerminalName"):
            pairs.append({
                "role_user": current_user,
                "role_assistant": asst_text
            })

    # Format into SFT conversation format
    sft_rows = []
    for p in pairs:
        # filter out pure raw logs
        if "HTTP Request:" in p["role_user"] or "Traceback (most recent call last):" in p["role_user"]:
            continue
        sft_rows.append([
            {"role": "user", "content": p["role_user"]},
            {"role": "assistant", "content": p["role_assistant"]}
        ])

    return sft_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and distill chat history into SFT dataset")
    parser.add_argument("--transcript", type=Path, default=DEFAULT_TRANSCRIPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    all_rows = []

    # 1. Add curated engineering conversations
    for conv in CURATED_SESSION_CONVERSATIONS:
        all_rows.append([
            {"role": "user", "content": conv["user"]},
            {"role": "assistant", "content": conv["assistant"]}
        ])

    # 2. Add extracted real session Q&A
    extracted = extract_from_transcript(args.transcript)
    all_rows.extend(extracted)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Successfully exported {len(all_rows)} distilled SFT conversations to {args.output}")


if __name__ == "__main__":
    main()
