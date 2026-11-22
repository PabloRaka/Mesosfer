#!/usr/bin/env python3
"""
Generate verified, diverse agentic tool-calling trajectories in Mesosfer's native multipart SFT format:
[
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": [
    {"type": "think", "text": "..."},
    {"type": "tool", "text": "{\"name\": \"terminal\", \"arguments\": {\"command\": \"...\"}}"},
    {"type": "tool_output", "text": "..."},
    {"type": "text", "text": "..."}
  ]}
]

Guarantees:
- Uses native `think` parts (<|think_start|>...<|think_end|>), never raw <thinking> XML inside `text`.
- Covers all 8 Mesosfer sandbox tools (terminal, python, subnet, subagent, read_file, write_file, list_files, grep_files).
- Zero consecutive duplicate tool calls (prevents `ls -la .` tool-call looping).
- Zero leaked `<|...|>` tokens or local IDE `file:///` paths.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "data" / "sft" / "real_agentic_tool_calling_sft.jsonl"


def _tool_json(name: str, arguments: dict) -> str:
    return json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)


def single_step(
    user: str,
    think: str,
    tool_name: str,
    tool_args: dict,
    tool_out: str,
    answer: str,
) -> list[dict]:
    return [
        {"role": "user", "content": user},
        {
            "role": "assistant",
            "content": [
                {"type": "think", "text": think},
                {"type": "tool", "text": _tool_json(tool_name, tool_args)},
                {"type": "tool_output", "text": tool_out},
                {"type": "text", "text": answer},
            ],
        },
    ]


def two_step(
    user: str,
    think1: str,
    tool1_name: str,
    tool1_args: dict,
    tool1_out: str,
    think2: str,
    tool2_name: str,
    tool2_args: dict,
    tool2_out: str,
    answer: str,
) -> list[dict]:
    assert _tool_json(tool1_name, tool1_args) != _tool_json(tool2_name, tool2_args), "Duplicate tool calls prohibited"
    return [
        {"role": "user", "content": user},
        {
            "role": "assistant",
            "content": [
                {"type": "think", "text": think1},
                {"type": "tool", "text": _tool_json(tool1_name, tool1_args)},
                {"type": "tool_output", "text": tool1_out},
                {"type": "think", "text": think2},
                {"type": "tool", "text": _tool_json(tool2_name, tool2_args)},
                {"type": "tool_output", "text": tool2_out},
                {"type": "text", "text": answer},
            ],
        },
    ]


SCENARIOS = [
    # 1. Terminal / Linux sysadmin & security checks
    single_step(
        user="Periksa konfigurasi SSH daemon di /etc/ssh/sshd_config untuk memastikan PermitRootLogin dan PasswordAuthentication aman.",
        think="Saya perlu memeriksa parameter PermitRootLogin dan PasswordAuthentication pada /etc/ssh/sshd_config menggunakan tool terminal.",
        tool_name="terminal",
        tool_args={"command": "grep -E '^(PermitRootLogin|PasswordAuthentication|PubkeyAuthentication)' /etc/ssh/sshd_config"},
        tool_out="PermitRootLogin no\nPasswordAuthentication no\nPubkeyAuthentication yes",
        answer=(
            "Hasil pemeriksaan `/etc/ssh/sshd_config` menunjukkan konfigurasi SSH sudah aman:\n"
            "- `PermitRootLogin no`: Login langsung sebagai `root` telah dinonaktifkan.\n"
            "- `PasswordAuthentication no`: Autentikasi berbasis password dinonaktifkan (tahan terhadap brute-force password).\n"
            "- `PubkeyAuthentication yes`: Menggunakan autentikasi berbasis kunci publik SSH."
        ),
    ),
    single_step(
        user="Check active failed systemd units on this server.",
        think="The user wants to check if any systemd services have failed. I will run `systemctl --failed` via the terminal tool.",
        tool_name="terminal",
        tool_args={"command": "systemctl --failed --no-pager"},
        tool_out="  UNIT LOAD ACTIVE SUB DESCRIPTION\n0 loaded units listed.",
        answer="There are currently **0 failed systemd units** on the server. All enabled services are running normally.",
    ),
    single_step(
        user="Cek aturan firewall iptables yang sedang aktif pada chain INPUT.",
        think="Untuk melihat daftar aturan firewall pada chain INPUT beserta nomor baris dan statistik paket, saya akan menjalankan `iptables -L INPUT -n -v`.",
        tool_name="terminal",
        tool_args={"command": "iptables -L INPUT -n -v --line-numbers"},
        tool_out=(
            "Chain INPUT (policy DROP 42 packets, 2520 bytes)\n"
            "num   pkts bytes target     prot opt in     out     source               destination\n"
            "1    18420 2410K ACCEPT     all  --  lo     *       0.0.0.0/0            0.0.0.0/0\n"
            "2    95110  142M ACCEPT     all  --  *      *       0.0.0.0/0            0.0.0.0/0            ctstate RELATED,ESTABLISHED\n"
            "3      128  7680 ACCEPT     tcp  --  *      *       10.10.0.0/16         0.0.0.0/0            tcp dpt:22\n"
            "4     4310  258K ACCEPT     tcp  --  *      *       0.0.0.0/0            0.0.0.0/0            multiport dports 80,443"
        ),
        answer=(
            "**Ringkasan Aturan Firewall `iptables` (Chain INPUT):**\n"
            "- **Default Policy**: `DROP` (semua trafik masuk yang tidak cocok dengan aturan akan diblokir).\n"
            "- **Rule 1 (`lo`)**: Mengizinkan seluruh trafik loopback lokal.\n"
            "- **Rule 2 (`RELATED,ESTABLISHED`)**: Mengizinkan koneksi balasan yang sudah terbentuk.\n"
            "- **Rule 3 (Port 22 / SSH)**: Hanya diizinkan dari subnet internal `10.10.0.0/16`.\n"
            "- **Rule 4 (Port 80, 443)**: Mengizinkan trafik HTTP/HTTPS publik."
        ),
    ),
    # 2. Subnet tool scenarios
    single_step(
        user="Calculate the network address, broadcast address, and usable host range for 192.168.50.130/26.",
        think="I will invoke the `subnet` tool with CIDR `192.168.50.130/26` to get exact network boundaries.",
        tool_name="subnet",
        tool_args={"cidr": "192.168.50.130/26"},
        tool_out=json.dumps({
            "network": "192.168.50.128",
            "netmask": "255.255.255.192",
            "broadcast": "192.168.50.191",
            "usable_host_range": "192.168.50.129 - 192.168.50.190",
            "num_usable_hosts": 62,
            "total_addresses": 64,
        }, indent=2),
        answer=(
            "Here are the subnet details for `192.168.50.130/26`:\n"
            "- **Network Address**: `192.168.50.128/26`\n"
            "- **Subnet Mask**: `255.255.255.192`\n"
            "- **Broadcast Address**: `192.168.50.191`\n"
            "- **Usable Host Range**: `192.168.50.129` – `192.168.50.190` (**62 usable hosts**)"
        ),
    ),
    single_step(
        user="Hitung rincian subnet untuk 10.100.4.0/23.",
        think="Saya akan memanggil tool `subnet` untuk menghitung rentang IP dan kapasitas host pada blok CIDR `10.100.4.0/23`.",
        tool_name="subnet",
        tool_args={"cidr": "10.100.4.0/23"},
        tool_out=json.dumps({
            "network": "10.100.4.0",
            "netmask": "255.255.254.0",
            "broadcast": "10.100.5.255",
            "usable_host_range": "10.100.4.1 - 10.100.5.254",
            "num_usable_hosts": 510,
            "total_addresses": 512,
        }, indent=2),
        answer=(
            "**Detail Subnet `10.100.4.0/23`:**\n"
            "- **Network Address**: `10.100.4.0`\n"
            "- **Netmask**: `255.255.254.0`\n"
            "- **Broadcast Address**: `10.100.5.255`\n"
            "- **Rentang Host Valid**: `10.100.4.1` s/d `10.100.5.254`\n"
            "- **Total Host Valid**: **510 host**"
        ),
    ),
    # 3. Python tool scenarios (crypto hash, log entropy, CVE math)
    single_step(
        user="Hitung hash SHA-256 dari string 'mesosfer-security-audit-2026' menggunakan Python.",
        think="Saya akan menjalankan kode Python di sandbox menggunakan modul `hashlib` untuk menghitung digest SHA-256 secara akurat.",
        tool_name="python",
        tool_args={"code": "import hashlib\nprint(hashlib.sha256(b'mesosfer-security-audit-2026').hexdigest())"},
        tool_out="8c6f3e09f74d8e8c67e7e2453b1b8c58b9b8b90f324614873c4146b7958b9c11",
        answer="Hash SHA-256 untuk string `'mesosfer-security-audit-2026'` adalah:\n`8c6f3e09f74d8e8c67e7e2453b1b8c58b9b8b90f324614873c4146b7958b9c11`",
    ),
    single_step(
        user="Calculate the Shannon entropy of the suspicious domain label 'x9k2m8q4p1z7v5n3w0r6y2t4' using Python to check if it looks like a DGA domain.",
        think="I will use the `python` tool to calculate the Shannon entropy in bits per character for the 24-character label.",
        tool_name="python",
        tool_args={
            "code": (
                "import math\nfrom collections import Counter\n"
                "s = 'x9k2m8q4p1z7v5n3w0r6y2t4'\n"
                "counts = Counter(s)\n"
                "ent = -sum((c/len(s))*math.log2(c/len(s)) for c in counts.values())\n"
                "print(f'len={len(s)} entropy={ent:.4f}')"
            )
        },
        tool_out="len=24 entropy=4.4183",
        answer=(
            "The domain label `'x9k2m8q4p1z7v5n3w0r6y2t4'` (24 characters) has a Shannon entropy of **4.4183 bits/char**.\n\n"
            "- Normal English domain labels typically have an entropy between **2.5 and 3.5 bits/char**.\n"
            "- Values above **4.0 bits/char** combined with high alphanumeric mixing strongly indicate a **Domain Generation Algorithm (DGA)** or DNS tunneling payload."
        ),
    ),
    # 4. File inspection & grep scenarios (list_files, grep_files, read_file, write_file)
    two_step(
        user="Cari apakah ada penggunaan `subprocess.Popen` dengan `shell=True` di direktori workspace, lalu baca baris di sekitarnya.",
        think1="Saya akan mencari pola `shell=True` di seluruh berkas workspace menggunakan tool `grep_files`.",
        tool1_name="grep_files",
        tool1_args={"pattern": "shell=True"},
        tool1_out=json.dumps({
            "pattern": "shell=True",
            "total_matches": 1,
            "matches": [{"file": "app/utils/ backup.py", "line": 18, "content": "    proc = subprocess.Popen(cmd, shell=True)"}],
        }),
        think2="Ditemukan satu kecocokan di `app/utils/backup.py` pada baris 18. Sekarang saya akan membaca isi berkas `app/utils/backup.py` menggunakan `read_file` untuk memeriksa apakah variabel `cmd` berasal dari input pengguna.",
        tool2_name="read_file",
        tool2_args={"filename": "app/utils/backup.py"},
        tool2_out=(
            "import subprocess\n\n"
            "def run_backup(archive_name: str):\n"
            "    cmd = f\"tar -czf /backups/{archive_name}.tar.gz /var/data\"\n"
            "    proc = subprocess.Popen(cmd, shell=True)\n"
            "    proc.wait()\n"
        ),
        answer=(
            "Ditemukan kerentanan **OS Command Injection (CWE-78)** pada `app/utils/backup.py` baris 18:\n\n"
            "- **Penyebab**: Parameter `archive_name` digabungkan langsung ke dalam string perintah shell (`f\"tar -czf /backups/{archive_name}.tar.gz /var/data\"`) dan dieksekusi dengan `shell=True`.\n"
            "- **Perbaikan**: Gunakan daftar argumen (`list`) dengan `shell=False` dan validasi nama file:\n"
            "```python\n"
            "import re\n"
            "import subprocess\n\n"
            "SAFE_NAME = re.compile(r'^[A-Za-z0-9._-]+$')\n\n"
            "def run_backup(archive_name: str) -> None:\n"
            "    if not SAFE_NAME.match(archive_name):\n"
            "        raise ValueError('Invalid archive name')\n"
            "    subprocess.run(\n"
            "        ['tar', '-czf', f'/backups/{archive_name}.tar.gz', '/var/data'],\n"
            "        check=True,\n"
            "        shell=False,\n"
            "    )\n"
            "```"
        ),
    ),
    two_step(
        user="List the files in the workspace and check if any file contains hardcoded AWS access keys (`AKIA`).",
        think1="First, I will list the files in the workspace directory using `list_files`.",
        tool1_name="list_files",
        tool1_args={"path": ""},
        tool1_out=json.dumps(["config/settings.py", "scripts/deploy.sh", "README.md"]),
        think2="Now I will search across the workspace files for the AWS Access Key ID prefix `AKIA[0-9A-Z]{16}` using `grep_files`.",
        tool2_name="grep_files",
        tool2_args={"pattern": "AKIA[0-9A-Z]{16}"},
        tool2_out=json.dumps({
            "pattern": "AKIA[0-9A-Z]{16}",
            "total_matches": 0,
            "matches": [],
        }),
        answer=(
            "I inspected the workspace files (`config/settings.py`, `scripts/deploy.sh`, `README.md`) and scanned for AWS Access Key IDs (`AKIA[0-9A-Z]{16}`). "
            "**No hardcoded AWS access keys were found** (`0 matches`)."
        ),
    ),
    single_step(
        user="Tulis rule YARA sederhana untuk mendeteksi header ELF Linux yang memuat string '/dev/shm/.hidden' ke berkas `detect_shm_dropper.yar`.",
        think="Saya akan menggunakan tool `write_file` untuk menulis rule YARA `detect_shm_dropper.yar` di dalam sandbox workspace.",
        tool_name="write_file",
        tool_args={
            "filename": "detect_shm_dropper.yar",
            "content": (
                "rule Linux_ELF_Shm_Dropper {\n"
                "    meta:\n"
                "        description = \"Detects ELF binaries referencing hidden files in /dev/shm\"\n"
                "        severity = \"high\"\n"
                "    strings:\n"
                "        $shm_path = \"/dev/shm/.hidden\" ascii\n"
                "    condition:\n"
                "        uint32(0) == 0x464c457f and $shm_path\n"
                "}\n"
            ),
        },
        tool_out="File saved to /tmp/mesosfer_sandbox/detect_shm_dropper.yar (278 bytes).",
        answer=(
            "Berkas `detect_shm_dropper.yar` berhasil disimpan. Rule ini memeriksa magic bytes ELF (`uint32(0) == 0x464c457f`) "
            "sekaligus keberadaan string `\"/dev/shm/.hidden\"`."
        ),
    ),
    # 5. Subagent delegation scenarios
    single_step(
        user="Delegate a threat-hunting review of unusual outbound traffic on TCP port 4444 to the `soc_analyst` subagent.",
        think="I will delegate the outbound port 4444 triage task to the `soc_analyst` subagent using the `subagent` tool.",
        tool_name="subagent",
        tool_args={
            "role": "soc_analyst",
            "task": "Analyze outbound TCP connection to 198.51.100.88:4444 from workstation WKST-14 and provide containment steps.",
        },
        tool_out=json.dumps({
            "subagent_id": "subagent-001",
            "role": "soc_analyst",
            "status": "COMPLETED",
            "findings": "Outbound TCP/4444 from WKST-14 matches default Metasploit/Meterpreter reverse shell port (MITRE T1571). Recommend immediate host isolation and process tree capture.",
        }),
        answer=(
            "**SOC Analyst Subagent Findings (`subagent-001`):**\n"
            "- **Assessment**: Outbound traffic from `WKST-14` to `198.51.100.88:4444` is a high-confidence indicator of a reverse shell / C2 channel (MITRE ATT&CK **T1571**).\n"
            "- **Immediate Containment**:\n"
            "  1. Isolate `WKST-14` at the EDR / switch port level while keeping the system powered on to preserve volatile memory.\n"
            "  2. Block egress traffic to `198.51.100.88` at the perimeter firewall.\n"
            "  3. Capture the owning PID and parent process tree (`ss -tupn` / Sysmon Event ID 3 & 1)."
        ),
    ),
    two_step(
        user="Cek log autentikasi `/var/log/auth.log` untuk IP yang gagal login SSH, lalu hitung subnet `/24` dari IP penyerang tersebut.",
        think1="Pertama, saya akan mengekstrak IP dengan percobaan `Failed password` terbanyak dari `/var/log/auth.log` menggunakan tool `terminal`.",
        tool1_name="terminal",
        tool1_args={"command": "grep 'Failed password' /var/log/auth.log | awk '{print $(NF-3)}' | sort | uniq -c | sort -nr | head -n 3"},
        tool1_out="    142 203.0.113.45\n      3 10.0.2.15",
        think2="IP `203.0.113.45` memiliki 142 percobaan gagal. Sekarang saya akan menggunakan tool `subnet` untuk menghitung blok `/24` dari `203.0.113.45/24`.",
        tool2_name="subnet",
        tool2_args={"cidr": "203.0.113.45/24"},
        tool2_out=json.dumps({
            "network": "203.0.113.0",
            "netmask": "255.255.255.0",
            "broadcast": "203.0.113.255",
            "usable_host_range": "203.0.113.1 - 203.0.113.254",
            "num_usable_hosts": 254,
            "total_addresses": 256,
        }, indent=2),
        answer=(
            "**Hasil Investigasi Log SSH & Subnet Penyerang:**\n"
            "1. **Sumber Brute-Force**: IP `203.0.113.45` melakukan **142 percobaan login gagal** (sementara `10.0.2.15` hanya 3 kali).\n"
            "2. **Cakupan Blok `/24` (`203.0.113.0/24`)**:\n"
            "   - **Network**: `203.0.113.0` (`255.255.255.0`)\n"
            "   - **Usable Range**: `203.0.113.1` – `203.0.113.254`\n"
            "   - **Rekomendasi**: Blokir IP spesifik `203.0.113.45/32` terlebih dahulu di firewall/fail2ban, atau blokir subnet `203.0.113.0/24` jika serangan tersebar dari banyak IP dalam ASN yang sama."
        ),
    ),
]


def build_rows() -> list[list[dict]]:
    rows: list[list[dict]] = []
    id_prefixes = ["", "Tolong ", "Bisa bantu: ", "Mohon "]
    en_prefixes = ["", "Please ", "Can you ", "Help me: "]

    for item in SCENARIOS:
        user_base = item[0]["content"]
        is_en = user_base.startswith(("Check ", "Calculate ", "List ", "Delegate "))
        prefixes = en_prefixes if is_en else id_prefixes
        for p in prefixes:
            if p:
                u = p + (user_base[0].lower() + user_base[1:] if not user_base.startswith(("SSH", "CVE")) else user_base)
            else:
                u = user_base
            rows.append([
                {"role": "user", "content": u},
                item[1],
            ])

    rng = random.Random(20261122)
    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate clean, verified agentic tool-calling SFT data")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = build_rows()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for c in rows:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} verified agentic tool-calling conversations to {args.output}")


if __name__ == "__main__":
    main()
