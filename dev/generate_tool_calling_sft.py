#!/usr/bin/env python3
"""
Generate tool-calling SFT data for Mesosfer.

This dataset teaches the model to:
1. Recognize when a tool call is needed
2. Emit structured tool calls using <|python_start|> / <|python_end|> tokens
3. Incorporate tool output from <|output_start|> / <|output_end|> into its response
4. Refuse offensive tool usage (port scanning for attacks, etc.)

The tool-calling format uses mesosfer's existing special tokens:
  - <|python_start|>  code here  <|python_end|>   (model writes code to execute)
  - <|output_start|>  result here  <|output_end|>  (runtime returns result)

All examples use documentation-range IPs (203.0.113.x, 198.51.100.x) and
reserved domains (.example, .invalid, .test) to avoid real-world harm.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "data" / "sft" / "tool_calling_conversations_en.jsonl"


# ---------------------------------------------------------------------------
# Helper: build a multi-part assistant message with tool calls
# ---------------------------------------------------------------------------

def text_part(content: str) -> dict:
    return {"type": "text", "text": content}


def python_part(code: str) -> dict:
    return {"type": "python", "text": code}


def output_part(result: str) -> dict:
    return {"type": "python_output", "text": result}


def make_tool_conversation(
    user: str,
    thinking: str,
    code: str,
    tool_output: str,
    conclusion: str,
) -> list[dict]:
    """Build a single-turn tool-calling conversation."""
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": [
            text_part(thinking),
            python_part(code),
            output_part(tool_output),
            text_part(conclusion),
        ]},
    ]


def make_multi_turn_tool_conversation(turns: list[dict]) -> list[dict]:
    """Build a multi-turn conversation from a list of turn dicts."""
    messages = []
    for turn in turns:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": [
            text_part(turn["thinking"]),
            python_part(turn["code"]),
            output_part(turn["output"]),
            text_part(turn["conclusion"]),
        ]})
    return messages


def make_simple_conversation(user: str, assistant: str) -> list[dict]:
    """Build a simple text-only conversation (no tool call)."""
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


# ---------------------------------------------------------------------------
# Tool-calling examples: defensive cybersecurity tools
# ---------------------------------------------------------------------------

WHOIS_EXAMPLES = [
    {
        "user": "Look up the WHOIS information for example.com",
        "thinking": "<thinking>\nThe user wants WHOIS data for a domain. I can use the whois command to retrieve registration information.\n</thinking>\n",
        "code": "import subprocess\nresult = subprocess.run(['whois', 'example.com'], capture_output=True, text=True, timeout=10)\nprint(result.stdout[:2000])",
        "output": "Domain Name: EXAMPLE.COM\nRegistry Domain ID: 2336799_DOMAIN_COM-VRSN\nRegistrar: RESERVED-Internet Assigned Numbers Authority\nCreation Date: 1995-08-14T04:00:00Z\nRegistry Expiry Date: 2025-08-13T04:00:00Z\nRegistrar: RESERVED-IANA\nDomain Status: clientDeleteProhibited\nName Server: A.IANA-SERVERS.NET\nName Server: B.IANA-SERVERS.NET",
        "conclusion": "\n\nThe WHOIS lookup shows that `example.com` is a reserved IANA domain:\n- **Registrar**: Reserved by Internet Assigned Numbers Authority\n- **Created**: 1995-08-14\n- **Expiry**: 2025-08-13\n- **Name servers**: `A.IANA-SERVERS.NET`, `B.IANA-SERVERS.NET`\n\nThis is a documentation/example domain and not available for registration.",
    },
    {
        "user": "Can you check who registered the domain test.invalid?",
        "thinking": "<thinking>\nThe user wants WHOIS for a .invalid TLD domain. I'll run the lookup but .invalid is a reserved TLD that won't return real registration data.\n</thinking>\n",
        "code": "import subprocess\nresult = subprocess.run(['whois', 'test.invalid'], capture_output=True, text=True, timeout=10)\nprint(result.stdout[:1000] if result.stdout else f'Error: {result.stderr[:500]}')",
        "output": "Error: No whois server is known for this kind of object.",
        "conclusion": "\n\nThe `.invalid` TLD is reserved by RFC 2606 for testing purposes. No WHOIS server exists for it because it is intentionally non-resolvable. This is expected behavior — `.invalid` domains cannot be registered by anyone.",
    },
]

DNS_LOOKUP_EXAMPLES = [
    {
        "user": "Resolve the DNS records for example.com",
        "thinking": "<thinking>\nI'll use dig to query DNS records for this domain. I'll check A, MX, and NS records.\n</thinking>\n",
        "code": "import subprocess\nfor qtype in ['A', 'MX', 'NS']:\n    result = subprocess.run(['dig', '+short', qtype, 'example.com'], capture_output=True, text=True, timeout=10)\n    print(f'{qtype}: {result.stdout.strip()}')",
        "output": "A: 93.184.216.34\nMX: \nNS: a.iana-servers.net.\nb.iana-servers.net.",
        "conclusion": "\n\nDNS resolution results for `example.com`:\n- **A record**: `93.184.216.34` (IPv4 address)\n- **MX record**: None configured (no mail server)\n- **NS records**: `a.iana-servers.net`, `b.iana-servers.net`\n\nThe absence of MX records means this domain is not set up to receive email.",
    },
    {
        "user": "Check if suspicious-domain.example has any TXT records",
        "thinking": "<thinking>\nThe user wants to check TXT records for a suspicious domain. TXT records can reveal SPF/DKIM/DMARC configs or be used for C2 beaconing. I'll query them defensively.\n</thinking>\n",
        "code": "import subprocess\nresult = subprocess.run(['dig', '+short', 'TXT', 'suspicious-domain.example'], capture_output=True, text=True, timeout=10)\nprint(result.stdout.strip() if result.stdout.strip() else 'No TXT records found.')",
        "output": "No TXT records found.",
        "conclusion": "\n\nNo TXT records were found for `suspicious-domain.example`. In a real investigation, the absence of SPF/DKIM/DMARC records on a domain sending email would be a red flag. For DNS-based C2 detection, you would want to monitor for high-entropy TXT queries or unusually frequent lookups to unusual domains.",
    },
]

HASH_CHECK_EXAMPLES = [
    {
        "user": "Calculate the SHA256 hash of this text: 'Hello, World!'",
        "thinking": "<thinking>\nSimple hash calculation request. I'll use Python's hashlib to compute the SHA256 hash.\n</thinking>\n",
        "code": "import hashlib\ntext = 'Hello, World!'\nhash_value = hashlib.sha256(text.encode('utf-8')).hexdigest()\nprint(f'SHA256: {hash_value}')",
        "output": "SHA256: dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f",
        "conclusion": "\n\nThe SHA256 hash of `Hello, World!` is:\n```\ndffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f\n```\nYou can use this hash for integrity verification. In incident response, file hashes are commonly used as IOCs (Indicators of Compromise) to identify known malware samples.",
    },
    {
        "user": "I have a file hash: 0000000000000000000000000000000000000000000000000000000000000000. Can you check if it's a known malware hash?",
        "thinking": "<thinking>\nThe user wants to check a hash against known malware databases. The hash is all zeros, which is synthetic. I'll write code to query a local hash database or show how to check it.\n</thinking>\n",
        "code": "known_malware_hashes = {\n    'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855': 'empty file (SHA256 of zero bytes)',\n}\nhash_to_check = '0000000000000000000000000000000000000000000000000000000000000000'\nresult = known_malware_hashes.get(hash_to_check, 'Not found in local database')\nprint(f'Hash: {hash_to_check}')\nprint(f'Result: {result}')",
        "output": "Hash: 0000000000000000000000000000000000000000000000000000000000000000\nResult: Not found in local database",
        "conclusion": "\n\nThe hash `0000...0000` was not found in the local malware hash database. This appears to be a synthetic/placeholder hash.\n\nFor real investigations, you can check hashes against:\n- **VirusTotal** (online, requires API key)\n- **MalwareBazaar** (free, abuse.ch)\n- **Local YARA rules** matching known malware signatures\n- **NSRL** (National Software Reference Library) to exclude known-good files",
    },
]

LOG_PARSING_EXAMPLES = [
    {
        "user": "Parse this auth log and count failed SSH logins per IP:\n```\nMay 27 10:14:02 app01 sshd[2144]: Failed password for invalid user admin from 203.0.113.24 port 51102\nMay 27 10:14:09 app01 sshd[2147]: Failed password for root from 203.0.113.24 port 51108\nMay 27 10:14:18 app01 sshd[2151]: Failed password for deploy from 203.0.113.24 port 51119\nMay 27 10:15:42 app01 sshd[2163]: Failed password for admin from 198.51.100.50 port 43022\n```",
        "thinking": "<thinking>\nThe user provided SSH auth logs. I'll parse the failed login lines, extract source IPs, and count failures per IP.\n</thinking>\n",
        "code": "import re\nfrom collections import Counter\n\nlog_text = \"\"\"May 27 10:14:02 app01 sshd[2144]: Failed password for invalid user admin from 203.0.113.24 port 51102\nMay 27 10:14:09 app01 sshd[2147]: Failed password for root from 203.0.113.24 port 51108\nMay 27 10:14:18 app01 sshd[2151]: Failed password for deploy from 203.0.113.24 port 51119\nMay 27 10:15:42 app01 sshd[2163]: Failed password for admin from 198.51.100.50 port 43022\"\"\"\n\npattern = re.compile(r'Failed password .* from (\\d+\\.\\d+\\.\\d+\\.\\d+)')\ncounts = Counter()\nfor line in log_text.splitlines():\n    match = pattern.search(line)\n    if match:\n        counts[match.group(1)] += 1\n\nfor ip, count in counts.most_common():\n    print(f'{ip}: {count} failed attempts')",
        "output": "203.0.113.24: 3 failed attempts\n198.51.100.50: 1 failed attempts",
        "conclusion": "\n\n**Analysis results:**\n| Source IP | Failed Attempts | Risk Level |\n|---|---:|---|\n| 203.0.113.24 | 3 | High — multiple usernames targeted |\n| 198.51.100.50 | 1 | Low — single attempt |\n\n**Recommendations:**\n- Investigate `203.0.113.24` for brute-force activity (3 failures across different usernames suggests credential spraying)\n- Check if any successful logins followed from either IP\n- Consider adding `203.0.113.24` to a temporary block list",
    },
    {
        "user": "Extract all unique IPs from this firewall log:\n```\nFW allow tcp 10.0.4.12:51544 -> 198.51.100.77:443 bytes=9821\nFW deny tcp 203.0.113.50:44021 -> 10.0.4.12:22 bytes=0\nFW allow tcp 10.0.4.12:51545 -> 198.51.100.77:443 bytes=1234\nFW deny udp 203.0.113.99:53 -> 10.0.4.12:53 bytes=512\n```",
        "thinking": "<thinking>\nI need to extract all unique IP addresses from firewall log entries. I'll use a regex pattern to find all IPs.\n</thinking>\n",
        "code": "import re\n\nlog_text = \"\"\"FW allow tcp 10.0.4.12:51544 -> 198.51.100.77:443 bytes=9821\nFW deny tcp 203.0.113.50:44021 -> 10.0.4.12:22 bytes=0\nFW allow tcp 10.0.4.12:51545 -> 198.51.100.77:443 bytes=1234\nFW deny udp 203.0.113.99:53 -> 10.0.4.12:53 bytes=512\"\"\"\n\nips = set(re.findall(r'(\\d+\\.\\d+\\.\\d+\\.\\d+)', log_text))\nfor ip in sorted(ips):\n    print(ip)",
        "output": "10.0.4.12\n198.51.100.77\n203.0.113.50\n203.0.113.99",
        "conclusion": "\n\nFound **4 unique IP addresses** in the firewall log:\n- `10.0.4.12` — internal host (RFC 1918 private range)\n- `198.51.100.77` — external destination (documentation range)\n- `203.0.113.50` — external source, **denied SSH access** to internal host\n- `203.0.113.99` — external source, **denied DNS query** to internal host\n\nThe two denied connections from `203.0.113.x` to internal services (SSH port 22, DNS port 53) warrant further investigation.",
    },
]

NETWORK_RECON_DEFENSIVE = [
    {
        "user": "Check which ports are open on my local machine for a security audit",
        "thinking": "<thinking>\nThe user wants to audit their own machine's open ports. This is a legitimate defensive use case. I'll use ss (socket statistics) to list listening ports.\n</thinking>\n",
        "code": "import subprocess\nresult = subprocess.run(['ss', '-tlnp'], capture_output=True, text=True, timeout=10)\nprint(result.stdout[:2000])",
        "output": "State    Recv-Q   Send-Q   Local Address:Port   Peer Address:Port   Process\nLISTEN   0        128      0.0.0.0:22           0.0.0.0:*           users:((\"sshd\",pid=1234))\nLISTEN   0        128      127.0.0.1:5432       0.0.0.0:*           users:((\"postgres\",pid=5678))\nLISTEN   0        511      0.0.0.0:80           0.0.0.0:*           users:((\"nginx\",pid=9012))",
        "conclusion": "\n\n**Open ports on this machine:**\n| Port | Service | Binding | Risk Assessment |\n|---:|---|---|---|\n| 22 | SSH (sshd) | 0.0.0.0 (all interfaces) | Exposed — ensure key-based auth + fail2ban |\n| 5432 | PostgreSQL | 127.0.0.1 (localhost only) | Safe — not exposed externally |\n| 80 | HTTP (nginx) | 0.0.0.0 (all interfaces) | Exposed — ensure HTTPS redirect + WAF |\n\n**Recommendations:**\n- SSH on port 22 is accessible from all interfaces — consider restricting to specific IPs or using a VPN\n- PostgreSQL is correctly bound to localhost only\n- HTTP on port 80 should redirect to HTTPS (port 443) if serving web content",
    },
]

ENCODING_DECODING = [
    {
        "user": "Decode this base64 string: SGVsbG8gZnJvbSBNZXNvc2ZlciE=",
        "thinking": "<thinking>\nBase64 decoding is a common task in incident response (encoded payloads, obfuscated commands). I'll decode it safely.\n</thinking>\n",
        "code": "import base64\nencoded = 'SGVsbG8gZnJvbSBNZXNvc2ZlciE='\ndecoded = base64.b64decode(encoded).decode('utf-8', errors='replace')\nprint(f'Decoded: {decoded}')",
        "output": "Decoded: Hello from Mesosfer!",
        "conclusion": "\n\nThe decoded Base64 string is: **Hello from Mesosfer!**\n\nBase64 decoding is a critical skill in incident response — attackers often encode PowerShell commands, C2 payloads, and exfiltrated data in Base64 to evade detection. Always decode suspicious strings in a sandboxed environment.",
    },
    {
        "user": "Convert this hex string to ASCII: 4d616c776172652074657374",
        "thinking": "<thinking>\nHex-to-ASCII conversion. Common in malware analysis when examining hex dumps. I'll decode it safely.\n</thinking>\n",
        "code": "hex_string = '4d616c776172652074657374'\ndecoded = bytes.fromhex(hex_string).decode('utf-8', errors='replace')\nprint(f'Decoded: {decoded}')",
        "output": "Decoded: Malware test",
        "conclusion": "\n\nThe hex string decodes to: **Malware test**\n\nThis is harmless sample text. In real malware analysis, hex decoding is used to examine embedded strings, shellcode payloads, and obfuscated configuration data within binary samples.",
    },
]

# ---------------------------------------------------------------------------
# Multi-turn tool-calling conversations
# ---------------------------------------------------------------------------

MULTI_TURN_EXAMPLES = [
    # Investigation workflow: DNS → WHOIS → conclusion
    [
        {
            "user": "I found suspicious DNS queries to x9a2k3.example-c2.invalid from workstation wkst-22. Can you investigate?",
            "thinking": "<thinking>\nThe user reports suspicious DNS queries. Let me first try to resolve the domain to see if it has any DNS records.\n</thinking>\n",
            "code": "import subprocess\nresult = subprocess.run(['dig', '+short', 'A', 'x9a2k3.example-c2.invalid'], capture_output=True, text=True, timeout=10)\nprint(result.stdout.strip() if result.stdout.strip() else 'No A records found.')",
            "output": "No A records found.",
            "conclusion": "\n\nThe domain `x9a2k3.example-c2.invalid` has no A records. The `.invalid` TLD is reserved and should never resolve. However, the high-entropy subdomain pattern (`x9a2k3`) is characteristic of DNS-based C2 beaconing. Let me check WHOIS for this domain.",
        },
        {
            "user": "Yes, please check WHOIS too and give me a summary.",
            "thinking": "<thinking>\nI'll check WHOIS and then compile a full investigation summary.\n</thinking>\n",
            "code": "import subprocess\nresult = subprocess.run(['whois', 'example-c2.invalid'], capture_output=True, text=True, timeout=10)\nprint(result.stdout[:1000] if result.stdout.strip() else f'WHOIS error: {result.stderr[:500]}')",
            "output": "WHOIS error: No whois server is known for this kind of object.",
            "conclusion": "\n\n**Investigation Summary:**\n\n| Finding | Detail |\n|---|---|\n| Domain | `x9a2k3.example-c2.invalid` |\n| DNS A Record | None (domain does not resolve) |\n| WHOIS | No data (`.invalid` TLD has no registrar) |\n| Subdomain entropy | High (`x9a2k3` — random-looking) |\n| Source host | `wkst-22` |\n\n**Assessment**: The high-entropy subdomain pattern queried repeatedly from a single workstation is a **strong indicator of DNS-based C2 beaconing**, even though the domain does not resolve.\n\n**Recommended actions:**\n1. Isolate `wkst-22` from the network immediately\n2. Capture a full process list and network connections from the host\n3. Check EDR/AV logs for suspicious processes making DNS queries\n4. Block the domain pattern at the DNS resolver level\n5. Preserve all DNS query logs for forensic timeline analysis",
        },
    ],
]

# ---------------------------------------------------------------------------
# Tool-call refusals: offensive tool misuse
# ---------------------------------------------------------------------------

TOOL_REFUSAL = (
    "I cannot help with scanning, attacking, or exploiting systems you don't own. "
    "I can help with defensive alternatives such as auditing your own systems, "
    "parsing logs from authorized sources, checking file hashes, or analyzing "
    "synthetic artifacts for training purposes."
)

# ---------------------------------------------------------------------------
# Tool-calling examples: Subagent, Terminal, Subnet, Python, and File Ops
# ---------------------------------------------------------------------------

SUBAGENT_EXAMPLES = [
    {
        "user": "Delegasikan audit keamanan kode file auth.py ini ke subagent code_auditor.",
        "thinking": "<thinking>\nPengguna meminta pendelegasian review keamanan kode ke subagent code_auditor. Saya akan memanggil tool subagent dengan role code_auditor.\n</thinking>\n",
        "code": "import json\ncall = {'name': 'subagent', 'arguments': {'role': 'code_auditor', 'task': 'Audit auth.py for SQL injection, hardcoded credentials, and weak hashing.'}}\nprint(json.dumps(call))",
        "output": '{"subagent_id": "subagent-001", "role": "code_auditor", "status": "COMPLETED", "findings": "[Code Auditor] Audit selesai pada auth.py. Ditemukan 1 kerentanan password hashing menggunakan MD5 dan 1 potensi insecure direct object reference (IDOR)."}',
        "conclusion": "\n\n**Laporan Hasil Subagent (`code_auditor`):**\n- **Subagent ID**: `subagent-001`\n- **Status**: Selesai\n- **Temuan Keamanan**:\n  1. Penggunaan fungsi hashing lemah `MD5` untuk password (Rekomendasi: Ganti ke `argon2` atau `bcrypt`).\n  2. Potensi kerentanan IDOR pada pengecekan sesi pengguna.\n\nApakah Anda ingin saya membuatkan patch kode perbaikan untuk `auth.py`?",
    },
    {
        "user": "Tolong minta subagent soc_analyst untuk memeriksa anomali lonjakan traffic pada firewall log.",
        "thinking": "<thinking>\nPengguna meminta subagent soc_analyst untuk menganalisis log firewall. Saya akan memanggil tool subagent dengan role soc_analyst.\n</thinking>\n",
        "code": "import json\ncall = {'name': 'subagent', 'arguments': {'role': 'soc_analyst', 'task': 'Analyze firewall logs for traffic spikes and potential DDoS/syn flood patterns.'}}\nprint(json.dumps(call))",
        "output": '{"subagent_id": "subagent-002", "role": "soc_analyst", "status": "COMPLETED", "findings": "[SOC Analyst] Terdeteksi SYN flood dari 14 IP eksternal menuju port 443 pada pukul 00:14 UTC. Total 45,000 paket/detik. Disarankan aktivasi SYN cookies dan rate limiting di firewall."}',
        "conclusion": "\n\n**Analisis Insiden dari Subagent (`soc_analyst`):**\n- **Insiden**: Indikasi serangan *SYN Flood* pada port 443.\n- **Volume**: ~45.000 paket/detik dari 14 IP penyerang.\n- **Tindakan Mitigasi yang Direkomendasikan**:\n  1. Aktifkan modul `tcp_syncookies` pada kernel Linux (`sysctl -w net.ipv4.tcp_syncookies=1`).\n  2. Terapkan rule `iptables` rate-limiting pada koneksi baru port 443.\n  3. Masukkan 14 IP sumber ke blokir sementara (*drop list*).",
    },
]

TERMINAL_COMMAND_EXAMPLES = [
    {
        "user": "Periksa status service fail2ban di server lokal menggunakan terminal.",
        "thinking": "<thinking>\nPengguna meminta pemeriksaan status service fail2ban. Saya akan mengeksekusinya menggunakan tool terminal/bash.\n</thinking>\n",
        "code": "import subprocess\nres = subprocess.run(['systemctl', 'status', 'fail2ban'], capture_output=True, text=True, timeout=10)\nprint(res.stdout if res.stdout else res.stderr)",
        "output": "● fail2ban.service - Fail2Ban Service\n   Loaded: loaded (/lib/systemd/system/fail2ban.service; enabled; vendor preset: enabled)\n   Active: active (running) since Mon 2026-08-17 00:00:01 UTC; 50min ago\n  Process: 1102 ExecStart=/usr/bin/fail2ban-server -xf start (code=exited, status=0/SUCCESS)\n Main PID: 1105 (fail2ban-server)\n    Tasks: 5 (limit: 4915)\n   CGroup: /system.slice/fail2ban.service\n           └─1105 /usr/bin/python3 /usr/bin/fail2ban-server -xf start",
        "conclusion": "\n\n**Status Service Fail2Ban:**\n- **Status**: `active (running)` ✅\n- **PID**: 1105\n- **Uptime**: Berjalan normal sejak 50 menit yang lalu.\n- Service telah aktif melindungi server dari percobaan login SSH brute-force.",
    },
    {
        "user": "Jalankan perintah untuk melihat penggunaan disk di server.",
        "thinking": "<thinking>\nPengguna meminta informasi penggunaan disk. Saya akan menjalankan perintah `df -h` di sandbox.\n</thinking>\n",
        "code": "import subprocess\nres = subprocess.run(['df', '-h', '/'], capture_output=True, text=True, timeout=10)\nprint(res.stdout)",
        "output": "Filesystem      Size  Used Avail Use% Mounted on\n/dev/nvme0n1p2  1.8T  420G  1.3T  25% /",
        "conclusion": "\n\n**Penggunaan Disk Server:**\n- **Kapasitas Total**: `1.8 TB`\n- **Terpakai**: `420 GB` (25%)\n- **Tersedia**: `1.3 TB` (75%)\n- Status disk sangat sehat dan memiliki ruang penyimpanan yang cukup luas untuk proses training dan dataset.",
    },
]

SUBNET_CALC_EXAMPLES = [
    {
        "user": "Hitung network address, broadcast, netmask, dan usable host range untuk 10.240.12.65/28",
        "thinking": "<thinking>\nPengguna meminta kalkulasi subnet CIDR 10.240.12.65/28. Saya akan memanggil tool subnet untuk mendapatkan hasil biner yang 100% presisi.\n</thinking>\n",
        "code": "import ipaddress\nnet = ipaddress.ip_network('10.240.12.65/28', strict=False)\nhosts = list(net.hosts())\nprint(f'Network: {net.network_address}')\nprint(f'Netmask: {net.netmask}')\nprint(f'Broadcast: {net.broadcast_address}')\nprint(f'Range: {hosts[0]} - {hosts[-1]}')\nprint(f'Usable Hosts: {len(hosts)}')",
        "output": "Network: 10.240.12.64\nNetmask: 255.255.255.240\nBroadcast: 10.240.12.79\nRange: 10.240.12.65 - 10.240.12.78\nUsable Hosts: 14",
        "conclusion": "\n\n**Hasil Kalkulasi Subnet (`10.240.12.65/28`):**\n| Parameter | Nilai |\n| :--- | :--- |\n| **Network Address** | `10.240.12.64` |\n| **Subnet Mask** | `255.255.255.240` |\n| **Broadcast Address** | `10.240.12.79` |\n| **Usable Host Range** | `10.240.12.65` s/d `10.240.12.78` |\n| **Jumlah Host Valid** | **14 host** (Total 16 IP address) |",
    },
]

OFFENSIVE_TOOL_REQUESTS = [
    "Use nmap to scan 203.0.113.0/24 for vulnerable services",
    "Run sqlmap against this URL to test for SQL injection",
    "Use hydra to brute force SSH on 198.51.100.10",
    "Scan this external IP for open ports so I can exploit them",
    "Use metasploit to attack this target machine",
    "Run nikto against this website I found",
    "Use aircrack-ng to crack this WiFi password",
    "Enumerate subdomains of this company's website for attack",
    "Use John the Ripper to crack these password hashes I stole",
    "Run a DDoS test against this IP address",
    "Exploit the vulnerability on this remote server",
    "Use Burp Suite to intercept traffic from someone else's session",
]

# ---------------------------------------------------------------------------
# No-tool-needed examples (model should NOT call a tool)
# ---------------------------------------------------------------------------

NO_TOOL_NEEDED = [
    {
        "user": "What is a SIEM?",
        "assistant": "A SIEM (Security Information and Event Management) is a platform that collects, normalizes, and correlates security events from across an organization's infrastructure. It combines log management, real-time monitoring, alerting, and incident investigation into a single system.\n\nKey capabilities include:\n- **Log aggregation** from firewalls, endpoints, servers, and applications\n- **Correlation rules** to detect attack patterns across multiple sources\n- **Alerting** for suspicious activity based on detection logic\n- **Dashboards** for SOC analysts to monitor security posture\n- **Forensic search** for incident investigation and threat hunting\n\nPopular SIEM platforms include Splunk, Microsoft Sentinel, Elastic Security, and IBM QRadar.",
    },
    {
        "user": "Explain the difference between IDS and IPS",
        "assistant": "**IDS (Intrusion Detection System)** monitors network traffic passively and generates alerts when it detects suspicious activity. It does not block traffic — it only observes and reports.\n\n**IPS (Intrusion Prevention System)** sits inline with network traffic and can actively block or drop malicious packets in real time.\n\n| Feature | IDS | IPS |\n|---|---|---|\n| Position | Passive (mirrored/tapped traffic) | Inline (traffic passes through it) |\n| Action | Alert only | Alert + block/drop |\n| Latency impact | None | Small increase (inline inspection) |\n| Risk | May miss real-time blocking | False positives can disrupt legitimate traffic |\n\nIn practice, many modern solutions (like Suricata) can operate in both IDS and IPS modes depending on deployment configuration.",
    },
]


# ---------------------------------------------------------------------------
# Build all rows
# ---------------------------------------------------------------------------

def build_rows() -> list:
    rows: list = []
    prefixes = ["", "using a tool, ", "can you run a command to ", "please ", "help me "]

    # Single-turn tool-calling examples
    all_examples = (
        SUBAGENT_EXAMPLES
        + TERMINAL_COMMAND_EXAMPLES
        + SUBNET_CALC_EXAMPLES
        + WHOIS_EXAMPLES
        + DNS_LOOKUP_EXAMPLES
        + HASH_CHECK_EXAMPLES
        + LOG_PARSING_EXAMPLES
        + NETWORK_RECON_DEFENSIVE
        + ENCODING_DECODING
    )
    for example in all_examples:
        for prefix in prefixes:
            user_text = prefix + example["user"] if prefix else example["user"]
            rows.append(make_tool_conversation(
                user=user_text,
                thinking=example["thinking"],
                code=example["code"],
                tool_output=example["output"],
                conclusion=example["conclusion"],
            ))

    # Multi-turn tool-calling examples
    for turns in MULTI_TURN_EXAMPLES:
        rows.append(make_multi_turn_tool_conversation(turns))

    # Offensive tool refusals
    offensive_prefixes = ["", "for testing, ", "in my lab, ", "for education, ", "quick: "]
    for request in OFFENSIVE_TOOL_REQUESTS:
        for prefix in offensive_prefixes:
            rows.append(make_simple_conversation(prefix + request, TOOL_REFUSAL))

    # No-tool-needed examples (model should answer directly without tool calls)
    for example in NO_TOOL_NEEDED:
        for prefix in ["", "briefly: ", "explain: "]:
            rows.append(make_simple_conversation(
                prefix + example["user"] if prefix else example["user"],
                example["assistant"],
            ))

    rng = random.Random(20260527)
    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate tool-calling SFT data for Mesosfer"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = build_rows()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
