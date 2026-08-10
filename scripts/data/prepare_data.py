#!/usr/bin/env python3
"""
Prepare a high-quality, cybersecurity-focused pretraining dataset for mesosfer.
Version 3: Replaces gated/broken datasets with fully public alternatives.

Usage:
    python -m scripts.data.prepare_data
    python -m scripts.data.prepare_data --max-tokens 1000000000  # limit to 1B for quick test
    python -m scripts.data.prepare_data --sources project_zero,unit42  # specific sources only
    python -m scripts.data.prepare_data --status              # check download progress
    python -m scripts.data.prepare_data --fresh               # discard previous progress and restart
"""

import os
import sys
import json
import glob
import hashlib
import shutil
import tarfile
import subprocess
import argparse
import logging
import time
import gzip
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from collections import defaultdict
from collections.abc import Mapping
from fnmatch import fnmatch
from pathlib import Path

import pandas as pd
try:
    from datasets import load_dataset
except ModuleNotFoundError:
    load_dataset = None

if load_dataset is None:
    # The warning is repeated later when a HF source is hit, but we fail
    # immediately so the user can install the package before running a
    # multi‑hour pipeline.
    logger = logging.getLogger(__name__)
    logger.warning(
        "The 'datasets' package is not installed. Hugging Face dataset sources "
        "will be skipped. To use them, run: pip install datasets"
    )

try:
    from mesosfer.utils.common import get_base_dir
except ModuleNotFoundError:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    def get_base_dir():
        if os.environ.get("mesosfer_BASE_DIR"):
            mesosfer_dir = os.environ["mesosfer_BASE_DIR"]
        else:
            mesosfer_dir = os.path.join(os.path.expanduser("~"), ".cache", "mesosfer")
        os.makedirs(mesosfer_dir, exist_ok=True)
        return mesosfer_dir

logger = logging.getLogger(__name__)


def _check_free_space(path: Path, required_bytes: int) -> None:
    """Raise RuntimeError if free disk space below required_bytes."""
    try:
        stat = shutil.disk_usage(str(path.parent if path.is_file() else path))
        if stat.free < required_bytes:
            raise RuntimeError(
                f"Insufficient free disk space on {path}: "
                f"need {required_bytes / (1024**3):.2f} GiB, "
                f"only {stat.free / (1024**3):.2f} GiB available"
            )
    except OSError as e:
        logger.warning("Unable to check disk space for %s: %s", path, e)

# =============================================================================
# Load .env from the mesosfer project directory
# =============================================================================

def _load_env_file():
    """Load a ``.env`` file.

    The original implementation only looked two levels up from this file,
    which fails when the script is executed from a different working
    directory. We now search in two locations: the repository root and the
    current working directory. The first file found is loaded. If no file
    is present we simply continue – missing environment variables will be
    warned about later.
    """
    possible_paths = [
        Path(__file__).resolve().parents[2] / ".env",  # project root
        Path.cwd() / ".env",  # cwd (useful when invoking from repo root)
    ]
    env_path = next((p for p in possible_paths if p.is_file()), None)
    if env_path is None:
        logger.debug("No .env file found in project root or cwd")
        return
    logger.debug("Loading .env from %s", env_path)
    with env_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value

_load_env_file()

HF_TOKEN: str | None = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None

if HF_TOKEN and HF_TOKEN != "your_hugging_face_token_here":
    logger.info("✓ Hugging Face token loaded from environment / .env file")
else:
    HF_TOKEN = None
    logger.warning("⚠ No Hugging Face token found. Public datasets will still work.")

# =============================================================================
# Dataset2 security vocabulary and source schema
# =============================================================================

TIER_GOLD = "GOLD"
TIER_SILVER = "SILVER"
TIER_BRONZE_CLEAN = "BRONZE_CLEAN"

CAUSAL_MARKERS = (
    "allows attacker to",
    "allows remote attackers",
    "exploits",
    "bypasses",
    "leads to",
    "results in",
    "can be exploited",
    "triggered by",
    "due to",
)

MECHANISM_TERMS = (
    "buffer overflow",
    "heap overflow",
    "stack overflow",
    "use-after-free",
    "double free",
    "integer overflow",
    "format string",
    "rop chain",
    "race condition",
    "deserialization",
    "path traversal",
)

MARKETING_TERMS = (
    "book a demo",
    "free trial",
    "contact sales",
    "request a demo",
    "pricing",
    "customer story",
    "webinar",
)

SECURITY_STACK_TERMS = (
    "firewall",
    "siem",
    "soar",
    "edr",
    "xdr",
    "yara",
    "sigma",
    "zeek",
    "suricata",
    "ids",
    "ips",
)

SECURITY_CODE_TERMS = (
    "shellcode",
    "ghidra",
    "ida pro",
    "radare2",
    "syscall",
    "vtable",
    "gadget",
    "payload",
    "metasploit",
    "pwntools",
)


@dataclass(frozen=True)
class Source:
    name: str
    source_type: str
    domain: str
    primary_subdomain: str
    expected_tier: str
    estimated_tokens: int
    max_tokens: int
    description: str
    config: dict


def _scaled_tokens(base_tokens, scale):
    return max(1, int(base_tokens * scale))


def _source_definitions(target_tokens=500_000_000):
    scale = target_tokens / 500_000_000
    return (
        Source(
            name="project_zero",
            source_type="rss",
            domain="cybersecurity",
            primary_subdomain="prose_intel",
            expected_tier=TIER_GOLD,
            estimated_tokens=_scaled_tokens(30_000_000, scale),
            max_tokens=_scaled_tokens(40_000_000, scale),
            description="Google Project Zero vulnerability research feed",
            config={"url": "https://googleprojectzero.blogspot.com/feeds/posts/default"},
        ),
        Source(
            name="unit42",
            source_type="rss",
            domain="cybersecurity",
            primary_subdomain="prose_intel",
            expected_tier=TIER_GOLD,
            estimated_tokens=_scaled_tokens(30_000_000, scale),
            max_tokens=_scaled_tokens(40_000_000, scale),
            description="Palo Alto Unit 42 threat research feed",
            config={"url": "https://unit42.paloaltonetworks.com/feed/"},
        ),
        Source(
            name="mandiant",
            source_type="rss",
            domain="cybersecurity",
            primary_subdomain="prose_intel",
            expected_tier=TIER_GOLD,
            estimated_tokens=_scaled_tokens(30_000_000, scale),
            max_tokens=_scaled_tokens(40_000_000, scale),
            description="Mandiant threat intelligence research feed",
            config={"url": "https://www.mandiant.com/resources/blog/rss.xml"},
        ),
        Source(
            name="cloudflare_security",
            source_type="rss",
            domain="cybersecurity",
            primary_subdomain="prose_intel",
            expected_tier=TIER_SILVER,
            estimated_tokens=_scaled_tokens(20_000_000, scale),
            max_tokens=_scaled_tokens(30_000_000, scale),
            description="Cloudflare security and incident research feed",
            config={"url": "https://blog.cloudflare.com/tag/security/rss/"},
        ),
        Source(
            name="talos",
            source_type="rss",
            domain="cybersecurity",
            primary_subdomain="prose_intel",
            expected_tier=TIER_GOLD,
            estimated_tokens=_scaled_tokens(30_000_000, scale),
            max_tokens=_scaled_tokens(40_000_000, scale),
            description="Cisco Talos threat intelligence feed",
            config={"url": "https://blog.talosintelligence.com/rss/"},
        ),
        Source(
            name="dfir_report",
            source_type="rss",
            domain="cybersecurity",
            primary_subdomain="prose_intel",
            expected_tier=TIER_GOLD,
            estimated_tokens=_scaled_tokens(25_000_000, scale),
            max_tokens=_scaled_tokens(35_000_000, scale),
            description="The DFIR Report intrusion analysis feed",
            config={"url": "https://thedfirreport.com/feed/"},
        ),
        Source(
            name="academic_security_papers",
            source_type="pdf_manifest",
            domain="cybersecurity",
            primary_subdomain="prose_intel",
            expected_tier=TIER_GOLD,
            estimated_tokens=_scaled_tokens(80_000_000, scale),
            max_tokens=_scaled_tokens(120_000_000, scale),
            description="Top-tier security conference papers from local PDF manifests",
            config={"manifest_paths": ["data/manifests/security_papers*.jsonl"]},
        ),
        Source(
            name="threat_report_pdfs",
            source_type="pdf_manifest",
            domain="cybersecurity",
            primary_subdomain="prose_intel",
            expected_tier=TIER_SILVER,
            estimated_tokens=_scaled_tokens(60_000_000, scale),
            max_tokens=_scaled_tokens(90_000_000, scale),
            description="Vendor threat report PDFs from local manifests",
            config={"manifest_paths": ["data/manifests/threat_reports*.jsonl"]},
        ),
        Source(
            name="github_advisory",
            source_type="github_repo",
            domain="cybersecurity",
            primary_subdomain="advisories",
            expected_tier=TIER_GOLD,
            estimated_tokens=_scaled_tokens(60_000_000, scale),
            max_tokens=_scaled_tokens(100_000_000, scale),
            description="GitHub Security Advisory database",
            config={"repo": "github/advisory-database", "include": ["advisories/**/*.json"]},
        ),
        Source(
            name="cisa_kev",
            source_type="url_json",
            domain="cybersecurity",
            primary_subdomain="advisories",
            expected_tier=TIER_GOLD,
            estimated_tokens=_scaled_tokens(5_000_000, scale),
            max_tokens=_scaled_tokens(10_000_000, scale),
            description="CISA Known Exploited Vulnerabilities catalog",
            config={"url": "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"},
        ),
        Source(
            name="mitre_attack_stix",
            source_type="github_repo",
            domain="cybersecurity",
            primary_subdomain="advisories",
            expected_tier=TIER_GOLD,
            estimated_tokens=_scaled_tokens(40_000_000, scale),
            max_tokens=_scaled_tokens(70_000_000, scale),
            description="MITRE ATT&CK STIX knowledge base",
            config={"repo": "mitre-attack/attack-stix-data", "include": ["enterprise-attack/*.json"]},
        ),
        Source(
            name="sigmahq_rules",
            source_type="github_repo",
            domain="cybersecurity",
            primary_subdomain="detections",
            expected_tier=TIER_SILVER,
            estimated_tokens=_scaled_tokens(40_000_000, scale),
            max_tokens=_scaled_tokens(70_000_000, scale),
            description="SigmaHQ detection rules",
            config={"repo": "SigmaHQ/sigma", "include": ["rules/**/*.yml", "rules/**/*.yaml"]},
        ),
        Source(
            name="elastic_rules",
            source_type="github_repo",
            domain="cybersecurity",
            primary_subdomain="detections",
            expected_tier=TIER_SILVER,
            estimated_tokens=_scaled_tokens(40_000_000, scale),
            max_tokens=_scaled_tokens(70_000_000, scale),
            description="Elastic detection rules",
            config={"repo": "elastic/detection-rules", "include": ["rules/**/*.toml"]},
        ),
        Source(
            name="splunk_rules",
            source_type="github_repo",
            domain="cybersecurity",
            primary_subdomain="detections",
            expected_tier=TIER_SILVER,
            estimated_tokens=_scaled_tokens(30_000_000, scale),
            max_tokens=_scaled_tokens(60_000_000, scale),
            description="Splunk security detections",
            config={"repo": "splunk/security_content", "include": ["detections/**/*.yml"]},
        ),
        Source(
            name="zeek_scripts",
            source_type="github_repo",
            domain="cybersecurity",
            primary_subdomain="detections",
            expected_tier=TIER_SILVER,
            estimated_tokens=_scaled_tokens(20_000_000, scale),
            max_tokens=_scaled_tokens(40_000_000, scale),
            description="Zeek scripts and packages",
            config={"repo": "zeek/zeek", "include": ["scripts/**/*.zeek"]},
        ),
        Source(
            name="metasploit",
            source_type="github_repo",
            domain="code",
            primary_subdomain="exploit_code",
            expected_tier=TIER_SILVER,
            estimated_tokens=_scaled_tokens(100_000_000, scale),
            max_tokens=_scaled_tokens(180_000_000, scale),
            description="Metasploit Framework modules and exploit utilities",
            config={"repo": "rapid7/metasploit-framework", "include": ["modules/**/*.rb", "lib/**/*.rb"]},
        ),
        Source(
            name="exploitdb",
            source_type="github_repo",
            domain="code",
            primary_subdomain="exploit_code",
            expected_tier=TIER_SILVER,
            estimated_tokens=_scaled_tokens(80_000_000, scale),
            max_tokens=_scaled_tokens(140_000_000, scale),
            description="Exploit Database proof-of-concept corpus",
            config={"repo": "offensive-security/exploitdb", "include": ["exploits/**/*"]},
        ),
    )


def _source_to_dataset_config(source):
    config = dict(source.config)
    config.update(
        {
            "source_type": source.source_type,
            "description": source.description,
            "category": source.domain,
            "max_tokens": source.max_tokens,
            "expected_tier": source.expected_tier,
            "primary_subdomain": source.primary_subdomain,
            "estimated_tokens": source.estimated_tokens,
            "dataset_group": "dataset2",
        }
    )
    return config

# =============================================================================
# Domain sampling weights for temperature-based interleaved shuffling.
# A weight is a priority WITHIN its bucket only: it decides which sources of a
# bucket are drawn more often, never how large the bucket itself is. The size of
# each bucket is TARGET_DOMAIN_SHARE below.
#
# Guidelines (relative, inside a bucket):
#   2.0  = high priority (exploit, vuln patches, SOC-critical)
#   1.5  = elevated (structured cyber knowledge)
#   1.0  = normal (general code, math, instruction)
#   0.7  = reduced (wiki, general web — avoid dominating)
#   0.5  = low (telemetry, bulk text)
# =============================================================================

DOMAIN_SAMPLING_WEIGHTS = {
    # Cybersecurity
    "all_cve_records":         1.8,
    "circl_vuln_patch":        2.3,
    "primus_nemotron_cc":      2.1,
    "primus_fineweb":          1.9,
    "nvd_cve":                 1.8,
    "local_incident_response": 2.2,
    "local_soc_synthetic":     2.1,
    "local_reverse_engineering": 2.0,
    "local_cloud_security":    1.9,
    "local_security_logs":     1.9,
    "project_zero":            2.0,
    "unit42":                  1.9,
    "mandiant":                2.0,
    "cloudflare_security":     1.6,
    "talos":                   1.9,
    "dfir_report":             2.0,
    "academic_security_papers": 1.8,
    "threat_report_pdfs":      1.5,
    "github_advisory":         1.9,
    "cisa_kev":                2.0,
    "mitre_attack_stix":       1.8,
    "sigmahq_rules":           1.6,
    "elastic_rules":           1.6,
    "splunk_rules":            1.5,
    "zeek_scripts":            1.5,
    "brightdata_cybersec":     2.0,  # Real-time threat intel via BrightData Proxy/API
    "primus_seed":             2.2,  # Trend Micro curated cybersecurity seed corpus
    # General knowledge (English)
    "climbmix":                0.6,  # very large, keep proportion down
    "wikipedia":               0.5,  # huge corpus, don't let it dominate
    "fineweb_edu":             0.7,
    # Indonesian — its own share bucket (see TARGET_DOMAIN_SHARE), weights only split it.
    "wikipedia_id":            0.5,
    "fineweb2_id":             0.6,
    # Code — important for secure coding
    "secure_code_python":      1.4,
    "secure_code_c":           1.4,
    "secure_code_cpp":         1.2,
    "secure_code_rust":        1.2,
    "secure_code_go":          1.2,
    "secure_code_shell":       1.4,
    "secure_code_javascript":  1.3,
    "secure_code_typescript":  1.3,
    "secure_code_java":        1.3,
    "secure_code_php":         1.3,
    "code_sql":                1.5,  # injection is the most common vuln class
    "code_powershell":         1.6,  # attacker tooling on Windows, shows up in real logs
    "code_assembly":           1.6,  # reverse engineering, pairs with local_reverse_engineering
    "code_csharp":             1.2,
    "code_jupyter":            1.0,  # ML/AI code
    "metasploit":              1.7,
    "exploitdb":               1.8,
    "swallow_code_v2":         1.3,
    # Instruction / reasoning
    "finemath":                0.9,
    "nemotron_cc_math":        1.0,  # NVIDIA high-quality math pretraining (score 4+)
}

# =============================================================================
# Target share of the mixed corpus per bucket. This is the knob for "how much
# of each thing does the model see"; DOMAIN_SAMPLING_WEIGHTS only ranks sources
# inside a bucket. Shares are measured in characters written (≈ tokens) and are
# renormalized over whichever buckets are actually present in a run, so a
# --sources subset still gets a sane mix.
# =============================================================================

TARGET_DOMAIN_SHARE = {
    "general":       0.40,  # English general knowledge
    "code":          0.30,
    "cybersecurity": 0.10,
    "instruction":   0.10,  # math / reasoning
    "indonesian":    0.10,
}

# Indonesian sources are category "general" in DATASET_SOURCES; they get their own
# share bucket so the Indonesian slice does not eat the English general budget.
INDONESIAN_SOURCES = ("wikipedia_id", "fineweb2_id")


def _source_bucket(name):
    source = DATASET_SOURCES.get(name)
    if source is None:
        return None  # stale name from an old progress.json
    return "indonesian" if name in INDONESIAN_SOURCES else source["category"]

# =============================================================================
# Dataset source definitions
# =============================================================================

DATASET_SOURCES = {
    # =========================================================================
    # CYBERSECURITY (Public Replacements)
    # =========================================================================
    # max_tokens here is an UPPER BOUND. Small datasets (circl 39K rows) will
    # not reach this limit.
    # Total actual ~4.5-5B tokens, compute-optimal for d20 on L4 (ratio ~11x).
    # NOTE: Instruction/chat/DPO-style HF datasets (trendyol_cyber, nist_cybersec,
    # fenrir_v2, cybernative_vuln_dpo, openhermes, code_feedback, numinamath_cot,
    # competition_math) were moved to scripts/data/download_sft_data.py — they are
    # SFT data, not pretraining corpora.
    "all_cve_records": {
        "source_type": "nvd_json_feeds",
        "description": "NVD CVE 2.0 JSON feeds, yearly bootstrap from 2002-current",
        "category": "cybersecurity",
        "max_tokens": 6_000_000_000,
        "start_year": 2002,
        "end_year": None,
    },
    "circl_vuln_patch": {
        "hf_name": "CIRCL/vulnerability-cwe-patch",
        "description": "39K vulns with real patches from GitHub/GitLab",
        "category": "cybersecurity",
        "max_tokens": 2_400_000_000,
        "text_column": None,
        "split": "train",
        "streaming": False,
        "format": "vuln_patch",
    },
    "primus_nemotron_cc": {
        "hf_name": "trend-cybertron/Primus-Nemotron-CC",
        "description": "7.6B tokens of cybersecurity text filtered from Nemotron-CC (gated, requires HF token)",
        "category": "cybersecurity",
        "max_tokens": 7_600_000_000,
        "text_column": "text",
        "split": "train",
        "streaming": True,
    },
    "primus_fineweb": {
        "hf_name": "trendmicro-ailab/Primus-FineWeb",
        "description": "Cybersecurity-filtered FineWeb pretraining corpus (gated, requires HF token)",
        "category": "cybersecurity",
        "max_tokens": 5_000_000_000,
        "text_column": "text",
        "split": "train",
        "streaming": True,
    },
    "nvd_cve": {
        "source_type": "circl_ndjson_dump",
        "description": "CIRCL Vulnerability-Lookup CVE List v5 dump",
        "category": "cybersecurity",
        "max_tokens": 4_800_000_000,
        "dump_url": "https://vulnerability.circl.lu/dumps/cvelistv5.ndjson",
    },
    # =========================================================================
    # LOCAL CYBERSECURITY PRETRAINING DATA (data/sft intentionally excluded)
    # =========================================================================
    "local_incident_response": {
        "source_type": "local_files",
        "description": "Local incident response reports and runbooks",
        "category": "cybersecurity",
        "max_tokens": 1_440_000_000,
        "local_paths": ["data/synthetic-ir/*.jsonl"],
    },
    "local_soc_synthetic": {
        "source_type": "local_files",
        "description": "Local synthetic SOC analyst conversations and cases",
        "category": "cybersecurity",
        "max_tokens": 1_440_000_000,
        "local_paths": ["data/synthetic-soc/*.jsonl"],
    },
    "local_reverse_engineering": {
        "source_type": "local_files",
        "description": "Local reverse engineering and exploitation analysis",
        "category": "cybersecurity",
        "max_tokens": 1_200_000_000,
        "local_paths": ["data/reverse-engineering/*.jsonl"],
    },
    "local_cloud_security": {
        "source_type": "local_files",
        # Raw JSON cloud logs (cloudtrail, azure, gcp) are converted to natural language
        # narratives by scripts/data/convert_logs_to_nl.py before training.
        # Run: python -m scripts.data.convert_logs_to_nl
        # Output goes to data/cloud_nl/*.jsonl — use that path here.
        "description": "Local AWS, Azure, and GCP audit/security events (NL narratives)",
        "category": "cybersecurity",
        "max_tokens": 1_200_000_000,
        "local_paths": ["data/cloud_nl/*.jsonl"],
    },
    "local_security_logs": {
        "source_type": "local_files",
        # Raw log files (auth.log, sysmon.xml, conn.log, etc.) are converted to natural
        # language narratives by scripts/data/convert_logs_to_nl.py before training.
        # Run: python -m scripts.data.convert_logs_to_nl
        # Output goes to data/log_nl/*.jsonl — use that path here.
        "description": "Local auth, web, Windows, Sysmon, CEF, and network security logs (NL narratives)",
        "category": "cybersecurity",
        "max_tokens": 1_200_000_000,
        "local_paths": [
            "data/log_nl/*.jsonl",
        ],
    },
    # =========================================================================
    # GENERAL KNOWLEDGE
    # =========================================================================
    "climbmix": {
        "hf_name": "karpathy/climbmix-400b-shuffle",
        "description": "High-quality general pretraining data",
        "category": "general",
        "max_tokens": 5_000_000_000,
        "text_column": "text",
        "split": "train",
        "streaming": True,
    },
    "wikipedia": {
        "hf_name": "wikimedia/wikipedia",
        "description": "English Wikipedia articles",
        "category": "general",
        "max_tokens": 3_000_000_000,
        "text_column": "text",
        "split": "train",
        "streaming": True,
        "hf_subset": "20231101.en",
    },
    "fineweb_edu": {
        "hf_name": "HuggingFaceFW/fineweb-edu",
        "description": "Educational web content",
        "category": "general",
        "max_tokens": 4_000_000_000,
        "text_column": "text",
        "split": "train",
        "streaming": True,
        "hf_subset": "sample-10BT",
    },
    # Indonesian foundation. Deliberately small (~3.2B, ~3% of the mix): enough to give the
    # 96K tokenizer real Indonesian sub-words and to ground the base model, without displacing
    # the cybersecurity/English budget. Indonesian SFT (data/sft/*.jsonl) builds on top of this —
    # SFT alone cannot teach a language the base model never saw.
    "wikipedia_id": {
        "hf_name": "wikimedia/wikipedia",
        "description": "Indonesian Wikipedia articles",
        "category": "general",
        "max_tokens": 200_000_000,
        "text_column": "text",
        "split": "train",
        "streaming": True,
        "hf_subset": "20231101.id",
    },
    "fineweb2_id": {
        "hf_name": "HuggingFaceFW/fineweb-2",
        "description": "Indonesian web text, FineWeb-2 filtering pipeline",
        "category": "general",
        "max_tokens": 3_000_000_000,
        "text_column": "text",
        "split": "train",
        "streaming": True,
        "hf_subset": "ind_Latn",
    },
"secure_code_python": {
    "hf_name": "bigcode/the-stack-dedup",
    "description": "Python security-related repositories",
    "category": "code",
    "max_tokens": 4_000_000_000,
    "text_column": "content",
    "split": "train",
    "streaming": True,
    "data_dir": "data/python",
},

"secure_code_c": {
    "hf_name": "bigcode/the-stack-dedup",
    "description": "C security-related repositories",
    "category": "code",
    "max_tokens": 3_000_000_000,
    "text_column": "content",
    "split": "train",
    "streaming": True,
    "data_dir": "data/c",
},

"secure_code_cpp": {
    "hf_name": "bigcode/the-stack-dedup",
    "description": "C++ security-related repositories",
    "category": "code",
    "max_tokens": 3_000_000_000,
    "text_column": "content",
    "split": "train",
    "streaming": True,
    "data_dir": "data/cpp",
},

"secure_code_rust": {
    "hf_name": "bigcode/the-stack-dedup",
    "description": "Rust security-related repositories",
    "category": "code",
    "max_tokens": 2_400_000_000,
    "text_column": "content",
    "split": "train",
    "streaming": True,
    "data_dir": "data/rust",
},

"secure_code_go": {
    "hf_name": "bigcode/the-stack-dedup",
    "description": "Go security-related repositories",
    "category": "code",
    "max_tokens": 2_400_000_000,
    "text_column": "content",
    "split": "train",
    "streaming": True,
    "data_dir": "data/go",
},

"secure_code_shell": {
    "hf_name": "bigcode/the-stack-dedup",
    "description": "Shell security-related repositories",
    "category": "code",
    "max_tokens": 2_000_000_000,
    "text_column": "content",
    "split": "train",
    "streaming": True,
    "data_dir": "data/shell",
},

"secure_code_javascript": {
    "hf_name": "bigcode/the-stack-dedup",
    "description": "JavaScript security-related repositories",
    "category": "code",
    "max_tokens": 2_400_000_000,
    "text_column": "content",
    "split": "train",
    "streaming": True,
    "data_dir": "data/javascript",
},

"secure_code_typescript": {
    "hf_name": "bigcode/the-stack-dedup",
    "description": "TypeScript security-related repositories",
    "category": "code",
    "max_tokens": 2_000_000_000,
    "text_column": "content",
    "split": "train",
    "streaming": True,
    "data_dir": "data/typescript",
},

"secure_code_java": {
    "hf_name": "bigcode/the-stack-dedup",
    "description": "Java security-related repositories",
    "category": "code",
    "max_tokens": 2_000_000_000,
    "text_column": "content",
    "split": "train",
    "streaming": True,
    "data_dir": "data/java",
},

"secure_code_php": {
    "hf_name": "bigcode/the-stack-dedup",
    "description": "PHP security-related repositories",
    "category": "code",
    "max_tokens": 1_600_000_000,
    "text_column": "content",
    "split": "train",
    "streaming": True,
    "data_dir": "data/php",
},

# The `code_*` entries below are the same thing as `secure_code_*` above: plain
# per-language directories of bigcode/the-stack-dedup, no security filter. The older
# name is kept only because `completed_sources` in progress.json is keyed by source
# name, so renaming would make every finished shard look un-downloaded.
"code_sql": {
    "hf_name": "bigcode/the-stack-dedup",
    "description": "SQL — queries, schemas, stored procedures (context for injection classes)",
    "category": "code",
    "max_tokens": 1_500_000_000,
    "text_column": "content",
    "split": "train",
    "streaming": True,
    "data_dir": "data/sql",
},

"code_powershell": {
    "hf_name": "bigcode/the-stack-dedup",
    "description": "PowerShell — Windows administration and offensive tooling",
    "category": "code",
    "max_tokens": 800_000_000,
    "text_column": "content",
    "split": "train",
    "streaming": True,
    "data_dir": "data/powershell",
},

"code_assembly": {
    "hf_name": "bigcode/the-stack-dedup",
    "description": "Assembly — disassembly listings and low-level routines (reverse engineering)",
    "category": "code",
    "max_tokens": 800_000_000,
    "text_column": "content",
    "split": "train",
    "streaming": True,
    "data_dir": "data/assembly",
},

"code_csharp": {
    "hf_name": "bigcode/the-stack-dedup",
    "description": "C# — .NET applications and tooling",
    "category": "code",
    "max_tokens": 2_000_000_000,
    "text_column": "content",
    "split": "train",
    "streaming": True,
    "data_dir": "data/c-sharp",
},

"code_jupyter": {
    # Notebooks are where machine-learning and AI code actually lives on GitHub:
    # training loops, data pipelines, model evaluation, interleaved with prose.
    "hf_name": "bigcode/the-stack-dedup",
    "description": "Jupyter notebooks — machine learning and AI code with inline explanation",
    "category": "code",
    "max_tokens": 2_000_000_000,
    "text_column": "content",
    "split": "train",
    "streaming": True,
    "data_dir": "data/jupyter-notebook",
},

"swallow_code_v2": {
    # Apache-2.0 | ~19.6M rows after LLM rewrite pipeline | ~16.1B tokens total
    # Source: The-Stack-v2 Python snippets refined through 4-stage pipeline:
    #   syntax validation → length filter → LLM scoring → LLM rewrite
    # Paper: https://arxiv.org/abs/2505.02881 (TokyoTech LLM, 2025)
    # Using stage4-llm-rewrite: highest quality subset (self-contained, algorithmically efficient)
    "hf_name": "tokyotech-llm/swallow-code-v2",
    "description": "High-quality Python code from The-Stack-v2, refined via 4-stage LLM pipeline (TokyoTech, Apache-2.0)",
    "category": "code",
    "max_tokens": 5_000_000_000,
    "text_column": "text",
    "split": "train",
    "streaming": True,
    "hf_subset": "stage4-llm-rewrite",
},

"finemath": {
    "hf_name": "HuggingFaceTB/finemath",
    "hf_subset": "finemath-4plus",
    "description": "Mathematical reasoning corpus",
    "category": "instruction",
    "max_tokens": 3_000_000_000,
    "text_column": "text",
    "split": "train",
    "streaming": True,
},

"nemotron_cc_math": {
    "hf_name": "nvidia/Nemotron-CC-Math-v1",
    "hf_subset": "4plus",
    "description": "High-quality math pretraining corpus extracted from Common Crawl (NVIDIA, 52B tokens, score 4+)",
    "category": "instruction",
    "max_tokens": 18_000_000_000,
    "text_column": "text",
    "split": "train",
    "streaming": True,
},

# =========================================================================
# NEW DATASETS (Depth 32 branch — replacing 404'd sources)
# =========================================================================
"primus_seed": {
    "hf_name": "trendmicro-ailab/Primus-Seed",
    "description": "Curated cybersecurity seed corpus from Trend Micro AI Lab (Wikipedia, MITRE, NIST, security blogs)",
    "category": "cybersecurity",
    "max_tokens": 2_000_000_000,
    "text_column": "text",
    "split": "train",
    "streaming": True,
},

# NOTE: primus_reasoning (trendmicro-ailab/Primus-Reasoning) is a chain-of-thought
# REASONING dataset (post-training), not a pretraining corpus. It lives in
# scripts/data/download_sft_data.py under the "primus_reasoning" SFT source.

"brightdata_cybersec": {
    "source_type": "brightdata_scraper",
    "description": "Real-time threat intel and secure coding blogs scraped via BrightData API/Proxy",
    "category": "cybersecurity",
    "max_tokens": 500_000_000,
    "target_urls": [
        "https://googleprojectzero.blogspot.com/",
        "https://thedfirreport.com/",
    ],
},
}

for _source in _source_definitions():
    DATASET_SOURCES.setdefault(_source.name, _source_to_dataset_config(_source))

CHARS_PER_TOKEN = 4.0
CHECKPOINT_EVERY_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB
NVD_FEED_BASE_URL = "https://nvd.nist.gov/feeds/json/cve/2.0"

# =============================================================================
# Category helpers
# =============================================================================

def _category_emoji(category):
    """Map dataset category to emoji."""
    emoji_map = {
        "cybersecurity": "🔒",
        "code": "💻",
        "general": "📚",
        "instruction": "🧠",
        "indonesian": "🇮🇩",
    }
    return emoji_map.get(category, "📋")

# =============================================================================
# Formatting helpers
# =============================================================================

def format_instruction_row(row, filter_keywords=None):
    if filter_keywords:
        text_dump = str(row).lower()
        if not any(k in text_dump for k in filter_keywords):
            return None
    messages_text = format_messages_row(row)
    if messages_text:
        return messages_text
    user_text = row.get("instruction", row.get("question", row.get("user", row.get("User", row.get("prompt", row.get("problem", row.get("query", row.get("input", ""))))))))
    bot_text = row.get("response", row.get("answer", row.get("assistant", row.get("Assistant", row.get("output", row.get("solution", row.get("code", "")))))))
    if not user_text or not bot_text:
        return None
    return f"{user_text}\n\n{bot_text}"

def format_dpo_row(row):
    prompt = row.get("prompt", row.get("question", ""))
    chosen = row.get("chosen", row.get("accepted", row.get("response", "")))
    if prompt and chosen:
        if isinstance(chosen, list) and len(chosen) > 0 and isinstance(chosen[0], dict):
             # handle complex chosen formats
             chosen = str(chosen)
        return f"{prompt}\n\n{chosen}"
    return None

def format_cve_row(row):
    text = row.get("text", "")
    if text:
        return text
    cve_id = row.get("cve_id", row.get("id", ""))
    description = row.get("description", row.get("desc", ""))
    if cve_id and description:
        return f"CVE: {cve_id}\n{description}"
    return None

def format_vuln_patch_row(row):
    desc = row.get("description", "")
    patches = row.get("patches", [])
    cwe = row.get("cwe", [])
    if not desc:
        return None
    if isinstance(cwe, str):
        cwe_str = cwe
    elif isinstance(cwe, list):
        cwe_str = ", ".join(
            c.get("name", "") if isinstance(c, dict) else str(c)
            for c in cwe
        )
    else:
        cwe_str = str(cwe) if cwe else ""

    if isinstance(patches, list) and patches:
        first_patch = patches[0]
        patch_text = first_patch.get("commit_message", "") if isinstance(first_patch, dict) else str(first_patch)
    else:
        patch_text = ""
    return f"Vulnerability: {desc}\nCWE: {cwe_str}\nPatch: {patch_text}".strip()

def format_chat_row(row):
    messages_text = format_messages_row(row)
    if messages_text:
        return messages_text
    user = row.get("user", row.get("User", row.get("question", "")))
    assistant = row.get("assistant", row.get("Assistant", row.get("response", "")))
    if not user or not assistant:
        return None
    return f"{user}\n\n{assistant}"

def _coerce_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()

def format_messages_row(row):
    messages = None
    if isinstance(row, list):
        messages = row
    elif isinstance(row, dict):
        for key in ("messages", "conversations", "conversation", "dialogue", "chat"):
            if key in row:
                messages = row.get(key)
                break
    if messages is None:
        return None
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            return messages.strip() or None
    if isinstance(messages, dict):
        for key in ("messages", "conversations", "conversation", "dialogue", "chat"):
            if key in messages:
                messages = messages[key]
                break
    if not isinstance(messages, list):
        return None

    parts = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(
            message.get("role")
            or message.get("from")
            or message.get("speaker")
            or message.get("author")
            or ""
        ).lower()
        content = _coerce_text(
            message.get("content")
            or message.get("value")
            or message.get("text")
            or message.get("message")
            or ""
        )
        if not content or role == "system":
            continue
        if role in {"human", "user"}:
            role = "user"
        elif role in {"gpt", "assistant", "bot", "model"}:
            role = "assistant"
        if role:
            parts.append(f"{role}: {content}")
        else:
            parts.append(content)
    return "\n\n".join(parts) if parts else None

def format_generic_row(row):
    parts = []
    for key, value in row.items():
        text = _coerce_text(value)
        if not text:
            continue
        parts.append(f"{key}: {text}")
    return "\n".join(parts) if parts else None

def _repo_root():
    return Path(__file__).resolve().parents[2]

def _resolve_local_files(source_config):
    files = []
    root = _repo_root()
    for pattern in source_config.get("local_paths", []):
        pattern_path = Path(pattern)
        glob_pattern = str(pattern_path if pattern_path.is_absolute() else root / pattern_path)
        for path in glob.glob(glob_pattern):
            normalized = Path(path).resolve()
            try:
                relative = normalized.relative_to(root)
            except ValueError:
                relative = normalized
            if "data/sft" in relative.as_posix():
                continue
            if normalized.is_file():
                files.append(str(normalized))
    return sorted(dict.fromkeys(files))

def count_local_files(source_config):
    return len(_resolve_local_files(source_config))

def _format_local_record(record):
    if isinstance(record, dict):
        text = _coerce_text(record.get("text"))
        if text:
            return text
        messages_text = format_messages_row(record)
        if messages_text:
            return messages_text
        return format_generic_row(record)
    if isinstance(record, list):
        messages_text = format_messages_row({"messages": record})
        if messages_text:
            return messages_text
        return _coerce_text(record)
    return _coerce_text(record)

def _with_local_source_label(source_name, path, text):
    return f"Source: {source_name}\nFile: {Path(path).name}\n\n{text.strip()}"

def stream_local_file_texts(source_name, source_config, max_chars, skip_docs=0):
    """Yield pretraining text from local non-SFT cybersecurity files."""
    char_count = 0
    doc_count = 0
    yielded_docs = 0

    for path in _resolve_local_files(source_config):
        suffix = Path(path).suffix.lower()

        if suffix == ".jsonl":
            with open(path, encoding="utf-8", errors="replace") as f:
                for line_number, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as e:
                        if line_number <= 5:
                            logger.warning(f"  Bad local JSONL line {path}:{line_number}: {e}")
                        continue

                    text = _format_local_record(record)
                    if not text or len(text.strip()) < 50:
                        continue

                    doc_count += 1
                    if doc_count <= skip_docs:
                        continue

                    labeled_text = _with_local_source_label(source_name, path, text)
                    if char_count + len(labeled_text) > max_chars:
                        return
                    char_count += len(labeled_text)
                    yielded_docs += 1
                    yield labeled_text
            continue

        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read().strip()
        if not text or len(text) < 50:
            continue

        doc_count += 1
        if doc_count <= skip_docs:
            continue

        labeled_text = _with_local_source_label(source_name, path, text)
        if char_count + len(labeled_text) > max_chars:
            return
        char_count += len(labeled_text)
        yielded_docs += 1
        yield labeled_text

    est_tokens = int(char_count / CHARS_PER_TOKEN)
    logger.info(f"  ✓ {source_name} local complete: {yielded_docs:,} docs, ~{est_tokens:,} tokens")

def _first_english_description(descriptions):
    if not isinstance(descriptions, list):
        return ""
    fallback = ""
    for desc in descriptions:
        if not isinstance(desc, dict):
            continue
        value = _coerce_text(desc.get("value"))
        if not value:
            continue
        if not fallback:
            fallback = value
        if str(desc.get("lang", "")).lower() == "en":
            return value
    return fallback

def _unique_join(values, limit=None):
    seen = set()
    deduped = []
    for value in values:
        text = _coerce_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
        if limit and len(deduped) >= limit:
            break
    return ", ".join(deduped)

def _extract_nvd_cvss(metrics):
    if not isinstance(metrics, dict):
        return ""
    for metric_key in ("cvssMetricV40", "cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        metric_rows = metrics.get(metric_key)
        if not isinstance(metric_rows, list) or not metric_rows:
            continue
        metric = metric_rows[0]
        if not isinstance(metric, dict):
            continue
        cvss_data = metric.get("cvssData", {})
        if not isinstance(cvss_data, dict):
            cvss_data = {}
        severity = cvss_data.get("baseSeverity") or metric.get("baseSeverity")
        score = cvss_data.get("baseScore")
        vector = cvss_data.get("vectorString")
        version = cvss_data.get("version") or metric_key.replace("cvssMetric", "CVSS ")
        parts = [str(version)]
        if severity:
            parts.append(str(severity))
        if score is not None:
            parts.append(f"score {score}")
        if vector:
            parts.append(str(vector))
        return " | ".join(parts)
    return ""

def _extract_nvd_cwes(weaknesses):
    cwes = []
    if not isinstance(weaknesses, list):
        return cwes
    for weakness in weaknesses:
        if not isinstance(weakness, dict):
            continue
        for desc in weakness.get("description", []):
            if not isinstance(desc, dict):
                continue
            value = _coerce_text(desc.get("value"))
            if value:
                cwes.append(value)
    return cwes

def _walk_nvd_config_nodes(nodes):
    if not isinstance(nodes, list):
        return
    for node in nodes:
        if not isinstance(node, dict):
            continue
        for match in node.get("cpeMatch", []) or []:
            if isinstance(match, dict) and match.get("vulnerable"):
                criteria = match.get("criteria") or match.get("matchCriteriaId")
                if criteria:
                    yield criteria
        yield from _walk_nvd_config_nodes(node.get("children", []))

def _extract_nvd_references(references):
    refs = []
    if not isinstance(references, list):
        return refs
    for ref in references:
        if not isinstance(ref, dict):
            continue
        url = ref.get("url")
        tags = ref.get("tags")
        if url and tags:
            refs.append(f"{url} ({', '.join(str(t) for t in tags[:3])})")
        elif url:
            refs.append(url)
    return refs

def format_nvd_cve_row(row):
    cve = row.get("cve", row) if isinstance(row, dict) else {}
    if not isinstance(cve, dict):
        return None

    root_id = row.get("id") if isinstance(row, dict) else None
    cve_id = cve.get("id") or cve.get("cveId") or root_id
    description = _first_english_description(cve.get("descriptions"))
    if not cve_id or not description:
        return None

    cwes = _extract_nvd_cwes(cve.get("weaknesses"))
    configurations = cve.get("configurations", [])
    cpes = []
    if isinstance(configurations, list):
        for config in configurations:
            if isinstance(config, dict):
                cpes.extend(_walk_nvd_config_nodes(config.get("nodes", [])))

    lines = [f"CVE: {cve_id}", f"Description: {description}"]
    for key, label in (
        ("published", "Published"),
        ("lastModified", "Last Modified"),
        ("vulnStatus", "Status"),
        ("sourceIdentifier", "Source"),
    ):
        value = cve.get(key)
        if value:
            lines.append(f"{label}: {value}")

    cvss = _extract_nvd_cvss(cve.get("metrics"))
    if cvss:
        lines.append(f"CVSS: {cvss}")

    cwe_text = _unique_join(cwes, limit=10)
    if cwe_text:
        lines.append(f"CWE: {cwe_text}")

    cpe_text = _unique_join(cpes, limit=20)
    if cpe_text:
        lines.append(f"Affected CPEs: {cpe_text}")

    ref_text = _unique_join(_extract_nvd_references(cve.get("references")), limit=10)
    if ref_text:
        lines.append(f"References: {ref_text}")

    return "\n".join(lines)

def _extract_cvelist_problem_types(problem_types):
    values = []
    if not isinstance(problem_types, list):
        return values
    for problem in problem_types:
        if not isinstance(problem, dict):
            continue
        for desc in problem.get("descriptions", []) or []:
            if not isinstance(desc, dict):
                continue
            cwe_id = desc.get("cweId")
            value = desc.get("description") or desc.get("value")
            if cwe_id and value:
                values.append(f"{cwe_id} {value}")
            elif cwe_id or value:
                values.append(cwe_id or value)
    return values

def _extract_cvelist_affected(affected):
    values = []
    if not isinstance(affected, list):
        return values
    for item in affected:
        if not isinstance(item, dict):
            continue
        vendor = _coerce_text(item.get("vendor"))
        product = _coerce_text(item.get("product"))
        versions = []
        for version in item.get("versions", []) or []:
            if isinstance(version, dict):
                version_text = version.get("version") or version.get("lessThan") or version.get("status")
                if version_text:
                    versions.append(str(version_text))
            else:
                versions.append(str(version))
        subject = " ".join(part for part in (vendor, product) if part)
        version_text = _unique_join(versions, limit=5)
        if subject and version_text:
            values.append(f"{subject} ({version_text})")
        elif subject:
            values.append(subject)
    return values

def format_cvelist_row(row):
    if not isinstance(row, dict):
        return None
    metadata = row.get("cveMetadata", {}) if isinstance(row.get("cveMetadata"), dict) else {}
    containers = row.get("containers", {}) if isinstance(row.get("containers"), dict) else {}
    cna = containers.get("cna", {}) if isinstance(containers.get("cna"), dict) else {}

    cve_id = metadata.get("cveId") or row.get("cve_id") or row.get("id")
    description = _first_english_description(cna.get("descriptions"))
    if not description:
        description = _coerce_text(cna.get("title") or row.get("summary") or row.get("description"))
    if not cve_id or not description:
        return None

    lines = [f"CVE: {cve_id}", f"Description: {description}"]
    title = _coerce_text(cna.get("title"))
    if title and title != description:
        lines.insert(1, f"Title: {title}")

    for key, label in (
        ("datePublished", "Published"),
        ("dateUpdated", "Updated"),
        ("state", "State"),
        ("assignerShortName", "Assigner"),
    ):
        value = metadata.get(key)
        if value:
            lines.append(f"{label}: {value}")

    problem_types = _unique_join(_extract_cvelist_problem_types(cna.get("problemTypes")), limit=10)
    if problem_types:
        lines.append(f"Problem Types: {problem_types}")

    affected = _unique_join(_extract_cvelist_affected(cna.get("affected")), limit=20)
    if affected:
        lines.append(f"Affected: {affected}")

    refs = []
    for ref in cna.get("references", []) or []:
        if isinstance(ref, dict) and ref.get("url"):
            refs.append(ref["url"])
    ref_text = _unique_join(refs, limit=10)
    if ref_text:
        lines.append(f"References: {ref_text}")

    return "\n".join(lines)

def format_vulnerability_record(row):
    return (
        format_nvd_cve_row(row)
        or format_cvelist_row(row)
        or format_cve_row(row)
        or format_generic_row(row)
    )

# =============================================================================
# Core data processing
# =============================================================================

def _urlopen(url, retries: int = 3, backoff: float = 1.5):
    """Open a URL with exponential‑backoff retry.

    The original implementation performed a single request and raised on any
    failure, which caused the whole pipeline to abort on transient network
    glitches. This helper now retries *retries* times, waiting ``backoff * attempt``
    seconds between attempts, and finally propagates the last exception if all
    attempts fail.
    """
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "mesosfer-data-prep/3.0 (+https://nvd.nist.gov/vuln/data-feeds)"},
            )
            return urllib.request.urlopen(request, timeout=120)
        except Exception as exc:  # pragma: no cover – exercised via retry path
            logger.warning("Attempt %d/%d: failed to fetch %s – %s", attempt, retries, url, exc)
            if attempt == retries:
                raise
            time.sleep(backoff * attempt)

def stream_nvd_feed_texts(source_name, source_config, max_chars):
    start_year = int(source_config.get("start_year", 2002))
    end_year = source_config.get("end_year") or datetime.now(timezone.utc).year
    end_year = int(end_year)
    char_count = 0
    doc_count = 0

    for year in range(start_year, end_year + 1):
        url = f"{NVD_FEED_BASE_URL}/nvdcve-2.0-{year}.json.gz"
        logger.info(f"  Downloading NVD CVE feed {year}: {url}")
        try:
            with _urlopen(url) as response:
                with gzip.GzipFile(fileobj=response) as gz:
                    feed = json.load(gz)
        except Exception as e:
            logger.warning(f"  Failed to download/parse NVD feed {year}: {e}")
            continue

        vulnerabilities = feed.get("vulnerabilities", [])
        if not isinstance(vulnerabilities, list):
            logger.warning(f"  NVD feed {year} has unexpected shape; skipping.")
            continue

        for item in vulnerabilities:
            text = format_nvd_cve_row(item)
            if not text or len(text.strip()) < 50:
                continue
            text = text.strip()
            char_count += len(text)
            doc_count += 1
            yield text

            if char_count >= max_chars:
                est_tokens = int(char_count / CHARS_PER_TOKEN)
                logger.info(f"  {source_name}: reached limit at {doc_count:,} docs, ~{est_tokens:,} tokens")
                return

        est_tokens = int(char_count / CHARS_PER_TOKEN)
        logger.info(f"  {source_name}: through {year}, {doc_count:,} docs, ~{est_tokens:,} tokens")

    est_tokens = int(char_count / CHARS_PER_TOKEN)
    logger.info(f"  ✓ {source_name} complete: {doc_count:,} docs, ~{est_tokens:,} tokens")


def _xml_text(element, local_name):
    for child in list(element):
        if child.tag.split("}")[-1] == local_name and child.text:
            return child.text.strip()
    return ""


def stream_rss_feed_texts(source_name, source_config, max_chars):
    url = source_config["url"]
    char_count = 0
    doc_count = 0

    logger.info(f"  Downloading RSS feed for {source_name}: {url}")
    try:
        with _urlopen(url) as response:
            root = ET.parse(response).getroot()
    except Exception as e:
        logger.warning(f"  Failed to download/parse RSS feed {url}: {e}")
        return

    entries = []
    for item in root.iter():
        local_tag = item.tag.split("}")[-1]
        if local_tag in {"item", "entry"}:
            entries.append(item)

    for entry in entries:
        title = _xml_text(entry, "title")
        summary = _xml_text(entry, "summary") or _xml_text(entry, "description")
        content = _xml_text(entry, "encoded") or _xml_text(entry, "content")
        link = _xml_text(entry, "link")
        text = "\n".join(part for part in (title, summary, content, link) if part).strip()
        if not text or len(text) < 50:
            continue
        if not is_high_quality_security_text(text, source_name):
            continue

        char_count += len(text)
        doc_count += 1
        yield text

        if char_count >= max_chars:
            break

    est_tokens = int(char_count / CHARS_PER_TOKEN)
    logger.info(f"  ✓ {source_name} RSS complete: {doc_count:,} docs, ~{est_tokens:,} tokens")


def stream_url_json_texts(source_name, source_config, max_chars):
    url = source_config["url"]
    logger.info(f"  Downloading JSON source for {source_name}: {url}")
    try:
        with _urlopen(url) as response:
            payload = json.load(response)
    except Exception as e:
        logger.warning(f"  Failed to download/parse JSON source {url}: {e}")
        return

    records = payload
    if isinstance(payload, dict):
        for key in ("vulnerabilities", "items", "data", "objects"):
            if isinstance(payload.get(key), list):
                records = payload[key]
                break
    if not isinstance(records, list):
        records = [records]

    char_count = 0
    doc_count = 0
    for record in records:
        text = _format_local_record(record)
        if not text or len(text) < 50:
            continue
        if not is_high_quality_security_text(text, source_name):
            continue
        char_count += len(text)
        doc_count += 1
        yield text
        if char_count >= max_chars:
            break

    est_tokens = int(char_count / CHARS_PER_TOKEN)
    logger.info(f"  ✓ {source_name} JSON complete: {doc_count:,} docs, ~{est_tokens:,} tokens")


def stream_circl_ndjson_texts(source_name, source_config, max_chars):
    url = source_config["dump_url"]
    char_count = 0
    doc_count = 0

    logger.info(f"  Downloading CIRCL Vulnerability-Lookup dump: {url}")
    try:
        response = _urlopen(url)
    except Exception as e:
        logger.warning(f"  Failed to open CIRCL dump {url}: {e}")
        return

    with response:
        raw_stream = response
        if url.endswith(".gz"):
            raw_stream = gzip.GzipFile(fileobj=response)

        for line_number, raw_line in enumerate(raw_stream, start=1):
            if not raw_line:
                continue
            try:
                line = raw_line.decode("utf-8").strip()
            except UnicodeDecodeError:
                line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                if line_number <= 5:
                    logger.warning(f"  Bad CIRCL NDJSON line {line_number}: {e}")
                continue

            text = format_vulnerability_record(row)
            if not text or len(text.strip()) < 50:
                continue

            text = text.strip()
            char_count += len(text)
            doc_count += 1
            yield text

            if char_count >= max_chars:
                est_tokens = int(char_count / CHARS_PER_TOKEN)
                logger.info(f"  {source_name}: reached limit at {doc_count:,} docs, ~{est_tokens:,} tokens")
                return

            if doc_count % 100_000 == 0:
                est_tokens = int(char_count / CHARS_PER_TOKEN)
                logger.info(f"  {source_name}: {doc_count:,} docs, ~{est_tokens:,} tokens")

    est_tokens = int(char_count / CHARS_PER_TOKEN)
    logger.info(f"  ✓ {source_name} complete: {doc_count:,} docs, ~{est_tokens:,} tokens")


def _is_dataset2_source(source_name, source_config=None):
    if source_config is None:
        source_config = DATASET_SOURCES.get(source_name, {})
    return source_config.get("dataset_group") == "dataset2"


def is_high_quality_security_text(text, source_name):
    lowered = text.lower()

    LOW_QUALITY_PATTERNS = [
        "node_modules",
        "package-lock.json",
        "yarn.lock",
        ".min.js",
        "generated automatically",
        "auto-generated",
        "webpack",
        "dist/",
        "build/",
    ]

    SECURITY_KEYWORDS = [
        "cve",
        "exploit",
        "payload",
        "shellcode",
        "overflow",
        "buffer overflow",
        "heap overflow",
        "xss",
        "rce",
        "lfi",
        "rfi",
        "csrf",
        "ssrf",
        "sqli",
        "sql injection",
        "privilege escalation",
        "reverse shell",
        "deserialization",
        "yara",
        "malware",
        "fuzz",
        "pentest",
        "mitre",
        "owasp",
        "metasploit",
        "backdoor",
        "rootkit",
        "c2",
        "command injection",
        "debug",
        "troubleshoot",
        "segfault",
        "kernel",
        "docker",
        "kubernetes",
        "tls",
        "ssh",
        "memory corruption",
        "forensics",
        "packet",
        "wireshark",
        "heap spray",
        "rop",
        "uaf",
        "format string",
    ]
    is_dataset2 = _is_dataset2_source(source_name)
    if is_dataset2:
        SECURITY_KEYWORDS.extend(CAUSAL_MARKERS)
        SECURITY_KEYWORDS.extend(MECHANISM_TERMS)
        SECURITY_KEYWORDS.extend(SECURITY_STACK_TERMS)
        SECURITY_KEYWORDS.extend(SECURITY_CODE_TERMS)

    # skip low quality junk
    if any(p in lowered for p in LOW_QUALITY_PATTERNS):
        return False

    if is_dataset2 and any(p in lowered for p in MARKETING_TERMS) and not any(
        p in lowered for p in CAUSAL_MARKERS + MECHANISM_TERMS + SECURITY_CODE_TERMS
    ):
        return False

    # skip gigantic garbage blobs (raised from 200k - C/C++ files can be large)
    if len(text) > 500_000:
        return False

    # stricter filtering for code datasets
    SECURITY_CODESETS = [
        "secure_code_python",
        "secure_code_c",
        "secure_code_cpp",
        "secure_code_rust",
        "secure_code_go",
        "secure_code_shell",
        "secure_code_javascript",
        "secure_code_typescript",
        "secure_code_java",
        "secure_code_php",
    ]

    if source_name in SECURITY_CODESETS:
        if not any(k in lowered for k in SECURITY_KEYWORDS):
            return False

    return True


def _github_clone_dir(repo):
    """Get cache directory for cloned github repo."""
    base_dir = get_base_dir()
    cache_root = os.path.join(base_dir, "github_cache")
    os.makedirs(cache_root, exist_ok=True)
    safe_name = repo.replace("/", "__")
    return os.path.join(cache_root, safe_name)


def _git_clone_or_pull(repo, target_dir):
    """Shallow clone repo if not exists, otherwise just verify it exists.

    Uses --depth=1 to minimize disk usage. Returns True on success.
    """
    if os.path.exists(os.path.join(target_dir, ".git")):
        logger.info(f"  Repo cache hit: {target_dir}")
        return True

    if os.path.exists(target_dir):
        # Stale dir without .git — remove and re-clone
        shutil.rmtree(target_dir, ignore_errors=True)

    url = f"https://github.com/{repo}.git"
    logger.info(f"  Cloning {url} (shallow) into {target_dir}")
    try:
        subprocess.run(
            ["git", "clone", "--depth=1", "--quiet", url, target_dir],
            check=True,
            timeout=600,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.warning(f"  git clone failed for {repo}: {e.stderr.decode(errors='replace')[:500]}")
        return False
    except subprocess.TimeoutExpired:
        logger.warning(f"  git clone timed out for {repo}")
        return False
    except FileNotFoundError:
        logger.warning("  git command not found — install git to use github_repo sources")
        return False


def _iter_repo_files(repo_dir, include_patterns):
    """Yield file paths in repo_dir matching any of the glob patterns.

    Patterns use ** for recursive match (e.g. 'rules/**/*.yml').
    Skips .git directory.
    """
    repo_path = Path(repo_dir)
    seen = set()
    for pattern in include_patterns:
        # Path.glob handles ** when used with rglob-style pattern
        for path in repo_path.glob(pattern):
            if not path.is_file():
                continue
            if ".git" in path.parts:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path


def stream_github_repo_texts(source_name, source_config, max_chars):
    """Yield pretraining text from files in a cloned GitHub repo."""
    repo = source_config["repo"]
    include_patterns = source_config.get("include", [])
    if not include_patterns:
        logger.warning(f"  {source_name}: no include patterns configured, skipping")
        return

    target_dir = _github_clone_dir(repo)
    if not _git_clone_or_pull(repo, target_dir):
        return

    char_count = 0
    doc_count = 0

    for path in _iter_repo_files(target_dir, include_patterns):
        try:
            # Skip files larger than 1MB (likely binary/blob)
            if path.stat().st_size > 1_000_000:
                continue
            with open(path, encoding="utf-8", errors="replace") as f:
                content = f.read()
        except (OSError, UnicodeDecodeError) as e:
            logger.debug(f"  Failed to read {path}: {e}")
            continue

        text = content.strip()
        if len(text) < 50:
            continue

        # Add file context as prefix
        rel_path = path.relative_to(target_dir)
        labeled = f"# Source: {repo}\n# File: {rel_path}\n\n{text}"

        if not is_high_quality_security_text(labeled, source_name):
            continue

        if char_count + len(labeled) > max_chars:
            break

        char_count += len(labeled)
        doc_count += 1
        yield labeled

    est_tokens = int(char_count / CHARS_PER_TOKEN)
    logger.info(f"  ✓ {source_name} github_repo complete: {doc_count:,} docs, ~{est_tokens:,} tokens")


def stream_brightdata_scraper_texts(source_name, source_config, max_chars):
    """
    Stream real-time cybersecurity or coding documentation using BrightData API / Proxy.
    Loads API credentials directly from environment variables (.env).
    """
    api_key = os.environ.get("BRIGHTDATA_API_KEY")
    proxy_url = os.environ.get("BRIGHTDATA_PROXY_URL")

    if not api_key and not proxy_url:
        logger.warning(
            f"  ⚠ BrightData credentials not found in .env (BRIGHTDATA_API_KEY or BRIGHTDATA_PROXY_URL). "
            f"Skipping {source_name}."
        )
        return

    logger.info(f"  🚀 Starting BrightData scraping for {source_name}...")
    target_urls = source_config.get("target_urls", [])
    if not target_urls:
        logger.warning(f"  No target_urls configured for BrightData source {source_name}.")
        return

    char_count = 0
    doc_count = 0

    for url in target_urls:
        logger.info(f"  [BrightData] Scraping {url} via proxy/API...")
        try:
            if proxy_url:
                proxy_handler = urllib.request.ProxyHandler({'http': proxy_url, 'https': proxy_url})
                opener = urllib.request.build_opener(proxy_handler)
            else:
                opener = urllib.request.build_opener()

            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Authorization": f"Bearer {api_key}" if api_key else ""
                }
            )

            with opener.open(req, timeout=120) as response:
                html = response.read().decode('utf-8', errors='replace')

            import re
            text_content = re.sub(r'<script.*?>.*?</script>', '', html, flags=re.DOTALL)
            text_content = re.sub(r'<style.*?>.*?</style>', '', text_content, flags=re.DOTALL)
            text_content = re.sub(r'<.*?>', '', text_content)
            text_content = re.sub(r'\s+', ' ', text_content).strip()

            if len(text_content) < 100:
                continue

            labeled = f"# Source: BrightData Scraper ({url})\n\n{text_content}"
            if not is_high_quality_security_text(labeled, source_name):
                continue

            char_count += len(labeled)
            doc_count += 1
            yield labeled

            if char_count >= max_chars:
                break

        except Exception as e:
            logger.warning(f"  [BrightData] Failed to scrape {url}: {e}")
            continue

    est_tokens = int(char_count / CHARS_PER_TOKEN)
    logger.info(f"  ✓ {source_name} complete: {doc_count:,} docs, ~{est_tokens:,} tokens")


def stream_dataset_texts(source_name, source_config, global_max_tokens=None, skip_docs=0):
    max_tokens = source_config.get("max_tokens")
    if global_max_tokens is not None and max_tokens is not None:
        max_tokens = min(max_tokens, global_max_tokens)
    elif global_max_tokens is not None:
        max_tokens = global_max_tokens
    max_chars = int(max_tokens * CHARS_PER_TOKEN) if max_tokens else float('inf')

    source_type = source_config.get("source_type")
    if source_type == "brightdata_scraper":
        yield from stream_brightdata_scraper_texts(source_name, source_config, max_chars)
        return
    if source_type == "nvd_json_feeds":
        yield from stream_nvd_feed_texts(source_name, source_config, max_chars)
        return
    if source_type == "circl_ndjson_dump":
        yield from stream_circl_ndjson_texts(source_name, source_config, max_chars)
        return
    if source_type == "local_files":
        yield from stream_local_file_texts(source_name, source_config, max_chars, skip_docs=skip_docs)
        return
    if source_type == "rss":
        yield from stream_rss_feed_texts(source_name, source_config, max_chars)
        return
    if source_type == "url_json":
        yield from stream_url_json_texts(source_name, source_config, max_chars)
        return
    if source_type == "github_repo":
        yield from stream_github_repo_texts(source_name, source_config, max_chars)
        return
    if source_type == "pdf_manifest":
        logger.warning("Source type %s for %s is registered for dry-run/planning but has no downloader yet.", source_type, source_name)
        return

    if load_dataset is None:
        logger.warning("The 'datasets' package is not installed; skipping Hugging Face source %s.", source_name)
        return

    hf_name = source_config["hf_name"]

    text_column = source_config.get("text_column")
    fmt = source_config.get("format")
    streaming = source_config.get("streaming", True)
    split = source_config.get("split")
    hf_subset = source_config.get("hf_subset")
    data_files = source_config.get("data_files")
    filter_keywords = source_config.get("filter_keywords")

    logger.info(f"Loading {source_name}: {hf_name} (streaming={streaming})")

    load_kwargs = {"streaming": streaming}
    if hf_subset:
        load_kwargs["name"] = hf_subset
    if data_files:
        load_kwargs["data_files"] = data_files
    if split:
        load_kwargs["split"] = split
    if HF_TOKEN:
        load_kwargs["token"] = HF_TOKEN

    kwargs = {}
    if "data_dir" in source_config:
        kwargs["data_dir"] = source_config["data_dir"]

    try:
        dataset = load_dataset(hf_name, **load_kwargs, **kwargs)
    except Exception as e:
        logger.warning(f"Failed to load {source_name} ({hf_name}): {e}")
        logger.warning(f"Skipping {source_name}...")
        return

    if hasattr(dataset, 'keys') and split is None:
        available_splits = list(dataset.keys())
        split = "train" if "train" in available_splits else available_splits[0]
        dataset = dataset[split]
        logger.info(f"  Auto-selected split: {split}")

    char_count = 0
    doc_count = 0

    if skip_docs > 0 and hasattr(dataset, 'skip'):
        logger.info(f"  Resuming {source_name}: skipping {skip_docs} previously processed documents...")
        dataset = dataset.skip(skip_docs)

    for row in dataset:
        if fmt == "instruction":
            text = format_instruction_row(row, filter_keywords)
        elif fmt == "dpo":
            text = format_dpo_row(row)
        elif fmt == "vuln_patch":
            text = format_vuln_patch_row(row)
        elif fmt == "chat":
            text = format_chat_row(row)
        elif fmt == "messages":
            text = format_messages_row(row)
        elif fmt == "generic":
            text = format_generic_row(row)
        elif text_column and text_column in row:
            text = _coerce_text(row[text_column])
        else:
            text = format_vulnerability_record(row)

        if not text or len(text.strip()) < 50:
            continue

        text = text.strip()

        if not is_high_quality_security_text(text, source_name):
            continue

        char_count += len(text)
        doc_count += 1
        yield text

        if char_count >= max_chars:
            break

        if doc_count % 100_000 == 0:
            est_tokens = int(char_count / CHARS_PER_TOKEN)
            logger.info(f"  {source_name}: {doc_count:,} docs, ~{est_tokens:,} tokens")

    est_tokens = int(char_count / CHARS_PER_TOKEN)
    logger.info(f"  ✓ {source_name} complete: {doc_count:,} docs, ~{est_tokens:,} tokens")


def write_shard(texts, shard_path):
    """Write a list of texts to a parquet shard.

    Handles permission errors gracefully so the pipeline can continue.
    """
    tmp_path = shard_path + ".tmp"
    df = pd.DataFrame({"text": texts})
    try:
        df.to_parquet(tmp_path, index=False, engine="pyarrow")
        os.replace(tmp_path, shard_path)
        logger.info(f"  Wrote {len(texts):,} docs to {shard_path}")
    except PermissionError as e:
        logger.error("Permission denied while writing %s: %s", shard_path, e)
        raise
    except OSError as e:
        logger.error("IO error while writing %s: %s", shard_path, e)
        raise

# =============================================================================
# Progress / resume helpers
# =============================================================================

PROGRESS_FILENAME = "progress.json"

def _progress_path(output_dir):
    return os.path.join(output_dir, PROGRESS_FILENAME)

def load_progress(output_dir):
    path = _progress_path(output_dir)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

def _source_doc_count(progress, source_name):
    stats = progress.get("stats", {}) if isinstance(progress, Mapping) else {}
    source_stats = stats.get(source_name, {}) if isinstance(stats, Mapping) else {}
    try:
        return int(source_stats.get("docs", 0))
    except (TypeError, ValueError):
        return 0

def normalize_completed_sources(progress, require_docs=True):
    if not isinstance(progress, Mapping):
        return []
    raw_completed = progress.get("completed_sources", [])
    if not isinstance(raw_completed, list):
        return []
    completed = set(raw_completed)
    return [
        name for name in DATASET_SOURCES
        if name in completed and (not require_docs or _source_doc_count(progress, name) > 0)
    ]

def add_completed_source(completed_sources, source_name):
    completed = set(completed_sources)
    completed.add(source_name)
    return [name for name in DATASET_SOURCES if name in completed]

def save_progress(output_dir, progress):
    progress = dict(progress)
    progress["completed_sources"] = normalize_completed_sources(progress, require_docs=True)
    path = _progress_path(output_dir)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(progress, f, indent=2)
    os.replace(tmp, path)

def cleanup_tmp_files(output_dir):
    for tmp in glob.glob(os.path.join(output_dir, "*.tmp")):
        logger.info(f"  Removing leftover tmp file: {tmp}")
        os.remove(tmp)

def cleanup_generated_data_files(output_dir):
    for pattern in ("*.parquet", "*.tmp", "*.valpart"):
        for path in glob.glob(os.path.join(output_dir, pattern)):
            logger.info(f"  Removing generated data file: {path}")
            os.remove(path)

def val_part_path(output_dir, part_idx):
    return os.path.join(output_dir, f"val_part_{part_idx:05d}.valpart")

def list_val_part_files(output_dir):
    return sorted(glob.glob(os.path.join(output_dir, "val_part_*.valpart")))

def split_train_val_docs(docs, val_ratio=0.05):
    """Split document list into train/val sets.

    Guarantees at least one training document even when the source is very
    small, preventing the degenerate case where all docs end up in validation.
    """
    if not docs:
        return [], []
    if not (0 < val_ratio < 1):
        logger.warning("Invalid val_ratio %s, falling back to 0.05", val_ratio)
        val_ratio = 0.05
    val_size = max(1, int(len(docs) * val_ratio))
    # Never allocate the entire set to validation
    if val_size >= len(docs):
        val_size = len(docs) - 1
    if val_size <= 0:
        return docs, []
    return docs[:-val_size], docs[-val_size:]

def write_val_part(docs, output_dir, part_idx):
    if not docs:
        return part_idx
    write_shard(docs, val_part_path(output_dir, part_idx))
    return part_idx + 1

def write_final_val_shard(output_dir, shard_path):
    val_parts = list_val_part_files(output_dir)
    if not val_parts:
        logger.warning("  No validation docs collected; validation shard will not be written.")
        return False

    val_docs = []
    for path in val_parts:
        val_docs.extend(pd.read_parquet(path)["text"].tolist())
    write_shard(val_docs, shard_path)

    for path in val_parts:
        os.remove(path)
    return True

def create_checkpoint(output_dir, checkpoint_dir, checkpoint_number):
    # Check free disk before creating checkpoint (needs at least 6 GiB)
    _check_free_space(Path(checkpoint_dir), 6 * 1024 ** 3)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_name = f"data_checkpoint_{checkpoint_number:03d}"
    archive_path = os.path.join(checkpoint_dir, checkpoint_name)

    logger.info(f"  💾 Creating checkpoint #{checkpoint_number} → {archive_path}.tar.gz")
    archive_full = archive_path + ".tar.gz"
    tmp_archive = archive_full + ".tmp"
    base_name = os.path.basename(output_dir)

    # Built by hand rather than shutil.make_archive so transient files can be skipped.
    # make_archive walks the directory and stats each entry; *.valpart and *.tmp are
    # written and deleted as shards flush (write_final_val_shard removes them all after
    # merging), so one can disappear between the listing and the stat, and the resulting
    # FileNotFoundError used to kill the whole run mid-download. They are also worthless
    # inside a checkpoint: only the finished .parquet shards and progress.json matter.
    with tarfile.open(tmp_archive, "w:gz") as tar:
        for entry in sorted(os.listdir(output_dir)):
            if entry.endswith((".valpart", ".tmp")):
                continue
            try:
                tar.add(os.path.join(output_dir, entry), arcname=os.path.join(base_name, entry))
            except FileNotFoundError:
                logger.warning(f"  Skipping file that vanished during checkpoint: {entry}")
    os.replace(tmp_archive, archive_full)

    if not os.path.isfile(archive_full):
        raise RuntimeError(f"Checkpoint archive not found after creation: {archive_full}")
    size_gb = os.path.getsize(archive_full) / (1024**3)
    logger.info(f"  💾 Checkpoint saved: {archive_full} ({size_gb:.2f} GB)")
    return archive_full

def get_total_parquet_bytes(output_dir):
    total = 0
    for f in os.listdir(output_dir):
        if f.endswith(".parquet"):
            total += os.path.getsize(os.path.join(output_dir, f))
    return total

def print_status(output_dir):
    progress = load_progress(output_dir)
    if progress is None:
        print(f"\n  No progress found in {output_dir}.")
        print("  Run without --status to start downloading.\n")
        return

    completed = normalize_completed_sources(progress, require_docs=True)
    raw_completed_list = progress.get("completed_sources", [])
    if not isinstance(raw_completed_list, list):
        raw_completed_list = []
    raw_completed = set(raw_completed_list)
    completed_set = set(completed)
    all_sources = list(DATASET_SOURCES.keys())
    requested_sources = progress.get("requested_sources") or all_sources
    requested_sources = [name for name in requested_sources if name in DATASET_SOURCES]
    requested_completed = completed_set & set(requested_sources)
    is_done = progress.get("finished", False)

    print()
    print("=" * 70)
    print("  mesosfer DATA PREPARATION — PROGRESS")
    print("=" * 70)
    print()

    for name in all_sources:
        src = DATASET_SOURCES[name]
        cat_emoji = _category_emoji(src["category"])
        source_stats = progress.get("stats", {}).get(name)
        docs = _source_doc_count(progress, name)
        chars = source_stats.get("chars", 0) if source_stats else 0
        if name in completed_set:
            est_tok = int(chars / CHARS_PER_TOKEN) if chars else 0
            print(f"  {cat_emoji} {name:25s}  ✅  {docs:>10,} docs  ~{est_tok:>12,} tokens")
        elif name in raw_completed and docs == 0:
            print(f"  {cat_emoji} {name:25s}  ⚠️   {docs:>10,} docs  will retry")
        else:
            print(f"  {cat_emoji} {name:25s}  ⏳  pending")

    print()
    print(f"  Shards written so far : {progress.get('next_shard_idx', 0)}")
    print(f"  Sources completed     : {len(requested_completed)} / {len(requested_sources)}")
    if is_done and len(requested_completed) == len(requested_sources):
        print("  Overall status        : ✅ FINISHED")
    elif is_done:
        print("  Overall status        : ⚠️  FINISHED WITH PENDING/EMPTY SOURCES")
    else:
        print("  Overall status        : 🔄 IN PROGRESS (re-run to resume)")
    print()

def _effective_source_cap_tokens(source_config):
    return source_config.get("max_tokens")

def print_dry_run_summary(args, source_names, output_dir):
    print()
    print("=" * 70)
    print("  mesosfer DATA PREPARATION — DRY RUN")
    print("=" * 70)
    print("  [DRY RUN] No data will be downloaded or written.")
    print(f"  Output: {output_dir}")
    print(f"  Shuffle mode: {args.shuffle_mode}")
    if args.max_tokens is not None:
        print(f"  Global max tokens: ~{args.max_tokens:,}")
    print(f"  Sources: {len(source_names)}")
    print()

    bucket_caps = defaultdict(int)
    for name in source_names:
        src = DATASET_SOURCES[name]
        cat = _source_bucket(name)
        emoji = _category_emoji(cat)
        weight = DOMAIN_SAMPLING_WEIGHTS.get(name, 1.0)
        cap = _effective_source_cap_tokens(src)
        cap_label = f"~{cap / 1e6:.0f}M tokens" if isinstance(cap, int) else "all available"
        bucket_caps[cat] += cap if isinstance(cap, int) else 0

        local_label = ""
        if src.get("source_type") == "local_files":
            local_label = f" | {count_local_files(src)} local files"

        print(f"  {emoji} {name:28s} | w={weight:.1f} | {cap_label:14s}{local_label} | {src['description']}")

    if bucket_caps:
        target_total = sum(TARGET_DOMAIN_SHARE.get(b, 0.0) for b in bucket_caps) or 1.0
        print()
        print("  Planned mix (target share of --max-tokens vs what the sources can supply):")
        for bucket, cap in sorted(bucket_caps.items()):
            share = TARGET_DOMAIN_SHARE.get(bucket, 0.0) / target_total
            emoji = _category_emoji(bucket)
            want = f"~{int(args.max_tokens * share):,} tokens" if args.max_tokens else "budget-limited"
            short = "  ⚠️  under-supplied" if args.max_tokens and cap < args.max_tokens * share else ""
            print(f"  {emoji} {bucket.capitalize():15s}: {100 * share:5.1f}% | want {want:>20s} | "
                  f"available ~{cap:,} tokens{short}")
    print()

# =============================================================================
# Interleaved domain-mixed shuffle
# =============================================================================

def _build_sampling_probs(source_names, temperature=1.0):
    """Build normalized sampling probabilities from DOMAIN_SAMPLING_WEIGHTS.

    temperature > 1.0 flattens the distribution (more uniform).
    temperature < 1.0 sharpens it (high-weight domains sampled even more).
    temperature = 1.0 uses weights as-is.
    """
    import math
    raw = [DOMAIN_SAMPLING_WEIGHTS.get(name, 1.0) for name in source_names]
    if temperature != 1.0:
        raw = [w ** (1.0 / temperature) for w in raw]
    total = sum(raw)
    return [w / total for w in raw]


def _build_bucket_samplers(source_names, temperature=1.0):
    """Group sources by share bucket, each with its own intra-bucket probabilities."""
    grouped = defaultdict(list)
    for name in source_names:
        grouped[_source_bucket(name)].append(name)
    return {b: (names, _build_sampling_probs(names, temperature)) for b, names in grouped.items()}


def _pick_bucket(samplers, stats):
    """Pick the bucket furthest below its target share of everything written so far.

    Greedy deficit selection instead of one flat weighted draw: it keeps the realized
    mix on target even though sources differ wildly in document length, and it makes
    the remaining buckets absorb the share of a bucket that runs dry.
    """
    written_by_bucket = defaultdict(int)
    for name, s in stats.items():
        written_by_bucket[_source_bucket(name)] += s["chars"]
    written = sum(written_by_bucket.values()) or 1
    target_total = sum(TARGET_DOMAIN_SHARE.get(b, 0.0) for b in samplers) or 1.0
    return max(samplers, key=lambda b: TARGET_DOMAIN_SHARE.get(b, 0.0) / target_total
               - written_by_bucket[b] / written)


def _expected_source_share(name, temperature=1.0):
    """Fraction of the whole corpus a single source is expected to get."""
    bucket = _source_bucket(name)
    peers = [n for n in DATASET_SOURCES if _source_bucket(n) == bucket]
    probs = dict(zip(peers, _build_sampling_probs(peers, temperature)))
    return TARGET_DOMAIN_SHARE.get(bucket, 0.0) * probs[name]


def interleaved_shuffle_main(args, source_names, output_dir):
    """Stream docs from all sources simultaneously with weighted interleaving.

    Instead of processing source-by-source (which creates domain-sequential shards),
    this opens ALL source iterators at once and picks which source to draw from
    using temperature-weighted random sampling.

    Each shard gets a MIX of domains. Within each shard, we write small 'runs'
    of the same domain (cluster_size docs) for locality, but the overall shard
    is domain-diverse.

    Example shard content (cluster_size=4):
        [exploit, exploit, exploit, exploit, cve, cve, cve, cve, wiki, wiki, ...]
    Instead of:
        [exploit * 100_000]  then  [wiki * 100_000]   ← old sequential mode
    """
    import random
    rng = random.Random(42)

    t0 = time.time()

    # Resume. Without this the function always started at shard_00000, so running it a
    # second time (e.g. to add sources with --sources) silently overwrote the shards the
    # previous run had written, and only the tail of the older run survived. The resume
    # code that used to exist lived below the interleaved dispatch in main(), i.e. only
    # on the sequential path, so the default mode never reached it.
    # --fresh already deleted progress.json and the parquet files before we get here, so
    # finding a progress file at this point unambiguously means "continue".
    prev = load_progress(output_dir)
    requested_all = list(source_names)
    already_done: set = set()
    if prev is not None:
        already_done = set(normalize_completed_sources(prev, require_docs=True))
        requested_all = sorted(set(prev.get("requested_sources") or []) | set(source_names))
        pending = [n for n in source_names if n not in already_done]
        skipped = [n for n in source_names if n in already_done]
        shard_idx = prev.get("next_shard_idx", 0)
        val_part_idx = prev.get("next_val_part_idx", 0)
        stats = defaultdict(lambda: {"docs": 0, "chars": 0})
        for name, counts in prev.get("stats", {}).items():
            stats[name] = counts
        checkpoint_number = prev.get("checkpoint_number", 0)
        bytes_at_last_checkpoint = prev.get("bytes_at_last_checkpoint", 0)
        logger.info(f"Resuming interleaved run at shard_idx={shard_idx} "
                    f"({len(already_done)} source(s) already complete).")
        if skipped:
            logger.info(f"  Skipping already-complete sources: {', '.join(skipped)}")
        if not pending:
            print("\n  ✅ All requested sources are already complete. Use --fresh to redo them.\n")
            return
        source_names = pending
    else:
        shard_idx = 0
        val_part_idx = 0
        stats = defaultdict(lambda: {"docs": 0, "chars": 0})
        checkpoint_number = 0
        bytes_at_last_checkpoint = 0

    global_max_chars = int(args.max_tokens * CHARS_PER_TOKEN) if args.max_tokens else None
    # Cumulative across runs: --max-tokens is a total budget, not a per-invocation one.
    global_chars = sum(s["chars"] for s in stats.values())
    if global_max_chars is not None and global_chars >= global_max_chars:
        print(f"\n  ⚠️  Token budget already reached "
              f"(~{int(global_chars / CHARS_PER_TOKEN):,} >= {args.max_tokens:,}).")
        print("  Raise --max-tokens to give the new sources room, or use --fresh.\n")
        return
    cluster_size = getattr(args, 'cluster_size', 32)  # docs of same domain run
    temperature = getattr(args, 'sampling_temperature', 1.2)

    checkpoint_dir = args.checkpoint_dir or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    checkpoint_every_bytes = int(args.checkpoint_every_gb * 1024 * 1024 * 1024)

    # Open all source iterators
    logger.info(f"Opening {len(source_names)} source iterators for interleaved streaming...")
    logger.info(f"  Cluster size: {cluster_size} docs per domain run")
    logger.info(f"  Sampling temperature: {temperature}")
    source_iters = {}
    for name in source_names:
        config = DATASET_SOURCES[name]
        source_iters[name] = iter(stream_dataset_texts(name, config))

    # Active sources (those not yet exhausted)
    active_sources = list(source_names)
    samplers = _build_bucket_samplers(active_sources, temperature)

    current_shard = []
    exhausted_count = 0

    print()
    print("  🔀 INTERLEAVED SHUFFLE MODE")
    print(f"  Cluster size: {cluster_size} | Temperature: {temperature}")
    print(f"  Active sources: {len(active_sources)}")
    print("  Target mix: " + " | ".join(
        f"{b} {100 * TARGET_DOMAIN_SHARE.get(b, 0.0):.0f}%" for b in sorted(samplers)))
    print()

    while active_sources:
        if global_max_chars is not None and global_chars >= global_max_chars:
            logger.info("Global --max-tokens limit reached.")
            break

        # Pick the bucket that is behind on its target share, then a source inside it
        bucket = _pick_bucket(samplers, stats)
        bucket_names, bucket_probs = samplers[bucket]
        chosen_name = rng.choices(bucket_names, weights=bucket_probs, k=1)[0]
        it = source_iters[chosen_name]

        # Draw a cluster of docs from the chosen source
        docs_drawn = 0
        for _ in range(cluster_size):
            try:
                text = next(it)
            except StopIteration:
                # Source exhausted — remove it
                logger.info(f"  ✓ {chosen_name} exhausted after {stats[chosen_name]['docs']:,} docs")
                active_sources.remove(chosen_name)
                del source_iters[chosen_name]
                if active_sources:
                    samplers = _build_bucket_samplers(active_sources, temperature)
                exhausted_count += 1
                break

            if global_max_chars is not None and global_chars + len(text) > global_max_chars:
                global_chars = global_max_chars
                break

            current_shard.append(text)
            stats[chosen_name]["docs"] += 1
            stats[chosen_name]["chars"] += len(text)
            global_chars += len(text)
            docs_drawn += 1

            # Flush shard when full
            if len(current_shard) >= args.shard_size:
                # Shuffle WITHIN the shard for extra randomness
                rng.shuffle(current_shard)
                train_docs, val_docs = split_train_val_docs(current_shard, args.val_ratio)
                shard_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.parquet")
                if train_docs:
                    write_shard(train_docs, shard_path)
                    shard_idx += 1
                val_part_idx = write_val_part(val_docs, output_dir, val_part_idx)
                current_shard = []

                current_bytes = get_total_parquet_bytes(output_dir)
                if current_bytes - bytes_at_last_checkpoint >= checkpoint_every_bytes:
                    create_checkpoint(output_dir, checkpoint_dir, checkpoint_number)
                    bytes_at_last_checkpoint = current_bytes
                    checkpoint_number += 1

        # Progress logging every 500k docs
        total_docs = sum(s["docs"] for s in stats.values())
        if total_docs % 500_000 < cluster_size:
            est_tokens = int(global_chars / CHARS_PER_TOKEN)
            logger.info(f"  Interleaved progress: {total_docs:,} docs, ~{est_tokens:,} tokens, "
                        f"{len(active_sources)} sources active")

    # Flush remaining docs
    if current_shard:
        rng.shuffle(current_shard)
        train_remaining, val_docs = split_train_val_docs(current_shard, args.val_ratio)
        if train_remaining:
            shard_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.parquet")
            write_shard(train_remaining, shard_path)
            shard_idx += 1
        val_part_idx = write_val_part(val_docs, output_dir, val_part_idx)

    # Write final validation shard
    if list_val_part_files(output_dir):
        val_shard_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.parquet")
        if write_final_val_shard(output_dir, val_shard_path):
            shard_idx += 1

    elapsed = time.time() - t0

    # Summary
    print()
    print("=" * 85)
    print("  INTERLEAVED SHUFFLE SUMMARY (RINGKASAN DATA PRETRAINING)")
    print("=" * 85)
    print(f"  {'Source (Sumber)':28s} | {'Weight':>6s} | {'Rows/Docs (Baris)':>18s} | {'Est. Tokens (Token)':>22s}")
    print("-" * 85)
    total_docs = 0
    total_chars = 0
    # Every source with data on disk, not just the ones this invocation handled, so the
    # totals and category shares describe the whole dataset across incremental batches.
    summary_sources = [n for n in DATASET_SOURCES if stats.get(n, {}).get("docs")]
    for source_name in summary_sources:
        s = stats[source_name]
        est_tokens = int(s["chars"] / CHARS_PER_TOKEN)
        total_docs += s["docs"]
        total_chars += s["chars"]
        cat = DATASET_SOURCES[source_name]["category"]
        emoji = _category_emoji(cat)
        weight = DOMAIN_SAMPLING_WEIGHTS.get(source_name, 1.0)
        source_label = f"{emoji} {source_name}"
        print(f"  {source_label:28s} | w={weight:.1f} | {s['docs']:18,} | {est_tokens:22,}")

    total_tokens = int(total_chars / CHARS_PER_TOKEN)
    print("-" * 85)
    print(f"  {'TOTAL':28s} |      | {total_docs:18,} | {total_tokens:22,}")
    print("-" * 85)
    print(f"  Shards written: {shard_idx}")
    print(f"  Output directory: {output_dir}")
    print(f"  Time elapsed: {elapsed/60:.1f} minutes")
    print()

    # Realized mix vs TARGET_DOMAIN_SHARE
    if total_tokens > 0:
        for bucket, target in TARGET_DOMAIN_SHARE.items():
            tokens = sum(int(stats[n]["chars"] / CHARS_PER_TOKEN)
                         for n in summary_sources if _source_bucket(n) == bucket)
            if tokens <= 0:
                continue
            emoji = _category_emoji(bucket)
            pct = 100 * tokens / total_tokens
            print(f"  {emoji} {bucket.capitalize():15s}: ~{tokens:,} tokens ({pct:.1f}%, target {100 * target:.0f}%)")
    print()

    # Final checkpoint, subject to the same threshold as the periodic ones. This used to
    # run unconditionally, which meant --checkpoint-every-gb could not switch checkpoints
    # off: each archive is a full tar.gz of the output dir, so N incremental --sources
    # batches left N snapshots behind. Seven batches over a 15 GB dataset produced 46 GB
    # of archives and filled the disk.
    final_bytes = get_total_parquet_bytes(output_dir)
    if global_chars > 0 and (final_bytes - bytes_at_last_checkpoint) >= checkpoint_every_bytes:
        create_checkpoint(output_dir, checkpoint_dir, checkpoint_number)
    elif global_chars > 0:
        logger.info(f"  Skipping final checkpoint (< {args.checkpoint_every_gb:g} GB since the last one). "
                    f"Resume uses progress.json, not the archives.")

    save_progress(output_dir, {
        # Union, not replacement: overwriting these was what made each --sources batch
        # erase the record of the previous ones, so --status only ever showed the last run.
        "completed_sources": sorted(already_done | set(source_names)),
        "next_shard_idx": shard_idx,
        "stats": dict(stats),
        "finished": True,
        "shuffle_mode": "interleaved",
        "bytes_at_last_checkpoint": get_total_parquet_bytes(output_dir),
        "checkpoint_number": checkpoint_number + 1,
        "next_val_part_idx": val_part_idx,
        "requested_sources": requested_all,
    })

    verify_shard_balance(output_dir)


def download_climbmix_shards(num_files, num_workers):
    """
    Downloads ClimbMix general pretraining shards directly from HuggingFace.
    This replaces the legacy mesosfer.data.dataset downloading CLI to unify all downloads.
    """
    import requests
    from concurrent.futures import ThreadPoolExecutor

    base_dir = get_base_dir()
    climbmix_dir = os.path.join(base_dir, "base_data_climbmix")
    os.makedirs(climbmix_dir, exist_ok=True)

    base_url = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
    max_shard = 6542

    num_train = max_shard if num_files == -1 else min(num_files, max_shard)
    ids = list(range(num_train))
    if max_shard not in ids:
        ids.append(max_shard)  # Always download the validation shard

    logger.info(f"Downloading {len(ids)} ClimbMix shards using {num_workers} parallel workers...")
    logger.info(f"Target directory: {climbmix_dir}")

    def download_shard(index):
        filename = f"shard_{index:05d}.parquet"
        filepath = os.path.join(climbmix_dir, filename)
        if os.path.exists(filepath):
            return True

        url = f"{base_url}/{filename}"
        temp_path = filepath + ".tmp"

        for attempt in range(1, 6):
            try:
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                os.replace(temp_path, filepath)
                logger.info(f"Successfully downloaded {filename}")
                return True
            except Exception as e:
                logger.warning(f"Attempt {attempt}/5 failed for {filename}: {e}")
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                time.sleep(2 ** attempt)
        return False

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(download_shard, ids))

    successful = sum(1 for r in results if r)
    logger.info(f"Done! Downloaded: {successful}/{len(ids)} shards to {climbmix_dir}")


# =============================================================================
# =============================================================================
# Helper functions for depth-based parameter scaling
# =============================================================================

def _get_vocab_size():
    try:
        base_dir = get_base_dir()
        tokenizer_path = os.path.join(base_dir, "tokenizer", "tokenizer.json")
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                vocab = data.get("model", {}).get("vocab", {})
                if vocab:
                    return len(vocab)
    except Exception:
        pass
    return 96000  # Default fallback


def has_ve(layer_idx, n_layer, ve_layers=-1):
    if ve_layers is None or ve_layers < 0:
        return layer_idx % 2 == (n_layer - 1) % 2
    return layer_idx < ve_layers or layer_idx == n_layer - 1


def estimate_tokens_for_depth(depth, target_param_data_ratio=15, aspect_ratio=128, head_dim=128, ve_layers=-1):
    vocab_size = _get_vocab_size()
    padded_vocab_size = ((vocab_size + 63) // 64) * 64
    
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    num_kv_head = num_heads
    
    num_layers_with_ve = sum(1 for i in range(depth) if has_ve(i, depth, ve_layers))
    
    transformer_matrices = 12 * model_dim * model_dim * depth + num_layers_with_ve * (12 * num_kv_head)
    lm_head = padded_vocab_size * model_dim
    num_scaling_params = transformer_matrices + lm_head
    
    target_tokens = int(target_param_data_ratio * num_scaling_params)
    return target_tokens


# Depths that get a CLI shortcut (--12 / --d12 etc). Any other depth still works via --depth N.
DEPTH_SHORTCUTS = (12, 16, 20, 24, 32)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Download and prepare the mesosfer cybersecurity pretraining dataset")
    parser.add_argument("--output-dir", type=str, default="base_data_cybersecurity",
                        help="Output directory name under mesosfer_BASE_DIR")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Global maximum tokens across ALL sources")
    parser.add_argument("--shard-size", type=int, default=100_000,
                        help="Number of documents per shard")
    parser.add_argument("--sources", type=str, default=None,
                        help="Comma-separated list of sources to download")
    parser.add_argument("--val-ratio", type=float, default=0.008,
                        help="Fraction of data to use for validation (0.8%% optimal for 100B+ token runs)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only show what would be downloaded, don't actually download")
    parser.add_argument("--status", action="store_true",
                        help="Show download progress and exit")
    parser.add_argument("--fresh", action="store_true",
                        help="Discard any previous progress and start from scratch")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory to save checkpoints")
    parser.add_argument("--checkpoint-every-gb", type=float, default=10.0,
                        help="Create a checkpoint every N GB of data written (must be > 0, default 10.0)")
    parser.add_argument("--shuffle-mode", type=str, default="interleaved",
                        choices=["sequential", "interleaved"],
                        help="'interleaved' = domain-mixed shards (recommended), "
                             "'sequential' = one source at a time (legacy)")
    parser.add_argument("--cluster-size", type=int, default=32,
                        help="Number of consecutive docs from same domain per run "
                             "in interleaved mode (default: 32, higher = smoother domain transitions)")
    parser.add_argument("--sampling-temperature", type=float, default=1.2,
                        help="Temperature for domain sampling weights. "
                             ">1.0 = more uniform (reduces domain burst spikes), <1.0 = sharper (favor high-weight domains)")
    parser.add_argument("--download-climbmix", type=int, default=None,
                        help="Number of ClimbMix pretraining shards to download directly (e.g. 170)")
    parser.add_argument("--download-workers", type=int, default=4,
                        help="Number of parallel download workers (default: 4)")
    parser.add_argument("--depth", type=int, default=None,
                        help="Target Transformer depth to scale the dataset size to (e.g. 12, 16, 20, 24, 32)")
    parser.add_argument("--target-param-data-ratio", type=float, default=15.0,
                        help="Data-to-parameters ratio used to scale dataset size from depth (default: 15)")
    # model_dim = depth x aspect_ratio, so this shifts the token target by ~4x between 64 and 128.
    # It MUST match the --aspect-ratio you later pass to base_train, or the dataset is sized for a
    # different model than the one you train (i.e. exactly the over/under-training this flag prevents).
    parser.add_argument("--aspect-ratio", type=int, default=128,
                        help="Model dim = depth x aspect-ratio; must match base_train (default: 128)")
    # Shortcuts for specific depths: both --12 and --d12 map to --depth 12
    for _d in DEPTH_SHORTCUTS:
        parser.add_argument(f"--{_d}", f"--d{_d}", dest=f"depth_{_d}", action="store_true",
                            help=f"Shortcut for --depth {_d}")
    parser.add_argument("--sample-10k", action="store_true",
                      help="Prepare a tiny 10,000-token sample of datasets for fast testing")
    args = parser.parse_args()

    # Resolve depth shortcuts
    for _d in DEPTH_SHORTCUTS:
        if getattr(args, f"depth_{_d}"):
            args.depth = _d
            break

    # Scale max_tokens based on depth if specified
    if args.depth is not None:
        target_tokens = estimate_tokens_for_depth(
            args.depth,
            target_param_data_ratio=args.target_param_data_ratio,
            aspect_ratio=args.aspect_ratio,
        )
        logger.info(
            f"Target depth {args.depth} (aspect-ratio {args.aspect_ratio}, ratio {args.target_param_data_ratio:g}). "
            f"Calculated compute-optimal pretrain tokens: {target_tokens:,}"
        )
        if args.max_tokens is None:
            args.max_tokens = target_tokens
        else:
            logger.info(f"Using explicitly requested --max-tokens={args.max_tokens:,} instead of depth-derived limit.")

    # Override defaults if `--sample-10k` is active
    if args.sample_10k:
        if args.max_tokens is None:
            args.max_tokens = 10000
        if args.shard_size == 100_000:
            args.shard_size = 100
        if args.output_dir == "base_data_cybersecurity":
            args.output_dir = "base_data_cybersecurity_sample"

    # Validate checkpoint interval
    if args.checkpoint_every_gb <= 0:
        parser.error("--checkpoint-every-gb must be > 0")

    # Validate log level from environment
    env_level = os.getenv("PREPARE_DATA_LOG_LEVEL")
    if env_level is not None:
        numeric_level = getattr(logging, env_level.upper(), None)
        if not isinstance(numeric_level, int):
            logger.warning("Invalid PREPARE_DATA_LOG_LEVEL=%r, falling back to INFO", env_level)
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(numeric_level)

    base_dir = get_base_dir()

    if args.download_climbmix is not None:
        download_climbmix_shards(args.download_climbmix, args.download_workers)
        return

    # Automatically ensure ClimbMix shards are present before preparing cybersecurity/code datasets
    if not args.dry_run and not args.status:
        climbmix_dir = os.path.join(base_dir, "base_data_climbmix")
        existing_shards = [f for f in os.listdir(climbmix_dir) if f.endswith('.parquet')] if os.path.exists(climbmix_dir) else []
        if args.sample_10k:
            required_shards = 2
        elif args.max_tokens is not None:
            # Each shard is ~60M tokens. Download what ClimbMix's slice of the general
            # bucket actually asks for, plus 20% for variance — a flat 5% guess starved
            # the general bucket once its target share went up.
            climbmix_needed = int((args.max_tokens * _expected_source_share("climbmix") * 1.2) / 60_000_000)
            required_shards = max(2, min(climbmix_needed + 2, 171))
        else:
            required_shards = 171
        if len(existing_shards) < required_shards:
            shards_to_download = required_shards - 1
            logger.info(f"  ℹ️ ClimbMix general pretraining shards not found or incomplete (need {required_shards} shards). Downloading baseline {shards_to_download} shards automatically...")
            download_climbmix_shards(shards_to_download, args.download_workers)

    output_dir = os.path.join(base_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if args.status:
        print_status(output_dir)
        return

    if args.sources:
        source_names = [s.strip() for s in args.sources.split(",")]
        for name in source_names:
            if name not in DATASET_SOURCES:
                available = ", ".join(DATASET_SOURCES.keys())
                raise ValueError(f"Unknown source: {name}. Available: {available}")
    else:
        source_names = list(DATASET_SOURCES.keys())
        if args.sample_10k:
            # Skip extremely slow/heavy sources for quick sample testing
            source_names = [s for s in source_names if s not in ("all_cve_records", "circl_vuln_patch")]

    if args.dry_run:
        print_dry_run_summary(args, source_names, output_dir)
        return

    if args.fresh:
        prog_path = _progress_path(output_dir)
        if os.path.exists(prog_path):
            os.remove(prog_path)
            logger.info("Cleared previous progress (--fresh).")
        cleanup_generated_data_files(output_dir)

    # =========================================================================
    # Route: interleaved mode (default) — domain-mixed shards
    # =========================================================================
    if args.shuffle_mode == "interleaved":
        cleanup_tmp_files(output_dir)
        interleaved_shuffle_main(args, source_names, output_dir)
        return

    # =========================================================================
    # Legacy sequential mode below (--shuffle-mode sequential)
    # =========================================================================
    prev_progress = load_progress(output_dir)
    resuming = prev_progress is not None and not args.fresh
    reopen_finished_run = False
    completed_sources = normalize_completed_sources(prev_progress, require_docs=True) if resuming else []
    if resuming and prev_progress.get("finished"):
        uncompleted = [s for s in source_names if s not in completed_sources]
        if not uncompleted:
            print("\n  ✅ All sources already downloaded! Nothing to do.")
            print("  Use --fresh to discard progress and re-download.\n")
            return
        else:
            reopen_finished_run = True
            print(f"\n  🔄 Found {len(uncompleted)} new sources. Resuming to download them...")

    print()
    print("=" * 70)
    print("  mesosfer DATA PREPARATION v3")
    print("  Cybersecurity-focused pretraining dataset (Public)")
    if resuming:
        print("  🔄 RESUMING from previous run")
    print("=" * 70)
    print()
    
    total_target = 0
    for name in source_names:
        src = DATASET_SOURCES[name]
        cat = src["category"]
        max_tok = src.get("max_tokens", "all")
        if isinstance(max_tok, int):
            total_target += max_tok
            max_tok_str = f"~{max_tok/1e9:.1f}B tokens"
        else:
            max_tok_str = "all available"
        emoji = "🔒" if cat == "cybersecurity" else "📚"
        status = " ✅ done" if name in completed_sources else ""
        print(f"  {emoji} {name:25s} | {max_tok_str:20s} | {src['description']}{status}")
    
    if total_target > 0:
        print(f"\n  Total target: ~{total_target/1e9:.1f}B tokens")
        
    if args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
    else:
        checkpoint_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    checkpoint_every_bytes = int(args.checkpoint_every_gb * 1024 * 1024 * 1024)

    print(f"  Output: {output_dir}")
    print(f"  Shard size: {args.shard_size:,} docs per shard")
    print(f"  Checkpoints: every {args.checkpoint_every_gb:.0f} GB → {checkpoint_dir}")
    print()

    cleanup_tmp_files(output_dir)

    t0 = time.time()
    if resuming:
        shard_idx = prev_progress.get("next_shard_idx", 0)
        stats = defaultdict(lambda: {"docs": 0, "chars": 0})
        for k, v in prev_progress.get("stats", {}).items():
            stats[k] = v
        logger.info(f"Resuming from shard_idx={shard_idx}, {len(completed_sources)} sources already done.")
    else:
        shard_idx = 0
        stats = defaultdict(lambda: {"docs": 0, "chars": 0})
        completed_sources = []
        
    current_shard = []
    bytes_at_last_checkpoint = prev_progress.get("bytes_at_last_checkpoint", 0) if resuming else 0
    checkpoint_number = prev_progress.get("checkpoint_number", 0) if resuming else 0
    val_part_idx = prev_progress.get("next_val_part_idx", len(list_val_part_files(output_dir))) if resuming else 0
    global_max_chars = int(args.max_tokens * CHARS_PER_TOKEN) if args.max_tokens else None
    global_chars = sum(s["chars"] for s in stats.values())
    global_chars_at_start = global_chars
    reached_global_limit = global_max_chars is not None and global_chars >= global_max_chars

    if reopen_finished_run and shard_idx > 0:
        last_val_shard = os.path.join(output_dir, f"shard_{shard_idx - 1:05d}.parquet")
        if os.path.exists(last_val_shard):
            reopened_val_part = val_part_path(output_dir, val_part_idx)
            os.replace(last_val_shard, reopened_val_part)
            shard_idx -= 1
            val_part_idx += 1
            logger.info(f"Reopened previous validation shard for extension: {reopened_val_part}")

    for source_name in source_names:
        if reached_global_limit:
            logger.info("Global --max-tokens limit reached; stopping source processing.")
            break

        if source_name in completed_sources:
            logger.info(f"Skipping {source_name} (already completed in previous run)")
            continue

        source_config = DATASET_SOURCES[source_name]
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {source_name}")
        logger.info(f"{'='*60}")

        source_completed = True
        docs_before_source = stats[source_name]["docs"]
        for text in stream_dataset_texts(source_name, source_config, skip_docs=docs_before_source):
            if global_max_chars is not None and global_chars + len(text) > global_max_chars:
                reached_global_limit = True
                source_completed = False
                logger.info(f"Global --max-tokens limit reached while processing {source_name}.")
                break

            current_shard.append(text)
            stats[source_name]["docs"] += 1
            stats[source_name]["chars"] += len(text)
            global_chars += len(text)

            if len(current_shard) >= args.shard_size:
                train_docs, val_docs = split_train_val_docs(current_shard, args.val_ratio)
                shard_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.parquet")
                if train_docs:
                    write_shard(train_docs, shard_path)
                    shard_idx += 1
                val_part_idx = write_val_part(val_docs, output_dir, val_part_idx)
                current_shard = []

                current_bytes = get_total_parquet_bytes(output_dir)
                if current_bytes - bytes_at_last_checkpoint >= checkpoint_every_bytes:
                    create_checkpoint(output_dir, checkpoint_dir, checkpoint_number)
                    bytes_at_last_checkpoint = current_bytes
                    checkpoint_number += 1

        docs_written_for_source = stats[source_name]["docs"] - docs_before_source
        if source_completed and docs_written_for_source <= 0:
            source_completed = False
            logger.warning(f"  ⚠️ {source_name} produced 0 usable docs; leaving it pending for retry.")

        if source_completed:
            completed_sources = add_completed_source(completed_sources, source_name)
        save_progress(output_dir, {
            "completed_sources": completed_sources,
            "next_shard_idx": shard_idx,
            "stats": dict(stats),
            "finished": False,
            "bytes_at_last_checkpoint": bytes_at_last_checkpoint,
            "checkpoint_number": checkpoint_number,
            "next_val_part_idx": val_part_idx,
            "requested_sources": source_names,
        })
        logger.info(f"  ✅ Progress saved — {source_name} {'complete' if source_completed else 'partial'}.")

        if reached_global_limit:
            break

    if current_shard:
        train_remaining, val_docs = split_train_val_docs(current_shard, args.val_ratio)

        if train_remaining:
            shard_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.parquet")
            write_shard(train_remaining, shard_path)
            shard_idx += 1

        val_part_idx = write_val_part(val_docs, output_dir, val_part_idx)

    if list_val_part_files(output_dir):
        val_shard_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.parquet")
        if write_final_val_shard(output_dir, val_shard_path):
            shard_idx += 1

    elapsed = time.time() - t0

    print()
    print("=" * 76)
    print("  PRETRAINING DATA SUMMARY (RINGKASAN DATA PRETRAINING)")
    print("=" * 76)
    print(f"  {'Source (Sumber)':28s} | {'Rows/Docs (Baris)':>18s} | {'Est. Tokens (Token)':>22s}")
    print("-" * 76)
    total_docs = 0
    total_chars = 0
    for source_name in source_names:
        s = stats[source_name]
        est_tokens = int(s["chars"] / CHARS_PER_TOKEN)
        total_docs += s["docs"]
        total_chars += s["chars"]
        cat = DATASET_SOURCES[source_name]["category"]
        emoji = _category_emoji(cat)
        source_label = f"{emoji} {source_name}"
        print(f"  {source_label:28s} | {s['docs']:18,} | {est_tokens:22,}")

    total_tokens = int(total_chars / CHARS_PER_TOKEN)
    print("-" * 76)
    print(f"  {'TOTAL':28s} | {total_docs:18,} | {total_tokens:22,}")
    print("-" * 76)
    print(f"  Shards written: {shard_idx}")
    print(f"  Output directory: {output_dir}")
    print(f"  Time elapsed: {elapsed/60:.1f} minutes")
    print()

    # Realized mix by bucket. Sequential mode has no share control (it drains source by
    # source), so this is a report, not a target — use the interleaved mode for the mix.
    if total_tokens > 0:
        for bucket in TARGET_DOMAIN_SHARE:
            tokens = sum(int(stats[n]["chars"] / CHARS_PER_TOKEN)
                         for n in source_names if _source_bucket(n) == bucket)
            if tokens > 0:
                emoji = _category_emoji(bucket)
                print(f"  {emoji} {bucket.capitalize():15s}: ~{tokens:,} tokens "
                      f"({100 * tokens / total_tokens:.1f}%)")
    print()

    if global_chars > global_chars_at_start:
        create_checkpoint(output_dir, checkpoint_dir, checkpoint_number)
        checkpoint_number += 1
    else:
        logger.info("Skipping final checkpoint because no new documents were written.")

    save_progress(output_dir, {
        "completed_sources": list(completed_sources),
        "next_shard_idx": shard_idx,
        "stats": dict(stats),
        "finished": all(source in set(completed_sources) for source in source_names) and not reached_global_limit,
        "bytes_at_last_checkpoint": get_total_parquet_bytes(output_dir),
        "checkpoint_number": checkpoint_number,
        "next_val_part_idx": val_part_idx,
        "limited_by_max_tokens": reached_global_limit,
        "requested_sources": source_names,
    })

    verify_shard_balance(output_dir)

def verify_shard_balance(data_dir):
    import pyarrow.parquet as pq

    shard_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])

    if not shard_files:
        return

    print("=" * 70)
    print("  SHARD BALANCE VERIFICATION")
    print("=" * 70)

    shard_info = []
    total_row_groups = 0
    for f in shard_files:
        path = os.path.join(data_dir, f)
        pf = pq.ParquetFile(path)
        num_rg = pf.num_row_groups
        num_rows = pf.metadata.num_rows
        shard_info.append({"file": f, "row_groups": num_rg, "rows": num_rows})
        total_row_groups += num_rg

    row_counts = [s["rows"] for s in shard_info]
    if len(row_counts) > 1:
        avg = sum(row_counts) / len(row_counts)
        max_deviation = max(abs(r - avg) / avg * 100 for r in row_counts) if avg > 0 else 0
        balance_status = "✅ BALANCED" if max_deviation < 30 else "⚠️ UNEVEN"
        print(f"  Shards: {len(shard_info)} files, {total_row_groups} row_groups total")
        print(f"  Balance: {balance_status} (max deviation: {max_deviation:.1f}%)")
    else:
        print(f"  Shards: {len(shard_info)} file, {total_row_groups} row_groups")

    print()
    print("  GPU DISTRIBUTION PLAN:")
    print("  ─────────────────────────────────────────")

    train_shards = shard_info[:-1] if len(shard_info) > 1 else shard_info
    train_rg = sum(s["row_groups"] for s in train_shards)
    train_rows = sum(s["rows"] for s in train_shards)

    for num_gpus in [1, 2]:
        if num_gpus > 1 and train_rg < num_gpus:
            continue
        rg_per_gpu = train_rg // num_gpus
        rows_per_gpu = train_rows // num_gpus
        print(f"  {'🖥️' if num_gpus == 1 else '🖥️🖥️'} {num_gpus} GPU(s): "
              f"~{rg_per_gpu} row_groups/GPU, "
              f"~{rows_per_gpu:,} rows/GPU")

    print()
    print("  📝 Note: mesosfer's dataloader auto-distributes row_groups")
    print("     round-robin across GPUs (rank 0 gets rg 0,2,4,...)")
    print("     No manual data splitting needed — just run with torchrun!")
    print()


def _main_with_guard():
    """Wrap ``main()`` with ``KeyboardInterrupt`` handling.

    When the user presses Ctrl+C during a long preparation run we log the
    interruption and exit with a non‑zero code rather than leaving a half‑
    written shard or a corrupted progress file.
    """
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt received – aborting data preparation")
        sys.exit(130)


if __name__ == "__main__":
    _main_with_guard()
