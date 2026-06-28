#!/usr/bin/env python3
"""
Convert raw security log files to natural language narratives for LLM pretraining.

Each raw event/record is grouped with related events and converted into a
security-informative English narrative, then saved as JSONL {"text": "..."}.

Usage:
    python -m scripts.data.convert_logs_to_nl                    # convert all
    python -m scripts.data.convert_logs_to_nl --source data/log/auth.log
    python -m scripts.data.convert_logs_to_nl --output-dir /custom/output
    python -m scripts.data.convert_logs_to_nl --dry-run          # preview first 3 docs
"""

import argparse
import json
import logging
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # scripts/data/ -> scripts/ -> project root

DEFAULT_LOG_DIR = PROJECT_ROOT / "data" / "log"
DEFAULT_CLOUD_DIR = PROJECT_ROOT / "data" / "cloud"
DEFAULT_LOG_NL_DIR = PROJECT_ROOT / "data" / "log_nl"
DEFAULT_CLOUD_NL_DIR = PROJECT_ROOT / "data" / "cloud_nl"

# ---------------------------------------------------------------------------
# MITRE ATT&CK quick-reference
# ---------------------------------------------------------------------------

MITRE = {
    "brute_force": "T1110.001 (Brute Force: Password Guessing)",
    "credential_dump": "T1003.008 (OS Credential Dumping: /etc/passwd and /etc/shadow)",
    "sqli": "T1190 (Exploit Public-Facing Application)",
    "c2_beacon": "T1071.001 (Application Layer Protocol: Web Protocols)",
    "dns_tunnel": "T1071.004 (Application Layer Protocol: DNS)",
    "stop_logging": "T1562.008 (Impair Defenses: Disable Cloud Logs)",
    "create_cloud_account": "T1136.003 (Create Account: Cloud Account)",
    "valid_cloud_account": "T1078.004 (Valid Accounts: Cloud Accounts)",
    "obfuscated_exec": "T1059.001 (Command and Scripting Interpreter: PowerShell)",
    "persistence_registry": "T1547.001 (Boot or Logon Autostart Execution: Registry Run Keys)",
    "scheduled_task": "T1053.005 (Scheduled Task/Job: Scheduled Task)",
    "webshell": "T1505.003 (Server Software Component: Web Shell)",
    "privesc_sudo": "T1548.003 (Abuse Elevation Control Mechanism: Sudo and Sudo Caching)",
    "persistence_ssh_key": "T1098.004 (Account Manipulation: SSH Authorized Keys)",
    "download_exec": "T1105 (Ingress Tool Transfer)",
    "cron_persistence": "T1053.003 (Scheduled Task/Job: Cron)",
    "lateral_movement_smb": "T1021.002 (Remote Services: SMB/Windows Admin Shares)",
    "pass_the_hash": "T1550.002 (Use Alternate Authentication Material: Pass the Hash)",
    "gcp_iam_escalation": "T1078.004 (Valid Accounts: Cloud Accounts) / T1098 (Account Manipulation)",
    "data_exfil": "T1530 (Data from Cloud Storage Object)",
    "defense_evasion_gcp": "T1562 (Impair Defenses)",
    "azure_role_assign": "T1098.003 (Account Manipulation: Additional Cloud Roles)",
    "azure_app_create": "T1136.003 (Create Account: Cloud Account)",
    "s3_public": "T1530 (Data from Cloud Storage Object)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_read(path: Path) -> str:
    """Read a full file into memory (JSON/XML/TSV formats only, with size guard).

    Warns if the file exceeds 500 MiB and returns empty string to prevent OOM.
    Prefer ``_iter_lines()`` for line‑oriented text/JSONL logs.
    """
    if not path.is_file():
        logger.warning("Missing log file: %s – skipping", path)
        return ""
    try:
        size = path.stat().st_size
        if size > 500 * 1024 * 1024:
            logger.warning(
                "File %s is %.1f MiB – too large for in‑memory read, skipping",
                path, size / (1024 * 1024),
            )
            return ""
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.warning("Cannot read %s: %s", path, exc)
        return ""


def _iter_lines(path: Path) -> Iterable[str]:
    """Yield non‑empty, stripped lines from a text/JSONL file (streaming).

    Processes very large files without loading the entire content into memory.
    Empty lines and lines that become empty after stripping are skipped.
    """
    if not path.is_file():
        logger.warning("Missing log file: %s – skipping", path)
        return
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line
    except Exception as exc:
        logger.warning("Cannot read %s (streaming): %s", path, exc)


def _ts_to_dt(ts_str: str) -> datetime | None:
    """Parse ISO-8601 or epoch float timestamp to datetime (UTC)."""
    if not ts_str:
        return None
    try:
        return datetime.fromtimestamp(float(ts_str), tz=timezone.utc)
    except (ValueError, TypeError):
        pass
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f+0000",
        "%Y-%m-%dT%H:%M:%S+0000",
        "%Y-%m-%dT%H:%M:%S.%fZ",
    ):
        try:
            return datetime.strptime(ts_str[:26], fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _fmt_dt(dt: datetime | None) -> str:
    if dt is None:
        return "unknown time"
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def _word_count(text: str) -> int:
    return len(text.split())


def _conn_state_desc(state: str) -> str:
    mapping = {
        "SF": "a normal completed session (SYN→SYN-ACK→data→FIN)",
        "S0": "a SYN sent with no response — possible port scan or filtered port",
        "REJ": "a connection rejected by the destination (RST received)",
        "RSTO": "the connection was reset by the originator",
        "RSTR": "the connection was reset by the responder",
        "SH": "a half-open connection — SYN sent, SYN-ACK received, but no data (possible beacon)",
        "OTH": "a mid-stream connection with no SYN observed",
    }
    return mapping.get(state, f"state {state}")


# ===========================================================================
# FORMAT 1 – Auth / syslog (auth.log, auth2.log, privesc.log, syslog_benign)
# ===========================================================================

_SYSLOG_RE = re.compile(
    r"^(?P<month>\w{3})\s+(?P<day>\d+)\s+(?P<time>\d{2}:\d{2}:\d{2})\s+"
    r"(?P<host>\S+)\s+(?P<proc>[^:]+):\s+(?P<msg>.+)$"
)

_SSH_FAIL_RE = re.compile(
    r"Failed password for (?:invalid user )?(?P<user>\S+) from (?P<ip>[\d.]+) port (?P<port>\d+)"
)
_SSH_ACCEPT_RE = re.compile(
    r"Accepted (?:password|publickey) for (?P<user>\S+) from (?P<ip>[\d.]+) port (?P<port>\d+)"
)
_SUDO_RE = re.compile(
    r"sudo:\s+(?P<user>\S+)\s+:.*?COMMAND=(?P<cmd>.+)$"
)
_SUDO_CMD_RE = re.compile(
    r"(?:sudo:\s+)?(?P<user>\S+)\s+:.*?COMMAND=(?P<cmd>.+)$"
)
_SUDO_USER_RE = re.compile(r"USER=(?P<target>\S+)")
_BASH_CMD_RE = re.compile(r"bash\[\d+\]:\s+(?P<cmd>.+)$")
_CRON_CMD_RE = re.compile(r"CRON\[\d+\].*CMD\s+\((?P<cmd>.+)\)")


def _parse_syslog_line(line: str) -> dict | None:
    m = _SYSLOG_RE.match(line.strip())
    if not m:
        return None
    return {
        "month": m.group("month"),
        "day": m.group("day"),
        "time": m.group("time"),
        "host": m.group("host"),
        "proc": m.group("proc"),
        "msg": m.group("msg"),
        "raw": line.strip(),
    }


def _syslog_sort_key(ev: dict) -> str:
    months = {"Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
               "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
               "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"}
    m = months.get(ev["month"], "00")
    d = ev["day"].zfill(2)
    return f"{m}{d}{ev['time']}"


def _syslog_time_minutes(ev: dict) -> int:
    """Convert syslog event time to total minutes (day*1440 + hour*60 + min)."""
    try:
        t = ev["time"]  # HH:MM:SS
        h, m, s = t.split(":")
        day = int(ev.get("day", "1"))
        return day * 1440 + int(h) * 60 + int(m)
    except Exception:
        return 0


def _group_syslog_events(events: list[dict], window_minutes: int = 5) -> list[list[dict]]:
    """Group syslog events by host within a rolling time window.

    All events on the same host within `window_minutes` of each other
    are placed in the same group. This ensures sudo/bash/cron events
    following an SSH login are grouped with the login events.
    """
    by_host: dict[str, list[dict]] = defaultdict(list)
    for ev in events:
        by_host[ev["host"]].append(ev)

    groups: list[list[dict]] = []
    for host_events in by_host.values():
        host_events.sort(key=_syslog_sort_key)
        if not host_events:
            continue

        current_group: list[dict] = [host_events[0]]
        group_start_min = _syslog_time_minutes(host_events[0])

        for ev in host_events[1:]:
            ev_min = _syslog_time_minutes(ev)
            if ev_min - group_start_min <= window_minutes:
                current_group.append(ev)
            else:
                groups.append(current_group)
                current_group = [ev]
                group_start_min = ev_min

        if current_group:
            groups.append(current_group)

    return groups


def _narrate_syslog_group(group: list[dict]) -> str:
    """Convert a group of syslog events into a security narrative."""
    if not group:
        return ""

    host = group[0]["host"]
    first_time = f"{group[0]['month']} {group[0]['day']} at {group[0]['time']}"
    last_time = group[-1]["time"]

    ssh_fails: list[dict] = []
    ssh_accepts: list[dict] = []
    sudo_cmds: list[dict] = []
    bash_cmds: list[str] = []
    cron_cmds: list[str] = []
    other_msgs: list[str] = []

    for ev in group:
        msg = ev["msg"]
        proc = ev.get("proc", "")
        mf = _SSH_FAIL_RE.search(msg)
        ma = _SSH_ACCEPT_RE.search(msg)
        # Sudo: match either "sudo:" in msg or proc == "sudo"
        ms = None
        if "sudo" in proc.lower() or msg.startswith("sudo:"):
            ms = _SUDO_CMD_RE.search(msg)
        mb = _BASH_CMD_RE.search(msg)
        mc = _CRON_CMD_RE.search(msg)

        if mf:
            ssh_fails.append({"user": mf.group("user"), "ip": mf.group("ip"),
                               "port": mf.group("port"), "time": ev["time"]})
        elif ma:
            ssh_accepts.append({"user": ma.group("user"), "ip": ma.group("ip"),
                                 "port": ma.group("port"), "time": ev["time"]})
        elif ms:
            target_m = _SUDO_USER_RE.search(msg)
            target = target_m.group("target") if target_m else "root"
            sudo_cmds.append({"user": ms.group("user"), "cmd": ms.group("cmd").strip(),
                               "target": target, "time": ev["time"]})
        elif mb:
            bash_cmds.append(mb.group("cmd").strip())
        elif mc:
            cron_cmds.append(mc.group("cmd").strip())
        elif "session opened" not in msg and "session closed" not in msg:
            other_msgs.append(msg)

    parts: list[str] = []

    # --- SSH brute-force / login narrative ---
    if ssh_fails:
        ips = list({f["ip"] for f in ssh_fails})
        users = list({f["user"] for f in ssh_fails})
        ip_str = ips[0] if len(ips) == 1 else ", ".join(ips[:3]) + (" and others" if len(ips) > 3 else "")
        user_str = ", ".join(f'"{u}"' for u in users[:6])
        t_start = ssh_fails[0]["time"]
        t_end = ssh_fails[-1]["time"]
        n = len(ssh_fails)

        if n >= 3:
            parts.append(
                f"Between {t_start} and {t_end}, host {host} experienced a rapid SSH brute-force "
                f"attack from IP {ip_str}, with {n} failed login attempt{'s' if n > 1 else ''} "
                f"targeting username{'s' if len(users) > 1 else ''} including {user_str}. "
                f"The attacker cycled through credentials at short intervals, consistent with "
                f"automated tooling such as Hydra or Medusa (MITRE ATT&CK {MITRE['brute_force']})."
            )
        else:
            parts.append(
                f"On {first_time}, host {host} recorded {n} failed SSH login "
                f"attempt{'s' if n > 1 else ''} from IP {ip_str} targeting user{'s' if len(users) > 1 else ''} "
                f"{user_str}. This may indicate a targeted credential attack."
            )

    if ssh_accepts:
        for acc in ssh_accepts:
            context = ""
            if ssh_fails:
                context = (
                    f" This successful login followed the brute-force activity described above, "
                    f"suggesting the attacker eventually guessed or obtained valid credentials."
                )
            parts.append(
                f"At {acc['time']}, the SSH authentication succeeded for user \"{acc['user']}\" "
                f"from IP {acc['ip']} on port {acc['port']}.{context}"
            )

    # --- Sudo / privilege escalation narrative ---
    if sudo_cmds:
        for sc in sudo_cmds:
            cmd = sc["cmd"]
            is_shadow = "/etc/shadow" in cmd or "/etc/passwd" in cmd
            is_sudoers = "/etc/sudoers" in cmd
            is_shell = cmd.endswith("/bash") or cmd.endswith("/sh") or "/bin/bash" in cmd
            is_vim = "vim" in cmd or "nano" in cmd or "vi " in cmd
            is_find = cmd.strip().endswith("/find") or "/find " in cmd

            if is_shadow:
                parts.append(
                    f"At {sc['time']}, user \"{sc['user']}\" executed a privileged command via sudo "
                    f"to read {cmd.split()[-1]} as {sc['target']}. Reading the shadow password file "
                    f"is a classic credential harvesting step following initial access "
                    f"({MITRE['credential_dump']})."
                )
            elif is_sudoers:
                parts.append(
                    f"At {sc['time']}, user \"{sc['user']}\" used sudo to edit the sudoers file "
                    f"({cmd}) as {sc['target']}. Modifying sudoers is a persistence and privilege "
                    f"escalation technique that can grant permanent root access "
                    f"({MITRE['privesc_sudo']})."
                )
            elif is_shell:
                parts.append(
                    f"At {sc['time']}, user \"{sc['user']}\" spawned an interactive root shell via "
                    f"sudo ({cmd}). This grants full system control and is a strong indicator of "
                    f"post-exploitation activity ({MITRE['privesc_sudo']})."
                )
            elif is_find:
                parts.append(
                    f"At {sc['time']}, user \"{sc['user']}\" ran the 'find' utility as {sc['target']} "
                    f"via sudo. Running find as root is often used to locate sensitive files or "
                    f"SUID binaries for privilege escalation ({MITRE['privesc_sudo']})."
                )
            else:
                parts.append(
                    f"At {sc['time']}, user \"{sc['user']}\" executed \"{cmd}\" as {sc['target']} "
                    f"via sudo, indicating elevated privilege usage."
                )

    # --- Bash / shell command narrative ---
    if bash_cmds:
        for cmd in bash_cmds:
            if "authorized_keys" in cmd:
                parts.append(
                    f"A shell command was executed to append an SSH public key to "
                    f"/root/.ssh/authorized_keys, establishing persistent backdoor access "
                    f"({MITRE['persistence_ssh_key']})."
                )
            elif "uid=0" in cmd:
                parts.append(
                    f"The shell confirmed root-level execution (uid=0), indicating full "
                    f"system compromise."
                )
            else:
                parts.append(f"Shell command executed: {cmd[:120]}.")

    # --- Cron / scheduled task narrative ---
    if cron_cmds:
        for cmd in cron_cmds:
            if "curl" in cmd or "wget" in cmd or "payload" in cmd:
                parts.append(
                    f"A cron job executed a remote download-and-execute command: \"{cmd[:120]}\". "
                    f"This is a common technique for deploying malware or maintaining persistence "
                    f"({MITRE['download_exec']}, {MITRE['cron_persistence']})."
                )
            else:
                parts.append(f"A scheduled cron job ran: \"{cmd[:120]}\".")

    # --- Benign / other messages ---
    if not parts and other_msgs:
        summary = "; ".join(other_msgs[:3])
        parts.append(
            f"On {first_time}, host {host} logged routine system activity: {summary}. "
            f"No indicators of compromise were identified in this event group."
        )

    if not parts:
        return ""

    narrative = " ".join(parts)

    # Append overall assessment
    is_malicious = bool(ssh_fails and ssh_accepts) or bool(sudo_cmds) or bool(bash_cmds) or bool(cron_cmds)
    if is_malicious:
        narrative += (
            f" Overall, this sequence on host {host} represents a multi-stage attack: "
            f"initial access via SSH followed by privilege escalation and persistence mechanisms, "
            f"consistent with a hands-on-keyboard intrusion."
        )

    return narrative.strip()


def convert_syslog(path: Path) -> list[str]:
    """Parse a syslog-format file and return a list of narrative strings.

    Uses streaming line‑by‑line reader for memory efficiency on large logs.
    """
    events = []
    for line in _iter_lines(path):
        ev = _parse_syslog_line(line)
        if ev:
            events.append(ev)
    if not events:
        return []
    groups = _group_syslog_events(events)
    narratives = []
    for grp in groups:
        narr = _narrate_syslog_group(grp)
        if narr and _word_count(narr) >= 30:
            narratives.append(narr)
    return narratives


# ===========================================================================
# FORMAT 2 – Apache / Web access log + ModSecurity
# ===========================================================================

_APACHE_RE = re.compile(
    r'^(?P<ip>[\d.]+) \S+ \S+ \[(?P<ts>[^\]]+)\] '
    r'"(?P<method>\S+) (?P<uri>\S+) HTTP/[\d.]+" '
    r'(?P<status>\d+) (?P<size>\d+)'
    r'(?:\s+"(?P<referer>[^"]*)")?'
    r'(?:\s+"(?P<ua>[^"]*)")?'
)
_MODSEC_RE = re.compile(
    r'\[(?P<module>[^\]]+)\]\s+ModSecurity:\s+(?P<action>.+?)\s+\[client (?P<ip>[^\]]+)\]'
    r'.*?\[msg "(?P<msg>[^"]+)"\]'
    r'.*?\[uri "(?P<uri>[^"]+)"\]'
)
_PHP_ERR_RE = re.compile(r'\[php:error\].*?(?P<msg>(?:PDOException|Fatal error|Warning).+?)$')


def _decode_uri(uri: str) -> str:
    """Percent-decode a URI for readability."""
    try:
        from urllib.parse import unquote
        return unquote(uri)
    except Exception:
        return uri


def _classify_uri(uri: str) -> str:
    decoded = _decode_uri(uri).lower()
    if "union select" in decoded or "union%20select" in decoded:
        return "UNION-based SQL injection"
    if "or '1'='1" in decoded or "or 1=1" in decoded or "%27%20or" in decoded:
        return "boolean-based SQL injection"
    if "information_schema" in decoded:
        return "database schema enumeration"
    if "' or" in decoded or "1=1" in decoded:
        return "SQL injection"
    if "<script" in decoded or "javascript:" in decoded:
        return "XSS attempt"
    if "../" in decoded or "..%2f" in decoded:
        return "path traversal"
    if "cmd=" in decoded or "exec=" in decoded or "system(" in decoded:
        return "command injection"
    return "suspicious request"


def _group_apache_events(lines: list[str], window_minutes: int = 2) -> list[list[str]]:
    """Group Apache log lines by source IP."""
    by_ip: dict[str, list[str]] = defaultdict(list)
    for line in lines:
        m = _APACHE_RE.match(line)
        if m:
            by_ip[m.group("ip")].append(line)
        elif "[security2:error]" in line or "[php:error]" in line:
            # Attach ModSecurity/PHP errors to the last IP seen
            pass  # handled in narration
    groups = []
    for ip, ip_lines in by_ip.items():
        groups.append(ip_lines)
    return groups


def _narrate_apache_group(lines: list[str], all_lines: list[str]) -> str:
    """Narrate an Apache log group (all lines from one IP)."""
    if not lines:
        return ""

    first_m = _APACHE_RE.match(lines[0])
    if not first_m:
        return ""

    ip = first_m.group("ip")
    requests = []
    for line in lines:
        m = _APACHE_RE.match(line)
        if m:
            requests.append(m)

    if not requests:
        return ""

    # Collect ModSecurity alerts for this IP
    modsec_alerts: list[str] = []
    php_errors: list[str] = []
    for line in all_lines:
        if ip in line:
            mm = _MODSEC_RE.search(line)
            if mm:
                modsec_alerts.append(mm.group("msg"))
            mp = _PHP_ERR_RE.search(line)
            if mp:
                php_errors.append(mp.group("msg")[:120])

    first_ts = first_m.group("ts")
    last_ts = requests[-1].group("ts") if len(requests) > 1 else first_ts
    ua = first_m.group("ua") or "unknown"
    statuses = [r.group("status") for r in requests]
    uris = [r.group("uri") for r in requests]

    # Classify attack type
    attack_types = set()
    for uri in uris:
        at = _classify_uri(uri)
        if at != "suspicious request":
            attack_types.add(at)

    is_sqlmap = "sqlmap" in ua.lower()
    is_attack = bool(attack_types) or is_sqlmap or bool(modsec_alerts)

    parts: list[str] = []

    if is_attack:
        tool_str = f"using the automated tool sqlmap (version {ua.split('/')[1].split('#')[0] if '/' in ua else ua})" if is_sqlmap else f"with user-agent \"{ua[:60]}\""
        attack_str = ", ".join(sorted(attack_types)) if attack_types else "SQL injection"
        parts.append(
            f"Between {first_ts} and {last_ts}, IP {ip} launched a web application attack "
            f"against the server, {tool_str}. The attacker sent {len(requests)} request(s) "
            f"targeting {attack_str} ({MITRE['sqli']})."
        )

        # Describe specific payloads
        notable_uris = [u for u in uris if any(k in _decode_uri(u).lower()
                        for k in ["union", "select", "or '1'", "information_schema", "database()"])]
        if notable_uris:
            decoded_examples = [_decode_uri(u)[:100] for u in notable_uris[:2]]
            parts.append(
                f"Notable payloads included: {'; '.join(decoded_examples)}."
            )

        # ModSecurity response
        if modsec_alerts:
            blocked = [r for r in requests if r.group("status") == "403"]
            allowed = [r for r in requests if r.group("status") not in ("403", "404")]
            parts.append(
                f"ModSecurity detected and flagged the following: {'; '.join(set(modsec_alerts[:3]))}. "
                f"{len(blocked)} request(s) were blocked with HTTP 403."
            )
            if allowed:
                parts.append(
                    f"However, {len(allowed)} request(s) received non-blocking responses "
                    f"(HTTP {', '.join(set(r.group('status') for r in allowed))}), "
                    f"suggesting some payloads may have reached the application layer."
                )
        else:
            blocked = [r for r in requests if r.group("status") == "403"]
            if blocked:
                parts.append(f"{len(blocked)} request(s) were blocked with HTTP 403 by the WAF.")

        if php_errors:
            parts.append(
                f"PHP errors were triggered during the attack: {php_errors[0][:120]}. "
                f"This indicates the application may be vulnerable to SQL injection."
            )

        parts.append(
            f"This activity is consistent with automated SQL injection reconnaissance, "
            f"likely attempting to extract database credentials or enumerate schema structure."
        )
    else:
        # Benign traffic
        status_summary = ", ".join(f"HTTP {s}" for s in set(statuses))
        parts.append(
            f"Between {first_ts} and {last_ts}, IP {ip} made {len(requests)} web request(s) "
            f"to the server, receiving responses: {status_summary}. "
            f"No attack indicators were detected in this session."
        )

    return " ".join(parts).strip()


def convert_apache(path: Path) -> list[str]:
    all_lines = list(_iter_lines(path))
    access_lines = [l for l in all_lines if _APACHE_RE.match(l)]
    if not access_lines:
        return []
    groups = _group_apache_events(access_lines)
    narratives = []
    for grp in groups:
        narr = _narrate_apache_group(grp, all_lines)
        if narr and _word_count(narr) >= 30:
            narratives.append(narr)
    return narratives


# ===========================================================================
# FORMAT 3 – CEF (Common Event Format)
# ===========================================================================

_CEF_HEADER_RE = re.compile(
    r"^CEF:(?P<version>\d+)\|(?P<vendor>[^|]+)\|(?P<product>[^|]+)\|"
    r"(?P<dev_version>[^|]+)\|(?P<sig_id>[^|]+)\|(?P<name>[^|]+)\|"
    r"(?P<severity>[^|]+)\|(?P<ext>.+)$"
)


def _parse_cef_ext(ext: str) -> dict:
    """Parse CEF key=value extension fields."""
    result = {}
    # Handle quoted values and unquoted values
    pattern = re.compile(r'(\w+)=((?:[^=\s]+(?:\s+(?!\w+=))*)+)')
    for m in pattern.finditer(ext):
        result[m.group(1)] = m.group(2).strip()
    return result


def _parse_cef_line(line: str) -> dict | None:
    m = _CEF_HEADER_RE.match(line.strip())
    if not m:
        return None
    ext = _parse_cef_ext(m.group("ext"))
    return {
        "vendor": m.group("vendor"),
        "product": m.group("product"),
        "sig_id": m.group("sig_id"),
        "name": m.group("name"),
        "severity": m.group("severity"),
        "ext": ext,
        "raw": line.strip(),
    }


def _group_cef_events(events: list[dict], window_minutes: int = 15) -> list[list[dict]]:
    """Group CEF events by suser (source user)."""
    by_user: dict[str, list[dict]] = defaultdict(list)
    for ev in events:
        user = ev["ext"].get("suser", ev["ext"].get("duser", "unknown"))
        by_user[user].append(ev)
    return list(by_user.values())


def _narrate_cef_group(group: list[dict]) -> str:
    if not group:
        return ""

    first = group[0]
    user = first["ext"].get("suser", first["ext"].get("duser", "unknown"))
    src_ip = first["ext"].get("src", "unknown IP")
    rt = first["ext"].get("rt", "")

    vendors = list({ev["vendor"] for ev in group})
    products = list({ev["product"] for ev in group})
    event_names = [ev["name"] for ev in group]

    parts: list[str] = []

    # Describe the session
    time_str = f"at {rt}" if rt else ""
    parts.append(
        f"A multi-product security event sequence was recorded for user \"{user}\" "
        f"originating from IP {src_ip} {time_str}. "
        f"The following {len(group)} event(s) were logged across {', '.join(vendors)} products."
    )

    for ev in group:
        name = ev["name"]
        vendor = ev["vendor"]
        product = ev["product"]
        ext = ev["ext"]
        outcome = ext.get("outcome", "unknown")
        msg = ext.get("msg", "")
        shost = ext.get("shost", "")
        mfa = ext.get("cs2", "")
        mfa_label = ext.get("cs2Label", "")
        policy = ext.get("cs1", "")
        policy_label = ext.get("cs1Label", "")
        dst = ext.get("dst", "")
        dpt = ext.get("dpt", "")
        app = ext.get("app", "")

        if "Authentication" in name or "Login" in name:
            mfa_str = f" via {mfa}" if mfa and mfa_label == "MFA" else ""
            host_str = f" from workstation {shost}" if shost else ""
            parts.append(
                f"  • {vendor} {product}: \"{name}\" — user authenticated with outcome "
                f"\"{outcome}\"{mfa_str}{host_str}. {msg}"
            )
        elif "Policy" in name or "Conditional" in name:
            parts.append(
                f"  • {vendor} {product}: \"{name}\" — {policy_label} \"{policy}\" "
                f"was applied with compliance state \"{ext.get('cs2', 'unknown')}\" "
                f"(outcome: {outcome})."
            )
        elif "Heartbeat" in name or "Sensor" in name:
            sensor_ver = ext.get("cs1", "")
            platform = ext.get("cs2", "")
            parts.append(
                f"  • {vendor} {product}: Endpoint heartbeat received from {shost} "
                f"(sensor version {sensor_ver}, platform {platform})."
            )
        elif "TLS" in name or "Session" in name:
            parts.append(
                f"  • {vendor} {product}: TLS session to {dst}:{dpt} was {ext.get('act', 'processed')} "
                f"under rule \"{policy}\" (category: {ext.get('cs2', 'unknown')})."
            )
        else:
            parts.append(
                f"  • {vendor} {product}: \"{name}\" (outcome: {outcome}). {msg}"
            )

    # Overall assessment
    all_success = all(ev["ext"].get("outcome", "") == "success" for ev in group)
    if all_success:
        parts.append(
            f"All events in this session completed successfully. The activity appears to represent "
            f"a normal authenticated user session with MFA and endpoint compliance checks passing."
        )
    else:
        parts.append(
            f"Some events in this session did not complete successfully. "
            f"Further investigation may be warranted."
        )

    return "\n".join(parts).strip()


def convert_cef(path: Path) -> list[str]:
    events = []
    for line in _iter_lines(path):
        ev = _parse_cef_line(line)
        if ev:
            events.append(ev)
    if not events:
        return []
    groups = _group_cef_events(events)
    narratives = []
    for grp in groups:
        narr = _narrate_cef_group(grp)
        if narr and _word_count(narr) >= 30:
            narratives.append(narr)
    return narratives


# ===========================================================================
# FORMAT 4 – Zeek conn.log (TSV with #fields header)
# ===========================================================================

def _parse_zeek_conn(text: str) -> list[dict]:
    """Parse Zeek conn.log TSV format (with or without #fields header)."""
    fields: list[str] = []
    records: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#fields"):
            fields = line.split("\t")[1:]
        elif line.startswith("#"):
            continue
        else:
            parts = line.split("\t") if "\t" in line else line.split()
            if fields and len(parts) >= len(fields):
                records.append(dict(zip(fields, parts)))
            elif not fields:
                # Headerless conn2.log format (12 cols, no resp_bytes):
                # ts uid orig_h orig_p resp_h resp_p proto service duration orig_bytes conn_state local_orig
                col_names_12 = ["ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h",
                                 "id.resp_p", "proto", "service", "duration",
                                 "orig_bytes", "conn_state", "local_orig"]
                # Full format (20 cols, with resp_bytes):
                col_names_20 = ["ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h",
                                 "id.resp_p", "proto", "service", "duration",
                                 "orig_bytes", "resp_bytes", "conn_state"]
                if len(parts) >= 12:
                    # Detect by checking if position 10 looks like a conn_state (letters only)
                    if re.match(r'^[A-Z]+$', parts[10]):
                        rec = dict(zip(col_names_12, parts[:len(col_names_12)]))
                    else:
                        rec = dict(zip(col_names_20, parts[:len(col_names_20)]))
                    records.append(rec)
    return records


def _group_zeek_by_src(records: list[dict]) -> list[list[dict]]:
    """Group Zeek records by source IP."""
    by_src: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        src = rec.get("id.orig_h", "unknown")
        by_src[src].append(rec)
    return list(by_src.values())


def _narrate_zeek_group(group: list[dict]) -> str:
    if not group:
        return ""

    src_ip = group[0].get("id.orig_h", "unknown")
    parts: list[str] = []

    # Summarize connections
    conn_summaries: list[str] = []
    suspicious_ports = {22, 23, 445, 3389, 5900, 4444, 1337, 8080, 3333}
    has_suspicious = False

    for rec in group:
        dst_ip = rec.get("id.resp_h", "?")
        dst_port = rec.get("id.resp_p", "?")
        proto = rec.get("proto", "?")
        service = rec.get("service", "-")
        duration = rec.get("duration", "-")
        orig_bytes = rec.get("orig_bytes", "0")
        resp_bytes = rec.get("resp_bytes", "0")
        state = rec.get("conn_state", "?")
        state_desc = _conn_state_desc(state)

        try:
            port_int = int(dst_port)
            if port_int in suspicious_ports:
                has_suspicious = True
        except (ValueError, TypeError):
            pass

        try:
            dur_f = float(duration)
            dur_str = f"{dur_f:.2f} seconds"
        except (ValueError, TypeError):
            dur_str = duration

        svc_str = f" ({service})" if service and service not in ("-", "(empty)") else ""
        resp_bytes_str = f", {resp_bytes} bytes received" if resp_bytes and resp_bytes not in ("-", "(empty)", "0") else ""
        conn_summaries.append(
            f"  • {proto.upper()}{svc_str} to {dst_ip}:{dst_port} — "
            f"duration {dur_str}, {orig_bytes} bytes sent{resp_bytes_str}, "
            f"state: {state_desc}"
        )

    parts.append(
        f"Network activity profile for source IP {src_ip} recorded {len(group)} connection(s):"
    )
    parts.extend(conn_summaries)

    # Detect scanning patterns
    unique_dsts = {r.get("id.resp_h") for r in group}
    unique_ports = {r.get("id.resp_p") for r in group}
    rejected = [r for r in group if r.get("conn_state") in ("REJ", "S0", "RSTO")]

    if len(unique_dsts) > 3 or len(unique_ports) > 5:
        parts.append(
            f"\nThis host contacted {len(unique_dsts)} distinct destination(s) on "
            f"{len(unique_ports)} distinct port(s), which may indicate network reconnaissance "
            f"or lateral movement scanning."
        )
    if rejected:
        parts.append(
            f"{len(rejected)} connection(s) were rejected or received no response, "
            f"consistent with port scanning behavior."
        )

    # Detect beaconing (conn2.log pattern: same dst, regular intervals, SH state)
    sh_conns = [r for r in group if r.get("conn_state") == "SH"]
    if len(sh_conns) >= 3:
        dst_ips = {r.get("id.resp_h") for r in sh_conns}
        parts.append(
            f"\n{len(sh_conns)} half-open (SH) connections to {', '.join(dst_ips)} were observed "
            f"at regular intervals. This periodic beaconing pattern is strongly indicative of "
            f"C2 (command-and-control) communication ({MITRE['c2_beacon']})."
        )

    if has_suspicious:
        parts.append(
            f"\nConnections to sensitive service ports (SSH/22, RDP/3389, SMB/445, VNC/5900) "
            f"were observed, warranting further investigation for lateral movement or "
            f"unauthorized remote access."
        )

    return "\n".join(parts).strip()


def convert_zeek_conn(path: Path) -> list[str]:
    text = _safe_read(path)
    records = _parse_zeek_conn(text)
    if not records:
        return []
    groups = _group_zeek_by_src(records)
    narratives = []
    for grp in groups:
        narr = _narrate_zeek_group(grp)
        if narr and _word_count(narr) >= 30:
            narratives.append(narr)
    return narratives


# ===========================================================================
# FORMAT 5 – Suricata / mixed JSONL (log.jsonl, c2_dns.jsonl)
# ===========================================================================

def _parse_jsonl(text: str) -> list[dict]:
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            logger.warning("JSONL parse error: %s", e)
    return records


def _parse_jsonl_streaming(path: Path) -> Iterable[dict]:
    """Yield JSON objects from a JSONL file using a streaming reader.

    Memory‑efficient: only one line is kept in memory at a time.
    Non‑parseable lines are logged and skipped.
    """
    for line in _iter_lines(path):
        try:
            yield json.loads(line)
        except json.JSONDecodeError as e:
            logger.warning("JSONL parse error in %s: %s", path, e)


def _group_suricata_by_src(records: list[dict]) -> list[list[dict]]:
    """Group Suricata/Zeek JSONL records by source IP."""
    by_src: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        src = rec.get("src_ip") or rec.get("id.orig_h", "unknown")
        by_src[src].append(rec)
    return list(by_src.values())


def _narrate_suricata_group(group: list[dict]) -> str:
    if not group:
        return ""

    src_ip = group[0].get("src_ip") or group[0].get("id.orig_h", "unknown")
    parts: list[str] = []

    alerts: list[dict] = []
    flows: list[dict] = []
    dns_queries: list[dict] = []
    tls_events: list[dict] = []

    for rec in group:
        etype = rec.get("event_type", "")
        if etype == "alert":
            alerts.append(rec)
        elif etype == "flow":
            flows.append(rec)
        elif etype == "dns":
            dns_queries.append(rec)
        elif etype == "tls":
            tls_events.append(rec)
        elif "alert" in rec:
            alerts.append(rec)
        elif "query" in str(rec.get("dns", {})):
            dns_queries.append(rec)
        elif "answers" in rec:
            dns_queries.append(rec)

    first_ts = group[0].get("timestamp") or group[0].get("ts", "")
    time_str = f"Starting at {first_ts[:19]}" if first_ts else "At an unspecified time"

    if alerts:
        sigs = list({a.get("alert", {}).get("signature", a.get("signature", "unknown")) for a in alerts})
        dsts = list({a.get("dest_ip", "unknown") for a in alerts})
        dst_ports = list({str(a.get("dest_port", "?")) for a in alerts})
        severities = [a.get("alert", {}).get("severity", 0) for a in alerts if "alert" in a]
        max_sev = min(severities) if severities else 0  # Suricata: 1=highest

        parts.append(
            f"{time_str}, Suricata IDS detected {len(alerts)} alert(s) from internal host "
            f"{src_ip} communicating with external IP(s) {', '.join(dsts[:3])} "
            f"on port(s) {', '.join(dst_ports[:3])}."
        )

        for sig in sigs[:3]:
            if "C2 Beaconing" in sig or "Beacon" in sig:
                parts.append(
                    f"Signature \"{sig}\" was triggered, indicating the host may be infected "
                    f"with malware that is periodically checking in with a command-and-control "
                    f"server ({MITRE['c2_beacon']})."
                )
            elif "TROJAN" in sig.upper() or "MALWARE" in sig.upper():
                parts.append(
                    f"Signature \"{sig}\" matched, indicating trojan or malware activity. "
                    f"This is a high-confidence indicator of compromise."
                )
            else:
                parts.append(f"Signature \"{sig}\" was triggered.")

        if max_sev == 1:
            parts.append(
                f"Alert severity is critical (level 1), indicating high-confidence malicious activity."
            )

    if tls_events:
        for tls in tls_events[:2]:
            tls_data = tls.get("tls", {})
            sni = tls_data.get("sni", tls_data.get("subject", "unknown"))
            issuer = tls_data.get("issuerdn", "unknown")
            ja3 = tls_data.get("ja3", "")
            parts.append(
                f"TLS connection observed to \"{sni}\" (issuer: {issuer})"
                + (f" with JA3 fingerprint {ja3}" if ja3 else "") + "."
            )

    if dns_queries:
        queries = []
        for rec in dns_queries:
            q = rec.get("query") or rec.get("dns", {}).get("rrname") or ""
            qtype = rec.get("qtype_name") or rec.get("dns", {}).get("rrtype") or "A"
            answers = rec.get("answers", [])
            if q:
                queries.append((q, qtype, answers))

        if queries:
            parts.append(
                f"DNS activity from {src_ip} included {len(queries)} query/queries:"
            )
            for q, qtype, answers in queries[:4]:
                is_dga = len(q.split(".")[0]) > 15 or re.search(r'\d{4,}', q.split(".")[0])
                is_txt_tunnel = qtype == "TXT" and answers
                note = ""
                if is_dga:
                    note = f" The subdomain appears algorithmically generated (DGA), a common C2 technique."
                if is_txt_tunnel:
                    note += (
                        f" TXT record responses containing base64-encoded data suggest "
                        f"DNS tunneling for data exfiltration or C2 ({MITRE['dns_tunnel']})."
                    )
                parts.append(f"  • DNS {qtype} query for \"{q}\"{note}")

    if flows:
        total_bytes_out = sum(f.get("flow", {}).get("bytes_toserver", 0) for f in flows)
        total_bytes_in = sum(f.get("flow", {}).get("bytes_toclient", 0) for f in flows)
        parts.append(
            f"Network flow data shows {total_bytes_out} bytes sent to the external server "
            f"and {total_bytes_in} bytes received, consistent with periodic small check-in "
            f"beacons rather than large data transfers."
        )

    if not parts:
        return ""

    parts.append(
        f"The combination of C2 beaconing alerts, suspicious DNS queries, and regular "
        f"outbound connections from {src_ip} strongly suggests this host is compromised "
        f"and under active remote control."
    )

    return " ".join(parts).strip()


def convert_suricata_jsonl(path: Path) -> list[str]:
    records = list(_parse_jsonl_streaming(path))
    if not records:
        return []
    groups = _group_suricata_by_src(records)
    narratives = []
    for grp in groups:
        narr = _narrate_suricata_group(grp)
        if narr and _word_count(narr) >= 30:
            narratives.append(narr)
    return narratives


# ===========================================================================
# FORMAT 6 & 7 – Sysmon XML + Windows Security Event XML
# ===========================================================================

_WIN_NS = "http://schemas.microsoft.com/win/2004/08/events/event"


def _parse_xml_events(text: str) -> list[dict]:
    """Parse a file containing multiple <Event> elements (not wrapped in a root)."""
    events = []
    # Wrap in a root element to allow parsing multiple top-level elements
    wrapped = f"<Root>{text}</Root>"
    try:
        root = ET.fromstring(wrapped)
    except ET.ParseError as e:
        logger.warning("XML parse error: %s — attempting line-by-line", e)
        # Try parsing individual events
        for chunk in re.split(r'(?=<Event[\s>])', text):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                elem = ET.fromstring(chunk)
                events.append(_extract_event_dict(elem))
            except ET.ParseError:
                pass
        return events

    for elem in root:
        if elem.tag in (f"{{{_WIN_NS}}}Event", "Event"):
            events.append(_extract_event_dict(elem))
    return events


def _extract_event_dict(elem: ET.Element) -> dict:
    """Extract a flat dict from an <Event> XML element."""
    ns = _WIN_NS
    result: dict = {}

    # Handle both namespaced and non-namespaced elements
    def find(parent, tag):
        node = parent.find(f"{{{ns}}}{tag}")
        if node is None:
            node = parent.find(tag)
        return node

    def findall(parent, tag):
        nodes = parent.findall(f"{{{ns}}}{tag}")
        if not nodes:
            nodes = parent.findall(tag)
        return nodes

    system = find(elem, "System")
    if system is not None:
        eid = find(system, "EventID")
        result["EventID"] = eid.text.strip() if eid is not None and eid.text else ""
        tc = find(system, "TimeCreated")
        result["TimeCreated"] = tc.get("SystemTime", "") if tc is not None else ""
        comp = find(system, "Computer")
        result["Computer"] = comp.text.strip() if comp is not None and comp.text else ""
        prov = find(system, "Provider")
        result["Provider"] = prov.get("Name", "") if prov is not None else ""

    event_data = find(elem, "EventData")
    if event_data is not None:
        for data in findall(event_data, "Data"):
            name = data.get("Name", "")
            value = data.text.strip() if data.text else ""
            if name:
                result[name] = value

    return result


def _group_xml_events_by_computer(events: list[dict], window_minutes: int = 10) -> list[list[dict]]:
    """Group XML events by Computer within a time window."""
    by_computer: dict[str, list[dict]] = defaultdict(list)
    for ev in events:
        comp = ev.get("Computer", "unknown")
        by_computer[comp].append(ev)
    return list(by_computer.values())


def _narrate_sysmon_group(group: list[dict]) -> str:
    """Narrate a group of Sysmon events from the same computer."""
    if not group:
        return ""

    computer = group[0].get("Computer", "unknown")
    first_time = group[0].get("TimeCreated", "")[:19]
    parts: list[str] = []

    proc_creates: list[dict] = []
    net_conns: list[dict] = []
    file_creates: list[dict] = []
    reg_sets: list[dict] = []
    sched_tasks: list[dict] = []
    iis_events: list[dict] = []
    other: list[dict] = []

    for ev in group:
        eid = ev.get("EventID", "")
        provider = ev.get("Provider", "")
        if eid == "1":
            proc_creates.append(ev)
        elif eid == "3":
            net_conns.append(ev)
        elif eid == "7":
            other.append(ev)  # Image loaded
        elif eid == "11":
            file_creates.append(ev)
        elif eid == "13":
            reg_sets.append(ev)
        elif eid in ("22", "106"):
            sched_tasks.append(ev)
        elif eid == "6200" or "IIS" in provider:
            iis_events.append(ev)
        else:
            other.append(ev)

    # IIS upload (webshell)
    if iis_events:
        for ev in iis_events:
            method = ev.get("Method", "")
            uri = ev.get("UriStem", "")
            query = ev.get("UriQuery", "")
            client_ip = ev.get("ClientIP", "unknown")
            status = ev.get("HttpStatus", "?")
            parts.append(
                f"At {first_time}, IIS on {computer} received a {method} request to "
                f"\"{uri}\" (query: {query}) from IP {client_ip}, returning HTTP {status}. "
                f"Uploading files via web forms to server-accessible paths is the primary "
                f"webshell deployment vector ({MITRE['webshell']})."
            )

    # Process creation
    for ev in proc_creates:
        image = ev.get("Image", "unknown")
        cmdline = ev.get("CommandLine", "")
        user = ev.get("User", "unknown")
        parent = ev.get("ParentImage", "")
        ts = ev.get("TimeCreated", "")[:19]

        is_ps = "powershell" in image.lower()
        is_encoded = "-enc" in cmdline.lower() or "-encodedcommand" in cmdline.lower()
        is_hidden = "-w hidden" in cmdline.lower() or "-windowstyle hidden" in cmdline.lower()
        is_noprofile = "-nop" in cmdline.lower() or "-noprofile" in cmdline.lower()
        is_iis_parent = "w3wp" in parent.lower()
        is_outlook_parent = "outlook" in parent.lower()
        is_payload = "payload" in image.lower() or "temp" in image.lower()

        if is_ps and (is_encoded or is_hidden):
            flags = []
            if is_noprofile:
                flags.append("-NoProfile")
            if is_hidden:
                flags.append("-WindowStyle Hidden")
            if is_encoded:
                flags.append("-EncodedCommand (base64 obfuscated payload)")
            flag_str = ", ".join(flags)

            parent_context = ""
            if is_outlook_parent:
                parent_context = (
                    f" The parent process was Microsoft Outlook, suggesting this was triggered "
                    f"by a malicious email attachment or macro — a classic phishing-to-execution "
                    f"chain."
                )
            elif is_iis_parent:
                parent_context = (
                    f" The parent process was the IIS worker process (w3wp.exe), confirming "
                    f"this PowerShell was spawned by the previously uploaded webshell."
                )

            parts.append(
                f"At {ts}, user \"{user}\" on {computer} launched PowerShell with suspicious "
                f"flags: {flag_str}.{parent_context} "
                f"The use of encoded commands and hidden window mode is a strong indicator of "
                f"obfuscated malware execution or post-exploitation activity "
                f"({MITRE['obfuscated_exec']})."
            )
        elif is_payload:
            parts.append(
                f"At {ts}, a suspicious executable \"{image}\" was launched by user \"{user}\" "
                f"with command line \"{cmdline[:100]}\". "
                f"Executables dropped in temporary directories and run silently are characteristic "
                f"of malware deployment ({MITRE['download_exec']})."
            )
        else:
            parts.append(
                f"At {ts}, process \"{image.split(chr(92))[-1]}\" was created by user \"{user}\" "
                f"(parent: \"{parent.split(chr(92))[-1] if parent else 'unknown'}\", "
                f"command: \"{cmdline[:80]}\")."
            )

    # Network connections
    for ev in net_conns:
        image = ev.get("Image", "unknown")
        dst_ip = ev.get("DestinationIp", "?")
        dst_port = ev.get("DestinationPort", "?")
        proto = ev.get("Protocol", "tcp")
        ts = ev.get("TimeCreated", "")[:19]
        parts.append(
            f"At {ts}, process \"{image.split(chr(92))[-1]}\" initiated a {proto.upper()} "
            f"connection to {dst_ip}:{dst_port}. "
            + (f"Outbound connections from PowerShell to external IPs are a hallmark of "
               f"C2 communication ({MITRE['c2_beacon']})."
               if "powershell" in image.lower() else "")
        )

    # File creation
    for ev in file_creates:
        filename = ev.get("TargetFilename", "unknown")
        image = ev.get("Image", "unknown")
        user = ev.get("User", "unknown")
        ts = ev.get("TimeCreated", "")[:19]
        is_script = any(filename.lower().endswith(ext) for ext in (".ps1", ".vbs", ".bat", ".js", ".hta"))
        is_appdata = "appdata" in filename.lower() or "temp" in filename.lower()
        parts.append(
            f"At {ts}, file \"{filename}\" was created by \"{image.split(chr(92))[-1]}\" "
            f"(user: {user})."
            + (" Dropping scripts into AppData or Temp directories is a common persistence "
               f"and staging technique." if is_script and is_appdata else "")
        )

    # Registry modifications
    for ev in reg_sets:
        target = ev.get("TargetObject", "unknown")
        details = ev.get("Details", "")
        user = ev.get("User", "unknown")
        ts = ev.get("TimeCreated", "")[:19]
        is_run_key = "\\Run\\" in target or "\\RunOnce\\" in target
        parts.append(
            f"At {ts}, registry value \"{target}\" was set to \"{details[:80]}\" by user \"{user}\"."
            + (f" Modifying Run/RunOnce registry keys establishes persistence by executing "
               f"the payload on every user login ({MITRE['persistence_registry']})."
               if is_run_key else "")
        )

    # Scheduled tasks
    for ev in sched_tasks:
        task = ev.get("TaskName", ev.get("Data", "unknown"))
        user_ctx = ev.get("UserContext", "unknown")
        ts = ev.get("TimeCreated", "")[:19]
        parts.append(
            f"At {ts}, scheduled task \"{task}\" was registered under user context \"{user_ctx}\". "
            f"Creating scheduled tasks is a persistence mechanism ({MITRE['scheduled_task']})."
        )

    if not parts:
        return ""

    # Overall assessment
    is_malicious = bool(proc_creates and (net_conns or reg_sets or file_creates)) or bool(iis_events)
    if is_malicious:
        parts.append(
            f"The sequence of events on {computer} represents a complete attack chain: "
            f"initial access, code execution, C2 establishment, and persistence. "
            f"Immediate incident response is recommended."
        )

    return " ".join(parts).strip()


def _narrate_winevent_group(group: list[dict]) -> str:
    """Narrate a group of Windows Security Events from the same computer."""
    if not group:
        return ""

    computer = group[0].get("Computer", "unknown")
    parts: list[str] = []

    logon_success: list[dict] = []
    logon_fail: list[dict] = []
    special_priv: list[dict] = []
    proc_create: list[dict] = []
    sched_task: list[dict] = []
    account_change: list[dict] = []
    share_access: list[dict] = []
    service_install: list[dict] = []

    for ev in group:
        eid = ev.get("EventID", "")
        if eid == "4624":
            logon_success.append(ev)
        elif eid == "4625":
            logon_fail.append(ev)
        elif eid == "4672":
            special_priv.append(ev)
        elif eid == "4688":
            proc_create.append(ev)
        elif eid in ("4698", "4702"):
            sched_task.append(ev)
        elif eid in ("4720", "4732", "4728"):
            account_change.append(ev)
        elif eid == "5140":
            share_access.append(ev)
        elif eid == "7045":
            service_install.append(ev)

    logon_type_map = {
        "2": "interactive (local console)",
        "3": "network (remote file share or named pipe)",
        "4": "batch (scheduled task)",
        "5": "service",
        "7": "unlock",
        "8": "network cleartext",
        "9": "new credentials (runas)",
        "10": "remote interactive (RDP/Terminal Services)",
        "11": "cached interactive",
    }

    # Logon success
    for ev in logon_success:
        user = ev.get("TargetUserName", "unknown")
        domain = ev.get("TargetDomainName", "")
        logon_type = ev.get("LogonType", "?")
        auth_pkg = ev.get("AuthenticationPackageName", "")
        src_ip = ev.get("IpAddress", "")
        workstation = ev.get("WorkstationName", "")
        ts = ev.get("TimeCreated", "")[:19]
        logon_desc = logon_type_map.get(logon_type, f"type {logon_type}")

        ntlm_note = ""
        if auth_pkg.upper() == "NTLM" and logon_type == "3":
            ntlm_note = (
                f" Authentication used NTLM rather than Kerberos for a network logon, "
                f"which may indicate pass-the-hash or legacy authentication "
                f"({MITRE['pass_the_hash']})."
            )

        parts.append(
            f"At {ts}, Windows Security Event 4624 on {computer} recorded a successful "
            f"{logon_desc} logon for user \"{domain}\\{user}\" "
            + (f"from IP {src_ip} " if src_ip else "")
            + (f"(workstation: {workstation}) " if workstation else "")
            + f"using {auth_pkg}.{ntlm_note}"
        )

    # Special privileges
    for ev in special_priv:
        user = ev.get("SubjectUserName", "unknown")
        privs = ev.get("PrivilegeList", "")
        ts = ev.get("TimeCreated", "")[:19]
        dangerous_privs = [p for p in privs.split() if p in
                           ("SeDebugPrivilege", "SeImpersonatePrivilege", "SeTcbPrivilege",
                            "SeLoadDriverPrivilege", "SeTakeOwnershipPrivilege")]
        if dangerous_privs:
            parts.append(
                f"At {ts}, Event 4672 recorded that user \"{user}\" was assigned sensitive "
                f"privileges: {', '.join(dangerous_privs)}. "
                f"SeDebugPrivilege allows reading memory of other processes (used by credential "
                f"dumpers like Mimikatz); SeImpersonatePrivilege enables token impersonation "
                f"attacks. This is a high-risk privilege assignment."
            )
        else:
            parts.append(
                f"At {ts}, user \"{user}\" was assigned special privileges: {privs[:100]}."
            )

    # Logon failures
    if logon_fail:
        fail_users = list({ev.get("TargetUserName", "?") for ev in logon_fail})
        fail_ips = list({ev.get("IpAddress", "?") for ev in logon_fail})
        parts.append(
            f"{len(logon_fail)} failed logon attempt(s) (Event 4625) were recorded on {computer} "
            f"for user(s) {', '.join(f'\"' + u + '\"' for u in fail_users[:3])} "
            f"from IP(s) {', '.join(fail_ips[:3])}. "
            f"Repeated failures may indicate a brute-force or password spray attack."
        )

    # Network share access
    for ev in share_access:
        share = ev.get("ShareName", "unknown")
        user = ev.get("SubjectUserName", "unknown")
        src_ip = ev.get("IpAddress", "")
        ts = ev.get("TimeCreated", "")[:19]
        is_admin_share = "ADMIN$" in share or "C$" in share or "IPC$" in share
        parts.append(
            f"At {ts}, user \"{user}\" accessed network share \"{share}\" "
            + (f"from IP {src_ip}. " if src_ip else ". ")
            + (f"Access to administrative shares (ADMIN$, C$) is commonly used for lateral "
               f"movement via PsExec or similar tools ({MITRE['lateral_movement_smb']})."
               if is_admin_share else "")
        )

    # Service installation
    for ev in service_install:
        svc_name = ev.get("ServiceName", "unknown")
        image_path = ev.get("ImagePath", "")
        account = ev.get("AccountName", "")
        ts = ev.get("TimeCreated", "")[:19]
        is_encoded = "-enc" in image_path.lower() or "base64" in image_path.lower()
        parts.append(
            f"At {ts}, a new service \"{svc_name}\" was installed with image path "
            f"\"{image_path[:100]}\" running as \"{account}\"."
            + (f" The service command contains encoded PowerShell, a strong indicator of "
               f"malware persistence via service installation ({MITRE['obfuscated_exec']})."
               if is_encoded else "")
        )

    if not parts:
        return ""

    # Assess overall
    is_suspicious = bool(special_priv) or bool(share_access) or bool(service_install)
    if is_suspicious:
        parts.append(
            f"The combination of privileged logon, sensitive privilege assignment, and "
            f"administrative share access on {computer} is consistent with lateral movement "
            f"and post-exploitation activity by a threat actor with valid credentials."
        )

    return " ".join(parts).strip()


def convert_xml(path: Path) -> list[str]:
    """Auto-detect Sysmon vs Windows Security Event XML and convert."""
    text = _safe_read(path)
    events = _parse_xml_events(text)
    if not events:
        return []

    # Detect type by EventID range and Provider
    providers = {ev.get("Provider", "") for ev in events}
    is_sysmon = any("Sysmon" in p for p in providers) or any(
        ev.get("EventID") in ("1", "3", "7", "11", "13", "22") for ev in events
    )
    is_winevent = any("Security-Auditing" in p for p in providers) or any(
        ev.get("EventID") in ("4624", "4625", "4672", "4688", "4698", "4720") for ev in events
    )

    groups = _group_xml_events_by_computer(events)
    narratives = []
    for grp in groups:
        if is_sysmon or any(ev.get("EventID") in ("1", "3", "11", "13") for ev in grp):
            narr = _narrate_sysmon_group(grp)
        else:
            narr = _narrate_winevent_group(grp)
        if not narr:
            # Try both
            narr = _narrate_sysmon_group(grp) or _narrate_winevent_group(grp)
        if narr and _word_count(narr) >= 30:
            narratives.append(narr)
    return narratives


# ===========================================================================
# FORMAT 8 – AWS CloudTrail JSON
# ===========================================================================

_CLOUDTRAIL_MITRE = {
    "StopLogging": (MITRE["stop_logging"], "disabling CloudTrail logging to cover tracks"),
    "DeleteTrail": (MITRE["stop_logging"], "deleting a CloudTrail trail to eliminate audit evidence"),
    "CreateUser": (MITRE["create_cloud_account"], "creating a new IAM user for persistence"),
    "AttachUserPolicy": (MITRE["create_cloud_account"], "attaching a policy to a user, potentially granting elevated permissions"),
    "CreateAccessKey": (MITRE["valid_cloud_account"], "creating new access keys for persistent access"),
    "AssumeRole": (MITRE["valid_cloud_account"], "assuming an IAM role"),
    "AuthorizeSecurityGroupIngress": (None, "opening inbound firewall rules"),
    "RunInstances": (None, "launching EC2 instances"),
    "PutBucketPolicy": (MITRE["s3_public"], "modifying S3 bucket policy"),
    "PutBucketWebsite": (None, "configuring S3 static website hosting"),
}


def _get_cloudtrail_user(event: dict) -> str:
    uid = event.get("userIdentity", {})
    utype = uid.get("type", "")
    if utype == "IAMUser":
        return uid.get("userName", "unknown")
    elif utype == "AssumedRole":
        session = uid.get("sessionContext", {}).get("sessionIssuer", {})
        role = session.get("userName", uid.get("arn", "unknown").split("/")[-1])
        session_name = uid.get("principalId", "").split(":")[-1]
        return f"{role} (session: {session_name})"
    elif utype == "Root":
        return "root account"
    return uid.get("userName", uid.get("arn", "unknown").split("/")[-1])


def _group_cloudtrail_by_user(events: list[dict], window_minutes: int = 30) -> list[list[dict]]:
    """Group CloudTrail events by IAM user/role."""
    by_user: dict[str, list[dict]] = defaultdict(list)
    for ev in events:
        user = _get_cloudtrail_user(ev)
        by_user[user].append(ev)
    return list(by_user.values())


def _narrate_cloudtrail_group(group: list[dict]) -> str:
    if not group:
        return ""

    user = _get_cloudtrail_user(group[0])
    src_ips = list({ev.get("sourceIPAddress", "unknown") for ev in group})
    region = group[0].get("awsRegion", "unknown")
    first_time = group[0].get("eventTime", "")[:19]
    last_time = group[-1].get("eventTime", "")[:19]

    parts: list[str] = []
    parts.append(
        f"AWS CloudTrail recorded {len(group)} API call(s) by \"{user}\" "
        f"from IP(s) {', '.join(src_ips[:3])} in region {region} "
        f"between {first_time} and {last_time} UTC."
    )

    for ev in group:
        event_name = ev.get("eventName", "unknown")
        event_source = ev.get("eventSource", "").replace(".amazonaws.com", "")
        src_ip = ev.get("sourceIPAddress", "?")
        ts = ev.get("eventTime", "")[:19]
        params = ev.get("requestParameters") or {}
        response = ev.get("responseElements") or {}
        user_agent = ev.get("userAgent", "")

        mitre_info = _CLOUDTRAIL_MITRE.get(event_name)

        if event_name == "StopLogging":
            trail = params.get("name", "unknown")
            parts.append(
                f"At {ts}, the StopLogging API was called to disable logging for trail "
                f"\"{trail}\" via {event_source}. This is a critical defense evasion technique "
                f"({MITRE['stop_logging']}) commonly used by attackers to prevent their "
                f"subsequent actions from being recorded."
            )
        elif event_name == "DeleteTrail":
            trail = params.get("name", "unknown")
            parts.append(
                f"At {ts}, the DeleteTrail API was called to permanently delete trail "
                f"\"{trail}\". This eliminates all future audit logging ({MITRE['stop_logging']})."
            )
        elif event_name == "RunInstances":
            instances = response.get("instancesSet", {}).get("items", [])
            instance_ids = [i.get("instanceId", "?") for i in instances]
            instance_type = params.get("instancesSet", {}).get("items", [{}])[0].get("instanceType", "?")
            parts.append(
                f"At {ts}, {len(instances)} EC2 instance(s) of type {instance_type} were launched "
                f"(IDs: {', '.join(instance_ids[:3])}). "
                + (f"The use of Terraform as the user agent suggests infrastructure-as-code deployment."
                   if "terraform" in user_agent.lower() else "")
            )
        elif event_name == "AuthorizeSecurityGroupIngress":
            sg = params.get("groupId", "?")
            perms = params.get("ipPermissions", {}).get("items", [{}])
            for perm in perms[:2]:
                from_port = perm.get("fromPort", "?")
                to_port = perm.get("toPort", "?")
                cidr = perm.get("ipRanges", {}).get("items", [{}])[0].get("cidrIp", "?")
                parts.append(
                    f"At {ts}, security group {sg} was modified to allow inbound TCP "
                    f"port {from_port}-{to_port} from {cidr}. "
                    + ("Opening port 0.0.0.0/0 exposes the service to the entire internet, "
                       "which is a significant security risk."
                       if cidr == "0.0.0.0/0" else "")
                )
        elif event_name == "CreateUser":
            new_user = params.get("userName", "?")
            parts.append(
                f"At {ts}, a new IAM user \"{new_user}\" was created ({MITRE['create_cloud_account']}). "
                f"Creating IAM users is a persistence technique used by attackers to maintain "
                f"access even if the original compromised credentials are revoked."
            )
        elif event_name in ("PutBucketPolicy", "PutBucketWebsite"):
            bucket = params.get("bucketName", "?")
            if event_name == "PutBucketPolicy":
                policy_str = str(params.get("bucketPolicy", ""))
                is_public = '"Principal":"*"' in policy_str or '"Principal": "*"' in policy_str
                parts.append(
                    f"At {ts}, the S3 bucket policy for \"{bucket}\" was modified."
                    + (" The new policy grants public read access (Principal: *), "
                       f"potentially exposing sensitive data ({MITRE['s3_public']})."
                       if is_public else "")
                )
            else:
                parts.append(
                    f"At {ts}, static website hosting was configured for S3 bucket \"{bucket}\"."
                )
        elif mitre_info:
            mitre_ref, action_desc = mitre_info
            parts.append(
                f"At {ts}, {event_source} API call \"{event_name}\" was made, {action_desc}."
                + (f" ({mitre_ref})" if mitre_ref else "")
            )
        else:
            parts.append(
                f"At {ts}, {event_source} API \"{event_name}\" was called from {src_ip}."
            )

    # Assess overall risk
    event_names = [ev.get("eventName") for ev in group]
    is_high_risk = any(n in ("StopLogging", "DeleteTrail", "CreateUser") for n in event_names)
    has_unusual_ip = any(
        not ev.get("sourceIPAddress", "").startswith(("10.", "172.", "192.168."))
        for ev in group
    )

    if is_high_risk:
        parts.append(
            f"This activity sequence is highly suspicious: disabling audit logging followed by "
            f"infrastructure changes from an external IP is a classic attacker playbook for "
            f"establishing persistence while evading detection."
        )
    elif has_unusual_ip:
        parts.append(
            f"All API calls originated from external IP(s) {', '.join(src_ips[:3])}, "
            f"which warrants verification that this activity was authorized."
        )

    return " ".join(parts).strip()


def convert_cloudtrail(path: Path) -> list[str]:
    text = _safe_read(path)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("CloudTrail JSON parse error in %s: %s", path, e)
        return []

    # CloudTrail can be a list or {"Records": [...]}
    if isinstance(data, dict):
        events = data.get("Records", data.get("events", [data]))
    elif isinstance(data, list):
        events = data
    else:
        return []

    groups = _group_cloudtrail_by_user(events)
    narratives = []
    for grp in groups:
        narr = _narrate_cloudtrail_group(grp)
        if narr and _word_count(narr) >= 30:
            narratives.append(narr)
    return narratives


# ===========================================================================
# FORMAT 9 – Azure Activity Log JSON
# ===========================================================================

def _group_azure_by_caller(events: list[dict]) -> list[list[dict]]:
    by_caller: dict[str, list[dict]] = defaultdict(list)
    for ev in events:
        caller = ev.get("caller", "unknown")
        by_caller[caller].append(ev)
    return list(by_caller.values())


def _narrate_azure_group(group: list[dict]) -> str:
    if not group:
        return ""

    caller = group[0].get("caller", "unknown")
    src_ip = group[0].get("claims", {}).get("ipaddr", "unknown")
    first_time = group[0].get("eventTimestamp", "")[:19]
    last_time = group[-1].get("eventTimestamp", "")[:19]

    parts: list[str] = []
    parts.append(
        f"Azure Activity Log recorded {len(group)} administrative operation(s) by "
        f"\"{caller}\" from IP {src_ip} between {first_time} and {last_time} UTC."
    )

    for ev in group:
        op_name = ev.get("operationName", {})
        op_value = op_name.get("value", "") if isinstance(op_name, dict) else str(op_name)
        op_display = op_name.get("localizedValue", op_value) if isinstance(op_name, dict) else op_value
        status = ev.get("status", {}).get("value", "unknown") if isinstance(ev.get("status"), dict) else str(ev.get("status", "unknown"))
        ts = ev.get("eventTimestamp", "")[:19]
        rg = ev.get("resourceGroupName", "")
        props = ev.get("properties", {}) or {}
        level = ev.get("level", "")

        if "roleAssignments/write" in op_value:
            principal_type = props.get("principalType", "unknown")
            role_def = props.get("roleDefinitionId", "").split("/")[-1]
            scope = ev.get("authorization", {}).get("scope", "")
            parts.append(
                f"At {ts}, a role assignment was created for a {principal_type} "
                f"(role definition: {role_def}) at scope \"{scope}\" (status: {status}). "
                f"Granting roles to service principals at subscription scope is a privilege "
                f"escalation technique ({MITRE['azure_role_assign']})."
            )
        elif "applications/create" in op_value or "Application" in op_display:
            app_name = props.get("displayName", "unknown")
            app_id = props.get("appId", "?")
            parts.append(
                f"At {ts}, a new Azure AD application \"{app_name}\" (appId: {app_id}) was "
                f"created. Registering applications can be used to create persistent OAuth "
                f"tokens or service principals ({MITRE['azure_app_create']})."
            )
        elif "KeyVault" in op_value and "secrets" in op_value.lower():
            vault = ev.get("resourceId", "").split("/")[-1]
            secret_name = props.get("secretName", "")
            if "list" in op_value.lower():
                parts.append(
                    f"At {ts}, all secrets in Key Vault \"{vault}\" were listed (status: {status}). "
                    f"Enumerating Key Vault secrets is a credential access technique that may "
                    f"expose API keys, passwords, and certificates."
                )
            elif "set" in op_value.lower():
                parts.append(
                    f"At {ts}, secret \"{secret_name}\" was written to Key Vault \"{vault}\" "
                    f"(status: {status})."
                )
            else:
                parts.append(
                    f"At {ts}, Key Vault operation \"{op_display}\" was performed on \"{vault}\" "
                    f"(status: {status})."
                )
        elif "accessPolicies/write" in op_value:
            vault = ev.get("resourceId", "").split("/")[-1]
            access_policy = props.get("accessPolicy", {})
            perms = access_policy.get("permissions", {}).get("secrets", [])
            parts.append(
                f"At {ts}, the access policy for Key Vault \"{vault}\" was modified to grant "
                f"secret permissions [{', '.join(perms)}] to an additional principal. "
                f"Expanding Key Vault access is a persistence technique."
            )
        else:
            parts.append(
                f"At {ts}, operation \"{op_display}\" was performed "
                + (f"in resource group \"{rg}\" " if rg else "")
                + f"(status: {status}, level: {level})."
            )

    # Risk assessment
    op_values = [ev.get("operationName", {}).get("value", "") if isinstance(ev.get("operationName"), dict)
                 else str(ev.get("operationName", "")) for ev in group]
    is_high_risk = any("roleAssignments" in o or "applications/create" in o for o in op_values)
    if is_high_risk:
        parts.append(
            f"The combination of application creation and role assignment from an external IP "
            f"is a common pattern in Azure tenant compromise, where attackers create persistent "
            f"backdoor service principals with elevated permissions."
        )

    return " ".join(parts).strip()


def convert_azure(path: Path) -> list[str]:
    text = _safe_read(path)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("Azure JSON parse error in %s: %s", path, e)
        return []

    events = data if isinstance(data, list) else data.get("value", [data])
    groups = _group_azure_by_caller(events)
    narratives = []
    for grp in groups:
        narr = _narrate_azure_group(grp)
        if narr and _word_count(narr) >= 30:
            narratives.append(narr)
    return narratives


# ===========================================================================
# FORMAT 10 – GCP Audit Log JSON
# ===========================================================================

_GCP_MITRE = {
    "SetIamPolicy": (MITRE["gcp_iam_escalation"], "modifying IAM policy, potentially granting elevated permissions"),
    "UpdateNotificationConfig": (MITRE["defense_evasion_gcp"], "disabling security notification configurations"),
    "CreateServiceAccountKey": (MITRE["valid_cloud_account"], "creating a new service account key for persistent access"),
    "ListServiceAccountKeys": (None, "enumerating service account keys"),
    "storage.objects.list": (MITRE["data_exfil"], "listing objects in a storage bucket"),
    "storage.objects.get": (MITRE["data_exfil"], "downloading objects from a storage bucket"),
}


def _group_gcp_by_principal(events: list[dict]) -> list[list[dict]]:
    by_principal: dict[str, list[dict]] = defaultdict(list)
    for ev in events:
        principal = ev.get("protoPayload", {}).get("authenticationInfo", {}).get("principalEmail", "unknown")
        by_principal[principal].append(ev)
    return list(by_principal.values())


def _narrate_gcp_group(group: list[dict]) -> str:
    if not group:
        return ""

    first_payload = group[0].get("protoPayload", {})
    principal = first_payload.get("authenticationInfo", {}).get("principalEmail", "unknown")
    src_ip = first_payload.get("requestMetadata", {}).get("callerIp", "unknown")
    first_time = group[0].get("timestamp", "")[:19]
    last_time = group[-1].get("timestamp", "")[:19]

    parts: list[str] = []
    parts.append(
        f"GCP Audit Log recorded {len(group)} API call(s) by principal \"{principal}\" "
        f"from IP {src_ip} between {first_time} and {last_time} UTC."
    )

    for ev in group:
        payload = ev.get("protoPayload", {})
        method = payload.get("methodName", "unknown")
        service = payload.get("serviceName", "unknown")
        resource = payload.get("resourceName", "")
        ts = ev.get("timestamp", "")[:19]
        severity = ev.get("severity", "")
        user_agent = payload.get("requestMetadata", {}).get("callerSuppliedUserAgent", "")
        request = payload.get("request", {})

        # Short method name for lookup
        short_method = method.split(".")[-1] if "." in method else method

        mitre_info = _GCP_MITRE.get(short_method) or _GCP_MITRE.get(method)

        if short_method == "SetIamPolicy":
            bindings = request.get("policy", {}).get("bindings", [])
            for binding in bindings[:2]:
                role = binding.get("role", "?")
                members = binding.get("members", [])
                parts.append(
                    f"At {ts}, IAM policy on project \"{resource}\" was modified to grant "
                    f"role \"{role}\" to {', '.join(members[:3])}. "
                    f"Granting owner/editor roles to service accounts is a privilege escalation "
                    f"technique ({MITRE['gcp_iam_escalation']})."
                )
        elif "UpdateNotificationConfig" in method:
            config_name = payload.get("resourceName", "?")
            new_topic = request.get("notificationConfig", {}).get("pubsubTopic", "")
            parts.append(
                f"At {ts}, the Security Command Center notification config \"{config_name}\" "
                f"was updated to remove its Pub/Sub topic (set to empty: \"{new_topic}\"). "
                f"Disabling security alert notifications prevents defenders from receiving "
                f"real-time threat alerts ({MITRE['defense_evasion_gcp']})."
            )
        elif "CreateServiceAccountKey" in method:
            sa = resource.split("/serviceAccounts/")[-1] if "/serviceAccounts/" in resource else resource
            parts.append(
                f"At {ts}, a new key was created for service account \"{sa}\" via curl. "
                f"Creating service account keys provides persistent, long-lived credentials "
                f"that bypass MFA ({MITRE['valid_cloud_account']})."
            )
        elif "storage.objects.get" in method:
            obj = resource.split("/objects/")[-1] if "/objects/" in resource else resource
            obj_size = payload.get("metadata", {}).get("object_size", "unknown")
            try:
                size_mb = int(obj_size) / (1024 * 1024)
                size_str = f"{size_mb:.1f} MB"
            except (ValueError, TypeError):
                size_str = f"{obj_size} bytes"
            parts.append(
                f"At {ts}, object \"{obj}\" ({size_str}) was downloaded from GCS "
                f"using {user_agent.split('/')[0] if '/' in user_agent else user_agent}. "
                f"Downloading large data exports from production buckets may indicate "
                f"data exfiltration ({MITRE['data_exfil']})."
            )
        elif "storage.objects.list" in method:
            bucket = resource.split("/buckets/")[-1].split("/")[0] if "/buckets/" in resource else resource
            parts.append(
                f"At {ts}, the contents of GCS bucket \"{bucket}\" were enumerated. "
                f"Listing bucket contents is typically a precursor to data exfiltration."
            )
        elif "ListServiceAccountKeys" in method:
            sa = resource.split("/serviceAccounts/")[-1] if "/serviceAccounts/" in resource else resource
            parts.append(
                f"At {ts}, all keys for service account \"{sa}\" were listed via curl. "
                f"Enumerating service account keys may be reconnaissance for credential theft."
            )
        elif mitre_info:
            mitre_ref, action_desc = mitre_info
            parts.append(
                f"At {ts}, GCP API \"{method}\" was called on \"{resource}\", {action_desc}."
                + (f" ({mitre_ref})" if mitre_ref else "")
            )
        else:
            parts.append(
                f"At {ts}, {service} API \"{method}\" was called on resource \"{resource}\" "
                f"(severity: {severity})."
            )

    # Risk assessment
    methods = [ev.get("protoPayload", {}).get("methodName", "") for ev in group]
    is_high_risk = any("SetIamPolicy" in m or "CreateServiceAccountKey" in m
                       or "UpdateNotificationConfig" in m for m in methods)
    has_data_access = any("storage.objects" in m for m in methods)

    if is_high_risk and has_data_access:
        parts.append(
            f"This activity sequence — IAM escalation, disabling security notifications, "
            f"creating persistent credentials, and downloading production data — represents "
            f"a complete cloud compromise and data exfiltration scenario."
        )
    elif is_high_risk:
        parts.append(
            f"The IAM and security configuration changes from an external IP warrant "
            f"immediate investigation to determine if this represents unauthorized access."
        )

    return " ".join(parts).strip()


def convert_gcp(path: Path) -> list[str]:
    text = _safe_read(path)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("GCP JSON parse error in %s: %s", path, e)
        return []

    events = data if isinstance(data, list) else data.get("entries", [data])
    groups = _group_gcp_by_principal(events)
    narratives = []
    for grp in groups:
        narr = _narrate_gcp_group(grp)
        if narr and _word_count(narr) >= 30:
            narratives.append(narr)
    return narratives


# ===========================================================================
# Format auto-detection and dispatch
# ===========================================================================

def _detect_format(path: Path) -> str:
    """Detect log format from filename and content sniffing."""
    name = path.name.lower()
    suffix = path.suffix.lower()

    # By filename
    if name in ("auth.log", "auth2.log", "privesc.log", "syslog_benign.log"):
        return "syslog"
    if name in ("apache.log", "apache2.log"):
        return "apache"
    if name == "cef.log":
        return "cef"
    if name in ("conn.log", "conn2.log"):
        return "zeek_conn"
    if name in ("log.jsonl", "c2_dns.jsonl"):
        return "suricata_jsonl"
    if name in ("sysmon.xml", "webshell.xml"):
        return "sysmon_xml"
    if name in ("winevent.xml", "winevent2.xml"):
        return "winevent_xml"
    if name in ("cloudtrail.json", "cloudtrail2.json"):
        return "cloudtrail"
    if name in ("azure_activity.json", "azure_activity2.json"):
        return "azure"
    if name in ("gcp_audit.json", "gcp_audit2.json"):
        return "gcp"

    # By content sniffing
    try:
        snippet = path.read_text(encoding="utf-8", errors="replace")[:2000]
    except Exception:
        return "unknown"

    if "CEF:0|" in snippet:
        return "cef"
    if "#fields\tts\tuid" in snippet or "#separator" in snippet:
        return "zeek_conn"
    if '"event_type"' in snippet and ('"alert"' in snippet or '"flow"' in snippet):
        return "suricata_jsonl"
    if "<EventID>" in snippet or "xmlns" in snippet:
        if "Sysmon" in snippet or any(f"<EventID>{e}</EventID>" in snippet for e in ("1", "3", "11", "13")):
            return "sysmon_xml"
        return "winevent_xml"
    if '"eventName"' in snippet and '"awsRegion"' in snippet:
        return "cloudtrail"
    if '"operationName"' in snippet and '"caller"' in snippet:
        return "azure"
    if '"protoPayload"' in snippet and '"methodName"' in snippet:
        return "gcp"
    if suffix == ".jsonl" or (snippet.strip().startswith("{") and "\n{" in snippet):
        return "suricata_jsonl"
    if suffix == ".json":
        return "cloudtrail"  # fallback
    if suffix == ".xml":
        return "winevent_xml"
    if suffix == ".log":
        # Check for Apache combined log format
        if re.search(r'^\d+\.\d+\.\d+\.\d+ - - \[', snippet, re.MULTILINE):
            return "apache"
        return "syslog"

    return "unknown"


_FORMAT_CONVERTERS = {
    "syslog": convert_syslog,
    "apache": convert_apache,
    "cef": convert_cef,
    "zeek_conn": convert_zeek_conn,
    "suricata_jsonl": convert_suricata_jsonl,
    "sysmon_xml": convert_xml,
    "winevent_xml": convert_xml,
    "cloudtrail": convert_cloudtrail,
    "azure": convert_azure,
    "gcp": convert_gcp,
}


def convert_file(path: Path) -> list[str]:
    """Detect format and convert a single file to a list of narrative strings."""
    fmt = _detect_format(path)
    if fmt == "unknown":
        logger.warning("Unknown format for %s — skipping", path)
        return []
    converter = _FORMAT_CONVERTERS.get(fmt)
    if not converter:
        logger.warning("No converter for format '%s' (%s)", fmt, path)
        return []
    logger.info("Converting %s as %s", path.name, fmt)
    try:
        narratives = converter(path)
    except Exception as exc:
        logger.error("Error converting %s: %s", path, exc, exc_info=True)
        return []
    return narratives


# ===========================================================================
# Main pipeline
# ===========================================================================

def _collect_source_files(source: str | None) -> list[tuple[Path, Path]]:
    """
    Returns list of (source_path, output_dir) pairs.
    output_dir is the directory where the JSONL should be written.
    """
    if source:
        p = Path(source)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists():
            logger.error("Source file not found: %s", p)
            return []
        # Determine output dir based on parent
        if "cloud" in str(p.parent):
            out_dir = DEFAULT_CLOUD_NL_DIR
        else:
            out_dir = DEFAULT_LOG_NL_DIR
        return [(p, out_dir)]

    pairs: list[tuple[Path, Path]] = []
    for src_dir, out_dir in [
        (DEFAULT_LOG_DIR, DEFAULT_LOG_NL_DIR),
        (DEFAULT_CLOUD_DIR, DEFAULT_CLOUD_NL_DIR),
    ]:
        if src_dir.exists():
            for f in sorted(src_dir.iterdir()):
                if f.is_file() and not f.name.startswith("."):
                    pairs.append((f, out_dir))
    return pairs


def _output_path(src: Path, out_dir: Path) -> Path:
    """Derive output JSONL path from source file."""
    stem = src.stem
    return out_dir / f"{stem}.jsonl"


def run(
    source: str | None = None,
    output_dir: str | None = None,
    dry_run: bool = False,
) -> dict:
    """
    Main entry point.

    Returns stats dict: {files_processed, docs_generated, avg_word_count, skipped}.
    """
    pairs = _collect_source_files(source)
    if not pairs:
        logger.error("No source files found.")
        return {}

    stats = {
        "files_processed": 0,
        "files_skipped": 0,
        "docs_generated": 0,
        "total_words": 0,
    }

    preview_count = 0

    for src_path, default_out_dir in pairs:
        out_dir = Path(output_dir) if output_dir else default_out_dir

        narratives = convert_file(src_path)
        if not narratives:
            logger.warning("No narratives generated from %s", src_path.name)
            stats["files_skipped"] += 1
            continue

        stats["files_processed"] += 1
        stats["docs_generated"] += len(narratives)
        stats["total_words"] += sum(_word_count(n) for n in narratives)

        if dry_run:
            for narr in narratives:
                if preview_count >= 3:
                    break
                print(f"\n{'='*70}")
                print(f"SOURCE: {src_path.name}")
                print(f"WORDS:  {_word_count(narr)}")
                print(f"{'='*70}")
                print(narr)
                preview_count += 1
            if preview_count >= 3:
                break
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = _output_path(src_path, out_dir)
            with open(out_path, "w", encoding="utf-8") as f:
                for narr in narratives:
                    f.write(json.dumps({"text": narr}, ensure_ascii=False) + "\n")
            logger.info("  → Wrote %d docs to %s", len(narratives), out_path)

    avg_words = (stats["total_words"] // stats["docs_generated"]
                 if stats["docs_generated"] > 0 else 0)

    print(f"\n{'='*50}")
    print(f"  Files processed : {stats['files_processed']}")
    print(f"  Files skipped   : {stats['files_skipped']}")
    print(f"  Docs generated  : {stats['docs_generated']}")
    print(f"  Avg word count  : {avg_words}")
    print(f"{'='*50}\n")

    return {**stats, "avg_word_count": avg_words}


# ===========================================================================
# CLI entry point
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert raw security logs to natural language narratives for LLM pretraining.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--source", "-s",
        metavar="FILE",
        help="Process a single source file instead of all files.",
    )
    p.add_argument(
        "--output-dir", "-o",
        metavar="DIR",
        help="Override output directory (default: data/log_nl/ or data/cloud_nl/).",
    )
    p.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Preview first 3 documents without writing any files.",
    )
    return p


# ===========================================================================
# Unit tests (run when executed directly without --dry-run)
# ===========================================================================

def _run_unit_tests():
    """Basic sanity checks for each converter."""
    import traceback
    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal passed, failed
        if condition:
            print(f"  ✓ {name}")
            passed += 1
        else:
            print(f"  ✗ {name}" + (f": {detail}" if detail else ""))
            failed += 1

    print("\n--- Unit Tests ---")

    # Test 1: syslog parsing
    try:
        ev = _parse_syslog_line(
            "May 16 10:41:02 web01 sshd[21455]: Failed password for invalid user admin from 203.0.113.77 port 49822 ssh2"
        )
        check("syslog parse", ev is not None and ev["host"] == "web01")
        check("syslog SSH fail regex", _SSH_FAIL_RE.search(ev["msg"]) is not None)
    except Exception as e:
        check("syslog parse", False, str(e))

    # Test 2: CEF parsing
    try:
        cef_line = 'CEF:0|Okta|SSO|2026.5|30001|User Authentication Success|2|src=10.14.22.45 suser=amelia.tan@corp.local outcome=success'
        ev = _parse_cef_line(cef_line)
        check("CEF parse", ev is not None and ev["vendor"] == "Okta")
        check("CEF ext suser", ev["ext"].get("suser") == "amelia.tan@corp.local")
    except Exception as e:
        check("CEF parse", False, str(e))

    # Test 3: Zeek conn parsing
    try:
        zeek_text = (
            "#fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\tproto\tservice\tduration\torig_bytes\tresp_bytes\tconn_state\n"
            "1747433281.114522\tCk9j\t10.50.0.15\t42122\t10.50.10.21\t22\ttcp\tssh\t3.221441\t1248\t2310\tSF\n"
        )
        records = _parse_zeek_conn(zeek_text)
        check("Zeek conn parse", len(records) == 1)
        check("Zeek conn fields", records[0].get("id.orig_h") == "10.50.0.15")
    except Exception as e:
        check("Zeek conn parse", False, str(e))

    # Test 4: XML event parsing
    try:
        xml_text = """<Event xmlns="http://schemas.microsoft.com/win/2004/08/events/event">
  <System><EventID>1</EventID><TimeCreated SystemTime="2026-05-16T18:21:11Z"/><Computer>HR-WS22</Computer><Provider Name="Microsoft-Windows-Sysmon"/></System>
  <EventData><Data Name="Image">C:\\Windows\\powershell.exe</Data><Data Name="CommandLine">powershell.exe -enc abc</Data><Data Name="User">CORP\\user</Data></EventData>
</Event>"""
        events = _parse_xml_events(xml_text)
        check("XML parse", len(events) == 1)
        check("XML EventID", events[0].get("EventID") == "1")
        check("XML CommandLine", "-enc" in events[0].get("CommandLine", ""))
    except Exception as e:
        check("XML parse", False, str(e))

    # Test 5: CloudTrail parsing
    try:
        ct_data = [{"eventName": "StopLogging", "userIdentity": {"type": "IAMUser", "userName": "test-user"},
                    "eventTime": "2026-05-16T01:13:27Z", "awsRegion": "us-east-1",
                    "sourceIPAddress": "1.2.3.4", "requestParameters": {"name": "my-trail"}}]
        groups = _group_cloudtrail_by_user(ct_data)
        check("CloudTrail group", len(groups) == 1)
        narr = _narrate_cloudtrail_group(groups[0])
        check("CloudTrail narrative", "StopLogging" in narr and "T1562" in narr)
    except Exception as e:
        check("CloudTrail narrative", False, str(e))

    # Test 6: Output format validation
    try:
        test_narr = "This is a test narrative with enough words to pass the minimum threshold check."
        doc = json.dumps({"text": test_narr}, ensure_ascii=False)
        parsed = json.loads(doc)
        check("JSONL output format", "text" in parsed and parsed["text"] == test_narr)
    except Exception as e:
        check("JSONL output format", False, str(e))

    # Test 7: conn_state descriptions
    check("conn_state SF", "normal completed" in _conn_state_desc("SF"))
    check("conn_state S0", "no response" in _conn_state_desc("S0"))
    check("conn_state REJ", "rejected" in _conn_state_desc("REJ"))

    print(f"\n  Results: {passed} passed, {failed} failed\n")
    return failed == 0


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    if not args.dry_run and args.source is None and args.output_dir is None:
        # No arguments: run unit tests first, then full conversion
        ok = _run_unit_tests()
        if not ok:
            print("Unit tests failed. Fix issues before running full conversion.")
            sys.exit(1)

    run(
        source=args.source,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )
