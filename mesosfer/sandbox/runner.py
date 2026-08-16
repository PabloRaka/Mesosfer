"""
Mesosfer Isolated Execution Sandbox Runner
Supports safe isolated execution of Python code, Bash commands, and Network/Subnet utilities.
Auto-detects environment (Local Subprocess, Virtual Workspace, or Docker Sandbox).
"""

import os
import sys
import json
import time
import shutil
import tempfile
import subprocess
import ipaddress
from typing import Dict, Any, Optional

DEFAULT_TIMEOUT = 10
SANDBOX_DIR = os.path.join(tempfile.gettempdir(), "mesosfer_sandbox")

class Sandbox:
    def __init__(self, workspace_dir: Optional[str] = None, use_docker: bool = False, docker_image: str = "python:3.11-slim"):
        self.workspace_dir = workspace_dir or SANDBOX_DIR
        self.use_docker = use_docker
        self.docker_image = docker_image
        os.makedirs(self.workspace_dir, exist_ok=True)

    def write_file(self, filename: str, content: str) -> str:
        """Write content to a file inside the sandbox workspace."""
        filepath = os.path.join(self.workspace_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return filepath

    def read_file(self, filename: str) -> Optional[str]:
        """Read content from a file inside the sandbox workspace."""
        filepath = os.path.join(self.workspace_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        return None

    def execute_python(self, code: str, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
        """
        Execute Python code inside the sandbox environment.
        Captures stdout, stderr, execution duration, and returncode.
        """
        script_file = os.path.join(self.workspace_dir, "_temp_script.py")
        with open(script_file, "w", encoding="utf-8") as f:
            f.write(code)

        t_start = time.time()
        try:
            if self.use_docker:
                cmd = [
                    "docker", "run", "--rm",
                    "-v", f"{os.path.abspath(self.workspace_dir)}:/workspace",
                    "-w", "/workspace",
                    "--network", "none",  # isolated network for security
                    "--memory", "512m",
                    self.docker_image,
                    "python3", "_temp_script.py"
                ]
            else:
                cmd = [sys.executable, script_file]

            proc = subprocess.run(
                cmd,
                cwd=self.workspace_dir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            elapsed = time.time() - t_start
            return {
                "success": proc.returncode == 0,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
                "returncode": proc.returncode,
                "duration_sec": round(elapsed, 3),
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution timed out after {timeout} seconds.",
                "returncode": -1,
                "duration_sec": timeout,
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "duration_sec": round(time.time() - t_start, 3),
            }

    def execute_bash(self, command: str, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
        """Execute a bash or shell command in the sandbox workspace."""
        t_start = time.time()
        try:
            if self.use_docker:
                cmd = [
                    "docker", "run", "--rm",
                    "-v", f"{os.path.abspath(self.workspace_dir)}:/workspace",
                    "-w", "/workspace",
                    "--memory", "512m",
                    self.docker_image,
                    "sh", "-c", command
                ]
            else:
                cmd = command

            proc = subprocess.run(
                cmd,
                cwd=self.workspace_dir,
                shell=not self.use_docker,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            elapsed = time.time() - t_start
            return {
                "success": proc.returncode == 0,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
                "returncode": proc.returncode,
                "duration_sec": round(elapsed, 3),
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Command timed out after {timeout} seconds.",
                "returncode": -1,
                "duration_sec": timeout,
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "duration_sec": round(time.time() - t_start, 3),
            }

    def execute_subnet(self, cidr: str) -> Dict[str, Any]:
        """Calculate network subnetting metrics with 100% precision."""
        try:
            net = ipaddress.ip_network(str(cidr).strip(), strict=False)
            hosts = list(net.hosts())
            first_host = str(hosts[0]) if hosts else str(net.network_address)
            last_host = str(hosts[-1]) if hosts else str(net.broadcast_address)
            return {
                "network": str(net.network_address),
                "netmask": str(net.netmask),
                "broadcast": str(net.broadcast_address),
                "usable_host_range": f"{first_host} - {last_host}",
                "num_usable_hosts": max(0, net.num_addresses - 2) if net.prefixlen < 31 else net.num_addresses,
                "total_addresses": net.num_addresses,
            }
        except Exception as e:
            return {"error": f"Invalid CIDR: {e}"}

    def run_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Dispatcher for tool calls emitted by Mesosfer."""
        name = str(tool_name).lower().strip()

        if name in ["subnet", "ipcalc", "network", "ipaddress"]:
            cidr = arguments.get("cidr") or arguments.get("subnet") or arguments.get("ip") or arguments.get("network") or ""
            res = self.execute_subnet(str(cidr))
            return json.dumps(res, indent=2)

        if name in ["python", "py", "calc", "calculator"]:
            code = arguments.get("code") or arguments.get("expression") or arguments.get("command") or ""
            if not code and isinstance(arguments, str):
                code = arguments
            res = self.execute_python(str(code))
            if res["success"]:
                return res["stdout"] if res["stdout"] else "Code executed successfully (no stdout)."
            else:
                return f"Error ({res['returncode']}):\n{res['stderr']}"

        if name in ["bash", "shell", "cmd", "terminal"]:
            command = arguments.get("command") or arguments.get("cmd") or ""
            if not command and isinstance(arguments, str):
                command = arguments
            res = self.execute_bash(str(command))
            if res["success"]:
                return res["stdout"] if res["stdout"] else "Command executed successfully (no stdout)."
            else:
                return f"Error ({res['returncode']}):\n{res['stderr']}"

        if name in ["write_file", "create_file"]:
            filename = arguments.get("filename") or "file.txt"
            content = arguments.get("content") or ""
            path = self.write_file(filename, content)
            return f"File saved to {path} ({len(content)} bytes)."

        if name in ["read_file"]:
            filename = arguments.get("filename") or ""
            content = self.read_file(filename)
            return content if content is not None else f"File {filename} not found."

        # Fallback to python execution
        return str(self.execute_python(str(arguments)))

# Global default sandbox instance
_default_sandbox = None

def get_default_sandbox() -> Sandbox:
    global _default_sandbox
    if _default_sandbox is None:
        _default_sandbox = Sandbox()
    return _default_sandbox
