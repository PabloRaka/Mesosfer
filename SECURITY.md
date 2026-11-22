# Security Policy

## Supported Versions

Security fixes are applied to the `main` branch.

| Branch / Version | Supported |
| :--- | :---: |
| `main` | Yes |
| Older snapshots | No |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, use GitHub's private vulnerability reporting:
- https://github.com/PabloRaka/Mesosfer/security/advisories/new

Include:
- A clear description of the vulnerability and potential impact
- Steps to reproduce or a minimal proof of concept
- Affected files, commits, or hardware/runtime versions

We aim to acknowledge reports within 72 hours and coordinate a fix before public disclosure.

## Responsible & Defensive Use

Mesosfer is designed for defensive cybersecurity research, log analysis,
vulnerability remediation, and educational LLM engineering:

- Do not use this project or its models to automate unauthorized intrusion,
  exploitation, or denial-of-service attacks.
- Inference servers and `ToolHarness` integrations must never execute shell or
  network tool calls without explicit operator opt-in and sandboxing.
