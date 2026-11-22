# Contributing to Mesosfer

Thank you for your interest in contributing to **Mesosfer**!

## Development Setup

Mesosfer uses [`uv`](https://docs.astral.sh/uv/) and Python `3.12–3.13`:

```bash
# CPU-only (sufficient for unit tests and tokenizer/data work)
uv sync --extra cpu --group dev

# NVIDIA CUDA 12.8
uv sync --extra cuda --group dev

# AMD ROCm 7.2
uv sync --extra rocm --group dev
```

## Workflow

1. **Fork** the repository and create a descriptive branch from `main`:
   - `feat/<short-name>` for new features
   - `fix/<short-name>` for bug fixes
   - `docs/<short-name>` for documentation updates
2. Keep pull requests focused and minimal — one logical change per PR.
3. Run the unit test suite before pushing:
   ```bash
   uv run pytest -m "not slow"
   ```
4. Open a Pull Request targeting `main` and complete the PR checklist.

## Guidelines

- **Language:** Write all code, comments, commit messages, and documentation in English.
- **No secrets or large artifacts:** Never commit `.env` files, API keys, `HF_TOKEN`, model checkpoints (`.pt`), or raw dataset shards.
- **Keep docs in sync:** If you change model defaults, special tokens, or dataset sources, update `README.md` and `DATASET.md` in the same PR.
- **Defensive security posture:** Tool handlers and sandbox utilities must never auto-execute untrusted shell/network commands by default. See [SECURITY.md](SECURITY.md).
- **Tests:** Add or update unit tests under `tests/` for any non-trivial behavior change or bug fix.

## License

By contributing to Mesosfer, you agree that your contributions will be licensed under the [MIT License](LICENSE).
