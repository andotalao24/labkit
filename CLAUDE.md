# labkit

Inspect AI eval harness for controlled agent benchmarks (see README.md for the full guide).

- Python env: `.venv` (`pip install -e .`). Run evals with `inspect eval ...` from the repo root, on a
  machine that can start Apptainer instances (not a cluster login node).
- Sandbox: `src/labkit/sandbox/apptainer.py` (Inspect sandbox `apptainer`), default image
  `images/python-3.12-slim.sif`. Check it with `inspect eval tasks/sandbox_check.py --model mockllm/model`.
- Subscription runs (Codex CLI / Claude Code on the user's own login, no API key):
  `tasks/extraction_sub.py --model none -T agent=codex|claude`. Smoke test first with
  `tasks/subscription_check.py`. Audit with `scripts/verify_subscription.py`.
- Benchmark units live in `extractions/<paper>/<unit>/` (not in git). `analysis.json` and
  `conclusion.json` are ground truth: never put them in a prompt, the sandbox, or task metadata.
- The tested model must never be able to search the web. Keep every layer: no search tools offered,
  sandbox `--network none` (subscription runs: only the `labkit/egress.py` allowlist), and run the
  verify script on every log.
- Never commit `.env`, `logs/`, `extractions/`, `images/` or credentials.
