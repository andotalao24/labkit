# labkit: controlled agent evals on Inspect AI

Evaluate open and closed models across harnesses (a plain API call, Inspect agents, Codex CLI,
Claude Code) with explicit control over tools and web search. Built for clusters without Docker:
real execution runs in Apptainer.

You can run Codex CLI and Claude Code **on your own subscription login** (ChatGPT plan / Claude
plan) instead of API keys, inside a sandbox that has no internet except the CLI's own model backend.

```
src/labkit/
  solvers.py            agent(harness, tools, search): one solver with all the knobs
  subscription.py       run Codex CLI / Claude Code on a subscription login (no API key)
  egress.py             allowlist proxy: the only way out of an offline sandbox
  judge.py              ground-truth judge for extraction units (point-by-point verdicts)
  tools/search.py       search modes: off | native | tavily | controlled | replay
  tools/simulated.py    simulated files / bash / SQL tools (nothing executes)
  sandbox/apptainer.py  Inspect sandbox "apptainer" (network off, no host env / fs)
  scorers.py            tool_usage(): per-sample tool / search call counts
tasks/
  extraction.py         benchmark units, plain API call, no tools
  extraction_codex.py   benchmark units, Codex CLI via Inspect's bridge (API key)
  extraction_sub.py     benchmark units, Codex CLI or Claude Code on a subscription
  subscription_check.py smoke test: subscription login works, sandbox has no internet
  followup.py           ask a follow-up question in the conversation of an earlier run
  demo.py               smoke test; each sample needs one capability
  sandbox_check.py      verifies the sandbox guarantees, no model needed
scripts/
  verify_no_tools.py      audit an API-run log: no tools / no search in the raw requests
  verify_subscription.py  audit a subscription run: no search tool, no egress beyond the model host
  convo_html.py           render a log's conversation as HTML
  eval_local.sbatch       vLLM-serve an open model on a GPU node and run an eval (Slurm)
  matrix.sh               models x harness x tools x search grid
```

## Setup

Requirements: Linux, Python 3.10+, [Apptainer](https://apptainer.org) (or SingularityCE as `apptainer`).

```bash
git clone https://github.com/andotalao24/labkit && cd labkit
python -m venv .venv && source .venv/bin/activate
pip install -e .
mkdir -p images && apptainer pull images/python-3.12-slim.sif docker://python:3.12-slim   # default sandbox image
cp .env.example .env               # API keys you need (judge, API-model runs); Inspect reads .env
inspect eval tasks/sandbox_check.py --model mockllm/model   # all checks should be C
```

Run evals on a machine where you can start Apptainer instances (on a cluster, a compute node via
`srun`/`sbatch`, not the login node). Leftover instances: `inspect sandbox cleanup apptainer`.

## Benchmark units

The extraction tasks read units from `extractions/<paper>/<unit>/`. The benchmark data itself is
not in this repo. A unit folder holds:

| File | Sent to the model? | Content |
|---|---|---|
| `task.md` | yes | the prompt (sections: title, Images, Sample context, Experimental context, Known inputs, Task, Output format) |
| `image1.png`, `image2.png`, ... | yes | the images, sent in numeric order, each preceded by its file name |
| `analysis.json`, `conclusion.json` | **never** | ground truth, read only by the judge after the answer is final |

The model gets one user message: the `task.md` text, then `image1.png:` + image, `image2.png:` +
image, and so on. The repo adds no system prompt. `-T drop_sections=Images` leaves a `## ` section
out of the prompt (ablation). Select units with `-T unit=<paper>/<unit>[,<paper>/<unit>...]`.

## Run Codex CLI / Claude Code on your subscription

1. Install and log in on the machine that runs the eval, with your plan (not an API key):
   - Codex: `npm i -g @openai/codex`, then `codex login` and choose ChatGPT sign-in
     (writes `~/.codex/auth.json`).
   - Claude Code: install from <https://docs.claude.com/en/docs/claude-code>, run `claude` and `/login`
     with your Claude account (writes `~/.claude/.credentials.json`).
2. Smoke test (a few cheap requests on your plan):
   ```bash
   inspect eval tasks/subscription_check.py --model none -T agent=claude --log-dir logs/checks
   inspect eval tasks/subscription_check.py --model none -T agent=codex  --log-dir logs/checks
   ```
   `C` means the CLI answered on your login, its shell could not reach the internet, and the
   no-search audit was clean.
3. Run units:
   ```bash
   inspect eval tasks/extraction_sub.py --model none -T agent=claude -T effort=high \
       -T unit=<paper>/<unit> --log-dir logs/extraction_sub
   inspect eval tasks/extraction_sub.py --model none -T agent=codex -T effort=high \
       -T unit=<paper>/<unit> --log-dir logs/extraction_sub
   .venv/bin/python scripts/verify_subscription.py logs/extraction_sub/<file>.eval
   ```
   - `--model none`: the tested model is not an Inspect model. The CLI talks to its own backend
     with your login.
   - `-T model=...` picks the model (e.g. `claude-opus-5-5`, `gpt-6.2-sol`). Omit it for the CLI's
     default. The model actually used is recorded in each sample's metadata.
   - The judge (`-T judge=openai/gpt-6.1-sol` by default) uses an API key from `.env`.
     `-T judge=none` skips it. Re-score later with `inspect score` (see `src/labkit/judge.py`).
   - The CLI binaries are bound read-only from your install. Set `LABKIT_CLAUDE_BIN` /
     `LABKIT_CODEX_BIN` if they are not found on `PATH`.

How it works (`src/labkit/subscription.py`):
- The CLI runs headless once per sample in an Apptainer instance started with `--contain --cleanenv
  --no-mount bind-paths,cwd --net --network none`. There are no host env vars (API keys), no host
  `$HOME` and no project dirs in it. Only `task.md` and the images are copied to its `/workspace`.
  Your login file is copied into the sandbox's own home.
- The only exit is `src/labkit/egress.py`, a host-side proxy reached through a bind-mounted Unix
  socket. It accepts `CONNECT` only to the CLI's own model and login hosts (`ALLOW`: `chatgpt.com`,
  `auth.openai.com` / `api.anthropic.com`, `platform.claude.com`, `console.anthropic.com`).
  Everything else (curl, pip, python from the agent's shell) is refused, and every attempt is logged.
- Web search is a server-side tool, so the network layer cannot block it. It is switched off in the
  CLI: Codex gets `-c web_search="disabled"`. Claude Code gets an explicit `--tools` list without
  WebSearch/WebFetch, plus `--disallowedTools WebSearch,WebFetch` and no MCP servers.
- After each sample, the CLI's own event stream (and Codex's rollout files) is audited for any
  search tool offered or used. The sample fails if one is found.
- Saved per sample in `logs/subscription/runs/`: CLI event stream, stderr, egress log, prompt,
  Codex rollout files, audit and answer. Credentials are never copied there.
- If the CLI refreshes your OAuth token inside the sandbox, the newer token is written back to your
  host login file so the host login keeps working. Samples run one at a time
  (`max_samples=1`) because parallel copies of one login can race on token refresh.
- Harness differences: Claude Code gets the same message as the plain run (text + inline images).
  Codex gets the text with the images attached via `--image`, plus one line listing their order.
  Both CLIs bring their own built-in system prompts, which this repo does not set.

Check that your plan's terms allow automated use before running large batches.

## API-key runs and the knobs

| Knob | Values | Meaning |
|---|---|---|
| model | `anthropic/…`, `openai/…`, `google/…`, `openrouter/…`, `openai-api/local/<tag>` (vLLM) | `--model`; comma-separate for several |
| `harness` | `plain` | one model call, no tools |
| | `loop` | the model's native tool-calling loop |
| | `react` | Inspect ReAct agent (tool loop + `submit()`) |
| `tools` | `none` | no environment tools |
| | `sim` | simulated files / bash / SQL, nothing runs |
| | `sandbox` | real `bash` + `python` in an Apptainer instance |
| `search` | `off` | no search tool |
| | `native` | provider's own server-side search (differs per vendor) |
| | `tavily` | Inspect's stock Tavily search (same backend for all models) |
| | `controlled` | our Tavily tool: domain blocklist, per-sample query budget, disk cache |
| | `replay` | `controlled` answered only from the cache: offline, identical for every model |

```bash
inspect eval tasks/extraction.py --model openai/<model> --reasoning-effort high -T unit=<paper>/<unit>
.venv/bin/python scripts/verify_no_tools.py logs/<file>.eval
inspect eval tasks/extraction_codex.py --model openai/<model> -T unit=<paper>/<unit>   # Codex via API key
.venv/bin/python scripts/verify_no_tools.py --agent logs/<file>.eval
```

On our tasks the knobs are task args (`-T harness=react -T tools=sandbox -T search=off`).
On any other Inspect / inspect_evals task, swap in our solver:

```bash
inspect eval inspect_evals/gaia --model openai/<model> \
  --solver labkit/agent -S harness=react -S tools=sandbox -S search=controlled \
  --sandbox apptainer
```

`--sandbox apptainer:<config>` takes a `.sif`, a `docker://` image, an existing benchmark's
`compose.yaml` (its `image:` is pulled; `build:` needs a SIF built by hand), or a yaml
`{image, network, binds}`.

```bash
# API models, full grid
MODELS="anthropic/claude-opus-5-5,openai/<model>" HARNESSES="plain react" \
  TOOLS="none sandbox" SEARCH="off controlled" scripts/matrix.sh tasks/demo.py

# open model on a Slurm GPU node (needs vLLM; set VLLM_BIN if it is not on PATH)
MODEL=Qwen/Qwen3.5-4B TAG=qwen35_4b ARGS="-T harness=react -T tools=sandbox -T search=off" \
  sbatch --account=<account> scripts/eval_local.sbatch

inspect view --log-dir logs        # browse transcripts, tool calls, scores
```

## Why search is actually controlled

1. **Tool layer.** The model only has tools from `agent(...)`. Native provider search is only
   enabled with `search=native`. The `tool_usage` scorer records `search_calls` per sample.
2. **Network layer.** Sandbox instances start with `--net --network none`, so `curl`/`pip`/`urllib`
   from bash or python cannot reach the internet (`tasks/sandbox_check.py` verifies this).
   Subscription runs add only the allowlist proxy described above.
3. **Sandbox isolation.** `--cleanenv` keeps host env vars (API keys) out of the container, which
   matters because Apptainer passes them through by default. `--contain --no-mount bind-paths,cwd`
   keeps `$HOME`, project dirs and site binds out.
4. **Contamination.** `controlled`/`replay` filter blocklisted domains (`DEFAULT_BLOCKLIST` in
   `tools/search.py`, or `-S blocklist=[...]`) both at query time and when reading the cache.
5. **Audit.** `scripts/verify_no_tools.py` (API runs) and `scripts/verify_subscription.py`
   (subscription runs) check every saved run afterwards.

For reproducible search: run once with `search=controlled` to fill `data/search_cache/`, then
use `search=replay` for every model. All models then see byte-identical results with no network.

## Notes

- Sandbox processes run as your user (no root). Images needing root setup should be built as a SIF
  beforehand (`apptainer build x.sif docker://...` or a `.def` file).
- Apptainer passes `http(s)_proxy` into containers even with `--cleanenv`; the sandbox clears them
  in every exec.
- The in-image writable layer is disk-backed in `LABKIT_SANDBOX_TMP` (default `/tmp`). Use
  `/workspace` (a per-sample scratch dir, deleted afterwards) for anything big.
- vLLM tool-call parsers in `eval_local.sbatch` are per-family defaults. If a model's tool calls come
  back as plain text, set `TOOL_PARSER=...`.
- Inspect's `bash_session` / `text_editor` tools need Inspect's tool-support binary inside the
  image; the sandbox toolset uses `bash` + `python`, which only need bash/python3 in the image.
