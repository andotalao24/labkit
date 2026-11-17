"""Extraction units with Codex CLI or Claude Code on a SUBSCRIPTION login (no API keys).

  # Codex on your ChatGPT plan (host login: `codex login`)
  inspect eval tasks/extraction_sub.py --model none -T agent=codex -T model=gpt-6.1-sol -T effort=high \
      -T unit=gan2025_star_ps_p2vp/T2_gisans_4arm_depth --log-dir logs/extraction_sub

  # Claude Code on your Claude plan (host login: `claude` -> /login)
  inspect eval tasks/extraction_sub.py --model none -T agent=claude -T model=claude-opus-5-5 -T effort=high \
      -T unit=gan2025_star_ps_p2vp/T2_gisans_4arm_depth --log-dir logs/extraction_sub

`--model none`: the tested model is not an Inspect model; the CLI talks to its own backend with
your login (labkit.subscription). -T model / -T effort: omit for the CLI's defaults.
The judge (-T judge=..., default openai/gpt-6.1-sol) still uses an API key from .env; -T judge=none
skips it (re-score later with `inspect score`, see labkit/judge.py).

Same input as tasks/extraction.py: task.md text plus image1..N. Claude Code gets exactly the plain
message (text + images inline); Codex gets the text with the images attached via --image. The same
files are also in the agent's /workspace. Ground truth never enters the sandbox.

NO WEB SEARCH / NO INTERNET:
  1. Apptainer instance, --net --network none: the agent's shell/python has no network.
  2. Only exit: an allowlist proxy (labkit.egress.ALLOW) to the CLI's own model + login hosts;
     everything else is refused and logged (egress.jsonl).
  3. Search tools are off in the CLI: Codex web_search="disabled"; Claude Code --tools without
     WebSearch/WebFetch, --disallowedTools WebSearch,WebFetch, no MCP servers.
  4. The CLI's own event stream (+ Codex rollout files) is audited; the sample fails on any search.
     Re-check saved runs: .venv/bin/python scripts/verify_subscription.py logs/extraction_sub/<f>.eval

Run samples one at a time (default --max-samples 1 below): parallel copies of one OAuth login can
race on refresh-token rotation.
"""
from __future__ import annotations

from inspect_ai import Task, task

from extraction import _names, assert_no_tools, load_unit
from labkit.subscription import subscription_agent, subscription_sandbox


@task
def extraction_sub(
    unit: str = "gan2025_star_ps_p2vp/T2_gisans_4arm_depth",
    agent: str = "codex",
    model: str | None = None,
    effort: str | None = "high",
    judge: str | None = "openai/gpt-6.1-sol",
    judge_effort: str = "high",
    time_limit: int = 3600,
    drop_sections: str | list | None = None,
    image: str | None = None,
) -> Task:
    """agent: codex | claude. image: sandbox .sif (default images/python-3.12-slim.sif)."""
    units = [u.strip() for u in unit.split(",") if u.strip()]
    model = None if model in (None, "", "default", "none") else str(model)
    effort = None if effort in (None, "", "default", "none", False) else str(effort)
    scorer = None
    if judge and str(judge).lower() != "none":
        from labkit.judge import gt_judge

        scorer = gt_judge(judge_model=judge, reasoning_effort=judge_effort,
                          tag=f"{agent}-sub" + "".join(f"-no{n.strip()}" for n in _names(drop_sections)))
    return Task(
        dataset=[load_unit(u, drop_sections) for u in units],
        solver=[assert_no_tools(), subscription_agent(agent=agent, model=model, effort=effort,
                                                      time_limit=time_limit)],
        scorer=scorer,
        sandbox=subscription_sandbox(agent, image),
        time_limit=time_limit + 300,
        max_samples=1,
    )
