"""Extraction units with the Codex CLI harness (OpenAI's coding agent, via inspect_swe).

  inspect eval tasks/extraction_codex.py --model openai/gpt-6-astra --reasoning-effort high \
      -T unit=gan2025_star_ps_p2vp/T2_gisans_4arm_depth --log-dir logs/extraction_codex

Same input as tasks/extraction.py: task.md text plus image1..N (Codex receives the images with
`codex exec --image`). Codex runs inside an Apptainer instance and its model calls are bridged back
through Inspect, so every request/response is in the log. Ground truth never enters the sandbox.
Codex settings are its defaults except the ones below.

NO WEB SEARCH, enforced independently:
  1. codex_cli(web_search="disabled"): Codex does not offer its web_search tool to the model.
  2. no_search_guard filter: every bridged model request is checked; if any tool whose name looks
     like search/web/browse/fetch is offered, the sample fails before the request is sent.
  3. Apptainer instance with --net --network none: shell/python inside Codex cannot reach the
     internet (curl, pip, urllib all fail). Model calls still work because the bridge runs over
     sandbox exec, not the network.
  4. scripts/verify_no_tools.py --agent <log>: audits the raw API traffic afterwards.
(Codex's own network_access setting only applies in its workspace-write mode, which needs bwrap;
under Codex's default danger-full-access mode layer 3 is what blocks the network.)
"""
from __future__ import annotations

import re

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig, Model
from inspect_swe import codex_cli

from extraction import _names, assert_no_tools, load_unit
from labkit.sandbox.apptainer import ApptainerConfig

TAG = "codex"
SEARCH_LIKE = re.compile(r"search|web|brows|fetch|http|url|internet", re.I)


async def no_search_guard(model: Model, messages, tools, tool_choice, config: GenerateConfig):
    bad = [t.name for t in tools if SEARCH_LIKE.search(t.name or "")]
    if bad:
        raise RuntimeError(f"search-like tool offered to the model, aborting: {bad}")
    return None  # pass the request through unchanged


@task
def extraction_codex(
    unit: str = "gan2025_star_ps_p2vp/T2_gisans_4arm_depth",
    judge: str | None = "openai/gpt-6.1-sol",
    judge_effort: str = "high",
    reasoning_effort: str = "high",
    time_limit: int = 3600,
    drop_sections: str | list | None = None,
) -> Task:
    """drop_sections: task.md sections to leave out of the prompt, e.g. -T drop_sections=Images."""
    units = [u.strip() for u in unit.split(",") if u.strip()]
    scorer = None
    if judge and str(judge).lower() != "none":
        from labkit.judge import gt_judge

        scorer = gt_judge(judge_model=judge, reasoning_effort=judge_effort,
                          tag=TAG + "".join(f"-no{n.strip()}" for n in _names(drop_sections)))
    agent = codex_cli(
        web_search="disabled",
        filter=no_search_guard,
        network_access=False,
        config_overrides={"model_reasoning_effort": f'"{reasoning_effort}"'},
    )
    return Task(
        dataset=[load_unit(u, drop_sections) for u in units],
        solver=[assert_no_tools(), agent],
        scorer=scorer,
        sandbox=("apptainer", ApptainerConfig(network=False)),
        time_limit=time_limit,
    )
