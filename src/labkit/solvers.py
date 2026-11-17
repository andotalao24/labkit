"""The `agent` solver: one switch each for harness, tools and search.

Use it in our tasks, or swap it into any Inspect / inspect_evals task from the CLI:
  inspect eval inspect_evals/gaia --solver labkit/agent -S harness=react -S tools=sandbox -S search=off

harness
  plain     a single model call, no tools (pure prompting baseline)
  loop      the model's native tool-calling loop: generate until it stops calling tools
  react     Inspect's ReAct agent: tool loop + explicit submit() tool, continues if the model
            stops without submitting (the default for agentic benchmarks)

tools
  none      no environment tools
  sim       simulated tools (ToolShield style): in-memory files, canned bash/SQL output, nothing runs
  sandbox   real bash + python executed in the task's sandbox (use sandbox="apptainer")

search      see labkit.tools.search.SEARCH_MODES (off | native | tavily | controlled | replay)
"""
from __future__ import annotations

from inspect_ai.agent import as_solver, react
from inspect_ai.solver import Solver, chain, generate, solver, system_message, use_tools
from inspect_ai.tool import Tool, bash, python

from labkit.tools.search import SEARCH_MODES, search_tools
from labkit.tools.simulated import seed_vfs, simulated_tools

HARNESSES = ("plain", "loop", "react")
TOOLSETS = ("none", "sim", "sandbox")


def tool_set(tools: str = "none", timeout: int = 180, sql: bool = False) -> list[Tool]:
    if tools == "none":
        return []
    if tools == "sim":
        return simulated_tools(sql=sql)
    if tools == "sandbox":
        return [bash(timeout=timeout), python(timeout=timeout)]
    raise ValueError(f"tools must be one of {TOOLSETS}, got {tools!r}")


@solver
def agent(
    harness: str = "react",
    tools: str = "none",
    search: str = "off",
    system_prompt: str | None = None,
    tool_timeout: int = 180,
    max_queries: int = 10,
    blocklist: list[str] | None = None,
    sql: bool = False,
) -> Solver:
    if harness not in HARNESSES:
        raise ValueError(f"harness must be one of {HARNESSES}, got {harness!r}")
    if search in (False, None):  # `-T search=off` arrives as YAML boolean False
        search = "off"
    if search not in SEARCH_MODES:
        raise ValueError(f"search must be one of {SEARCH_MODES}, got {search!r}")

    search_kwargs = {"max_queries": max_queries, "blocklist": blocklist} if search in ("controlled", "replay") else {}
    all_tools = tool_set(tools, tool_timeout, sql) + search_tools(search, **search_kwargs)

    steps: list[Solver] = []
    if tools == "sim":
        steps.append(seed_vfs())
    if system_prompt and harness != "react":
        steps.append(system_message(system_prompt))

    if harness == "plain":
        if all_tools:
            raise ValueError("harness=plain takes no tools; use tools=none search=off, or harness=loop")
        steps.append(generate())
    elif harness == "loop":
        steps += [use_tools(all_tools), generate(tool_calls="loop")]
    else:
        kwargs = {"prompt": system_prompt} if system_prompt else {}
        steps.append(as_solver(react(tools=all_tools, **kwargs)))
    return chain(steps)
