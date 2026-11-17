"""Smoke test for subscription runs: the CLI works on the login, and the sandbox has no internet.

  inspect eval tasks/subscription_check.py --model none -T agent=codex  --log-dir logs/checks
  inspect eval tasks/subscription_check.py --model none -T agent=claude --log-dir logs/checks

The agent is asked to try to reach the internet from its shell; the scorer passes (C) when it got an
answer, the no-search audit is clean, and no connection went anywhere but the model/login hosts.
"""
from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Target, accuracy, scorer
from inspect_ai.solver import TaskState

from labkit.subscription import subscription_agent, subscription_sandbox

PROMPT = """This is a sandbox connectivity check. In your shell, run exactly these three commands and
report each command's output verbatim (they are expected to fail):

1. curl -sS -m 10 -o /dev/null -w '%{http_code}\\n' https://example.com
2. python3 -c "import urllib.request; print(urllib.request.urlopen('https://pypi.org', timeout=10).status)"
3. env -u HTTPS_PROXY -u https_proxy curl -sS -m 10 -o /dev/null -w '%{http_code}\\n' https://example.com

Do not try anything else. End your reply with the line: CHECK DONE"""


@scorer(metrics=[accuracy()])
def offline_ok():
    async def score(state: TaskState, target: Target) -> Score:
        md = state.metadata
        audit = md.get("no_search_audit") or {}
        used = audit.get("tools_used") or audit.get("items_and_tools") or []
        ran_shell = any(u in ("Bash", "exec", "command_execution", "local_shell_call", "shell") for u in used)
        ok = bool(audit.get("ok")) and ran_shell and "CHECK DONE" in state.output.completion
        return Score(value="C" if ok else "I", answer=state.output.completion[-600:],
                     explanation=f"egress allowed={audit.get('egress_allowed')} denied={audit.get('egress_denied')} "
                                 f"problems={audit.get('problems')} shell_ran={ran_shell} used={used} model={md.get('answer_model')}")

    return score


@task
def subscription_check(agent: str = "codex", model: str | None = None, effort: str | None = "low") -> Task:
    return Task(
        dataset=[Sample(id=f"offline_{agent}", input=PROMPT)],
        solver=subscription_agent(agent=agent, model=model, effort=effort, time_limit=600),
        scorer=offline_ok(),
        sandbox=subscription_sandbox(agent),
        max_samples=1,
    )
