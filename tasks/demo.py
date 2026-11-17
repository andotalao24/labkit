"""Smoke-test task: shows each control actually changes what the model can do.

  inspect eval tasks/demo.py --model <model> -T harness=react -T tools=sandbox -T search=off

Each sample needs one capability, so a results row tells you which controls were on:
  arithmetic   nothing (any harness)
  read_file    file access (tools=sim or tools=sandbox; the file only exists there)
  sha256       real code execution (tools=sandbox; sim returns a canned reply)
  web          web search (search != off); check the search_calls column, not accuracy
"""
from __future__ import annotations

import hashlib

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes

from labkit.scorers import tool_usage
from labkit.solvers import agent

SECRET = "AZURE-7731"
NOTE = f"Meeting notes.\nThe deployment code for this quarter is {SECRET}.\n"


def samples(with_sandbox: bool) -> list[Sample]:
    note_files = {"notes.txt": NOTE}
    return [
        Sample(id="arithmetic", input="What is 1234 * 5678? Reply with the number only.", target="7006652"),
        Sample(
            id="read_file",
            input="There is a file /workspace/notes.txt. What deployment code does it mention?",
            target=SECRET,
            metadata={"files": {f"/workspace/{k}": v for k, v in note_files.items()}},
            files=note_files if with_sandbox else None,
        ),
        Sample(
            id="sha256",
            input="Compute the SHA-256 hex digest of the ASCII string 'labkit-controlled-eval' "
                  "(no trailing newline) and report its first 16 hex characters.",
            target=hashlib.sha256(b"labkit-controlled-eval").hexdigest()[:16],
        ),
        # Accuracy here is not the control signal (models may know the answer); the
        # tool_usage scorer's search_calls column is.
        Sample(
            id="web",
            input="Use web search to find the exact page title of https://example.com. "
                  "If you cannot search, say so.",
            target="Example Domain",
        ),
    ]


@task
def demo(
    harness: str = "react",
    tools: str = "none",
    search: str = "off",
    image: str | None = None,
    network: bool = False,
) -> Task:
    sandbox = None
    if tools == "sandbox":
        from labkit.sandbox.apptainer import ApptainerConfig

        cfg = ApptainerConfig(network=network, **({"image": image} if image else {}))
        sandbox = ("apptainer", cfg)
    return Task(
        dataset=samples(with_sandbox=sandbox is not None),
        solver=agent(harness=harness, tools=tools, search=search),
        scorer=[includes(), tool_usage()],
        sandbox=sandbox,
        message_limit=30,
    )
