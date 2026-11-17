"""Checks the Apptainer sandbox's guarantees without calling any model.

  inspect eval tasks/sandbox_check.py --model mockllm/model            # network off (default)
  inspect eval tasks/sandbox_check.py --model mockllm/model -T network=true

Every check must score C. Checks run inside the sample's sandbox:
  no_egress      HTTPS to the internet fails            (skipped meaning-wise when network=true: expects success)
  no_host_env    host env vars (API keys, SLURM_*) are not visible
  no_host_fs     /projects and the host $HOME are not mounted
  files_io       write_file/read_file round-trip, incl. binary, and Sample.files landed in /workspace
  state_persists a file written by one exec is visible to the next
"""
from __future__ import annotations

import os

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.util import sandbox

from labkit.sandbox.apptainer import ApptainerConfig

FETCH = "import urllib.request as u; print(u.urlopen('https://example.com', timeout=5).status)"


@solver
def run_check():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        sb = sandbox()
        check = state.sample_id
        ok, detail = False, ""
        if check == "no_egress":
            r = await sb.exec(["python3", "-c", FETCH], timeout=30)
            reached = r.success and "200" in r.stdout
            want_net = state.metadata["network"]
            ok = reached == want_net
            detail = f"reached={reached} network={want_net} stderr={r.stderr[-200:]}"
        elif check == "no_host_env":
            r = await sb.exec(["env"])
            leaked = [k for k in os.environ if k.endswith(("_API_KEY", "_TOKEN")) or k.startswith("SLURM_")]
            leaked = [k for k in leaked if f"{k}=" in r.stdout]
            # proxy variables must be cleared (Apptainer passes them through --cleanenv)
            leaked += [ln.split("=")[0] for ln in r.stdout.splitlines()
                       if ln.lower().split("=")[0] in ("http_proxy", "https_proxy", "all_proxy")
                       and ln.split("=", 1)[1].strip()]
            ok, detail = not leaked, f"leaked={leaked}"
        elif check == "no_host_fs":
            home = os.path.expanduser("~")
            r = await sb.exec(["sh", "-c", f"ls /projects 2>/dev/null; ls {home} 2>/dev/null"])
            ok, detail = r.stdout.strip() == "", r.stdout[:200]
        elif check == "files_io":
            await sb.write_file("/tmp/sub/b.bin", bytes(range(256)))
            back = await sb.read_file("/tmp/sub/b.bin", text=False)
            seeded = await sb.read_file("seed.txt")
            ok = back == bytes(range(256)) and seeded == "seeded\n"
            detail = f"binary_ok={back == bytes(range(256))} seeded={seeded!r}"
        elif check == "localhost":
            # agent bridges (Codex, Claude Code) talk to an in-sandbox proxy on localhost
            code = ("import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); s.listen(); p=s.getsockname()[1];"
                    "socket.create_connection(('localhost',p),timeout=3); print('ok')")
            r = await sb.exec(["python3", "-c", code], timeout=30)
            ok, detail = r.stdout.strip() == "ok", (r.stdout + r.stderr)[-200:]
        elif check == "state_persists":
            await sb.exec(["sh", "-c", "echo kept > /opt/state.txt"])
            r = await sb.exec(["cat", "/opt/state.txt"])
            ok, detail = r.stdout.strip() == "kept", r.stdout + r.stderr
        state.metadata["ok"], state.metadata["detail"] = ok, detail
        state.output.completion = detail
        return state

    return solve


@scorer(metrics=[accuracy()])
def passed():
    async def score(state: TaskState, target: Target) -> Score:
        return Score(value=CORRECT if state.metadata["ok"] else INCORRECT, explanation=state.metadata["detail"])

    return score


@task
def sandbox_check(network: bool = False, image: str | None = None) -> Task:
    cfg = ApptainerConfig(network=network, **({"image": image} if image else {}))
    ids = ["no_egress", "no_host_env", "no_host_fs", "files_io", "state_persists", "localhost"]
    return Task(
        dataset=[Sample(id=i, input=i, metadata={"network": network}, files={"seed.txt": "seeded\n"}) for i in ids],
        solver=run_check(),
        scorer=passed(),
        sandbox=("apptainer", cfg),
    )
